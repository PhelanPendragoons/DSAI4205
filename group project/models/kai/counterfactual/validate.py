"""Offline integration check. Pass --tokenizer a local Qwen3 tokenizer snapshot."""
import argparse
import runpy
import io
import json
import os
import re
import tempfile
import zipfile
from pathlib import Path
from types import SimpleNamespace

parser = argparse.ArgumentParser()
parser.add_argument("--tokenizer", type=Path, required=True)
options = parser.parse_args()
os.environ.update(HF_HUB_OFFLINE="1", MPLBACKEND="Agg", TOKENIZERS_PARALLELISM="false")
import torch
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

torch.set_num_threads(2)
kai = Path(__file__).resolve().parents[1]
parts = runpy.run_path(str(kai / "build_notebook.py"))["recipes"]()
work = Path(tempfile.mkdtemp(prefix="dsai-counterfactual-check-"))
previous = {c["id"]: c["source"] for c in parts["rationale-distillation"] if c["cell_type"] == "code"}
cells = {c["id"]: c["source"] for c in parts["counterfactual"] if c["cell_type"] == "code"}
ns = {}

def run(key, code=None):
    print(key, flush=True)
    exec(compile(cells[key] if code is None else code, key, "exec"), ns)

os.environ["KAI_RUNS_ROOT"] = str(work / "kai-runs")
cfg = cells["config"].replace("SAVE_TO_DRIVE = True", "SAVE_TO_DRIVE = False")
cfg = cfg[:cfg.index("if not torch.cuda.is_available():")] + "USE_BF16=False\nMODEL_DTYPE=torch.float32\nIN_COLAB=False\n" + cfg[cfg.index("set_seed(SEED)"): ]
cfg = cfg.replace('TEACHER_CACHE_ROOT = ""', f"TEACHER_CACHE_ROOT = {str(work / 'reused-cache')!r}")
run("config", cfg)
ns["OUTPUT_ROOT"] = work / "runs"
ns["RUN_DIR"] = ns["OUTPUT_ROOT"] / ns["RUN_ID"]
ns["RUN_DIR"].mkdir(parents=True, exist_ok=True)
run("embedded-data")
run("data", previous["data"])
reference_split = ns["split_manifest"].copy()
full_train = ns["train_df"].copy()
train = ns["pd"].concat([
    full_train[full_train.Sentiment.eq(0) & full_train.text.str.contains(r"\b(hate|bad|worst|terrible)\b", regex=True, case=False)].head(2),
    full_train[full_train.Sentiment.eq(1) & full_train.text.str.contains(r"\b(love|good|best|great)\b", regex=True, case=False)].head(2),
    full_train[full_train.Sentiment.eq(2)].head(1), full_train[full_train.Sentiment.eq(3)].head(1)
]).reset_index(drop=True)
assert len(train) == 6
run("teacher-functions", previous["teacher-functions"])
# The previous notebook's unchanged request-key function produces these test cache entries.
prior_zip = work / "prior.zip"
with zipfile.ZipFile(prior_zip, "w") as archive:
    archive.write(ns["RUN_DIR"] / "split_manifest.csv", "split_manifest.csv")
    archive.writestr("experiment_protocol.json", json.dumps(dict(
        model_id=ns["MODEL_ID"], model_revision=options.tokenizer.name, split_hash=ns["SPLIT_HASH"],
        teacher_base_url=ns["TEACHER_BASE_URL"], teacher_body_example_without_text=ns["teacher_body"]("x", "a"),
        filter=dict(min_words=5, max_words=45, max_rationale_tokens=96), rationale_weight=.5, max_length=512)))
    for row in train.itertuples():
        for variant in ("a", "b"):
            key = ns["request_key"](row.text, variant)
            record = dict(status="ok", request_key=key, variant=variant, usage={}, parsed=dict(
                label=row.Sentiment, explanation="The words in this tweet support the stated sentiment.",
                evidence=[row.text[:10]], needs_missing_target=False))
            archive.writestr(f"teacher_cache/{key}.json", json.dumps(record))
ns["DISTILLATION_EXPORT"] = str(prior_zip)
run("import-export")
run("data")
assert reference_split.equals(ns["split_manifest"])
ns["train_df"], ns["val_df"] = train, ns["val_df"].groupby("Sentiment").head(1).reset_index(drop=True)
ns.update(TRAIN_BATCH_SIZE=1, EVAL_BATCH_SIZE=1, GRAD_ACCUMULATION=1, SMOKE_TEST=True,
          CF_MAX_PARENTS=4, SEEDS=[42])
run("teacher-functions")
assert not ns["ALLOW_TEACHER_REQUESTS"]
run("teacher-run")
assert ns["teacher_table"].accepted.all()
ns["model_info"] = lambda *a, **k: SimpleNamespace(sha=options.tokenizer.name)
tok_cell = cells["tokenization"].replace("AutoTokenizer.from_pretrained(MODEL_ID, revision=RESOLVED_REVISION)",
                                        f"AutoTokenizer.from_pretrained({str(options.tokenizer)!r}, local_files_only=True)")
run("tokenization", tok_cell)
# Use the real chat template and exercise real Qwen3 tokenization first, then shrink
# the tokenizer vocabulary and model together to keep the three-forward CPU check small.
compact = ns["tokenizer"].train_new_from_iterator(train.text.tolist() + [ns["SYSTEM_PROMPT"]], vocab_size=512)
compact_dir = work / "compact"
compact.save_pretrained(compact_dir)
run("tokenization", tok_cell.replace(str(options.tokenizer), str(compact_dir)))
run("multitask")
run("counterfactual-functions")
local_edit, verify = ns["local_edit"], ns["verification_reason"]
assert local_edit("I love this app", {"old_span": "love", "new_span": "hate"}) == ("I hate this app", "")
assert local_edit("love love", {"old_span": "love", "new_span": "hate"})[1]
assert local_edit("I love @brand", {"old_span": "@brand", "new_span": "@other"})[1]
assert verify(1, "I hate it", dict(source_label=1, edited_label=1)) == "wrong_polarity"
verdict = dict(source_label=1, edited_label=0, natural=True, same_context=True, clear_flip=True, evidence="missing")
assert verify(1, "I hate it", verdict) == "ungrounded_evidence"
by_text = train.set_index("text").Sentiment.to_dict()
calls = []
def mock_post(url, **kwargs):
    messages = kwargs["json"]["messages"]
    payload = json.loads(messages[1]["content"])
    if "tweet" in payload:
        original = payload["tweet"]
        label = int(by_text[original])
        choices = ("hate|bad|worst|terrible", "love|good|best|great")
        span = re.search(r"\b(" + choices[label] + r")\b", original, re.I).group()
        result = {"old_span": span, "new_span": "awesome" if label == 0 else "awful"}
    else:
        assert set(payload) == {"original_tweet", "edited_tweet"}
        original = payload["original_tweet"]
        label = int(by_text[original])
        result = dict(source_label=label, edited_label=1-label, natural=True, same_context=True,
                      clear_flip=True, evidence="awesome" if label == 0 else "awful")
    assert original in by_text and label in (0, 1)
    calls.append(original)
    return SimpleNamespace(status_code=200, json=lambda: dict(model="fixture", choices=[
        dict(finish_reason="stop", message=dict(content=json.dumps(result)))]))

client_cls = ns["CounterfactualTeacher"]
ns["CounterfactualTeacher"] = lambda cache: client_cls(cache, post=mock_post, key_getter=lambda: "fixture")
run("generate")
assert 0 < len(ns["accepted_cf"]) <= 4 and 0 < len(calls) <= 8
count = len(calls)
ns["ALLOW_CF_REQUESTS"] = False
run("generate")
assert len(calls) == count and ns["client"].calls == 0
blocked = set(ns["blocked_hashes"]) | {ns["text_group"](s) for s in ns["accepted_cf"].cf_text}
duplicates = ns["build_counterfactuals"](ns["parents"], ns["client"], blocked)
assert not duplicates.accepted.any() and duplicates.reason.eq("duplicate").any()
run("paired-training")
ds = ns["paired_datasets"]
assert len(ds[ns["AUGMENTED"]]) == len(ds[ns["CONTROL"]]) == 6
for i in range(6):
    assert ds[ns["AUGMENTED"]][i]["auxiliary"] == ds[ns["CONTROL"]][i]["auxiliary"]
    assert ds[ns["AUGMENTED"]][i]["labels"] == ds[ns["CONTROL"]][i]["labels"]
tok = ns["tokenizer"]
model_dir = work / "tiny"
tiny = Qwen3ForCausalLM(Qwen3Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=1,
    num_attention_heads=2, num_key_value_heads=1, head_dim=16, max_position_embeddings=1024,
    eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id, tie_word_embeddings=True))
tiny.save_pretrained(model_dir)
del tiny
ns.update(MODEL_ID=str(model_dir), RESOLVED_REVISION="main")
for name in ["metrics", "inference", "model-factory"]: run(name)
run("experiments", cells["experiments"].replace("fp16=not USE_BF16,", "fp16=False, use_cpu=True,")
    .replace("max_steps=2 if SMOKE_TEST", "max_steps=6 if SMOKE_TEST"))
assert len(ns["summaries"]) == 2 and ns["summaries"][0]["cf_rows"] == 0 < ns["summaries"][1]["cf_rows"]
from safetensors.torch import load_file
weights = [load_file(str(Path(s["adapter"]) / "adapter_model.safetensors")) for s in ns["summaries"]]
assert any(not torch.equal(weights[0][k], weights[1][k]) for k in weights[0])
ns["SMOKE_TEST"] = False
run("test", cells["test"].replace('test_df = read_snapshot("test")', 'test_df = read_snapshot("test").groupby("Sentiment").head(2)').replace("range(500)", "range(12)"))
comparison = ns["comparison"]
assert comparison.iloc[0].delta_macro_f1 == 0
assert abs(comparison.iloc[1].delta_macro_f1 - (comparison.iloc[1].test_macro_f1 - comparison.iloc[0].test_macro_f1)) < 1e-8
run("export")
with zipfile.ZipFile(ns["archive"]) as bundle:
    assert all(name in bundle.namelist() for name in ["comparison.csv", "paired_bootstrap.csv", "accepted_counterfactuals.csv", "cf_audit.csv", "split_manifest.csv"])
    assert any(name.startswith("counterfactual_cache/") for name in bundle.namelist())
print("PASS: previous export/cache, exact split, train-only editing, blinded verification, duplicate rejection, cache-only restart, matched training, distinct adapters, evaluation and export")
