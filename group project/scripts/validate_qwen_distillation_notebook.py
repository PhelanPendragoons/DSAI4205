"""CPU integration tests: mock teacher transport + real tokenizer + tiny Qwen3.

Never calls a paid teacher API. Fixture scores are not benchmark results.
Run in an environment containing notebook dependencies, nbformat and IPython:
python validate_qwen_distillation_notebook.py --work-dir /path/to/temporary/output
"""
import argparse
import ast
import copy
import json
import os
import faulthandler
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--work-dir", type=Path, required=True)
work = parser.parse_args().work_dir.resolve()
work.mkdir(parents=True, exist_ok=True)
faulthandler.enable()
os.environ.update(HF_HOME=str(work / "hf-cache"), MPLBACKEND="Agg", TOKENIZERS_PARALLELISM="false")
import nbformat
import numpy as np
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
torch.set_num_threads(2)
root = Path(__file__).resolve().parents[1]
nb = nbformat.read(root / "notebooks/Qwen3_0_6B_LoRA_Rationale_Distillation_Colab.ipynb", as_version=4)
nbformat.validate(nb)
sources = {c.id: c.source for c in nb.cells if c.cell_type == "code"}
for name, text in sources.items():
    if name != "install": ast.parse(text)
ns = {"__name__": "__validation__"}


def run(name, source=None):
    print("RUN", name, flush=True)
    exec(compile(source if source is not None else sources[name], name, "exec"), ns)


config = sources["config"].replace("SAVE_TO_DRIVE = True", "SAVE_TO_DRIVE = False")
start, end = config.index("if not torch.cuda.is_available():"), config.index("set_seed(SEED)")
config = config[:start] + "USE_BF16 = False\nMODEL_DTYPE = torch.float32\n" + config[end:]
config = config.replace('Path("/content/qwen_distillation_runs")', f"Path({str(work / 'runs')!r})")
run("config", config)
ns.update(SMOKE_TEST=True, GRAD_ACCUMULATION=1, TRAIN_BATCH_SIZE=1, EVAL_BATCH_SIZE=1, EPOCHS=1)
ns["CACHE_ROOT"] = ns["RUN_DIR"] / "teacher_cache_test"
run("embedded-data")
run("data")
full_train, full_val = ns["train_df"].copy(), ns["val_df"].copy()
manifest = ns["split_manifest"].copy()
baseline = nbformat.read(root / "notebooks/Qwen3_0_6B_LoRA_Sentiment_Colab.ipynb", as_version=4)
baseline_data = next(c.source for c in baseline.cells if c.id == "data")
exec(compile(baseline_data, "baseline-data", "exec"), ns)
assert manifest.equals(ns["split_manifest"]), "Distillation split must exactly equal original notebook split"
ns["SPLIT_MANIFEST_PATH"] = str(ns["RUN_DIR"] / "split_manifest.csv")
run("data")
assert manifest.equals(ns["split_manifest"])
ns["train_df"] = full_train.groupby("Sentiment", group_keys=False).head(2).reset_index(drop=True)
ns["val_df"] = full_val.groupby("Sentiment", group_keys=False).head(2).reset_index(drop=True)
run("teacher-functions")
calls = []
gold = ns["train_df"].set_index("text").Sentiment.to_dict()


def mock_post(url, **kwargs):
    body = kwargs["json"]
    assert set(json.loads(body["messages"][1]["content"])) == {"tweet"}
    text = json.loads(body["messages"][1]["content"])["tweet"]
    calls.append((url, body))
    parsed = {"label": int(gold[text]), "evidence": [text[:15]],
              "explanation": "The wording expresses an attitude supported by the quoted phrase in this tweet.",
              "needs_missing_target": False}
    payload = {"id": "mock-response", "model": "mock-teacher", "usage": {"total_tokens": 25},
               "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(parsed)}}]}
    class Response:
        status_code = 200
        def json(self): return payload
    return Response()


collect = ns["collect_teacher"]
ns["collect_teacher"] = lambda frame, cache: collect(frame, cache, post=mock_post, key_getter=lambda: "mock-not-a-secret")
run("teacher-run")
assert len(calls) == 16 and ns["teacher_table"].accepted.all()
ns["ALLOW_TEACHER_REQUESTS"] = False
def forbidden_key():
    raise AssertionError("Cache hit must not request an API key")
ns["collect_teacher"] = lambda frame, cache: collect(frame, cache, post=mock_post, key_getter=forbidden_key)
run("teacher-run")
assert len(calls) == 16
try:
    collect(ns["val_df"], work / "bad-cache", post=mock_post, key_getter=lambda: "unused")
    raise AssertionError("Validation data sent to teacher")
except AssertionError as exc:
    assert str(exc) != "Validation data sent to teacher"
try:
    collect(ns["train_df"], work / "missing-cache", post=mock_post, key_getter=lambda: "unused")
    raise AssertionError("Incomplete cache silently accepted")
except RuntimeError:
    pass
assert len(calls) == 16
mutated = copy.deepcopy(ns["teacher_records"])
mutated[0]["parsed"]["label"] = (mutated[0]["parsed"]["label"] + 1) % 4
mutated[2]["parsed"]["evidence"] = ["THIS QUOTE IS NOT IN THE TWEET"]
mutated[4]["parsed"]["needs_missing_target"] = True
mutated[6]["parsed"]["explanation"] = "bad"
audit = ns["filter_records"](ns["train_df"], mutated)
assert "unstable_label" in audit.iloc[0].rejection_reasons
assert "gold_disagreement" in audit.iloc[0].rejection_reasons
assert "ungrounded_quote" in audit.iloc[1].rejection_reasons
assert "missing_target" in audit.iloc[2].rejection_reasons
assert not audit.iloc[3].format_valid
assert audit.accepted.sum() == 4 and audit.format_valid.sum() == 7
for content, finish, status in [("", "stop", "invalid_json"), ("{}", "length", "unfinished")]:
    response = ns["parse_teacher_response"]({"choices": [{"finish_reason": finish, "message": {"content": content}}]})
    assert response["status"] == status
ns["teacher_table"] = audit
run("tokenization")
tok = ns["tokenizer"]
run("multitask")
assert all(len(ds) == 8 for ds in ns["paired_datasets"].values())
assert ns["paired_datasets"]["label_only"].aux_count == 0
assert ns["paired_datasets"]["rationale_unfiltered"].aux_count == 7
assert ns["paired_datasets"]["rationale_filtered"].aux_count == 4
# With uniform logits, each nonpadding target has CE log(V). Denominator is
# original batch size, not explanation count or explanation token count.
labels = torch.tensor([[-100, 1, 2, -100], [-100, 1, 2, 3]])
logits = torch.zeros(2, 4, 4, requires_grad=True)
loss = ns["rationale_mean_loss"](logits, labels, original_batch_size=4)
assert torch.isclose(loss, torch.tensor(np.log(4) / 2, dtype=torch.float32))
loss.backward()
assert torch.isfinite(logits.grad).all()
assert ns["encode_rationale"]("A tweet", "word " * 1000) is None
for ds in ns["paired_datasets"].values():
    batch = ns["collate_pairs"]([ds[i] for i in range(4)])
    assert batch["labels"].ne(-100).sum() == 8
    if "auxiliary" in batch:
        aux = batch["auxiliary"]
        assert torch.all(aux["labels"][aux["attention_mask"] == 0] == -100)
tiny_dir = work / "tiny-qwen3"
tiny = Qwen3ForCausalLM(Qwen3Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64,
    num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=16,
    max_position_embeddings=1024, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
    tie_word_embeddings=True))
tiny.save_pretrained(tiny_dir)
del tiny
ns["MODEL_ID"], ns["RESOLVED_REVISION"] = str(tiny_dir), "main"
run("metrics")
run("inference")
run("model-factory")
run("experiments", sources["experiments"].replace("fp16=not USE_BF16,", "fp16=False, use_cpu=True,"))
assert len(ns["summaries"]) == 3
assert all(s["optimizer_steps"] == 2 for s in ns["summaries"])
run("test")
assert not (ns["RUN_DIR"] / "comparison.csv").exists()
ns["SMOKE_TEST"] = False
run("test", sources["test"].replace('test_df = read_snapshot("test")', 'test_df = read_snapshot("test").groupby("Sentiment", group_keys=False).head(2).reset_index(drop=True)').replace("range(500)", "range(10)"))
ns["SMOKE_TEST"] = True
run("export")
assert Path(ns["archive"]).exists()
for summary in ns["summaries"]:
    adapter_file = Path(summary["adapter"]) / "adapter_model.safetensors"
    assert adapter_file.exists()
from safetensors.torch import load_file
weights = [load_file(str(Path(s["adapter"]) / "adapter_model.safetensors")) for s in ns["summaries"]]
assert any(not torch.equal(weights[0][key], weights[-1][key]) for key in weights[0])
summary = {"passed": True, "scope": "Mock teacher transport, real Qwen3 tokenizer, tiny random Qwen3 CPU training; no paid API or full pretrained/GPU experiment",
    "checks": ["notebook schema/syntax", "exact baseline split and manifest import", "teacher never receives gold labels or held-out rows",
        "cache reuse without API key", "incomplete-cache failure", "JSON and truncation handling", "agreement/quote/target/length filters",
        "all original rows retained", "per-example token-normalized auxiliary loss", "padding masks and token limits",
        "three actual Trainer runs", "identical initial adapters and optimizer steps", "auxiliary supervision changes learned weights",
        "classifier inference alignment", "adapter reload", "smoke skips test", "comparison/bootstrap/ZIP on fixture"],
    "split_counts": {"train": len(full_train), "validation": len(full_val), "test": 397}, "versions": ns["VERSIONS"]}
(work / "validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2), flush=True)
