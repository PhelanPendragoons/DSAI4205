"""Offline CPU integration check with tiny random Qwen3 models and synthetic caches.

Run: python models/kai/embedding-mlp/validate.py
No downloaded weights, teacher API, or benchmark claims are involved.
"""
import ast
import copy
import json
import os
import runpy
import tempfile
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.update(HF_HUB_OFFLINE="1", MPLBACKEND="Agg", TOKENIZERS_PARALLELISM="false")
import joblib
import nbformat
import numpy as np
import pandas as pd
import requests
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM, Qwen3Model

torch.set_num_threads(2)
kai = Path(__file__).resolve().parents[1]
project = kai.parents[1]
work = Path(tempfile.mkdtemp(prefix="kai-embedding-check-"))
os.environ["KAI_RUNS_ROOT"] = str(work / "runs")
builder = runpy.run_path(str(kai / "build_notebook.py"))
parts = builder["recipes"]()
prior = {c["id"]: c["source"] for c in parts["counterfactual"] if c["cell_type"] == "code"}
notebook = nbformat.read(project / "notebooks/kai/Kai_Sentiment_Experiments.ipynb", 4)
nbformat.validate(notebook)
for cell in notebook.cells:
    if cell.cell_type == "code":
        ast.parse(cell.source)
cells = {c.id: c.source for c in notebook.cells if c.cell_type == "code"}


def cpu_config(code):
    start = code.index("if not torch.cuda.is_available():")
    end = code.index("set_seed(SEED)", start)
    return code[:start] + "USE_BF16=False\nMODEL_DTYPE=torch.float32\nIN_COLAB=False\n" + code[end:]


fixture = {}
exec(cpu_config(prior["config"].replace("SAVE_TO_DRIVE = True", "SAVE_TO_DRIVE = False")), fixture)
exec(prior["embedded-data"], fixture)
fixture["prior_protocol"] = None
exec(prior["data"], fixture)
reference_manifest = fixture["split_manifest"].copy()
train = fixture["train_df"].groupby("Sentiment").head(2).reset_index(drop=True)
validation = fixture["val_df"].groupby("Sentiment").head(2).reset_index(drop=True)
exec(prior["teacher-functions"], fixture)

# Small local tokenizer with real chat boundary tokens and a deterministic template.
backend = Tokenizer(models.WordLevel(unk_token="<unk>"))
backend.pre_tokenizer = pre_tokenizers.Whitespace()
trainer = trainers.WordLevelTrainer(vocab_size=2048, special_tokens=["<pad>", "<unk>", "<|im_start|>", "<|im_end|>"])
backend.train_from_iterator(train.text.tolist() + validation.text.tolist() + [
    "0 1 2 3 negative positive irrelevant neutral system user assistant Tweet Instruct Query",
    "The words in this tweet support the stated sentiment. Counterfactual positive negative sample."], trainer)
tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>", pad_token="<pad>", eos_token="<|im_end|>")
tokenizer.chat_template = "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
model_dir, embedding_dir = work / "tiny-qwen", work / "tiny-embedding"
config = Qwen3Config(vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                    num_attention_heads=2, num_key_value_heads=1, head_dim=16, max_position_embeddings=1024,
                    eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id, tie_word_embeddings=True)
torch.manual_seed(7)
Qwen3ForCausalLM(config).save_pretrained(model_dir)
torch.manual_seed(9)
Qwen3Model(config).save_pretrained(embedding_dir)
for folder in [model_dir, embedding_dir]:
    tokenizer.save_pretrained(folder)

records = []
for row in train.itertuples():
    for variant in ("a", "b"):
        records.append(dict(status="ok", request_key=fixture["request_key"](row.text, variant), variant=variant,
            row_id=row.row_id, usage={}, parsed=dict(label=row.Sentiment,
            explanation="The words in this tweet support the stated sentiment.", evidence=[row.text[:10]], needs_missing_target=False)))
cf_rows = []
for row in train[train.Sentiment.isin([0, 1])].groupby("Sentiment").head(1).itertuples():
    cf_rows.append(dict(parent_row_id=row.row_id, source_text=row.text, source_label=row.Sentiment,
                       cf_text=row.text + (" Counterfactual positive sample." if row.Sentiment == 0 else " Counterfactual negative sample."),
                       cf_label=1-row.Sentiment, accepted=True))
protocol = dict(model_id=str(model_dir), model_revision="main", split_hash=fixture["SPLIT_HASH"],
    teacher_base_url=fixture["TEACHER_BASE_URL"], teacher_body_example_without_text=fixture["teacher_body"]("x", "a"),
    filter=dict(min_words=5, max_words=45, max_rationale_tokens=96), rationale_weight=.5, max_length=512,
    counterfactual=dict(accepted=len(cf_rows), weight=.5), smoke_test=True)
prior_zip = work / "prior.zip"
with zipfile.ZipFile(prior_zip, "w") as archive:
    archive.writestr("experiment_protocol.json", json.dumps(protocol))
    archive.writestr("split_manifest.csv", reference_manifest.to_csv(index=False))
    archive.writestr("teacher_responses.json", json.dumps(records))
    archive.writestr("accepted_counterfactuals.csv", pd.DataFrame(cf_rows).to_csv(index=False))

ns = {}


def execute(key, source=None):
    print("CHECK", key, flush=True)
    exec(compile(source if source is not None else cells[key], key, "exec"), ns)


def deny_network(*args, **kwargs):
    raise AssertionError("Offline test attempted a network request")


with patch.object(requests.sessions.Session, "request", deny_network):
    for key, source in cells.items():
        if key == "install":
            continue
        if key == "mode":
            source = source.replace('RUN_MODE = "label-only"', 'RUN_MODE = "embedding-mlp"', 1)
        if key == "config":
            source = cpu_config(source)
            source = source.replace('PRIOR_EXPORT = ""', f"PRIOR_EXPORT = {str(prior_zip)!r}")
        source = source.replace("fp16=not USE_BF16,", "fp16=False, use_cpu=True,")
        if key == "embedding-test":
            execute(key)  # Real smoke mode must skip test labels.
            assert "test_df" not in ns and "comparison" not in ns
            ns["SMOKE_TEST"] = False  # Exercise evaluation using tiny fixture rows, never real results.
            source = source.replace('test_df = read_snapshot("test")',
                                    'test_df = read_snapshot("test").groupby("Sentiment").head(2).reset_index(drop=True)')
        execute(key, source)
        if key == "config":
            ns.update(SMOKE_TEST=True, SEEDS=[42], TRAIN_BATCH_SIZE=2, EVAL_BATCH_SIZE=2, GRAD_ACCUMULATION=1,
                      EMBEDDING_MODEL_ID=str(embedding_dir), MLP_MAX_EPOCHS=3, MLP_PATIENCE=2,
                      EMBED_BOOTSTRAP_SAMPLES=10, EMBED_BATCH_SIZE=3)
            ns["model_info"] = lambda *a, **kw: SimpleNamespace(sha="main")
        if key == "data":
            assert ns["split_manifest"].equals(reference_manifest)
            assert (len(ns["train_df"]), len(ns["val_df"])) == (473, 118)
            ns["train_df"], ns["val_df"] = train, validation
        if key == "embedding-adaptation":
            summaries = ns["summaries"]
            assert len(summaries) == 3 and len({s["initial_adapter_hash"] for s in summaries}) == 1
            assert [s["rationale_rows"] for s in summaries] == [0, 8, 8]
            assert [s["cf_rows"] for s in summaries] == [0, 0, 2]
            assert len({s["adapter_sha256"] for s in summaries}) == 3
        if key == "embedding-mlp-fit":
            assert len(ns["mlp_summaries"]) == 5 and "test_df" not in ns
            for item in ns["mlp_summaries"]:
                bundle = joblib.load(ns["RUN_DIR"] / item["mlp_relative"])
                assert bundle["train_row_ids"] == train.row_id.tolist()
                with np.load(ns["FEATURE_ROOT"] / item["feature_caches"]["train"], allow_pickle=False) as data:
                    np.testing.assert_allclose(bundle["scaler"].mean_, data["features"].mean(axis=0), atol=1e-6)
                assert bundle["classifier"].coefs_[0].shape == (32, 128)

    assert len(ns["comparison"]) == 5 and len(ns["paired"]) == 5
    with zipfile.ZipFile(ns["archive"]) as bundle:
        names = bundle.namelist()
        for expected in ["comparison.csv", "paired_bootstrap.csv", "per_class_metrics.csv",
                         "split_manifest.csv", "accepted_counterfactuals.csv", "mlp_summaries.json"]:
            assert expected in names
        assert sum(name.endswith("mlp.joblib") for name in names) == 5
        assert sum(name.endswith("adapter_model.safetensors") for name in names) == 3

    # Finished adapters and MLPs resume without model loading/training; features hit cache.
    original_make = ns["make_model"]
    original_loader = ns["load_feature_decoder"]
    ns["make_model"] = deny_network
    ns["load_feature_decoder"] = deny_network
    ns["SMOKE_TEST"] = True
    execute("embedding-adaptation", cells["embedding-adaptation"].replace("fp16=not USE_BF16,", "fp16=False, use_cpu=True,"))
    execute("embedding-mlp-fit")
    ns["make_model"], ns["load_feature_decoder"] = original_make, original_loader

    # Reject changed provenance, flipped labels, held-out parents and changed cache identities.
    def rejects(call):
        try:
            call()
        except (ValueError, FileNotFoundError):
            return
        raise AssertionError("Expected provenance validation to fail")

    saved_protocol = json.loads((ns["RUN_DIR"] / "experiment_protocol.json").read_text(encoding="utf-8"))
    changed = copy.deepcopy(saved_protocol)
    changed["frozen_embeddings"]["mlp"]["hidden"] += 1
    rejects(lambda: ns["guard_embedding_protocol"](ns["RUN_DIR"] / "experiment_protocol.json", changed))
    bad_cf = ns["accepted_cf"].copy()
    bad_cf.loc[0, "cf_label"] = bad_cf.loc[0, "source_label"]
    rejects(lambda: ns["validate_cached_counterfactuals"](bad_cf, train, train.row_id, 2, ns["text_group"], set()))
    bad_cf = ns["accepted_cf"].copy()
    bad_cf.loc[0, "parent_row_id"] = validation.row_id.iloc[0]
    rejects(lambda: ns["validate_cached_counterfactuals"](bad_cf, train, train.row_id, 2, ns["text_group"], set()))
    rejects(lambda: ns["feature_cache"](work, {"dimension": 32}, train, [[1]] * len(train), [False] * len(train), lambda: None))
    tiny_frame = train[["row_id", "text"]].head(1)
    spec = {"dimension": 2, "adapter_sha256": "a"}
    _, first, _ = ns["feature_cache"](work / "cache-check", spec, tiny_frame, [[1]], [False], lambda: np.ones((1, 2)))
    _, same, hit = ns["feature_cache"](work / "cache-check", spec, tiny_frame, [[1]], [False], deny_network)
    assert same == first and hit
    _, changed, _ = ns["feature_cache"](work / "cache-check", {**spec, "adapter_sha256": "b"}, tiny_frame, [[1]], [False], lambda: np.ones((1, 2)))
    assert changed != first

    # Same input gets the same frozen vector alone/in a mixed batch, with either padding side.
    decoder = ns["load_feature_decoder"](ns["representation_spec"]("qwen_label", 42))
    encoded = [ns["prompt_ids"](t)[0] for t in train.text.iloc[:2]]
    for side in ["left", "right"]:
        tokenizer.padding_side = side
        one = ns["extract_frozen_vectors"](decoder, tokenizer, encoded, 1, "cpu")
        batch = ns["extract_frozen_vectors"](decoder, tokenizer, encoded, 2, "cpu")
        np.testing.assert_allclose(one, batch, atol=1e-5, rtol=1e-5)
        assert not any(p.requires_grad for p in decoder.parameters())
        assert all(p.grad is None for p in decoder.parameters())
    print("PASS: 5 arms, 3 trained/reloaded LoRA adapters, exact split, cache-only teachers, train-only scaler/MLP, "
          "frozen pooling, cache fingerprints, resume, test isolation, paired metrics and portable export", flush=True)
    print("Fixture artifacts:", work)
