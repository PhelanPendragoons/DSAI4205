"""Execute notebook logic on CPU with the real tokenizer and a tiny random Qwen3.

This is an integration check, NOT a benchmark of pretrained Qwen3-0.6B.
Install the notebook's dependencies plus nbformat and ipython first.
Usage: python validate_qwen_lora_notebook.py --work-dir PATH
Only the public tokenizer is downloaded (no 0.6B pretrained weights).
"""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--work-dir", type=Path, required=True)
args = parser.parse_args()
work = args.work_dir.resolve()
work.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(work / "hf-cache")
os.environ["MPLBACKEND"] = "Agg"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import nbformat
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

torch.set_num_threads(2)
path = Path(__file__).resolve().parents[1] / "models/label-only" / "Qwen3_0_6B_LoRA_Sentiment_Colab.ipynb"
nb = nbformat.read(path, as_version=4)
nbformat.validate(nb)
sources = {cell.id: cell.source for cell in nb.cells if cell.cell_type == "code"}
for name, source in sources.items():
    if name != "install":
        ast.parse(source, filename=name)


def replace(source, before, after):
    assert source.count(before) == 1, (before, source.count(before))
    return source.replace(before, after)


ns = {"__name__": "__notebook_validation__"}


def run(name, source=None):
    print(f"RUN {name}", flush=True)
    exec(compile(source if source is not None else sources[name], f"notebook:{name}", "exec"), ns)


config = sources["config"]
start = config.index("if not torch.cuda.is_available():")
end = config.index("set_seed(SEED)")
config = config[:start] + "USE_BF16 = False\nMODEL_DTYPE = torch.float32\n" + config[end:]
config = replace(config, 'Path("/content/qwen_lora_runs")', f"Path({str(work / 'runs')!r})")
run("config", config)
ns.update(SMOKE_TEST=True, EPOCHS=1, GRAD_ACCUMULATION=1, TRAIN_BATCH_SIZE=2, EVAL_BATCH_SIZE=2)
run("embedded-data")
run("data")
np = ns["np"]
full_train, full_val = ns["train_df"], ns["val_df"]
assert len(ns["pool"]) == 591
test_all = ns["read_snapshot"]("test")
assert len(test_all) == 397
assert not (set(ns["pool"].group_id) & set(test_all.Phrase.map(ns["text_group"])))
run("tokenization")
tok = ns["tokenizer"]
assert "<think>\n\n</think>" in tok.decode(ns["prompt_ids"]("Test")[0])
long_ids, clipped = ns["prompt_ids"]("Very long tweet! " * 2000)
assert clipped and len(long_ids) <= ns["MAX_LENGTH"] - 2
examples = [ns["encode_example"]("Short.", 0)[0], ns["encode_example"]("This is a somewhat longer tweet!", 3)[0]]
batch = ns["collate_examples"](examples)
assert (batch["labels"] != -100).sum().item() == 4
assert torch.all(batch["labels"][batch["attention_mask"] == 0] == -100)

# Keep a real full tokenizer vocabulary so code IDs, chat template and masking
# are exactly the same as in the delivered notebook.
tiny_dir = work / "tiny-qwen3"
tiny = Qwen3ForCausalLM(Qwen3Config(
    vocab_size=len(tok), hidden_size=32, intermediate_size=64,
    num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
    head_dim=16, max_position_embeddings=1024,
    bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
    tie_word_embeddings=True,
))
tiny.save_pretrained(tiny_dir)
del tiny
ns["MODEL_ID"] = str(tiny_dir)
ns["RESOLVED_REVISION"] = "main"
run("model")
before = {name: param.detach().clone() for name, param in ns["model"].named_parameters() if param.requires_grad}
frozen_name, frozen_param = next((n, p) for n, p in ns["model"].named_parameters() if not p.requires_grad)
frozen_before = frozen_param.detach().clone()
ns["train_dataset"] = ns["SentimentDataset"](full_train.head(8))
ns["val_dataset"] = ns["SentimentDataset"](full_val.head(8))
train = replace(sources["train"], "fp16=not USE_BF16,", "fp16=False, use_cpu=True,")
train = replace(train, "torch.cuda.reset_peak_memory_stats()", "pass  # CPU check")
run("train", train)
assert np.isfinite(ns["train_result"].training_loss)
assert any(not torch.equal(before[n], p) for n, p in ns["model"].named_parameters() if n in before)
assert torch.equal(frozen_before, dict(ns["model"].named_parameters())[frozen_name])
run("inference")

# Directly test that metric extraction uses the position BEFORE the label.
fake_labels = torch.tensor([[-100, -100, ns["LABEL_TOKEN_IDS"][2], ns["STOP_ID"]]])
fake_logits = torch.zeros(1, 4, len(tok))
fake_logits[0, 1, ns["LABEL_TOKEN_IDS"][2]] = 10
fake_logits[0, 2, ns["LABEL_TOKEN_IDS"][0]] = 100
assert ns["preprocess_logits_for_metrics"](fake_logits, fake_labels).argmax().item() == 2

run("test")  # Confirm SMOKE_TEST skips the held-out test evaluation.
assert not (ns["RUN_DIR"] / "test_metrics.json").exists()
# Exercise report / plot / baseline code on a small fixture, not publishable test scores.
test_source = replace(sources["test"], 'test_df = read_snapshot("test")', 'test_df = read_snapshot("test").groupby("Sentiment", group_keys=False).head(2).reset_index(drop=True)')
ns["SMOKE_TEST"] = False
run("test", test_source)
ns["SMOKE_TEST"] = True
run("export")
run("demo")
assert (ns["ADAPTER_DIR"] / "adapter_model.safetensors").exists()
assert Path(ns["archive_path"]).exists()
assert json.loads((ns["RUN_DIR"] / "reload_check.json").read_text())["passed"]
summary = {
    "passed": True,
    "validation_scope": "Real Qwen3 tokenizer + tiny randomly initialized Qwen3 on CPU; not pretrained-model/GPU validation",
    "checks": ["notebook schema and syntax", "embedded dataset hashes and counts", "group-disjoint split",
               "no normalized train/test overlap", "long-input truncation", "answer-only and padding masks",
               "actual Trainer forward/backward and checkpoint selection", "LoRA parameters updated; base frozen",
               "causal next-token metric alignment", "label-free inference equivalence", "smoke mode skips test",
               "test reporting and TF-IDF comparison on fixture", "adapter save/reload equivalence", "ZIP export"],
    "versions": ns["VERSIONS"], "split_counts": {"train": len(full_train), "validation": len(full_val), "test": len(test_all)},
}
(work / "validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2), flush=True)
