"""Five frozen-representation arms composed with the existing adaptation recipes."""
from textwrap import dedent
from pathlib import Path
import hashlib
import json

cells = []


def add(kind, name, body):
    cells.append({"cell_type": kind, "id": "embedding-" + name, "source": dedent(body).strip() + "\n"})


add("markdown", "note", """
## 冻结文本表示＋独立 MLP（RUN_MODE = embedding-mlp）

此模式复用完整反事实导出的划分、教师解释缓存和 accepted_counterfactuals.csv，不发出教师请求。
从同一基础 revision 重新训练标签、严格解释、严格解释＋反事实三个 adapter，使用相同初始化种子、
原始样本曝光次数、优化步数和验证选型规则；保留未微调的 Qwen 与专用 Qwen3-Embedding 两个参照。

| 组名 | 冻结的表示模型 | 独立分类器 |
|---|---|---|
| qwen_base | 原始 Qwen3-0.6B | MLP |
| qwen_label | 标签 LoRA | 相同结构 MLP |
| qwen_rationale | 严格短解释 LoRA | 相同结构 MLP |
| qwen_rationale_cf | 严格短解释＋已核验反事实 LoRA | 相同结构 MLP |
| qwen_embedding | Qwen3-Embedding-0.6B（不做本任务微调） | 相同结构 MLP |

前四组用同一分类提示，仅输入推文，提取回答开始位置（最后有效 token）的最终层状态；不输入真实答案、
教师解释或生成标签。第五组按官方格式使用 Instruct/Query 提示与最后有效 token pooling。
五组均 L2 归一化，再以**原始训练行**拟合 StandardScaler 和一个 128 单元 ReLU 隐藏层的 MLP。
反事实样本仅用于适配 Qwen，不加入 MLP 训练集，避免把表示改进与分类器新增数据混在一起。

三组 adapter 固定 epoch、关闭早停，再按生成式标签验证 macro-F1 选 checkpoint；随后各 MLP 按验证 macro-F1 早停。
MLP 不向 Qwen 反传梯度。所有组的 MLP 选型完成后，才统一打开测试标签。
这五组是同一独立 MLP 分析模型的表示对照，不能作为课程要求的五种不同分析模型计数。
""")
add("code", "helpers", helper_path.read_text(encoding="utf-8"))
add("code", "datasets", r'''
if ALLOW_TEACHER_REQUESTS or ALLOW_CF_REQUESTS:
    raise ValueError("embedding-mlp 必须只读既有教师缓存。")
if not prior_protocol or "counterfactual" not in prior_protocol:
    raise ValueError("请设置 PRIOR_EXPORT 为原 counterfactual 模式的完整导出。")
if prior_protocol.get("smoke_test") and not SMOKE_TEST:
    raise ValueError("快速检查导出不能用于正式实验，请选择完整反事实结果。")
if not SEEDS or len(set(SEEDS)) != len(SEEDS):
    raise ValueError("SEEDS 需为非空且不重复的整数列表。")
if min(EMBED_BATCH_SIZE, MLP_HIDDEN, MLP_MAX_EPOCHS, MLP_PATIENCE,
       MLP_BATCH_SIZE, EMBED_BOOTSTRAP_SAMPLES) <= 0 or MLP_ALPHA < 0 or MLP_LEARNING_RATE <= 0:
    raise ValueError("向量/MLP 配置值无效。")

explanations = teacher_table.loc[teacher_table.accepted].set_index("row_id").explanation.to_dict()
original_dataset = PairedDataset(train_df, explanations)
if not original_dataset.used_ids:
    raise ValueError("没有在 token 预算内可用的严格解释。")
accepted_file = imported / "accepted_counterfactuals.csv"
if not accepted_file.is_file():
    raise FileNotFoundError("完整反事实导出必须包含 accepted_counterfactuals.csv。")
# Only held-out text is read here for duplicate rejection, never held-out labels.
heldout_bytes = gzip.decompress(base64.b64decode(EMBEDDED_DATA["test"]))
assert hashlib.sha256(heldout_bytes).hexdigest() == DATA_MANIFEST["test"]["sha256"]
heldout_hashes = pd.read_csv(io.BytesIO(heldout_bytes), usecols=["Phrase"], keep_default_na=False).Phrase.map(text_group)
accepted_cf = validate_cached_counterfactuals(
    pd.read_csv(accepted_file, keep_default_na=False), train_df, original_dataset.used_ids,
    prior_protocol["counterfactual"]["accepted"], text_group, set(pool.group_id) | set(heldout_hashes))
for row in accepted_cf.itertuples():
    encoded, truncated = encode_example(row.cf_text, row.cf_label)
    if truncated or encoded["input_ids"][:-2] == prompt_ids(row.source_text)[0]:
        raise ValueError("缓存改写在当前 tokenizer 下被截断或与父文本相同。")
CF_WEIGHT = prior_protocol["counterfactual"]["weight"]
if CF_WEIGHT <= 0:
    raise ValueError("CF_WEIGHT 必须为正。")
accepted_cf.to_csv(RUN_DIR / "accepted_counterfactuals.csv", index=False)
EXPERIMENTS = ["label_only", "rationale_filtered", "rationale_filtered_cf"]
paired_datasets = {"label_only": PairedDataset(train_df, {}), "rationale_filtered": original_dataset,
                   "rationale_filtered_cf": CounterfactualDataset(original_dataset, train_df.row_id, accepted_cf)}
coverage = [{"experiment": name, "original_rows": len(ds), "rationale_rows": ds.aux_count,
             "cf_rows": getattr(ds, "cf_count", 0)} for name, ds in paired_datasets.items()]
pd.DataFrame(coverage).to_csv(RUN_DIR / "adaptation_coverage.csv", index=False)
write_json(RUN_DIR / "rationale_coverage.json", {"used_ids": original_dataset.used_ids,
           "token_budget_skipped_ids": original_dataset.skipped_ids})
display(pd.DataFrame(coverage))

from transformers import AutoConfig, AutoModel
import joblib
EMBEDDING_RESOLVED_REVISION = model_info(EMBEDDING_MODEL_ID, revision=EMBEDDING_REVISION).sha
base_config = AutoConfig.from_pretrained(MODEL_ID, revision=RESOLVED_REVISION)
embedding_config = AutoConfig.from_pretrained(EMBEDDING_MODEL_ID, revision=EMBEDDING_RESOLVED_REVISION)
if base_config.hidden_size != embedding_config.hidden_size:
    raise ValueError("五组向量维度需相同，以保持 MLP 参数量相同。请使用配套的 0.6B 模型。")
FEATURE_DIMENSION = base_config.hidden_size
EMBED_INSTRUCTION = "Classify the sentiment of the tweet as negative, positive, irrelevant, or neutral."
MLP_SETTINGS = {"hidden": MLP_HIDDEN, "max_epochs": 2 if SMOKE_TEST else MLP_MAX_EPOCHS,
                "patience": MLP_PATIENCE, "batch_size": MLP_BATCH_SIZE,
                "alpha": MLP_ALPHA, "learning_rate": MLP_LEARNING_RATE}
embedding_design = {
    "schema": 1, "mode": RUN_MODE, "data_manifest": DATA_MANIFEST,
    "embedding_model": EMBEDDING_MODEL_ID, "embedding_revision": EMBEDDING_RESOLVED_REVISION,
    "embedding_instruction": EMBED_INSTRUCTION, "feature_dimension": FEATURE_DIMENSION,
    "pooling": "last_non_padding_final_hidden_state", "normalization": "l2",
    "qwen_input": "classification_chat_prompt_without_answer", "embedding_input": "Instruct/Query",
    "batch_size": EMBED_BATCH_SIZE, "dtype": str(MODEL_DTYPE), "mlp": MLP_SETTINGS,
    "mlp_training_data": "original_train_only", "scaler_fit": "original_train_only",
    "adapter_selection": "generative_validation_macro_f1", "mlp_selection": "validation_macro_f1",
    "teacher_records_sha256": embedding_digest(sorted(teacher_records, key=lambda r: (str(r["row_id"]), r["variant"]))),
    "counterfactual_rows_sha256": embedding_digest(accepted_cf.to_dict("records")),
    "counterfactual_weight": CF_WEIGHT, "bootstrap_samples": EMBED_BOOTSTRAP_SAMPLES,
    "feature_code_sha256": EMBEDDING_HELPER_HASH,
}
''')

# The classifier/rationale losses, optimizer and checkpoint selection are exactly
# the existing implementations. Only the set of arms and export/evaluation differ.
experiments = sources["rationale-distillation"]["experiments"]
# Keep this representation ablation's fixed update budget while the original
# generative modes retain their shared validation-based early stopping policy.
early_stop = """    if not SMOKE_TEST:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE))
"""
assert early_stop in experiments
experiments = experiments.replace(early_stop, "")
experiments = experiments.replace('"early_stopping_patience": None if SMOKE_TEST else EARLY_STOPPING_PATIENCE',
                                  '"early_stopping_patience": None')
experiments = experiments.replace('"early_stopping_patience": EARLY_STOPPING_PATIENCE',
                                  '"early_stopping_patience": None')
experiments = experiments.replace('# 早停下各臂实际步数可以不同；统一的是 epoch 上限与 patience，不是强制同一步数。',
                                  'assert summary_df.groupby("seed").optimizer_steps.nunique().eq(1).all()')
experiments = experiments.replace("trainer = RationaleTrainer(", "trainer = CounterfactualTrainer(")
experiments = experiments.replace("data_collator=collate_pairs", "data_collator=collate_with_counterfactuals")
experiments = experiments.replace('"rationale_rows": paired_datasets[name].aux_count,',
                                '"rationale_rows": paired_datasets[name].aux_count,\n'
                                '               "cf_rows": getattr(paired_datasets[name], "cf_count", 0),')
experiments = experiments.replace('write_json(out / "summary.json", summary)',
                                'summary["adapter_sha256"] = adapter_digest(adapter)\n'
                                '    write_json(out / "summary.json", summary)')
experiments = experiments.replace('write_json(RUN_DIR / "experiment_protocol.json", protocol)', '''
protocol["frozen_embeddings"] = embedding_design
protocol["counterfactual"] = {"accepted": len(accepted_cf), "weight": CF_WEIGHT, "api_calls_this_run": 0}
guard_embedding_protocol(RUN_DIR / "experiment_protocol.json", protocol)''')
experiments = experiments.replace('summaries.append(json.loads(existing.read_text(encoding="utf-8")))', '''saved = json.loads(existing.read_text(encoding="utf-8"))
            saved["adapter"] = str(existing.parent / "adapter")
            if saved.get("adapter_sha256") != adapter_digest(saved["adapter"]):
                raise ValueError("已完成的 adapter 与摘要指纹不符。请使用完整结果续跑。")
            summaries.append(saved)''')
add("code", "adaptation", experiments)

add("code", "feature-functions", r'''
EMBED_ARMS = ["qwen_base", "qwen_label", "qwen_rationale", "qwen_rationale_cf", "qwen_embedding"]
adaptation_names = {"qwen_label": "label_only", "qwen_rationale": "rationale_filtered",
                    "qwen_rationale_cf": "rationale_filtered_cf"}
adaptation_lookup = {(s["experiment"], s["seed"]): s for s in summaries}
FEATURE_ROOT = RUN_DIR / "embeddings"
MLP_ROOT = RUN_DIR / "mlp"
MLP_ROOT.mkdir(exist_ok=True)

def representation_spec(arm, seed):
    dedicated = arm == "qwen_embedding"
    spec = {"arm": arm, "model_id": EMBEDDING_MODEL_ID if dedicated else MODEL_ID,
            "revision": EMBEDDING_RESOLVED_REVISION if dedicated else RESOLVED_REVISION,
            "dimension": FEATURE_DIMENSION, "max_length": MAX_LENGTH, "dtype": str(MODEL_DTYPE),
            "pooling": "last_non_padding_final_hidden_state", "normalization": "l2",
            "input": EMBED_INSTRUCTION if dedicated else SYSTEM_PROMPT,
            "input_format": "Instruct/Query" if dedicated else "classification_chat_without_answer",
            "batch_size": EMBED_BATCH_SIZE, "versions": VERSIONS,
            "feature_code_sha256": EMBEDDING_HELPER_HASH}
    if arm in adaptation_names:
        summary = adaptation_lookup[(adaptation_names[arm], seed)]
        spec["adapter_relative"] = Path(summary["adapter"]).relative_to(RUN_DIR).as_posix()
        spec["adapter_sha256"] = summary["adapter_sha256"]
    return spec

def load_feature_decoder(spec):
    if "adapter_relative" in spec:
        adapter = RUN_DIR / spec["adapter_relative"]
        if adapter_digest(adapter) != spec["adapter_sha256"]:
            raise ValueError("Adapter 权重已改变，拒绝复用旧向量。")
        base = AutoModelForCausalLM.from_pretrained(spec["model_id"], revision=spec["revision"],
                    torch_dtype=MODEL_DTYPE, attn_implementation="sdpa")
        adapted = PeftModel.from_pretrained(base, adapter, is_trainable=False)
        # LoRA remains attached to the decoder layers. The LM head is never evaluated.
        decoder = adapted.get_base_model().model
    else:
        decoder = AutoModel.from_pretrained(spec["model_id"], revision=spec["revision"],
                    torch_dtype=MODEL_DTYPE, attn_implementation="sdpa")
    return freeze_decoder(decoder).to("cuda" if torch.cuda.is_available() else "cpu")

def vectors_for_frames(arm, seed, frames):
    spec = representation_spec(arm, seed)
    feature_tokenizer = AutoTokenizer.from_pretrained(spec["model_id"], revision=spec["revision"])
    feature_tokenizer.padding_side = "left" if arm == "qwen_embedding" else "right"
    if feature_tokenizer.pad_token_id is None:
        feature_tokenizer.pad_token = feature_tokenizer.eos_token
    decoder, result = None, {}
    try:
        for partition, frame in frames.items():
            inputs, truncated = [], []
            for text in frame.text:
                if arm == "qwen_embedding":
                    rendered = f"Instruct: {EMBED_INSTRUCTION}\nQuery:{text}"
                    ids = feature_tokenizer.encode(rendered, add_special_tokens=True)
                    cut = len(ids) > MAX_LENGTH
                    ids = ids[:MAX_LENGTH]
                else:
                    ids, cut = prompt_ids(text)
                inputs.append(ids)
                truncated.append(cut)
            def compute():
                nonlocal decoder
                if decoder is None:
                    decoder = load_feature_decoder(spec)
                return extract_frozen_vectors(decoder, feature_tokenizer, inputs, EMBED_BATCH_SIZE,
                                              next(decoder.parameters()).device)
            values, path, hit = feature_cache(FEATURE_ROOT, spec, frame[["row_id", "text"]],
                                              inputs, truncated, compute)
            result[partition] = {"values": values, "cache": path.name, "truncated": truncated}
            print(arm, seed, partition, values.shape, "缓存命中" if hit else "已冻结提取", flush=True)
    finally:
        del decoder
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return spec, result
''')
add("markdown", "mlp-note", """
### 先完成五组 MLP 选型，再评估测试集

默认 SEEDS=[42]；正式比较可改成 [42,43,44]。同一个 seed 的五组 MLP 使用同样的结构、
初始化规则和训练行顺序，StandardScaler 仅拟合训练向量。专用 Embedding 模型的输入模板与普通
Qwen 不同，因此第五组是现成表示模型的参考，不是仅改变预训练权重的严格消融。

向量按输入文本、顺序、token、基础 revision、adapter 权重和提取配置缓存。
续跑保持全部配置不变，用 run_local.py --resume-dir 指向当前运行目录；已完成的 adapter、向量和 MLP 可复用。
""")
add("code", "mlp-fit", r'''
mlp_summaries = []
for seed in SEEDS:
    for arm in EMBED_ARMS:
        spec, features = vectors_for_frames(arm, seed, {"train": train_df, "validation": val_df})
        out = MLP_ROOT / f"{arm}_seed{seed}"
        out.mkdir(exist_ok=True)
        signature = embedding_digest({"spec": spec, "train": features["train"]["cache"],
            "validation": features["validation"]["cache"], "train_labels": train_df.Sentiment.tolist(),
            "validation_labels": val_df.Sentiment.tolist(), "settings": MLP_SETTINGS, "seed": seed})
        bundle_path, summary_path = out / "mlp.joblib", out / "summary.json"
        if summary_path.is_file() and bundle_path.is_file():
            item = json.loads(summary_path.read_text(encoding="utf-8"))
            if item["signature"] != signature:
                raise ValueError("已有 MLP 的向量、标签或配置不同，请使用新运行目录。")
            print("复用已完成 MLP", arm, seed)
        else:
            start = time.perf_counter()
            bundle = fit_independent_mlp(features["train"]["values"], train_df.Sentiment.to_numpy(),
                features["validation"]["values"], val_df.Sentiment.to_numpy(), seed, MLP_SETTINGS)
            bundle.update(signature=signature, representation=spec, train_row_ids=train_df.row_id.tolist())
            joblib.dump(bundle, bundle_path)
            reloaded = joblib.load(bundle_path)
            predicted, probs = predict_independent_mlp(bundle, features["validation"]["values"])
            _, restored_probs = predict_independent_mlp(reloaded, features["validation"]["values"])
            np.testing.assert_allclose(probs, restored_probs, rtol=1e-6, atol=1e-7)
            item = {"experiment": arm, "seed": seed, "signature": signature,
                    "validation_macro_f1": bundle["validation_macro_f1"], "best_epoch": bundle["best_epoch"],
                    "epochs_run": len(bundle["history"]), "training_seconds": time.perf_counter() - start,
                    "original_train_rows": len(train_df), "feature_dimension": FEATURE_DIMENSION,
                    "feature_caches": {k: v["cache"] for k, v in features.items()},
                    "representation": spec, "mlp_relative": bundle_path.relative_to(RUN_DIR).as_posix()}
            pd.DataFrame(bundle["history"]).to_csv(out / "training_log.csv", index=False)
            write_json(out / "validation_metrics.json", embedding_metrics(val_df.Sentiment, predicted, LABEL_NAMES))
            validation_rows = val_df[["row_id", "Sentiment"]].copy()
            validation_rows["predicted_id"] = predicted
            validation_rows.to_csv(out / "validation_predictions.csv", index=False)
            write_json(summary_path, item)
            del bundle, reloaded
        mlp_summaries.append(item)
        del features
write_json(RUN_DIR / "mlp_summaries.json", mlp_summaries)
display(pd.DataFrame(mlp_summaries)[["experiment", "seed", "validation_macro_f1", "best_epoch", "training_seconds"]])
''')
add("code", "test", r'''
if SMOKE_TEST:
    print("SMOKE_TEST：五组流程检查完成，不读取测试标签，不报告测试成绩。")
else:
    if len(mlp_summaries) != 5 * len(SEEDS):
        raise ValueError("必须先完成所有表示组的 MLP 选型。")
    test_df = read_snapshot("test")
    test_df["text"] = test_df.Phrase.map(normalize_text)
    assert not (set(pool.group_id) & set(test_df.text.map(text_group)))
    all_predictions, comparison_rows, class_rows = {}, [], []
    for item in mlp_summaries:
        arm, seed = item["experiment"], item["seed"]
        spec, features = vectors_for_frames(arm, seed, {"test": test_df})
        if spec != item["representation"]:
            raise ValueError("测试时的表示配置与 MLP 训练不一致。")
        bundle_path = RUN_DIR / item["mlp_relative"]
        bundle = joblib.load(bundle_path)
        if bundle["signature"] != item["signature"]:
            raise ValueError("MLP 文件与选型摘要不一致。")
        predicted, probabilities = predict_independent_mlp(bundle, features["test"]["values"])
        metrics = embedding_metrics(test_df.Sentiment, predicted, LABEL_NAMES)
        keep = test_df.cross_split_overlap.astype(int).eq(0).to_numpy()
        metrics["without_historical_overlap"] = embedding_metrics(
            test_df.Sentiment.to_numpy()[keep], predicted[keep], LABEL_NAMES)
        out = bundle_path.parent
        write_json(out / "test_metrics.json", metrics)
        table = test_df[["row_id", "Phrase", "Sentiment"]].copy()
        table["predicted_id"] = predicted
        table["truncated"] = features["test"]["truncated"]
        for i, name in enumerate(LABEL_NAMES):
            table[f"prob_{name}"] = probabilities[:, i]
            class_rows.append({"experiment": arm, "seed": seed, "label": name,
                               **metrics["classification_report"][name]})
        table.to_csv(out / "test_predictions.csv", index=False)
        all_predictions[(arm, seed)] = predicted
        comparison_rows.append({k: item[k] for k in ["experiment", "seed", "validation_macro_f1", "best_epoch", "training_seconds"]}
            | {"test_accuracy": metrics["accuracy"], "test_macro_f1": metrics["macro_f1"],
               "test_feature_cache": features["test"]["cache"]})
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay(np.asarray(metrics["confusion_matrix"]), display_labels=LABEL_NAMES).plot(ax=ax, colorbar=False)
        ax.set_title(f"{arm} + MLP / seed {seed}")
        fig.tight_layout()
        fig.savefig(out / "confusion_matrix.png", dpi=150)
        plt.close(fig)
        del features, bundle
    comparison = pd.DataFrame(comparison_rows)
    base_scores = comparison[comparison.experiment.eq("qwen_base")].set_index("seed").test_macro_f1
    comparison["delta_vs_base_macro_f1"] = comparison.test_macro_f1 - comparison.seed.map(base_scores)
    comparison.to_csv(RUN_DIR / "comparison.csv", index=False)
    pd.DataFrame(class_rows).to_csv(RUN_DIR / "per_class_metrics.csv", index=False)
    aggregate = comparison.groupby("experiment").agg(mean_macro_f1=("test_macro_f1", "mean"),
        std_macro_f1=("test_macro_f1", "std"), mean_accuracy=("test_accuracy", "mean"), seeds=("seed", "size"))
    aggregate.to_csv(RUN_DIR / "comparison_aggregate.csv")
    paired = pd.DataFrame(embedding_bootstrap(test_df.Sentiment.to_numpy(), all_predictions, SEEDS, EMBED_BOOTSTRAP_SAMPLES))
    paired.to_csv(RUN_DIR / "paired_bootstrap.csv", index=False)
    display(comparison)
    display(aggregate)
    display(paired)
''')
add("code", "export", r'''
EXPORT_DIR = RUN_DIR / "export"
EXPORT_DIR.mkdir(exist_ok=True)
for path in RUN_DIR.iterdir():
    if path.is_file() and path.suffix != ".zip":
        shutil.copy2(path, EXPORT_DIR / path.name)
for summary in summaries:
    source_folder = Path(summary["adapter"]).parent
    target = EXPORT_DIR / source_folder.name
    target.mkdir(exist_ok=True)
    shutil.copytree(source_folder / "adapter", target / "adapter", dirs_exist_ok=True)
    for path in source_folder.iterdir():
        if path.is_file():
            shutil.copy2(path, target / path.name)
for folder in [MLP_ROOT, FEATURE_ROOT]:
    shutil.copytree(folder, EXPORT_DIR / folder.name, dirs_exist_ok=True)
export_cache = EXPORT_DIR / "teacher_cache"
export_cache.mkdir(exist_ok=True)
for record in teacher_records:
    filename = record["request_key"] + ".json"
    shutil.copy2(CACHE_ROOT / filename, export_cache / filename)
(EXPORT_DIR / "README.txt").write_text(
    "Five frozen Qwen representations + independent sklearn MLP classifiers.\n"
    "See experiment_protocol.json and mlp_summaries.json for revisions, prompts, pooling and settings.\n"
    "Train-only scaling; MLPs use original training rows only. Test labels are used only after head selection.\n"
    "MLP bundles contain scaler + classifier; use the exact feature representation recorded in each bundle.\n"
    "Embedding NPZ files contain row_ids, truncation flags and content-addressed metadata.\n"
    "comparison.csv compares all arms with qwen_base. paired_bootstrap.csv also reports incremental contrasts.\n"
    "Bootstrap intervals measure test-sample uncertainty conditional on a seed, not training-seed uncertainty.\n"
    "Dedicated Qwen3-Embedding uses its own instruction format and is a reference, not a pure weight ablation.\n"
    "SMOKE_TEST runs skip test evaluation and must not be presented as benchmark results.\n",
    encoding="utf-8")
archive = shutil.make_archive(str(RUN_DIR / "qwen3_embedding_mlp"), "zip", EXPORT_DIR)
print("结果包:", archive)
try:
    from google.colab import files
    files.download(archive)
except ImportError:
    print("请从输出目录取得 ZIP。")
''')
add("markdown", "references", """
### 实验边界和依据

此前 Qwen 直接给标签打分的结果不能替代这里的 MLP 结果。先使用固定配置完成比较，再报告结果；
不根据测试集调参。一个 seed 的标准差留空；配对 bootstrap 不代表训练随机性。
解释监督和反事实增强不保证改善冻结表示，本模式用于检验这一假设。

[Qwen3-Embedding 官方用法](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) ·
[LLM2Vec](https://arxiv.org/abs/2404.05961) ·
[SetFit](https://arxiv.org/abs/2209.11055) ·
[Distilling Step-by-Step](https://aclanthology.org/2023.findings-acl.507/) ·
[Counterfactually-Augmented Data](https://arxiv.org/abs/1909.12434)
""")

# Include an implementation identity in cache and resume fingerprints.
helper_hash = hashlib.sha256(helper_path.read_bytes() + Path(__file__).read_bytes()
                            + json.dumps(sources, sort_keys=True).encode()).hexdigest()
cells[2]["source"] = f"EMBEDDING_HELPER_HASH = {helper_hash!r}\n" + cells[2]["source"]
