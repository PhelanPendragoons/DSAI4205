"""Sampling and provenance checks embedded in Kai's matched-ablation notebook cells."""
import hashlib
import json
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd


def previous_export(runs_root, mode, explicit_path):
    if explicit_path:
        candidates = [Path(explicit_path).expanduser()]
    else:
        origin = "counterfactual" if mode == "counterfactual-matched" else "rationale_distillation"
        marker = runs_root / f"last_{origin}.json"
        candidates = []
        if marker.is_file():
            record = json.loads(marker.read_text(encoding="utf-8"))
            for field in ("archive", "run_dir"):
                if record.get(field):
                    candidates.append(Path(record[field]))
            if record.get("run_dir"):
                # The results tree can be copied from Windows to another machine.
                old = PureWindowsPath(record["run_dir"])
                relocated = runs_root / old.parent.name / old.name
                candidates.append(relocated)
    for path in candidates:
        if path.is_file() and path.suffix.lower() == ".zip":
            return path
        for folder in (path / "export", path):
            if (folder / "experiment_protocol.json").is_file():
                return folder
    raise FileNotFoundError("找不到上次实验的完整结果。请在 5090 上设置 PRIOR_EXPORT，"
                            "选择对应的 ZIP、export 文件夹或运行目录。仅有 summary.md 不能还原逐条样本。")


def rationale_arms(train, table, dataset_cls, seed):
    valid = table.loc[table.format_valid].set_index("row_id", verify_integrity=True)
    usable = dataset_cls(train, valid.explanation.to_dict()).used_ids
    eligible = valid.loc[usable].sort_index()
    strict = eligible[eligible.accepted]
    if strict.empty:
        raise ValueError("没有在学生 token 预算内可用的严格解释。")
    rng = np.random.default_rng(seed)
    random_ids = []
    for label, count in strict.Sentiment.value_counts().sort_index().items():
        pool = eligible.index[eligible.Sentiment.eq(label)]
        random_ids.extend(rng.choice(pool.to_numpy(), size=int(count), replace=False))
    random_arm = eligible.loc[random_ids]
    arms = {"rationale_random_matched": random_arm, "rationale_strict_matched": strict}
    datasets, selection = {}, []
    for name, rows in arms.items():
        data = dataset_cls(train, rows.explanation.to_dict())
        assert set(data.used_ids) == set(rows.index) and len(data) == len(train)
        assert rows.Sentiment.value_counts().sort_index().equals(strict.Sentiment.value_counts().sort_index())
        datasets[name] = data
        selection.append(rows.reset_index().assign(experiment=name))
    return datasets, pd.concat(selection, ignore_index=True)


def counterfactual_arms(accepted, train, usable_ids, seed, expected_count):
    rows = accepted.sort_values("parent_row_id").reset_index(drop=True)
    parents = train.set_index("row_id", verify_integrity=True)
    if (len(rows) != expected_count or not rows.parent_row_id.is_unique or
            not rows.parent_row_id.isin(usable_ids).all() or not rows.accepted.eq(True).all()):
        raise ValueError("反事实表与原实验的接受数、训练父样本或核验标记不符。")
    if (not rows.source_label.isin([0, 1]).all() or not rows.cf_label.eq(1 - rows.source_label).all()
            or not rows.source_label.eq(rows.parent_row_id.map(parents.Sentiment)).all()
            or not rows.source_text.eq(rows.parent_row_id.map(parents.text)).all()
            or not rows.cf_text.str.strip().ne("").all() or not rows.cf_text.is_unique):
        raise ValueError("反事实文本、父样本标签或翻转方向与既有划分不符。")
    counts = rows.source_label.value_counts().reindex([0, 1], fill_value=0)
    half = int(counts.min())
    if half == 0 or counts[0] == counts[1]:
        raise ValueError(f"需要两个方向均有样本且原比例不平衡，当前计数为 {counts.to_dict()}。")
    total = 2 * half
    minority = int(counts.idxmin())
    # Nearest feasible unbalanced quota, constrained by the available majority rows.
    small = min(half - 1, int(np.floor(total * half / len(rows) + 0.5)))
    observed_quota = {minority: small, 1 - minority: total - small}
    rng = np.random.default_rng(seed)
    # Shared permutations maximize overlap of parents at the two different quotas.
    pools = {label: rng.permutation(rows.index[rows.source_label.eq(label)]) for label in (0, 1)}
    arms = {}
    for name, quota in (("cf_observed_ratio", observed_quota), ("cf_balanced", {0: half, 1: half})):
        ids = np.concatenate([pools[label][:quota[label]] for label in (0, 1)])
        arm = rows.loc[ids].copy()
        assert len(arm) == total and arm.source_label.value_counts().reindex([0, 1], fill_value=0).to_dict() == quota
        arms[name] = arm
    return arms


def record_selection(folder, mode, seed, split_hash, selection, id_column, label_column):
    ordered = selection.sort_values(["experiment", id_column])
    manifest = ordered.to_csv(index=False, lineterminator="\n").encode("utf-8")
    description = {"mode": mode, "sampling_seed": seed, "split_hash": split_hash,
                   "selection_sha256": hashlib.sha256(manifest).hexdigest()}
    path = folder / "matched_design.json"
    if (path.exists() and json.loads(path.read_text(encoding="utf-8")) != description
            or not path.exists() and any(folder.glob("*_seed*/summary.json"))):
        raise ValueError("续跑目录与本次抽样设计不一致，请使用新的运行目录。")
    counts = ordered.groupby(["experiment", label_column]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=range(4 if label_column == "Sentiment" else 2), fill_value=0)
    counts["total"] = counts.sum(axis=1)
    counts.to_csv(folder / "matched_counts.csv")
    (folder / "matched_selection.csv").write_bytes(manifest)
    path.write_text(json.dumps(description, indent=2), encoding="utf-8")
    print("共同入选样本数", ordered.groupby(id_column).experiment.nunique().eq(2).sum())
    return counts
