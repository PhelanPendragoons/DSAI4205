# TF-IDF + Logistic Regression

项目的线性基线使用清洗与词形还原后的推文。向量化器提取词级一元与二元特征，设置 `min_df=2`、`sublinear_tf=True`。逻辑回归使用 `C=1`。

用 591 条训练池样本拟合、在 397 条测试样本上评估的已有结果为 accuracy **0.5063**、macro-F1 **0.4662**。逐类指标见 [baseline_metrics.json](../../reports/baseline_metrics.json)。

在项目目录中运行 `python scripts/baseline_tfidf_lr.py` 可重建基线报告与图表。[训练脚本](../../scripts/baseline_tfidf_lr.py)

原始标签 LoRA notebook 还会在相同的 473 条训练行上重跑本模型，便于比较训练规模一致时的结果。[原始标签模型](../label-only/README.md)
