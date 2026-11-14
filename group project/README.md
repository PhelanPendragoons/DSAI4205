# Twitter 情感分类小组项目

本目录存放英文推文四分类的数据、代码、notebook 和分析报告。完整项目说明、课程交付要求、固定数据划分和共享基线结果见[仓库首页](../README.md)。

## 共享 baseline

TF-IDF + 逻辑回归在 397 条测试推文上取得的 macro-F1 为 0.4461（473 条训练子集）和 0.4662（591 条完整训练池）。在 Colab 的 CPU 运行时运行共享 notebook 即可复现。[模型说明](models/tfidf-lr/) · [基线 notebook](notebooks/TFIDF_LR_Baseline.ipynb) · [完整训练池指标](reports/baseline_metrics.json)

## 项目文件

| 内容 | 位置 |
|---|---|
| 课程说明和原始数据 | [项目要求](Project%20description%20%281%29.pdf) · [Twitter 数据集](Data%205%20-%20Twitter/) |
| 处理后的数据与固定划分 | [data/processed/](data/processed/) |
| 共享清理和分析脚本 | [scripts/](scripts/) |
| 分析报告与图表 | [reports/](reports/) · [figures/](figures/) |
| Kai | [models/kai/](models/kai/) · [notebooks/kai/Kai_Sentiment_Experiments.ipynb](notebooks/kai/Kai_Sentiment_Experiments.ipynb) |
