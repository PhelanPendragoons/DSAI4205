# Twitter 情感分类小组项目

本目录包含数据、建模代码和分析报告。任务为推文级四分类，标签编码为 0 Negative、1 Positive、2 Irrelevant、3 Neutral。项目背景与主要评估设置见[仓库首页](../README.md)。

## 运行模型

模型说明与实现位于 `models/<成员名>/`，Colab notebook 位于 `notebooks/<成员名>/`。添加模型时在自己的目录中修改代码和说明，并在此处登记运行入口。

| 成员 | 模型说明 | Notebook |
|---|---|---|
| Kai | [四组情感分类实验](models/kai/README.md) | [统一 Colab notebook](notebooks/kai/Kai_Sentiment_Experiments.ipynb) |

## 数据和报告

- [原始数据与数据说明](Data%205%20-%20Twitter/) 和 [课程项目说明](Project%20description%20%281%29.pdf)。
- [处理后的数据](data/processed/) 和[数据清理记录](reports/数据清理与分析.md)。原始训练池 596 条，清理后 591 条。
- [数据分析报告](reports/Twitter情感数据技术分析报告.md)、[图表](figures/) 和 [TF-IDF 测试结果](reports/baseline_metrics.json)。
- [共享的数据清理与分析脚本](scripts/)；各模型的运行与构建说明见对应成员目录。

默认模型划分为 **473 train / 118 validation / 397 test**。神经模型使用保留大小写、标点和表情的轻度清洗原文。比较各训练方法时，请使用同一数据划分和同一 notebook 中对应的对照结果。
