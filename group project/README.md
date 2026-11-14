# Twitter 情感分类小组项目

本目录包含数据、建模代码和分析报告。任务为推文级四分类，标签编码为 0 Negative、1 Positive、2 Irrelevant、3 Neutral。项目背景与主要评估设置见[仓库首页](../README.md)。

## 运行模型

每个模型目录包含自己的方法说明、运行入口和结果解读。

| 模型目录 | 输入与产物 |
|---|---|
| [TF-IDF + 逻辑回归](models/tfidf-lr/README.md) | 清洗后的推文，输出分类报告和词项权重 |
| [Qwen3 + LoRA 原始标签](models/label-only/README.md) | 推文及人工标签，输出分类 adapter 和固定划分 |
| [教师短解释蒸馏](models/rationale-distillation/README.md) | 原始标签和训练集解释，输出解释缓存与对照结果 |
| [解释蒸馏加反事实增强](models/counterfactual/README.md) | 复用已有划分和解释缓存，输出成对改写审计及相对仅解释蒸馏的指标差值 |

三个 Colab notebook 内嵌相同数据快照。历史发布的前两个 notebook 链接仍可通过 `notebooks/` 打开，构建脚本会同步这些入口。

## 数据和报告

- [原始数据与数据说明](Data%205%20-%20Twitter/) 和 [课程项目说明](Project%20description%20%281%29.pdf)。
- [处理后的数据](data/processed/) 和[数据清理记录](reports/数据清理与分析.md)。原始训练池 596 条，清理后 591 条。
- [数据分析报告](reports/Twitter情感数据技术分析报告.md)、[图表](figures/) 和 [TF-IDF 测试结果](reports/baseline_metrics.json)。
- [构建和验证脚本](scripts/) 可重新生成各 notebook。构建顺序为 `build_qwen_lora_notebook.py`、`build_qwen_distillation_notebook.py`、`build_qwen_counterfactual_notebook.py`。

默认模型划分为 **473 train / 118 validation / 397 test**。神经模型使用保留大小写、标点和表情的轻度清洗原文。比较各训练方法时，请使用同一数据划分和同一 notebook 中对应的对照结果。
