# Kai 的情感分类模型

本实验研究在只有几百条标注推文时，教师模型提供的短解释和情感翻转样本能否改善 Qwen3-0.6B 的四分类表现。标签编码为 0（Negative）、1（Positive）、2（Irrelevant）、3（Neutral）。

## 在 Colab 运行

三组 Qwen 实验共用一个 notebook。在 Colab 选择 GPU，在首格的 `RUN_MODE` 下拉框选择实验，再点「全部运行」。[打开 Colab](https://colab.research.google.com/github/once123456/DSAI4205/blob/main/group%20project/notebooks/kai/Kai_Sentiment_Experiments.ipynb) · [查看 notebook](../../notebooks/kai/Kai_Sentiment_Experiments.ipynb)

| RUN_MODE | 模型与实验 |
|---|---|
| `label-only` | [原始标签 LoRA 微调](label-only/README.md)，按验证集 macro-F1 早停 |
| `rationale-distillation` | [短解释蒸馏](rationale-distillation/README.md)，比较原始标签、未筛选解释和筛选解释三种训练方式 |
| `counterfactual` | [反事实增强](counterfactual/README.md)，比较仅用筛选解释训练与加入情感翻转样本后的表现 |

首次生成解释时，在 Colab Secrets 中设置 `TEACHER_API_KEY`，运行解释蒸馏并保留导出的 ZIP。接着运行反事实实验，将 `DISTILLATION_EXPORT` 指向该 ZIP，即可沿用同一份数据划分和教师解释缓存。首次生成反事实样本也会调用教师 API。

## 数据与评估

原始训练池有 596 条推文，处理重复文本和跨集重叠后保留 591 条。神经模型使用保留大小写、标点和表情的轻度清洗文本，默认划分为 **473 条训练、118 条验证、397 条测试**。比较方法时使用相同划分，按验证集 macro-F1 选取模型，在测试集报告 macro-F1、准确率和各类别指标。[原始数据](../../Data%205%20-%20Twitter/) · [处理后数据](../../data/processed/) · [清理记录](../../reports/数据清理与分析.md)

Qwen 的测试结果由 Colab 运行后导出。要与小组的线性基线比较，可将原始标签微调导出的 ZIP 提供给[共享基线 notebook](../../notebooks/TFIDF_LR_Baseline.ipynb)，按同一训练划分和测试样本比较。

项目还提供[课程说明](../../Project%20description%20%281%29.pdf)、[数据分析报告](../../reports/Twitter情感数据技术分析报告.md)、[图表](../../figures/) 和[数据处理脚本](../../scripts/)。

## 修改和构建

三个 Qwen 模型目录中的 `notebook.py` 提供对应实验单元，`validate.py` 用于小模型流程检查。解释蒸馏复用原始标签微调的分类逻辑，反事实实验复用解释蒸馏的筛选和训练逻辑。

在 `group project` 目录运行以下命令，将这些单元写入 `notebooks/kai/Kai_Sentiment_Experiments.ipynb`。提交修改时一并提交更新后的 notebook。

```bash
python models/kai/build_notebook.py
```
