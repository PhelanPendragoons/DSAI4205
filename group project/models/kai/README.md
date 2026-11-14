# Kai 的情感分类模型

本实验研究在只有几百条标注推文时，教师模型提供的短解释和情感翻转样本能否改善 Qwen3-0.6B 的四分类表现。标签编码为 0（Negative）、1（Positive）、2（Irrelevant）、3（Neutral）。

三组实验都用同一 Qwen 解码器对四个标签代码打分，改变的是训练监督。按课程要求统计进阶分析模型数量时，这三组属于同一模型的消融对照。课程要求的“词嵌入模型搭配独立分析分类器”还需由对应的模型实现体现。[课程建模要求](../../Project%20description%20%281%29.pdf)

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

## 对照实验与结果分析

| 问题 | 如何比较与解读 |
|---|---|
| 原始标签微调能否超过共享基线？ | 在相同的 473 条训练、397 条测试推文上，比较原始标签 LoRA 与共享基线的 macro-F1、准确率和各类召回率。该划分的基线 macro-F1 为 **0.4461**，准确率为 **0.4912**。 |
| 教师短解释是否有帮助？筛选起了什么作用？ | 在蒸馏模式内比较同一随机种子的三组结果。格式有效解释组与原始标签组之差衡量加入解释后的变化。筛选组与格式有效组之差衡量采用筛选规则后的整体变化。结合 `teacher_filter_summary.csv` 的覆盖率解读后一个差值，因为筛选会同时改变解释的内容和数量。 |
| 情感翻转样本能否在解释蒸馏之上带来增益？ | 两组共用原始标签、筛选解释缓存和训练划分，增强组额外学习通过核验的正负向改写。查看 `comparison.csv` 中相对仅解释组的 `delta_macro_f1`，并用 `cf_audit.csv` 统计两种翻转方向的接受数。分别检查四类 F1，以判断正负类变化是否伴随 Neutral 或 Irrelevant 的性能变化。 |

**已验证的结果。** 本地用缩小的 Qwen3 在 CPU 上跑通了三种模式。原始标签流程检查覆盖数据划分、LoRA 参数更新和 adapter 重载。蒸馏与反事实流程用模拟教师或缓存检查筛选、对照训练和结果导出。[原始标签检查记录](../../reports/qwen_lora_notebook_validation.json) · [蒸馏检查记录](../../reports/qwen_distillation_notebook_validation.json) · [反事实检查脚本](counterfactual/validate.py)

**正式测试结果。** 仓库目前尚未收录预训练 Qwen3-0.6B 在全部 397 条测试推文上的正式成绩，解释筛选和反事实增强的性能增益仍待验证。完整运行后，原始标签组的指标在 `test_metrics.json` 中。蒸馏和增强实验分别导出 `comparison.csv`、各类别报告和 `paired_bootstrap.csv`，后者给出同一随机种子下测试 macro-F1 差值的 95% 配对重采样区间。跨训练种子的波动另看 `comparison_aggregate.csv`。

## 修改和构建

三个 Qwen 模型目录中的 `notebook.py` 提供对应实验单元，`validate.py` 用于小模型流程检查。解释蒸馏复用原始标签微调的分类逻辑，反事实实验复用解释蒸馏的筛选和训练逻辑。

在 `group project` 目录运行以下命令，将这些单元写入 `notebooks/kai/Kai_Sentiment_Experiments.ipynb`。提交修改时一并提交更新后的 notebook。

```bash
python models/kai/build_notebook.py
```
