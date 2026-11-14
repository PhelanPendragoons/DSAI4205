# Kai 的情感分类模型

四种实验使用同一份数据快照和一个 Colab 入口。在首格下拉框选择 `RUN_MODE`，再点「全部运行」。[打开 Colab](https://colab.research.google.com/github/once123456/DSAI4205/blob/main/group%20project/notebooks/kai/Kai_Sentiment_Experiments.ipynb) · [查看 notebook](../../notebooks/kai/Kai_Sentiment_Experiments.ipynb)

| RUN_MODE | 模型与实验 |
|---|---|
| `tfidf-lr` | [TF-IDF + 逻辑回归](tfidf-lr/README.md)，比较 473 条和 591 条训练样本的线性基线 |
| `label-only` | [原始标签 LoRA 微调](label-only/README.md)，使用验证集早停 |
| `rationale-distillation` | [短解释蒸馏](rationale-distillation/README.md)，比较原始标签、未筛选解释和筛选解释 |
| `counterfactual` | [反事实增强](counterfactual/README.md)，比较仅筛选解释与加入情感翻转样本 |

## 修改和构建

线性模型目录中的 `experiment.py` 提供 Colab 单元，`baseline_tfidf_lr.py` 生成完整的基线报告。三个 Qwen 模型目录中的 `notebook.py` 提供对应实验单元，`validate.py` 用于小模型流程检查。解释蒸馏复用原始标签微调的分类逻辑，反事实实验复用解释蒸馏的筛选和训练逻辑。

在 `group project` 目录运行以下命令，将这些单元写入 `notebooks/kai/Kai_Sentiment_Experiments.ipynb`。提交修改时一并提交更新后的 notebook。

```bash
python models/kai/build_notebook.py
```
