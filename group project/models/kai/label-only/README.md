# Qwen3-0.6B + LoRA 原始标签微调

这个模型使用推文和人工标签学习四分类。输入保留原文的大小写、标点和表情。训练只监督回答中的标签代码和结束符，推理时选择分数最高的标签。

在首个单元将 `RUN_MODE` 设为 `label-only`，再点击「全部运行」。[在 Colab 打开 Kai 的统一 notebook](https://colab.research.google.com/github/once123456/DSAI4205/blob/main/group%20project/notebooks/kai/Kai_Sentiment_Experiments.ipynb) · [查看 notebook](../../../notebooks/kai/Kai_Sentiment_Experiments.ipynb)

在 Colab 选择 NVIDIA GPU 后按顺序运行。Notebook 内嵌数据，默认使用 473 条训练样本、118 条验证样本和 397 条测试样本，按验证 macro-F1 选取最佳 adapter。这个版本支持早停，并在相同训练子集上重跑 TF-IDF 对照。

下载结果 ZIP 可取得划分清单、LoRA adapter、原始配置、逐条测试预测和分类指标。教师解释实验会在其内部重新训练同预算的对照组。
