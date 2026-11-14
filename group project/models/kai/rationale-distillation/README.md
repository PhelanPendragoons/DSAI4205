# Qwen3-0.6B + LoRA 短解释蒸馏

教师在同一条训练推文上使用两个提示独立判断，给出类别、原文引文和短解释。筛选规则检查两次判断与人工标签的一致性、原文引文和目标实体信息。全部训练行保留原始分类监督，通过筛选的解释用于辅助生成任务。

在首个单元将 `RUN_MODE` 设为 `rationale-distillation`，再点击「全部运行」。[在 Colab 打开 Kai 的统一 notebook](https://colab.research.google.com/github/once123456/DSAI4205/blob/main/group%20project/notebooks/kai/Kai_Sentiment_Experiments.ipynb) · [查看 notebook](../../../notebooks/kai/Kai_Sentiment_Experiments.ipynb)

启用 Colab GPU，在 Secrets 中设置 `TEACHER_API_KEY`。默认使用 DeepSeek 兼容接口。默认划分有 473 条训练行，空缓存完整生成需要 946 个不同请求，重试另计。Drive 缓存支持续跑。

Notebook 对比原始标签、格式有效的解释以及筛选后的解释，三组使用相同训练预算。结果 ZIP 包含模型、比较表、逐条预测、教师筛选审计、`teacher_cache`、`split_manifest.csv` 和 `experiment_protocol.json`。保存此 ZIP，后续反事实实验可以直接导入。[反事实模型说明](../counterfactual/README.md)
