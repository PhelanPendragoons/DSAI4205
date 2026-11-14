# 筛选解释蒸馏与反事实增强

在四分类情感任务中，本实验检验一个局部改动是否有帮助。对于语义明确的正面或负面训练推文，教师替换一个短语，翻转情感，同时保留原句的话题、评价对象和写法。例如，手写示例 “The update fixed my problem” 可以改为 “The update caused my problem”。每个候选都要经过程序检查和教师核验。

在首个单元将 `RUN_MODE` 设为 `counterfactual`，再点击「全部运行」。[在 Colab 打开 Kai 的统一 notebook](https://colab.research.google.com/github/once123456/DSAI4205/blob/main/group%20project/notebooks/kai/Kai_Sentiment_Experiments.ipynb) · [查看 notebook](../../../notebooks/kai/Kai_Sentiment_Experiments.ipynb)

## 开始实验

1. 在 Colab 选择 GPU，将上一版解释蒸馏导出的 ZIP 放到 Google Drive。设置 `DISTILLATION_EXPORT` 为文件路径，例如 `/content/drive/MyDrive/qwen3_rationale_distillation.zip`。导入已解压的结果目录也可以。
2. 如果上一版采用默认 Drive 目录，直接运行也能找到已有的 `dsai4205_qwen_distillation/teacher_cache`。单独提供缓存时可设置 `TEACHER_CACHE_ROOT`，自定义过划分时填写对应的 `SPLIT_MANIFEST_PATH`。原始解释缓存默认只读。需要付费补齐缺项时开启 `ALLOW_TEACHER_REQUESTS`。
3. 将教师服务的密钥存入 Colab Secret `TEACHER_API_KEY`。从上一版 ZIP 导入时沿用其教师配置。首次生成反事实会产生新的请求，已有反事实缓存可通过 `CF_CACHE_ROOT` 复用。
4. 默认选最多 80 个父样本，两个翻转方向轮流取样。每个父样本最多产生一次生成和一次核验请求，重试另计。`CF_REQUEST_BUDGET` 限制一次生成单元执行的请求数，重跑可继续读取已完成的记录。

## 两组训练

| 组别 | 原始训练监督 | 额外监督 |
|---|---|---|
| `rationale_filtered` | 全部人工标签、筛选后的短解释 | 无 |
| `rationale_filtered_cf` | 相同的人工标签和短解释 | 使用合格改写预测翻转标签 |

默认 473 条训练、118 条验证、397 条测试。程序按父样本绑定改写，两组使用相同的初始化、每个 epoch 的原始样本数和优化步数。增强组多计算一次包含改写的前向与反向传播。默认解释权重和反事实权重均为 0.5。每个 batch 的反事实损失按有改写的原始行比例加权。

每个父样本最多接受一条改写。生成器提出原文中唯一的短语及替代词。核验器只看原句和改写句，独立给出两个标签、情感证据和语境判断。程序检查翻转方向、改写长度、原文重合度、证据、去重和学生的输入长度。测试文本仅用于本地重复指纹检查和最终评估。两个请求使用同一个教师，仍可能对同一条推文产生相同误判。

## 查看结果

- `comparison.csv` 包含各 seed 的测试 accuracy、macro-F1、相对仅解释蒸馏组的差值、实际增强数和训练时间。
- `paired_bootstrap.csv` 给出两组在同一测试样本上的 macro-F1 差值区间。`comparison_aggregate.csv` 汇总多个 seed。
- `cf_audit.csv` 可按父样本、方向或拒绝原因检查每次尝试。`accepted_counterfactuals.csv` 保存实际接受的改写。
- 每组目录包含 adapter、验证指标、测试分类报告、混淆矩阵和逐条预测。ZIP 同时包含划分清单、实验配置、原始解释缓存和反事实缓存。

正式效果以完整训练导出的比较表为准。本地集成检查使用模拟教师与缩小的 Qwen3。
