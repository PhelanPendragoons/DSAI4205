# DSAI4205 Group Project — Twitter Sentiment Classification

四分类：`0 Negative / 1 Positive / 2 Irrelevant / 3 Neutral`。
原始数据包含 596 条训练样本、397 条测试样本；既有去重流程保留 591 条训练池样本。

## Colab notebooks

| 实验 | Notebook | Colab |
|---|---|---|
| Qwen3-0.6B + LoRA，原始标签微调 | [原始标签版本](notebooks/Qwen3_0_6B_LoRA_Sentiment_Colab.ipynb) | [打开 Colab](https://colab.research.google.com/github/once123456/DSAI4205/blob/main/group%20project/notebooks/Qwen3_0_6B_LoRA_Sentiment_Colab.ipynb) |
| 教师短解释生成、筛选与 LoRA 蒸馏对照 | [解释蒸馏版本](notebooks/Qwen3_0_6B_LoRA_Rationale_Distillation_Colab.ipynb) | [打开 Colab](https://colab.research.google.com/github/once123456/DSAI4205/blob/main/group%20project/notebooks/Qwen3_0_6B_LoRA_Rationale_Distillation_Colab.ipynb) |

两个 notebook 都内嵌相同 CSV 快照，使用 Transformers Trainer 和 PEFT。
默认划分为 **473 train / 118 validation / 397 test**。
在 Colab 启用 NVIDIA GPU，从安装单元开始顺序运行。

解释蒸馏版本默认使用 DeepSeek 的 OpenAI 兼容 Chat Completions 接口。
在 Colab Secrets 中配置 `TEACHER_API_KEY`；其他服务修改地址、模型和额外参数。
完整教师生成通常需要 946 次请求，网络重试另计；请先检查请求上限与账户费用。
默认保存到 Google Drive，以便复用教师缓存。密钥不会写入导出文件。

## 实验约定

- 教师只读取训练文本，不接收正确标签，不接触验证或测试样本。
- 未筛选解释组仍执行最小格式、长度检查；筛选组另检查两次独立判断、原始标签、原文引文和缺失目标信息。
- 三组分类训练均保留全部原始训练样本，解释作为单独辅助任务的输出。
- 解释蒸馏 notebook 内的对照组从相同基础权重重新训练；固定 epoch 预算，按验证 macro-F1 选模型。
- 第一版原始标签 notebook 使用早停；两版训练预算设置不同，优先使用蒸馏 notebook 内部的对照结果。
- 复现实验应保存 `split_manifest.csv`、模型 revision、教师缓存、配置及随机种子。
- 测试数据仅用于最终报告。反事实增强尚未实现。

## 文件结构

- `Data 5 - Twitter/`：原始 CSV 与数据说明。
- `data/processed/`：处理后数据与删除行记录。
- `scripts/`：数据分析、TF-IDF baseline、notebook 构建与集成检查脚本。
- `notebooks/`：可独立上传运行的 Colab notebook。
- `reports/`：数据分析、已有 baseline 指标及 notebook 验证记录。
- `figures/`：EDA 和 TF-IDF baseline 图表。

已有 TF-IDF + LR 全测试集结果：accuracy **0.5063**，macro-F1 **0.4662**。
来源：[baseline_metrics.json](reports/baseline_metrics.json)。
Notebook 验证采用真实分词器与缩小随机初始化的 Qwen3；教师接口检查使用模拟响应。
这些是程序正确性检查，不代表正式模型效果。完整 Qwen3 GPU 实验指标需在 Colab 运行后获得。

## 重新构建 notebook

```text
python "group project/scripts/build_qwen_lora_notebook.py"
python "group project/scripts/build_qwen_distillation_notebook.py"
```

验证脚本需要 notebook 的依赖以及 `nbformat`、`IPython`，详细用法见脚本开头。
依赖安装、训练、推理和导出说明均在 notebook 内。
