# DSAI4205

这个课程项目研究英文 Twitter 推文的四分类情感识别。我们关心在几百条人工标注的条件下，教师提供的短解释和少量情感翻转样本能否改善小语言模型的表现。标签为 Negative、Positive、Irrelevant 和 Neutral。

## 数据与评估

原始训练集包含 596 条推文。处理重复文本和跨集重叠后，训练池保留 591 条，神经模型默认使用 473 条训练、118 条验证、397 条测试。按验证集 macro-F1 选择模型，并报告测试 macro-F1、accuracy 和各类别指标。数据说明、处理记录和分析图表列在[项目目录](group%20project/README.md)。

用完整训练池拟合的 TF-IDF + 逻辑回归在测试集上取得 macro-F1 0.4662、accuracy 0.5063。神经模型的实验结果由各自的 Colab notebook 导出。[基线报告](group%20project/reports/baseline_metrics.json)

## 成员实验

| 成员 | 模型 | 运行入口 |
|---|---|---|
| Kai | [线性基线、原始标签微调、短解释蒸馏和反事实增强](group%20project/models/kai/README.md) | [在 Colab 打开](https://colab.research.google.com/github/once123456/DSAI4205/blob/main/group%20project/notebooks/kai/Kai_Sentiment_Experiments.ipynb) |

模型和 notebook 按成员归档。其他组员可分别在 `group project/models/<成员名>/` 和 `group project/notebooks/<成员名>/` 添加各自的工作，再将入口补到上表。
