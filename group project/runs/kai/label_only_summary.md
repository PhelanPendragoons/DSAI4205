# label-only 结果汇总

- Run: `qwen_lora_runs/20261114T085807_400102Z_seed42`
- 模型: Qwen3-0.6B + LoRA；验证选 checkpoint-180
- 验证 macro-F1: **0.5374**；训练约 611s；峰值显存 2.62 GiB

## 测试集（397）

| 指标 | 本实验 | TF-IDF+LR 基线 | 差值 |
|---|---:|---:|---:|
| macro-F1 | **0.4850** | 0.4461 | +0.0389 |
| accuracy | **0.5088** | 0.4912 | +0.0176 |

### 各类别 F1

| 类别 | precision | recall | f1 |
|---|---:|---:|---:|
| Negative | 0.618 | 0.633 | 0.626 |
| Positive | 0.529 | 0.604 | 0.564 |
| Irrelevant | 0.253 | 0.318 | 0.282 |
| Neutral | 0.586 | 0.390 | 0.469 |

## 结论

原始标签 LoRA 在同划分测试上超过共享基线（macro-F1 +3.9 个百分点）。Irrelevant 仍最弱；Neutral 召回偏低。
