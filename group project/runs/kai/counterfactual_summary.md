# counterfactual 结果汇总

- Run: `qwen_counterfactual_runs/20261114T102655_279072Z_seed42`
- 导入蒸馏 ZIP：`qwen3_rationale_distillation.zip`（deepseek-v4-pro）
- 反事实 API：142 次新请求；父样本 80，通过核验 **46**
- ZIP：`qwen3_counterfactual.zip`

## 反事实接受情况

| source_label | attempted | accepted | rate |
|---|---:|---:|---:|
| Negative (0) | 40 | 14 | 35% |
| Positive (1) | 40 | 32 | 80% |

主要拒绝原因：`wrong_polarity` 15、`nonlocal_or_protected_edit` 13。

## 测试对照（相对 rationale_filtered）

| 实验 | test macro-F1 | test accuracy | delta_macro_f1 | 95% CI |
|---|---:|---:|---:|---|
| rationale_filtered | 0.4855 | 0.5365 | 0 | — |
| rationale_filtered_cf | 0.4873 | 0.4987 | **+0.0017** | [-0.0408, 0.0471] |

## 结论

加入 46 条通过核验的情感翻转样本后，macro-F1 仅 **+0.17 pp**，bootstrap 区间跨 0。负向父样本接受率偏低，增强信号有限；本设置下反事实增强未显示可靠增益。
