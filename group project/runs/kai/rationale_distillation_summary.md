# rationale-distillation 结果汇总

- Run: `qwen_distillation_runs/20261114T090915_982382Z_seed42`
- 教师: DeepSeek `deepseek-v4-pro`（thinking disabled），946 次请求，约 326k tokens
- ZIP: `qwen3_rationale_distillation.zip`

## 教师筛选覆盖率

| Sentiment | total | format_valid | accepted | acceptance_rate |
|---|---:|---:|---:|---:|
| Negative | 140 | 136 | 84 | 60.0% |
| Positive | 132 | 130 | 78 | 59.1% |
| Irrelevant | 85 | 83 | 6 | 7.1% |
| Neutral | 116 | 115 | 25 | 21.6% |

辅助解释行数：unfiltered **464** / filtered **193**（共 473 训练行）。

## 测试对照（seed=42，相对 label_only）

| 实验 | test macro-F1 | test accuracy | delta_macro_f1 | 95% bootstrap CI |
|---|---:|---:|---:|---|
| label_only | 0.4745 | 0.4912 | 0 | — |
| rationale_unfiltered | **0.4886** | 0.5063 | **+0.0141** | [-0.0195, 0.0508] |
| rationale_filtered | 0.4759 | 0.5063 | +0.0014 | [-0.0381, 0.0442] |

## 结论

- 加入格式有效短解释（unfiltered）相对同对照 label_only 有约 **+1.4** 个百分点 macro-F1，但 bootstrap 区间跨 0，单种子下尚不能断言显著。
- 严格筛选后 accepted 仅约 41% 训练行（尤其 Irrelevant/Neutral 很低），filtered 几乎回到 label_only（+0.14 pp），说明**筛得过严、覆盖不足**，本设置下筛选未带来额外增益。
- 与独立第一阶段 label-only（test macro-F1 0.485）相比，蒸馏内 label_only 为 0.475：同协议对照以本表为准；跨 run 波动约 1 pp。
