"""Build Kai's single, offline-contained Colab notebook from the model folders."""
import io
import json
import runpy
import tokenize
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "notebooks/kai/Kai_Sentiment_Experiments.ipynb"
LABEL, RAT, CF = "label-only", "rationale-distillation", "counterfactual"


def recipes():
    label = runpy.run_path(str(HERE / LABEL / "notebook.py"))["notebook"]
    rationale = runpy.run_path(str(HERE / RAT / "notebook.py"), {"original": label})["cells"]
    counterfactual = runpy.run_path(str(HERE / CF / "notebook.py"), {"prior": {"cells": rationale}})["cells"]
    return {LABEL: label["cells"], RAT: rationale, CF: counterfactual}


def indent_code(source):
    # Indent statements while preserving every byte inside multiline prompt literals.
    interior = set()
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.STRING:
            interior.update(range(token.start[0] + 1, token.end[0] + 1))
    return "\n".join(line if n in interior or not line else "    " + line
                     for n, line in enumerate(source.splitlines(), 1))


def build():
    parts = recipes()
    source = {mode: {cell["id"]: cell["source"] for cell in part} for mode, part in parts.items()}
    cells = []

    def add(kind, key, body, modes=()):
        if modes:
            body = f"if RUN_MODE in {tuple(modes)!r}:\n" + indent_code(body)
        cell = {"cell_type": kind, "id": key, "metadata": {}, "source": body.strip() + "\n"}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    def code(key, body, modes=()):
        add("code", key, body, modes)

    def step(mode, key, modes=None):
        code(f"{mode}-{key}", source[mode][key], modes or [mode])

    add("markdown", "intro", """# Kai 的 Twitter 情感分类实验

选择一种运行模式，然后在新的 Colab 运行时中点击「全部运行」。更换模式后从第一格重新开始。
数据快照内嵌于本文件。默认使用 473 条训练样本、118 条验证样本和 397 条测试样本。

| RUN_MODE | 本次执行 | 所需资源 |
|---|---|---|
| label-only | 原始标签 LoRA 微调、早停与测试评估 | GPU |
| rationale-distillation | 原始标签、未筛选解释、筛选解释对照 | GPU 和教师缓存或 API 密钥 |
| counterfactual | 筛选解释组与加入反事实改写组的配对比较 | GPU 和教师缓存或 API 密钥 |

从零生成解释，每条训练推文需要两次教师判断，共 946 个请求，重试另计。
反事实模式默认复用解释缓存，最多为 80 个训练样本生成并核验改写，最多 160 个不同的新请求。
如需教师 API，请在 Colab Secrets 设置 TEACHER_API_KEY。""")
    code("mode", '''# @title 选择本次实验
RUN_MODE = "label-only" # @param ["label-only", "rationale-distillation", "counterfactual"]
if RUN_MODE not in {"label-only", "rationale-distillation", "counterfactual"}:
    raise ValueError("请选择表中的运行模式。")
print("本次运行", RUN_MODE)''')
    packages = source[CF]["install"].strip().removeprefix("%pip ")
    code("install", f'get_ipython().run_line_magic("pip", {packages!r})')
    add("markdown", "settings", """## 实验配置与数据

神经模型请选择 GPU。解释蒸馏默认生成缺少的教师记录，反事实模式默认只读上一版解释缓存。
在反事实模式中，可将 DISTILLATION_EXPORT 指向上一版 ZIP 或已解压的结果文件夹。
自定义缓存位置请填写 TEACHER_CACHE_ROOT，反事实缓存使用 CF_CACHE_ROOT。
SAVE_TO_DRIVE 控制是否将结果持久保存到 Google Drive。""")
    cfg = source[CF]["config"]
    cfg = cfg.replace("SAVE_TO_DRIVE = True", 'SAVE_TO_DRIVE = RUN_MODE in {"rationale-distillation", "counterfactual"}')
    cfg = cfg.replace('ALLOW_TEACHER_REQUESTS = False', 'ALLOW_TEACHER_REQUESTS = RUN_MODE == "rationale-distillation"')
    cfg = cfg.replace('SEEDS = [42]', 'SEEDS = [42]\nRUN_UNFILTERED = True')
    start, end = cfg.index("if SAVE_TO_DRIVE:"), cfg.index("RUN_ID =")
    cfg = cfg[:start] + '''run_folder = {"label-only": "qwen_lora",
              "rationale-distillation": "qwen_distillation", "counterfactual": "qwen_counterfactual"}[RUN_MODE]
if SAVE_TO_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    OUTPUT_ROOT = Path("/content/drive/MyDrive") / ("dsai4205_" + run_folder)
else:
    OUTPUT_ROOT = Path("/content") / (run_folder + "_runs")
''' + cfg[end:]
    start = cfg.index("import gc\n")
    cfg = cfg[:start] + 'if RUN_MODE in ("rationale-distillation", "counterfactual"):\n' + indent_code(cfg[start:])
    code("config", cfg)
    code("prior-default", "prior_protocol = None")
    step(CF, "import-export")
    code("embedded-data", source[LABEL]["embedded-data"])
    code("data", source[CF]["data"])
    add("markdown", "teacher-note", source[RAT]["teacher-note"])
    for key in ["teacher-functions", "teacher-run"]:
        step(RAT, key, [RAT, CF])
    add("markdown", "token-note", source[LABEL]["tokenization-note"])
    code("label-only-tokenization", source[LABEL]["tokenization"])
    add("markdown", "label-note", "## 原始标签微调\n\n本模式按验证 macro-F1 选取最佳 adapter，支持早停。实验完成后下载 ZIP，最后一格可试用自定义推文。")
    for key in ["model", "train", "inference", "test", "export", "demo"]:
        step(LABEL, key)
    add("markdown", "aux-note", source[RAT]["training-note"])
    common, extra = source[RAT]["multitask"].split("\nEXPERIMENTS =", 1)
    assert source[CF]["multitask"].startswith(common.rstrip())
    code("shared-multitask", common, [RAT, CF])
    code("rationale-datasets", "EXPERIMENTS =" + extra, [RAT])
    code("cf-originals", source[CF]["multitask"][len(common.rstrip()):], [CF])
    add("markdown", "cf-note", source[CF]["generation-note"])
    for key in ["counterfactual-functions", "generate", "paired-training"]:
        step(CF, key)
    add("markdown", "compare-note", """## 训练对照组与导出

解释蒸馏模式以原始标签组为参照。反事实模式以仅筛选解释组为参照。
两种模式均从相同的基础模型分别训练本次对照组，按验证 macro-F1 选模型。
查看 comparison.csv 的 delta_macro_f1，配对区间保存在 paired_bootstrap.csv。""")
    for key in ["metrics", "inference", "model-factory"]:
        assert source[RAT][key] == source[CF][key]
        step(RAT, key, [RAT, CF])
    for mode in [RAT, CF]:
        for key in ["experiments", "test", "export"]:
            step(mode, key)
    metadata = {"accelerator": "GPU", "colab": {"name": OUT.name, "gpuType": "T4"},
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
    out = {"cells": cells, "metadata": metadata, "nbformat": 4, "nbformat_minor": 5}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    build()
