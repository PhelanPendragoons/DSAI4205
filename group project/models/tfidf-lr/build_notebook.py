"""Build the standalone CPU baseline notebook from the project data and split."""
import base64
import csv
import gzip
import io
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]


def snapshot(filename, columns):
    with (PROJECT / "data/processed" / filename).open(encoding="utf-8-sig", newline="") as data:
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(csv.DictReader(data))
    return base64.b64encode(gzip.compress(buffer.getvalue().encode(), mtime=0)).decode()


def build():
    columns = ["row_id", "Sentiment", "text_clean", "Phrase"]
    dataset = {"train": snapshot("train_cleaned.csv", columns), "test": snapshot("test_cleaned.csv", columns),
               "split": snapshot("split_manifest.csv", ["row_id", "Sentiment", "group_id", "partition"])}
    sections = [
        ("markdown", "intro", "# 小组共享基线 · TF-IDF + 逻辑回归\n\n在 Colab 的 CPU 运行时点击「全部运行」。"
         "实验分别使用固定训练子集和完整训练池拟合模型，并在同一测试集评估。"
         "如需与 Qwen 原始标签微调比较，将该次实验的结果 ZIP 上传至 Colab，在配置格填写其路径。"),
        ("code", "install", """get_ipython().run_line_magic("pip", 'install -q "pandas==2.2.3" "scikit-learn==1.6.1"')"""),
        ("code", "config", '# @title 可选的 Qwen 原始标签微调结果\nQWEN_EXPORT = "" # @param {type:"string"}'),
        ("code", "data", "SNAPSHOT = " + repr(dataset)),
        ("code", "experiment", (HERE / "experiment.py").read_text(encoding="utf-8")),
    ]
    cells = []
    for kind, name, source in sections:
        cell = {"cell_type": kind, "id": name, "metadata": {}, "source": source.rstrip() + "\n"}
        if kind == "code":
            cell.update(outputs=[], execution_count=None)
        cells.append(cell)
    result = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
              "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}}
    path = PROJECT / "notebooks/TFIDF_LR_Baseline.ipynb"
    path.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build()
