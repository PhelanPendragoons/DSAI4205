"""TF-IDF + logistic regression baseline for the four-class tweet task.

Trains on data/processed/train_cleaned.csv and evaluates on the cleaned test set.
The one test tweet that also appeared in the raw training file is scored again
after it is removed, because its train and test labels disagree.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline

BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data" / "processed"
FIG_DIR = BASE / "figures"
REPORT_DIR = BASE / "reports"

LABELS = {0: "Negative", 1: "Positive", 2: "Irrelevant", 3: "Neutral"}
LABEL_IDS = [0, 1, 2, 3]
LABEL_NAMES = [LABELS[i] for i in LABEL_IDS]


def load_split(name: str) -> pd.DataFrame:
    frame = pd.read_csv(DATA_DIR / name)
    frame["text_clean"] = frame["text_clean"].fillna("").astype(str)
    frame["Sentiment"] = frame["Sentiment"].astype(int)
    return frame


def score(y_true: pd.Series, y_pred: pd.Series) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=LABEL_IDS, zero_division=0
    )
    macro_p, macro_r, macro_f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    per_class = []
    for index, label in enumerate(LABEL_NAMES):
        per_class.append(
            {
                "label": label,
                "precision": round(float(precision[index]), 4),
                "recall": round(float(recall[index]), 4),
                "f1": round(float(f1[index]), 4),
                "support": int(support[index]),
            }
        )
    return {
        "n": int(len(y_true)),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_precision": round(float(macro_p), 4),
        "macro_recall": round(float(macro_r), 4),
        "macro_f1": round(float(macro_f), 4),
        "weighted_precision": round(float(weighted_p), 4),
        "weighted_recall": round(float(weighted_r), 4),
        "weighted_f1": round(float(weighted_f), 4),
        "per_class": per_class,
    }


def top_coefficients(pipeline: Pipeline, limit: int = 8) -> dict:
    vectorizer: TfidfVectorizer = pipeline.named_steps["tfidf"]
    classifier: LogisticRegression = pipeline.named_steps["clf"]
    features = vectorizer.get_feature_names_out()
    output = {}
    for class_index, label in enumerate(LABEL_NAMES):
        weights = classifier.coef_[class_index]
        order = weights.argsort()
        output[label] = {
            "towards_class": [
                {"term": features[i], "weight": round(float(weights[i]), 3)}
                for i in order[-limit:][::-1]
            ],
            "against_class": [
                {"term": features[i], "weight": round(float(weights[i]), 3)}
                for i in order[:limit]
            ],
        }
    return output


def _tfidf_panel(
    feature_matrix: np.ndarray,
    feature_names: np.ndarray,
    documents: list[str],
    max_words: int,
    max_documents: int,
) -> tuple[np.ndarray, list[str], list[str], str]:
    """Return the rows, terms, labels, and sampling note for one heatmap panel."""
    matrix = feature_matrix.toarray() if hasattr(feature_matrix, "toarray") else np.asarray(feature_matrix)
    matrix = np.asarray(matrix, dtype=float)
    names = np.asarray(feature_names)
    labels = list(documents)
    if matrix.ndim != 2:
        raise ValueError(f"feature_matrix must be 2-D, got shape {matrix.shape}")
    n_docs, n_terms = matrix.shape
    if n_docs == 0 or n_terms == 0:
        raise ValueError(f"feature_matrix is empty: shape {matrix.shape}")
    if len(labels) != n_docs:
        raise ValueError(
            f"documents has {len(labels)} labels but feature_matrix has {n_docs} rows"
        )
    if len(names) != n_terms:
        raise ValueError(
            f"feature_names has {len(names)} terms but feature_matrix has {n_terms} columns"
        )

    notes: list[str] = []
    if n_docs > max_documents:
        matrix = matrix[:max_documents]
        labels = labels[:max_documents]
        notes.append(f"first {max_documents} of {n_docs} documents")
    if matrix.shape[1] > max_words:
        totals = matrix.sum(axis=0)
        chosen = np.argsort(totals)[-max_words:]
        chosen.sort()
        matrix = matrix[:, chosen]
        names = names[chosen]
        notes.append(f"top {max_words} terms")
    return matrix, [str(name) for name in names], labels, "; ".join(notes)


def plot_tfidf_heatmap(
    feature_matrices: list[np.ndarray],
    feature_names: np.ndarray,
    documents: list[list[str]],
    output_path: Path | str,
    panel_titles: list[str] | None = None,
    max_words: int = 40,
    max_documents: int = 30,
) -> None:
    """Save train and test TF-IDF heatmaps side by side.

    Each panel keeps the first ``max_documents`` rows and the terms with the
    largest TF-IDF mass on those rows. Term order follows the original vocabulary.
    """
    if max_documents < 1:
        raise ValueError("max_documents must be a positive integer")
    if max_words < 1:
        raise ValueError("max_words must be a positive integer")
    if len(feature_matrices) != len(documents):
        raise ValueError("feature_matrices and documents must have the same length")
    if len(feature_matrices) == 0:
        raise ValueError("feature_matrices is empty")
    titles = list(panel_titles) if panel_titles is not None else [f"Panel {i}" for i in range(len(feature_matrices))]
    if len(titles) != len(feature_matrices):
        raise ValueError("panel_titles must match the number of matrices")

    panels = [
        _tfidf_panel(matrix, feature_names, labels, max_words, max_documents)
        for matrix, labels in zip(feature_matrices, documents)
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(18, 8), squeeze=False)
    for ax, (matrix, names, labels, note), title in zip(axes[0], panels, titles):
        sns.heatmap(
            matrix,
            cmap="viridis",
            xticklabels=names,
            yticklabels=labels,
            cbar_kws={"label": "TF-IDF value"},
            ax=ax,
        )
        ax.set_title(f"{title} ({note})" if note else title)
        ax.set_xlabel("Term")
        ax.set_ylabel("Document")
        ax.tick_params(axis="x", labelrotation=90, labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
    fig.suptitle("TF-IDF Feature Matrix Heatmap")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_confusion(y_true: pd.Series, y_pred: pd.Series) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=LABEL_IDS)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Baseline confusion matrix on the test set")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "13_baseline_confusion.png", dpi=160)
    plt.close(fig)


def main() -> None:
    train = load_split("train_cleaned.csv")
    test = load_split("test_cleaned.csv")
    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=1.0,
                    solver="lbfgs",
                    max_iter=2000,
                    random_state=7,
                ),
            ),
        ]
    )
    pipeline.fit(train["text_clean"], train["Sentiment"])
    vectorizer: TfidfVectorizer = pipeline.named_steps["tfidf"]
    X_train = vectorizer.transform(train["text_clean"])
    X_test = vectorizer.transform(test["text_clean"])
    feature_names = vectorizer.get_feature_names_out()
    plot_tfidf_heatmap(
        [X_train, X_test],
        feature_names,
        [
            [f"Doc {i}" for i in range(X_train.shape[0])],
            [f"Doc {i}" for i in range(X_test.shape[0])],
        ],
        FIG_DIR / "14_tfidf_heatmap.png",
        panel_titles=["Train", "Test"],
        max_words=40,
        max_documents=30,
    )
    stale_heatmap = FIG_DIR / "15_tfidf_heatmap_test.png"
    if stale_heatmap.exists():
        stale_heatmap.unlink()
    prediction = pipeline.predict(test["text_clean"])
    full = score(test["Sentiment"], prediction)

    keep = test["cross_split_overlap"].fillna(0).astype(int) == 0
    held_out = score(test.loc[keep, "Sentiment"], prediction[keep.to_numpy()])

    report = classification_report(
        test["Sentiment"],
        prediction,
        labels=LABEL_IDS,
        target_names=LABEL_NAMES,
        digits=4,
        zero_division=0,
    )
    (REPORT_DIR / "baseline_classification_report.txt").write_text(report, encoding="utf-8")
    plot_confusion(test["Sentiment"], prediction)

    result = {
        "model": "TfidfVectorizer(1-2 grams, min_df=2, sublinear_tf) + LogisticRegression(C=1)",
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "vocabulary_size": int(len(pipeline.named_steps["tfidf"].vocabulary_)),
        "full_test": full,
        "test_without_overlap_row": held_out,
        "top_coefficients": top_coefficients(pipeline),
    }
    (REPORT_DIR / "baseline_metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(report)
    print(
        "accuracy", full["accuracy"],
        "macro_f1", full["macro_f1"],
        "vocab", result["vocabulary_size"],
    )
    print("without overlap", held_out["accuracy"], held_out["macro_f1"])


if __name__ == "__main__":
    main()
