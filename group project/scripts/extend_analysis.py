"""Extra dataset analysis on the cleaned Twitter tables.

Reads data/processed and writes the figures and tables used in section 4
of the cleaning report. It does not modify the cleaned text.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from nltk.corpus import stopwords
from scipy import stats

BASE = Path(__file__).resolve().parents[1]
OUT_DIR = BASE / "data" / "processed"
FIG_DIR = BASE / "figures"
REPORT_DIR = BASE / "reports"

LABEL_ORDER = ["Negative", "Positive", "Irrelevant", "Neutral"]
LABEL_COLORS = {
    "Negative": "#c0392b",
    "Positive": "#1e8449",
    "Irrelevant": "#7f8c8d",
    "Neutral": "#2980b9",
}
STOPS = set(stopwords.words("english"))
NEGATION = {"no", "not", "nor", "never", "neither", "nobody", "nothing", "nowhere"}
CONTENT_STOPS = (STOPS - NEGATION) | {"atmention"}

GAME_WORDS = {
    "game", "play", "player", "xbox", "fifa", "warzone", "pubg", "overwatch",
    "madden", "nba", "valorant", "fortnite", "blizzard", "nintendo", "apex",
    "gta", "destiny", "warcraft", "hearthstone", "cod", "ps5", "minecraft",
    "roblox", "dota", "csgo", "valorant",
}
BRAND_WORDS = {
    "amazon", "verizon", "johnson", "microsoft", "google", "nvidia", "depot",
    "facebook", "youtube", "twitter", "tesla", "apple",
}
NUMERIC_FEATURES = [
    ("n_words", "Word count"),
    ("n_chars", "Character count"),
    ("exclaim_count", "Exclamation marks"),
    ("question_count", "Question marks"),
    ("caps_ratio", "Capital-letter ratio"),
    ("url_count", "URL count"),
    ("mention_count", "Mention count"),
]


def content_tokens(text: str) -> list[str]:
    return [
        tok
        for tok in str(text).split()
        if tok not in CONTENT_STOPS and not tok.isdigit() and len(tok) > 2
    ]


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def class_profile(train: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label in LABEL_ORDER:
        part = train[train["sentiment_label"] == label]
        texts = part["text_clean"].fillna("")
        has_negation = texts.map(lambda text: bool(NEGATION & set(str(text).split())))
        has_game = texts.map(lambda text: bool(GAME_WORDS & set(str(text).split())))
        has_brand = texts.map(lambda text: bool(BRAND_WORDS & set(str(text).split())))
        rows.append(
            {
                "sentiment": label,
                "n": int(len(part)),
                "median_words": float(part["n_words"].median()),
                "mean_words": round(float(part["n_words"].mean()), 2),
                "mean_exclaim": round(float(part["exclaim_count"].mean()), 3),
                "mean_question": round(float(part["question_count"].mean()), 3),
                "share_exclaim": round(float((part["exclaim_count"] > 0).mean()), 3),
                "share_question": round(float((part["question_count"] > 0).mean()), 3),
                "share_url": round(float((part["url_count"] > 0).mean()), 3),
                "share_mention": round(float((part["mention_count"] > 0).mean()), 3),
                "mean_caps_ratio": round(float(part["caps_ratio"].mean()), 3),
                "share_negation": round(float(has_negation.mean()), 3),
                "share_game_word": round(float(has_game.mean()), 3),
                "share_brand_word": round(float(has_brand.mean()), 3),
                "mean_content_types": round(
                    float(texts.map(lambda text: len(set(content_tokens(text)))).mean()), 2
                ),
            }
        )
    return pd.DataFrame(rows)


def split_similarity(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    train_counts = train["sentiment_label"].value_counts().reindex(LABEL_ORDER)
    test_counts = test["sentiment_label"].value_counts().reindex(LABEL_ORDER)
    table = pd.DataFrame({"train": train_counts, "test": test_counts})
    chi2, p_value, _, _ = stats.chi2_contingency(table.values)
    length_stat, length_p = stats.mannwhitneyu(
        train["n_words"], test["n_words"], alternative="two-sided"
    )
    return {
        "label_chi2": round(float(chi2), 3),
        "label_p": float(p_value),
        "length_u": round(float(length_stat), 1),
        "length_p": float(length_p),
        "train_median_words": float(train["n_words"].median()),
        "test_median_words": float(test["n_words"].median()),
    }


def correlation_matrix(train: pd.DataFrame) -> pd.DataFrame:
    columns = [col for col, _ in NUMERIC_FEATURES]
    labels = [label for _, label in NUMERIC_FEATURES]
    corr = train[columns].corr(method="pearson")
    corr.index = labels
    corr.columns = labels
    return corr


def plot_correlation(corr: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 6.6))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-0.6,
        vmax=0.6,
        square=True,
        cbar_kws={"label": "Pearson r"},
        ax=ax,
    )
    ax.set_title("Correlation among numeric text features (training set)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "09_feature_correlation.png", dpi=160)
    plt.close(fig)


def plot_profile_heatmap(profile: pd.DataFrame) -> None:
    value_cols = [
        "mean_words",
        "mean_exclaim",
        "mean_question",
        "share_url",
        "share_mention",
        "share_negation",
        "share_game_word",
        "share_brand_word",
    ]
    nice = {
        "mean_words": "Mean words",
        "mean_exclaim": "Mean !",
        "mean_question": "Mean ?",
        "share_url": "Share with URL",
        "share_mention": "Share with @",
        "share_negation": "Share with negation",
        "share_game_word": "Share with game word",
        "share_brand_word": "Share with brand word",
    }
    raw = profile.set_index("sentiment")[value_cols]
    centered = (raw - raw.mean()) / raw.std(ddof=0)
    centered.columns = [nice[col] for col in centered.columns]
    fig, ax = plt.subplots(figsize=(11.2, 5.4))
    sns.heatmap(
        centered.loc[LABEL_ORDER],
        annot=raw.loc[LABEL_ORDER].rename(columns=nice),
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        yticklabels=LABEL_ORDER,
        ax=ax,
    )
    ax.set_title("Number is the raw value; color is the distance from the four-class average")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelrotation=0)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "10_class_profile.png", dpi=160)
    plt.close(fig)


def token_counts(frame: pd.DataFrame) -> Counter:
    counts: Counter[str] = Counter()
    for text in frame["text_clean"].fillna(""):
        counts.update(content_tokens(text))
    return counts


def distinctive_words(train: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
    """Monroe-style log-odds: words used more by one class than by the other three."""
    counts = {
        label: token_counts(part)
        for label, part in train.groupby("sentiment_label")
    }
    doc_counts = {}
    for label, part in train.groupby("sentiment_label"):
        seen = Counter()
        for text in part["text_clean"].fillna(""):
            seen.update(set(content_tokens(text)))
        doc_counts[label] = seen
    background = Counter()
    for class_counts in counts.values():
        background.update(class_counts)
    total_tokens = sum(background.values())
    prior_strength = 50.0
    rows = []
    for label in LABEL_ORDER:
        class_counts = counts[label]
        other_counts: Counter[str] = Counter()
        for other, other_class in counts.items():
            if other != label:
                other_counts.update(other_class)
        n_c = sum(class_counts.values())
        n_o = sum(other_counts.values())
        n_docs = int((train["sentiment_label"] == label).sum())
        scored = []
        for word, background_count in background.items():
            y_c = class_counts[word]
            doc_hits = doc_counts[label][word]
            if y_c < 4 or doc_hits < 4:
                continue
            prior = prior_strength * background_count / total_tokens
            y_o = other_counts[word]
            a_c = y_c + prior
            b_c = n_c + prior_strength - y_c - prior
            a_o = y_o + prior
            b_o = n_o + prior_strength - y_o - prior
            if min(a_c, b_c, a_o, b_o) <= 0:
                continue
            delta = math.log(a_c / b_c) - math.log(a_o / b_o)
            z_score = delta / math.sqrt(1 / a_c + 1 / b_c + 1 / a_o + 1 / b_o)
            scored.append((word, z_score, doc_hits, n_docs))
        scored.sort(key=lambda item: item[1], reverse=True)
        for rank, (word, z_score, doc_hits, class_size) in enumerate(scored[:top_n], start=1):
            rows.append(
                {
                    "sentiment": label,
                    "rank": rank,
                    "word": word,
                    "z_score": round(z_score, 2),
                    "docs_in_class": doc_hits,
                    "class_size": class_size,
                    "share_in_class": round(doc_hits / class_size, 3),
                }
            )
    return pd.DataFrame(rows)


def plot_distinctive(words: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, label in zip(axes.ravel(), LABEL_ORDER):
        part = words[words["sentiment"] == label].sort_values("z_score")
        ax.barh(part["word"], part["z_score"], color=LABEL_COLORS[label])
        ax.set_title(label)
        ax.set_xlabel("Distinctiveness (log-odds z)")
        style_axes(ax)
    fig.suptitle("Words that separate each class from the other three", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "11_distinctive_words.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def vocab_overlap(train: pd.DataFrame) -> pd.DataFrame:
    sets = {}
    for label, part in train.groupby("sentiment_label"):
        counts = Counter(tok for text in part["text_clean"].fillna("") for tok in set(content_tokens(text)))
        sets[label] = {word for word, count in counts.items() if count >= 3}
    rows = []
    for i, left in enumerate(LABEL_ORDER):
        for right in LABEL_ORDER[i + 1 :]:
            union = sets[left] | sets[right]
            inter = sets[left] & sets[right]
            rows.append(
                {
                    "class_a": left,
                    "class_b": right,
                    "shared_words": len(inter),
                    "jaccard": round(len(inter) / len(union), 3) if union else 0.0,
                }
            )
    return pd.DataFrame(rows)


def plot_overlap(overlap: pd.DataFrame) -> None:
    matrix = pd.DataFrame(1.0, index=LABEL_ORDER, columns=LABEL_ORDER)
    for _, row in overlap.iterrows():
        matrix.loc[row["class_a"], row["class_b"]] = row["jaccard"]
        matrix.loc[row["class_b"], row["class_a"]] = row["jaccard"]
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0,
        vmax=1,
        square=True,
        ax=ax,
    )
    ax.set_title("Vocabulary overlap between classes (Jaccard)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "12_vocab_overlap.png", dpi=160)
    plt.close(fig)


def outlier_table(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    long_rows = frame[frame["length_outlier"] == 1][
        ["row_id", "sentiment_label", "n_words", "Phrase"]
    ].copy()
    long_rows.insert(0, "split", split)
    return long_rows


def main() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    train = pd.read_csv(OUT_DIR / "train_cleaned.csv")
    test = pd.read_csv(OUT_DIR / "test_cleaned.csv")

    profile = class_profile(train)
    profile.to_csv(REPORT_DIR / "class_profile.csv", index=False, encoding="utf-8-sig")
    corr = correlation_matrix(train)
    corr.to_csv(REPORT_DIR / "feature_correlation.csv", encoding="utf-8-sig")
    words = distinctive_words(train)
    words.to_csv(REPORT_DIR / "distinctive_words.csv", index=False, encoding="utf-8-sig")
    overlap = vocab_overlap(train)
    overlap.to_csv(REPORT_DIR / "vocab_overlap.csv", index=False, encoding="utf-8-sig")
    outliers = pd.concat(
        [outlier_table(train, "train"), outlier_table(test, "test")],
        ignore_index=True,
    )
    outliers.to_csv(REPORT_DIR / "long_tweets.csv", index=False, encoding="utf-8-sig")

    plot_correlation(corr)
    plot_profile_heatmap(profile)
    plot_distinctive(words)
    plot_overlap(overlap)

    summary = {
        "split_similarity": split_similarity(train, test),
        "strongest_correlations": strongest_pairs(corr),
    }
    (REPORT_DIR / "extended_stats.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(profile.to_string(index=False))
    print(overlap.to_string(index=False))
    print(json.dumps(summary, indent=2))
    print(words.groupby("sentiment")["word"].apply(list).to_dict())


def strongest_pairs(corr: pd.DataFrame) -> list[dict]:
    pairs = []
    columns = list(corr.columns)
    for i, left in enumerate(columns):
        for right in columns[i + 1 :]:
            pairs.append(
                {
                    "feature_a": left,
                    "feature_b": right,
                    "r": round(float(corr.loc[left, right]), 3),
                }
            )
    pairs.sort(key=lambda item: abs(item["r"]), reverse=True)
    return pairs[:6]


if __name__ == "__main__":
    main()
