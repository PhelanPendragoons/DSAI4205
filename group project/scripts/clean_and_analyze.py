"""Clean the Twitter sentiment data and write the EDA figures.

The original CSVs are left untouched. Cleaned tables go to data/processed,
figures go to figures, and numeric summaries go to reports/eda_stats.json.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import emoji
import ftfy
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from scipy import stats
from wordcloud import WordCloud

BASE = Path(__file__).resolve().parents[1]
RAW_DIR = BASE / "Data 5 - Twitter"
OUT_DIR = BASE / "data" / "processed"
FIG_DIR = BASE / "figures"
REPORT_DIR = BASE / "reports"

LABELS = {
    0: "Negative",
    1: "Positive",
    2: "Irrelevant",
    3: "Neutral",
}
LABEL_ORDER = ["Negative", "Positive", "Irrelevant", "Neutral"]
LABEL_COLORS = {
    "Negative": "#c0392b",
    "Positive": "#1e8449",
    "Irrelevant": "#7f8c8d",
    "Neutral": "#2980b9",
}

CONTRACTIONS = {
    "won't": "will not",
    "can't": "can not",
    "cannot": "can not",
    "ain't": "is not",
    "aren't": "are not",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it's": "it is",
    "let's": "let us",
    "mightn't": "might not",
    "mustn't": "must not",
    "shan't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "shouldn't": "should not",
    "that's": "that is",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we would",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what's": "what is",
    "where's": "where is",
    "who's": "who is",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have",
    "y'all": "you all",
    "im": "i am",
    "dont": "do not",
    "doesnt": "does not",
    "didnt": "did not",
    "cant": "can not",
    "wont": "will not",
    "isnt": "is not",
    "arent": "are not",
    "wasnt": "was not",
    "werent": "were not",
    "havent": "have not",
    "hasnt": "has not",
    "hadnt": "had not",
    "wouldnt": "would not",
    "couldnt": "could not",
    "shouldnt": "should not",
}

# Whole-token replacements. Longer keys are applied first.
CONTRACTION_RE = re.compile(
    r"\b(" + "|".join(sorted(map(re.escape, CONTRACTIONS), key=len, reverse=True)) + r")\b"
)

URL_RE = re.compile(
    r"""(?xi)
    (?:https?://|www\.)\S+
    | \b[\w.-]+\.(?:com|org|net|io|co|uk|it|be|ly|me|tv|gg|tt)(?:/\S*)?
    | \b[\w.-]+\.(?:org|co)\.in(?:/\S*)?
    """
)
MENTION_RE = re.compile(r"@\s*([A-Za-z0-9_]{2,})")
UNK_RE = re.compile(r"<\s*/?\s*unk\s*>", re.IGNORECASE)
REDACTION_RE = re.compile(r"\brhandlerr\b", re.IGNORECASE)
ELONG_RE = re.compile(r"([a-z])\1{2,}")
NON_ASCII_RE = re.compile(r"[^\x00-\x7F]")
BROKEN_APOSTROPHE_RE = re.compile(
    r"([A-Za-z])\s*\ufffd\s*(ve|re|ll|t|s|m|d)\b",
    re.IGNORECASE,
)

NEGATION_WORDS = {
    "no",
    "not",
    "nor",
    "never",
    "neither",
    "nobody",
    "nothing",
    "nowhere",
    "without",
}
DISPLAY_STOPS = set(stopwords.words("english")) - NEGATION_WORDS
DISPLAY_STOPS.update(
    {"atmention", "unk", "rhandlerr", "http", "https", "www", "co", "amp"}
)

LEMMATIZER = WordNetLemmatizer()
# Closed-class words are not lemmatized. WordNet maps a few of them to the
# wrong lemma (for example "us" -> "u" and "as" -> "a").
FUNCTION_WORDS = set(stopwords.words("english"))


def repair_encoding(text: str) -> str:
    """Repair the common UTF-8-as-Latin1 damage in this export.

    Fully intact sequences such as the curly apostrophe in "I'm" are recovered
    by ftfy. Sequences whose last byte was already replaced by "?" cannot be
    recovered as the original character; those are turned into an apostrophe
    when they sit inside a contraction, and removed otherwise.
    """
    text = text.replace("\u00e2\u0080?", "'")
    text = text.replace("\u00e2\u20ac?", " ")
    text = ftfy.fix_text(text)
    text = text.replace("\u00e2\u20ac?", " ")
    text = text.replace("\u00e2\u0080?", "'")
    text = BROKEN_APOSTROPHE_RE.sub(r"\1'\2", text)
    text = text.replace("\ufffd", " ")
    text = text.replace("\u00a0", " ")
    text = text.replace("£", " gbp ")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text


def _wordnet_pos(tag: str) -> str:
    if tag.startswith("J"):
        return wordnet.ADJ
    if tag.startswith("V"):
        return wordnet.VERB
    if tag.startswith("N"):
        return wordnet.NOUN
    if tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def lemmatize_tokens(tokens: list[str]) -> list[str]:
    if not tokens:
        return []
    tagged = pos_tag(tokens)
    lemmas = []
    for token, tag in tagged:
        if (
            token in FUNCTION_WORDS
            or token == "atmention"
            or token.isdigit()
            or len(token) <= 2
        ):
            lemmas.append(token)
            continue
        lemma = LEMMATIZER.lemmatize(token, _wordnet_pos(tag))
        if len(lemma) <= 1:
            lemmas.append(token)
        else:
            lemmas.append(lemma)
    return lemmas


def clean_text(text: str) -> str:
    text = repair_encoding(text)
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" atmention ", text)
    text = UNK_RE.sub(" ", text)
    text = REDACTION_RE.sub(" ", text)
    text = text.replace("_", " ")
    text = CONTRACTION_RE.sub(lambda match: CONTRACTIONS[match.group(0)], text)
    text = re.sub(r"\b([a-z]+)'s\b", r"\1", text)
    text = ELONG_RE.sub(r"\1\1", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    kept = []
    for token in text.split():
        if token.isdigit() and len(token) >= 6:
            continue
        if token.isalpha() and len(token) == 1 and token not in {"a", "i"}:
            continue
        kept.append(token)
    return " ".join(lemmatize_tokens(kept))


def extract_entities(text: str) -> list[str]:
    return [handle.lower() for handle in MENTION_RE.findall(text)]


def raw_features(text: str) -> dict:
    letters = [ch for ch in text if ch.isalpha()]
    caps_ratio = (
        sum(ch.isupper() for ch in letters) / len(letters) if letters else 0.0
    )
    return {
        "n_chars": len(text),
        "n_words": len(text.split()),
        "exclaim_count": text.count("!"),
        "question_count": text.count("?"),
        "caps_ratio": round(caps_ratio, 4),
        "url_count": len(URL_RE.findall(text)),
        "mention_count": len(MENTION_RE.findall(text)),
        "entities": "|".join(extract_entities(text)),
        "has_unk": int(bool(UNK_RE.search(text))),
        "has_non_ascii": int(bool(NON_ASCII_RE.search(text))),
        "has_elongation": int(bool(re.search(r"([A-Za-z])\1{2,}", text))),
        "has_redaction": int(bool(REDACTION_RE.search(text))),
    }


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(RAW_DIR / "traindata.csv")
    test = pd.read_csv(RAW_DIR / "testdata.csv")
    for frame, split in ((train, "train"), (test, "test")):
        frame["split"] = split
        frame["row_id"] = [f"{split}_{i}" for i in frame.index]
        frame["Phrase"] = frame["Phrase"].astype(str)
        frame["Sentiment"] = frame["Sentiment"].astype(int)
    return train, test


def deduplicate_train(
    train: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep one row per exact phrase. Conflicting labels use the majority label."""
    removed = []
    kept_rows = []
    notes = {}
    for phrase, group in train.groupby("Phrase", sort=False):
        counts = group["Sentiment"].value_counts()
        majority = int(counts.idxmax())
        keep = group.iloc[0].copy()
        keep["Sentiment"] = majority
        if len(group) > 1:
            conflict = int(group["Sentiment"].nunique() > 1)
            notes[keep["row_id"]] = (
                f"kept 1 of {len(group)}; majority label {majority}"
                + ("; labels disagreed" if conflict else "")
            )
            for _, row in group.iloc[1:].iterrows():
                reason = (
                    "duplicate_label_conflict"
                    if int(row["Sentiment"]) != majority
                    else "exact_duplicate"
                )
                removed.append(
                    {
                        "row_id": row["row_id"],
                        "split": "train",
                        "Phrase": phrase,
                        "Sentiment": int(row["Sentiment"]),
                        "reason": reason,
                        "majority_label": majority,
                    }
                )
        kept_rows.append(keep)
    cleaned = pd.DataFrame(kept_rows).reset_index(drop=True)
    cleaned["dedup_note"] = cleaned["row_id"].map(notes).fillna("")
    removed_df = pd.DataFrame(removed)
    return cleaned, removed_df


def mark_overlap(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Remove train rows whose phrase also appears in test, and record why."""
    overlap = set(train["Phrase"]) & set(test["Phrase"])
    leaked = train[train["Phrase"].isin(overlap)].copy()
    records = []
    for _, row in leaked.iterrows():
        test_labels = sorted(
            test.loc[test["Phrase"] == row["Phrase"], "Sentiment"].unique().tolist()
        )
        records.append(
            {
                "row_id": row["row_id"],
                "split": "train",
                "Phrase": row["Phrase"],
                "Sentiment": int(row["Sentiment"]),
                "reason": "train_test_overlap",
                "test_labels": "|".join(str(x) for x in test_labels),
            }
        )
    remaining = train[~train["Phrase"].isin(overlap)].copy()
    return remaining, pd.DataFrame(records), overlap


def build_clean_frame(frame: pd.DataFrame, outlier_cutoff: float) -> pd.DataFrame:
    features = pd.DataFrame(list(frame["Phrase"].map(raw_features)))
    out = pd.concat([frame.reset_index(drop=True), features], axis=1)
    out["text_clean"] = out["Phrase"].map(clean_text)
    out["n_words_clean"] = out["text_clean"].str.split().str.len().fillna(0).astype(int)
    out["sentiment_label"] = out["Sentiment"].map(LABELS)
    out["length_outlier"] = (out["n_words"] > outlier_cutoff).astype(int)
    out["empty_after_clean"] = (out["n_words_clean"] == 0).astype(int)
    return out


def cramers_v(chi2: float, n: int, n_rows: int, n_cols: int) -> float:
    denom = n * (min(n_rows, n_cols) - 1)
    if denom <= 0:
        return 0.0
    return float(np.sqrt(chi2 / denom))


def association_table(frame: pd.DataFrame, split_name: str) -> list[dict]:
    binary_cols = [
        "has_exclaim",
        "has_question",
        "has_url",
        "has_mention",
        "high_caps",
        "has_unk",
        "has_non_ascii",
        "has_elongation",
        "length_outlier",
    ]
    work = frame.copy()
    work["has_exclaim"] = (work["exclaim_count"] > 0).astype(int)
    work["has_question"] = (work["question_count"] > 0).astype(int)
    work["has_url"] = (work["url_count"] > 0).astype(int)
    work["has_mention"] = (work["mention_count"] > 0).astype(int)
    work["high_caps"] = (work["caps_ratio"] >= 0.3).astype(int)
    rows = []
    for col in binary_cols:
        table = pd.crosstab(work[col], work["sentiment_label"])
        table = table.reindex(columns=LABEL_ORDER, fill_value=0)
        chi2, p_value, _, _ = stats.chi2_contingency(table.values)
        rates = {}
        for label in LABEL_ORDER:
            subset = work[work["sentiment_label"] == label]
            rates[label] = float(subset[col].mean()) if len(subset) else 0.0
        rows.append(
            {
                "split": split_name,
                "feature": col,
                "chi2": round(float(chi2), 3),
                "p_value": float(p_value),
                "cramers_v": round(cramers_v(chi2, len(work), *table.shape), 3),
                **{f"rate_{label.lower()}": round(rates[label], 3) for label in LABEL_ORDER},
            }
        )
    return rows


def top_tokens(texts: pd.Series, ngram: int, limit: int) -> list[tuple[str, int]]:
    counts: Counter[str] = Counter()
    for text in texts.fillna(""):
        tokens = [tok for tok in str(text).split() if tok not in DISPLAY_STOPS]
        if ngram == 1:
            counts.update(tokens)
        else:
            counts.update(
                " ".join(pair)
                for pair in zip(tokens, tokens[1:])
                if pair[0] not in DISPLAY_STOPS or pair[1] not in DISPLAY_STOPS
            )
    return counts.most_common(limit)


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_class_balance(train: pd.DataFrame, test: pd.DataFrame) -> None:
    rows = []
    for split, frame in (("Train", train), ("Test", test)):
        counts = frame["sentiment_label"].value_counts()
        for label in LABEL_ORDER:
            rows.append({"Split": split, "Sentiment": label, "Count": int(counts.get(label, 0))})
    plot_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    sns.barplot(
        data=plot_df,
        x="Sentiment",
        y="Count",
        hue="Split",
        order=LABEL_ORDER,
        palette=["#2c3e50", "#85c1e9"],
        ax=ax,
    )
    ax.set_title("Class distribution in the train and test sets")
    ax.set_xlabel("")
    ax.set_ylabel("Number of tweets")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_class_distribution.png", dpi=160)
    plt.close(fig)


def plot_length(train: pd.DataFrame, test: pd.DataFrame, cutoff: float) -> None:
    combined = pd.concat([train, test], ignore_index=True)
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    sns.boxplot(
        data=combined,
        x="sentiment_label",
        y="n_words",
        order=LABEL_ORDER,
        hue="sentiment_label",
        palette=LABEL_COLORS,
        legend=False,
        ax=ax,
        fliersize=3,
    )
    ax.axhline(cutoff, color="#922b21", ls="--", lw=1, label=f"Long-text flag ({cutoff:.0f} words)")
    ax.set_title("Tweet length by sentiment")
    ax.set_xlabel("")
    ax.set_ylabel("Words in the original tweet")
    ax.legend(frameon=False)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_length_by_sentiment.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for split, frame, color in (
        ("Train", train, "#2c3e50"),
        ("Test", test, "#5dade2"),
    ):
        ax.hist(frame["n_words"], bins=18, alpha=0.55, label=split, color=color)
    ax.set_title("Word-count distribution of the two splits")
    ax.set_xlabel("Words in the original tweet")
    ax.set_ylabel("Number of tweets")
    ax.legend(frameon=False)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_length_train_test.png", dpi=160)
    plt.close(fig)


def plot_surface_rates(train: pd.DataFrame) -> None:
    features = [
        ("exclaim_count", "Has !"),
        ("question_count", "Has ?"),
        ("url_count", "Has URL"),
        ("mention_count", "Has @mention"),
        ("caps_ratio", "Caps ratio >= 0.3"),
        ("has_elongation", "Elongated word"),
    ]
    rows = []
    for col, name in features:
        for label in LABEL_ORDER:
            subset = train[train["sentiment_label"] == label]
            if col == "caps_ratio":
                rate = float((subset[col] >= 0.3).mean())
            elif col == "has_elongation":
                rate = float(subset[col].mean())
            else:
                rate = float((subset[col] > 0).mean())
            rows.append({"Feature": name, "Sentiment": label, "Share": rate})
    plot_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    sns.barplot(
        data=plot_df,
        x="Feature",
        y="Share",
        hue="Sentiment",
        hue_order=LABEL_ORDER,
        palette=LABEL_COLORS,
        ax=ax,
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share of tweets")
    ax.set_xlabel("")
    ax.set_title("Surface features by sentiment (training set)")
    ax.tick_params(axis="x", rotation=15)
    ax.legend(frameon=False, ncol=4, loc="upper right")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_surface_features.png", dpi=160)
    plt.close(fig)


def plot_top_words(train: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, label in zip(axes.ravel(), LABEL_ORDER):
        pairs = top_tokens(train.loc[train["sentiment_label"] == label, "text_clean"], 1, 8)
        words = [word for word, _ in pairs][::-1]
        counts = [count for _, count in pairs][::-1]
        ax.barh(words, counts, color=LABEL_COLORS[label])
        ax.set_title(label)
        style_axes(ax)
    fig.suptitle("Most frequent content words after cleaning", y=1.01)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_top_words_by_class.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    for ax, label in zip(axes, ("Negative", "Positive")):
        pairs = top_tokens(train.loc[train["sentiment_label"] == label, "text_clean"], 2, 8)
        phrases = [phrase for phrase, _ in pairs][::-1]
        counts = [count for _, count in pairs][::-1]
        ax.barh(phrases, counts, color=LABEL_COLORS[label])
        ax.set_title(label)
        style_axes(ax)
    fig.suptitle("Frequent two-word phrases: negative and positive", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_top_bigrams.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_entities(train: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in train.iterrows():
        if not row["entities"]:
            continue
        for handle in str(row["entities"]).split("|"):
            records.append({"entity": handle, "sentiment_label": row["sentiment_label"]})
    entity_df = pd.DataFrame(records)
    if entity_df.empty:
        return entity_df
    totals = entity_df["entity"].value_counts()
    keep = totals[totals >= 4].head(12).index
    plot_src = entity_df[entity_df["entity"].isin(keep)]
    pivot = (
        pd.crosstab(plot_src["entity"], plot_src["sentiment_label"])
        .reindex(columns=LABEL_ORDER, fill_value=0)
        .loc[totals[keep].sort_values(ascending=True).index]
    )
    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    left = np.zeros(len(pivot))
    for label in LABEL_ORDER:
        values = pivot[label].to_numpy()
        ax.barh(pivot.index, values, left=left, color=LABEL_COLORS[label], label=label)
        left = left + values
    ax.set_xlabel("Mentions in the training set")
    ax.set_title("Sentiment of the most mentioned accounts")
    ax.legend(frameon=False, ncol=2)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_entity_sentiment.png", dpi=160)
    plt.close(fig)
    return pivot.sort_values(LABEL_ORDER, ascending=False)


def plot_wordclouds(train: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    colormaps = {
        "Negative": "Reds",
        "Positive": "Greens",
        "Irrelevant": "Greys",
        "Neutral": "Blues",
    }
    for ax, label in zip(axes.ravel(), LABEL_ORDER):
        text = " ".join(train.loc[train["sentiment_label"] == label, "text_clean"].tolist())
        cloud = WordCloud(
            width=700,
            height=420,
            background_color="white",
            colormap=colormaps[label],
            stopwords=DISPLAY_STOPS,
            collocations=False,
            max_words=70,
            random_state=7,
            min_font_size=10,
        ).generate(text)
        ax.imshow(cloud, interpolation="bilinear")
        ax.set_title(label)
        ax.axis("off")
    fig.suptitle("Word clouds from cleaned text (display stopwords removed)", y=1.01)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "08_wordclouds.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def length_tests(frame: pd.DataFrame) -> dict:
    groups = [
        frame.loc[frame["sentiment_label"] == label, "n_words"].to_numpy()
        for label in LABEL_ORDER
    ]
    h_stat, p_value = stats.kruskal(*groups)
    return {
        "kruskal_h": round(float(h_stat), 3),
        "kruskal_p": float(p_value),
        "median_by_class": {
            label: float(frame.loc[frame["sentiment_label"] == label, "n_words"].median())
            for label in LABEL_ORDER
        },
        "mean_by_class": {
            label: round(float(frame.loc[frame["sentiment_label"] == label, "n_words"].mean()), 2)
            for label in LABEL_ORDER
        },
    }


def class_counts(frame: pd.DataFrame) -> dict:
    counts = frame["sentiment_label"].value_counts()
    return {label: int(counts.get(label, 0)) for label in LABEL_ORDER}


def main() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["axes.unicode_minus"] = False
    for folder in (OUT_DIR, FIG_DIR, REPORT_DIR):
        folder.mkdir(parents=True, exist_ok=True)

    train_raw, test_raw = load_raw()
    raw_train_n = len(train_raw)
    raw_test_n = len(test_raw)

    train_dedup, removed_dups = deduplicate_train(train_raw)
    train_model, removed_overlap, overlap_phrases = mark_overlap(train_dedup, test_raw)
    removed = pd.concat([removed_dups, removed_overlap], ignore_index=True)

    q1 = float(train_model["Phrase"].str.split().str.len().quantile(0.25))
    q3 = float(train_model["Phrase"].str.split().str.len().quantile(0.75))
    outlier_cutoff = q3 + 1.5 * (q3 - q1)

    train = build_clean_frame(train_model, outlier_cutoff)
    test = build_clean_frame(test_raw, outlier_cutoff)
    test["cross_split_overlap"] = test["Phrase"].isin(overlap_phrases).astype(int)
    train["cross_split_overlap"] = 0
    test["dedup_note"] = ""

    keep_cols = [
        "row_id",
        "split",
        "Phrase",
        "text_clean",
        "Sentiment",
        "sentiment_label",
        "n_chars",
        "n_words",
        "n_words_clean",
        "exclaim_count",
        "question_count",
        "caps_ratio",
        "url_count",
        "mention_count",
        "entities",
        "has_unk",
        "has_non_ascii",
        "has_elongation",
        "has_redaction",
        "length_outlier",
        "empty_after_clean",
        "cross_split_overlap",
        "dedup_note",
    ]
    train[keep_cols].to_csv(OUT_DIR / "train_cleaned.csv", index=False, encoding="utf-8-sig")
    test[keep_cols].to_csv(OUT_DIR / "test_cleaned.csv", index=False, encoding="utf-8-sig")
    removed.to_csv(OUT_DIR / "removed_rows.csv", index=False, encoding="utf-8-sig")

    examples = []
    flags = {
        "encoding": train_raw["Phrase"].map(lambda s: bool(NON_ASCII_RE.search(s))),
        "url": train_raw["Phrase"].map(lambda s: bool(URL_RE.search(s))),
        "unk": train_raw["Phrase"].map(lambda s: bool(UNK_RE.search(s))),
        "emoji_or_broken": train_raw["Phrase"].map(lambda s: "ð" in s or "â" in s),
    }
    seen = set()
    for reason, mask in flags.items():
        for _, row in train_raw.loc[mask].head(2).iterrows():
            if row["row_id"] in seen:
                continue
            seen.add(row["row_id"])
            examples.append(
                {
                    "reason": reason,
                    "Sentiment": int(row["Sentiment"]),
                    "Phrase": row["Phrase"],
                    "text_clean": clean_text(row["Phrase"]),
                }
            )
    pd.DataFrame(examples).to_csv(
        REPORT_DIR / "cleaning_examples.csv", index=False, encoding="utf-8-sig"
    )

    sample_rows = []
    for label, group in train.groupby("sentiment_label"):
        take = group.sort_values("n_words").iloc[:: max(len(group) // 6, 1)].head(6)
        for _, row in take.iterrows():
            sample_rows.append(
                {
                    "sentiment_label": label,
                    "Phrase": row["Phrase"],
                    "text_clean": row["text_clean"],
                }
            )
    pd.DataFrame(sample_rows).to_csv(
        REPORT_DIR / "class_samples.csv", index=False, encoding="utf-8-sig"
    )

    plot_class_balance(train, test)
    plot_length(train, test, outlier_cutoff)
    plot_surface_rates(train)
    plot_top_words(train)
    entity_pivot = plot_entities(train)
    plot_wordclouds(train)

    associations = association_table(train, "train") + association_table(test, "test")
    pd.DataFrame(associations).to_csv(
        REPORT_DIR / "association_tests.csv", index=False, encoding="utf-8-sig"
    )
    if not entity_pivot.empty:
        entity_pivot.to_csv(REPORT_DIR / "top_entities.csv", encoding="utf-8-sig")

    raw_vocab = Counter(tok.lower() for text in train_raw["Phrase"] for tok in re.findall(r"[A-Za-z']+", text))
    clean_vocab = Counter(tok for text in train["text_clean"] for tok in text.split())

    summary = {
        "raw_rows": {"train": raw_train_n, "test": raw_test_n},
        "cleaned_rows": {"train": int(len(train)), "test": int(len(test))},
        "removed_rows": int(len(removed)),
        "raw_class_counts": {
            "train": class_counts(train_raw.assign(sentiment_label=train_raw["Sentiment"].map(LABELS))),
            "test": class_counts(test_raw.assign(sentiment_label=test_raw["Sentiment"].map(LABELS))),
        },
        "clean_class_counts": {"train": class_counts(train), "test": class_counts(test)},
        "outlier_cutoff_words": outlier_cutoff,
        "length_outliers": {
            "train": int(train["length_outlier"].sum()),
            "test": int(test["length_outlier"].sum()),
        },
        "empty_after_clean": {
            "train": int(train["empty_after_clean"].sum()),
            "test": int(test["empty_after_clean"].sum()),
        },
        "issue_counts_raw_train": {
            "non_ascii": int(train_raw["Phrase"].map(lambda s: bool(NON_ASCII_RE.search(s))).sum()),
            "unk": int(train_raw["Phrase"].map(lambda s: bool(UNK_RE.search(s))).sum()),
            "url": int(train_raw["Phrase"].map(lambda s: bool(URL_RE.search(s))).sum()),
            "mention": int(train_raw["Phrase"].map(lambda s: bool(MENTION_RE.search(s))).sum()),
            "elongation": int(train_raw["Phrase"].map(lambda s: bool(re.search(r"([A-Za-z])\1{2,}", s))).sum()),
            "redaction": int(train_raw["Phrase"].map(lambda s: bool(REDACTION_RE.search(s))).sum()),
        },
        "issue_counts_raw_test": {
            "non_ascii": int(test_raw["Phrase"].map(lambda s: bool(NON_ASCII_RE.search(s))).sum()),
            "unk": int(test_raw["Phrase"].map(lambda s: bool(UNK_RE.search(s))).sum()),
            "url": int(test_raw["Phrase"].map(lambda s: bool(URL_RE.search(s))).sum()),
            "mention": int(test_raw["Phrase"].map(lambda s: bool(MENTION_RE.search(s))).sum()),
            "elongation": int(test_raw["Phrase"].map(lambda s: bool(re.search(r"([A-Za-z])\1{2,}", s))).sum()),
            "redaction": int(test_raw["Phrase"].map(lambda s: bool(REDACTION_RE.search(s))).sum()),
        },
        "vocab_train_raw_alpha": len(raw_vocab),
        "vocab_train_clean": len(clean_vocab),
        "mean_words_raw_train": round(float(train["n_words"].mean()), 2),
        "mean_words_clean_train": round(float(train["n_words_clean"].mean()), 2),
        "length_tests_train": length_tests(train),
        "length_tests_test": length_tests(test),
        "top_words": {
            label: top_tokens(train.loc[train["sentiment_label"] == label, "text_clean"], 1, 10)
            for label in LABEL_ORDER
        },
        "top_bigrams": {
            label: top_tokens(train.loc[train["sentiment_label"] == label, "text_clean"], 2, 8)
            for label in ("Negative", "Positive", "Irrelevant", "Neutral")
        },
        "overlap_phrases": sorted(overlap_phrases),
        "label_map": {str(k): v for k, v in LABELS.items()},
    }
    (REPORT_DIR / "eda_stats.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({k: summary[k] for k in (
        "raw_rows", "cleaned_rows", "removed_rows", "clean_class_counts",
        "outlier_cutoff_words", "length_outliers", "empty_after_clean",
        "vocab_train_raw_alpha", "vocab_train_clean",
        "mean_words_raw_train", "mean_words_clean_train",
    )}, indent=2))
    print("length train", summary["length_tests_train"])
    print("wrote outputs to", OUT_DIR)
    from extend_analysis import main as write_extended_analysis

    write_extended_analysis()


if __name__ == "__main__":
    main()
