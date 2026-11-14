"""Profile the raw Twitter sentiment CSVs before cleaning."""
import re
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "Data 5 - Twitter"

train = pd.read_csv(DATA / "traindata.csv")
test = pd.read_csv(DATA / "testdata.csv")
df = pd.concat(
    [train.assign(split="train"), test.assign(split="test")],
    ignore_index=True,
)
text = df["Phrase"].astype(str)

patterns = {
    "mojibake": r"â€|â€™|â€œ|ðŸ|Ã.|Â.|�",
    "unk_token": r"<unk>",
    "url": r"https?://|www\.|t\.co/|pic\.twitter|dlvr\.it|youtube\.com",
    "mention": r"@\s*\w+",
    "hashtag": r"#\w+",
    "html_entity": r"&[a-zA-Z]+;|&#\d+;",
    "html_tag": r"<[^>\s][^>]*>",
    "elongated": r"(.)\1{3,}",
    "multi_punct": r"[!?]{2,}",
    "non_ascii": r"[^\x00-\x7F]",
    "digits": r"\d",
    "all_caps_word": r"\b[A-Z]{3,}\b",
}

print("rows", len(df))
for name, pat in patterns.items():
    mask = text.str.contains(pat, regex=True, na=False)
    train_n = int(mask[df["split"] == "train"].sum())
    test_n = int(mask[df["split"] == "test"].sum())
    print(f"{name:16} train {train_n:4}  test {test_n:4}  total {train_n + test_n:4}")

wc = text.str.split().str.len()
print("\nword count by split")
print(wc.groupby(df["split"]).describe())
print("very short <=3 words", wc[wc <= 3].groupby(df.loc[wc <= 3, "split"]).size().to_dict())
print("very long >=60 words", wc[wc >= 60].groupby(df.loc[wc >= 60, "split"]).size().to_dict())

print("\n=== sample mojibake ===")
moji = text[text.str.contains(r"â€|â€™|ðŸ|Ã.|Â", regex=True)]
for s in moji.head(6):
    print(repr(s[:220]))
    print("---")

print("\n=== sample unk ===")
for s in text[text.str.contains("<unk>")].head(4):
    print(repr(s[:220]))
    print("---")

print("\n=== normalized near-duplicates ===")
norm = (
    text.str.lower()
    .str.replace(r"https?://\S+|www\.\S+", " ", regex=True)
    .str.replace(r"[^a-z0-9\s]", " ", regex=True)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)
df = df.assign(norm=norm)
for split, part in df.groupby("split"):
    dup_mask = part["norm"].duplicated(keep=False) & part["norm"].ne("")
    print(split, "normalized-dup rows", int(dup_mask.sum()), "unique phrases", part.loc[dup_mask, "norm"].nunique())

conflict = (
    df.groupby("norm")
    .filter(lambda g: g["Sentiment"].nunique() > 1)
)
print("normalized groups with conflicting labels", conflict["norm"].nunique(), "rows", len(conflict))
if len(conflict):
    for key, g in conflict.groupby("norm"):
        print("LABELS", sorted(g["Sentiment"].unique().tolist()), "splits", g["split"].unique().tolist())
        print(repr(key[:180]))
