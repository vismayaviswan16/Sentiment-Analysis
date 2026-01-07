# src/main.py
# Works with your file name: data/tourism_dataset.csv
# Auto-detects common column names for caption/text, hashtags, and (optionally) timestamp.

import os
import re
import pandas as pd
from pathlib import Path
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
ROOT = Path(__file__).resolve().parents[1]  # goes one level up from /src
DATA_PATH = ROOT / "data" / "tourism_dataset.csv"
OUTPUT_DIR = ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------- HELPERS ----------
def pick_column(df, candidates):
    # exact match (case-insensitive)
    for c in df.columns:
        if c.lower() in candidates:
            return c
    # fuzzy contains
    lower_cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        for lc, orig in lower_cols.items():
            if cand in lc:
                return orig
    return None

def clean_caption(text: str) -> str:
    text = str(text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)  # URLs
    text = re.sub(r"@\w+", " ", text)              # mentions
    text = re.sub(r"[^A-Za-z\s#]", " ", text)      # keep letters and '#'
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()

def parse_tags(x):
    s = str(x).strip()
    if not s:
        return []
    # If looks like a list-string, pull tokens safely
    if s.startswith("[") and s.endswith("]"):
        items = re.findall(r"#?\w+", s.lower())
        return [t.lstrip("#") for t in items if t]
    # Otherwise split on whitespace and keep hashtag tokens
    return [t.lstrip("#").lower() for t in s.split() if t.startswith("#")]

# ---------- LOAD ----------
if not DATA_PATH.exists():
    raise FileNotFoundError(f"File not found: {DATA_PATH.resolve()}")

df = pd.read_csv(DATA_PATH)

# ---------- AUTO-DETECT COLUMNS ----------
caption_col = pick_column(df, {
    "caption","text","description","content","body","post_text","message"
})
hashtags_col = pick_column(df, {
    "hashtags","tags","hash_tags","hash tag","hash-tag"
})
time_col = pick_column(df, {
    "timestamp","time","date","created_at","posted_at","post_date","datetime"
})

# Required: caption/text
if caption_col is None:
    raise ValueError(
        "Could not find a caption/text column. "
        "Add a column named 'caption' (your CSV already has 'caption' if you used the provided one)."
    )

# ---------- CLEAN ----------
df["clean_caption"] = df[caption_col].astype(str).apply(clean_caption)

# Hashtags list (optional)
if hashtags_col is not None:
    df["hashtags_list"] = df[hashtags_col].apply(parse_tags)
else:
    df["hashtags_list"] = [[] for _ in range(len(df))]

# Timestamp parse (optional)
if time_col is not None:
    # try multiple common formats, coerce errors to NaT
    df["timestamp_parsed"] = pd.to_datetime(df[time_col], errors="coerce", infer_datetime_format=True)
else:
    df["timestamp_parsed"] = pd.NaT

# ---------- SENTIMENT ----------
analyzer = SentimentIntensityAnalyzer()
df["compound"] = df["clean_caption"].apply(lambda x: analyzer.polarity_scores(x)["compound"])
df["sentiment"] = df["compound"].apply(
    lambda c: "positive" if c > 0.05 else ("negative" if c < -0.05 else "neutral")
)

# ---------- SAVE RESULTS ----------
out_csv = OUTPUT_DIR / "sentiment_results.csv"
df.to_csv(out_csv, index=False)

# ---------- SIMPLE PLOTS ----------
# Sentiment distribution
plt.figure()
df["sentiment"].value_counts().plot.pie(autopct="%1.1f%%", ylabel="")
plt.title("Tourism Sentiment Distribution")
plt.tight_layout()
plt.savefig(FIG_DIR / "sentiment_distribution.png", dpi=200)
plt.close()

# Monthly trend (only if timestamps exist)
if df["timestamp_parsed"].notna().any():
    monthly = (
        df.dropna(subset=["timestamp_parsed"])
          .set_index("timestamp_parsed")
          .resample("M")["compound"].mean()
    )
    if not monthly.empty:
        plt.figure()
        monthly.plot()
        plt.xlabel("Month")
        plt.ylabel("Average Sentiment (compound)")
        plt.title("Monthly Tourism Sentiment Trend")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "monthly_trend.png", dpi=200)
        plt.close()

print(f"Done.\nSaved: {out_csv}\nFigures in: {FIG_DIR}")
