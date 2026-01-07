# app.py — Streamlit site for refined tourism sentiment analytics
import re
import pandas as pd
import numpy as np
from pathlib import Path
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Tourism Sentiment Analytics", page_icon="🧭", layout="wide")

OUTPUT_DIR = Path("outputs")
DATA_PATH = OUTPUT_DIR / "sentiment_results.csv"

@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Safety: ensure expected cols
    for col in ["clean_caption","compound","sentiment"]:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}. Run src/main.py first.")
    # Parse timestamp if present
    ts_col = None
    for c in ["timestamp_parsed","timestamp","date","created_at","posted_at","post_date","datetime","time"]:
        if c in df.columns:
            ts_col = c
            break
    if ts_col:
        df["timestamp_parsed"] = pd.to_datetime(df[ts_col], errors="coerce", infer_datetime_format=True)
    else:
        df["timestamp_parsed"] = pd.NaT
    # Normalize hashtags_list
    if "hashtags_list" in df.columns:
        df["hashtags_list"] = df["hashtags_list"].apply(
            lambda x: [t.strip().lower() for t in re.findall(r"#?([\w]+)", str(x))]
        )
    else:
        df["hashtags_list"] = [[] for _ in range(len(df))]
    return df

df = load_data(DATA_PATH)

model_used = df["model_name"].iloc[0] if "model_name" in df.columns else "unknown"
st.caption(f"Model: {model_used}")

# ---------- SIDEBAR FILTERS ----------
st.sidebar.header("Filters")
sentiment_pick = st.sidebar.multiselect(
    "Sentiment", ["positive","neutral","negative"], default=["positive","neutral","negative"]
)

keyword = st.sidebar.text_input("Keyword search (in caption)", value="").strip().lower()

# Build hashtag universe for multi-select
all_tags = sorted({t for row in df["hashtags_list"] for t in row})
tags_pick = st.sidebar.multiselect("Hashtags", all_tags)

# Date filter if timestamps exist
if df["timestamp_parsed"].notna().any():
    min_d = pd.to_datetime(df["timestamp_parsed"].min())
    max_d = pd.to_datetime(df["timestamp_parsed"].max())
    date_range = st.sidebar.date_input(
        "Date range", value=(min_d.date(), max_d.date()), min_value=min_d.date(), max_value=max_d.date()
    )
else:
    date_range = None

# Apply filters
f = df.copy()
f = f[f["sentiment"].isin(sentiment_pick)]
if keyword:
    f = f[f["clean_caption"].str.contains(re.escape(keyword))]
if tags_pick:
    f = f[f["hashtags_list"].apply(lambda r: any(t in r for t in tags_pick))]
if date_range and df["timestamp_parsed"].notna().any():
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)
    f = f[(f["timestamp_parsed"] >= start) & (f["timestamp_parsed"] < end)]

# ---------- METRICS ----------
c1, c2, c3, c4 = st.columns(4)
total_n = len(f)
pos = int((f["sentiment"] == "positive").sum())
neu = int((f["sentiment"] == "neutral").sum())
neg = int((f["sentiment"] == "negative").sum())
c1.metric("Posts (filtered)", total_n)
c2.metric("Positive", pos)
c3.metric("Neutral", neu)
c4.metric("Negative", neg)

st.markdown("---")

# ---------- CHARTS ----------
g1, g2 = st.columns([1,1])

with g1:
    st.subheader("Sentiment distribution")
    if total_n:
        vc = f["sentiment"].value_counts().reindex(["positive","neutral","negative"]).fillna(0)
        fig = plt.figure()
        vc.plot.pie(autopct="%1.1f%%", ylabel="")
        st.pyplot(fig)
    else:
        st.info("No data after filters.")

with g2:
    st.subheader("Top hashtags")
    # flatten tags, count
    tag_counts = {}
    for row in f["hashtags_list"]:
        for t in row:
            tag_counts[t] = tag_counts.get(t, 0) + 1
    if tag_counts:
        top_tags = pd.Series(tag_counts).sort_values(ascending=False).head(20)
        fig2 = plt.figure()
        top_tags[::-1].plot.barh()  # horizontal bar, most at bottom to top
        plt.xlabel("Count")
        plt.ylabel("Hashtag")
        st.pyplot(fig2)
    else:
        st.info("No hashtags present.")

# Trend only if timestamps exist
if f["timestamp_parsed"].notna().any():
    st.subheader("Monthly sentiment trend (compound mean)")
    monthly = (
        f.dropna(subset=["timestamp_parsed"])
         .set_index("timestamp_parsed")
         .resample("M")["compound"].mean()
    )
    if not monthly.empty:
        fig3 = plt.figure()
        monthly.plot()
        plt.xlabel("Month")
        plt.ylabel("Avg compound")
        st.pyplot(fig3)
    else:
        st.info("No time data available for trend.")

st.markdown("---")

# ---------- TABLE + EXPORT ----------
st.subheader("Posts (preview)")
show_cols = [c for c in ["clean_caption","sentiment","compound","hashtags_list","timestamp_parsed"] if c in f.columns]
st.dataframe(f[show_cols].head(200), use_container_width=True)

def to_csv_bytes(df_):
    return df_.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download filtered CSV",
    data=to_csv_bytes(f),
    file_name="tourism_sentiment_filtered.csv",
    mime="text/csv"
)

# ---------- INSIGHTS ----------
st.subheader("Quick insights")
if total_n == 0:
    st.write("• Your filters nuked everything. Loosen them.")
else:
    # crude “insight” bullets
    top3 = []
    if tag_counts:
        top3 = list(pd.Series(tag_counts).sort_values(ascending=False).head(3).index)
    st.write(f"• Positive/Neutral/Negative split = {pos}/{neu}/{neg}.")
    if top3:
        st.write(f"• Top hashtags in this slice: {', '.join('#'+t for t in top3)}.")
    if f['compound'].mean() > 0.05:
        st.write("• Overall tone: mildly positive.")
    elif f['compound'].mean() < -0.05:
        st.write("• Overall tone: mildly negative.")
    else:
        st.write("• Overall tone: mixed/neutral.")
