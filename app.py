# app.py
# Final stable Streamlit app (no plt.pause, no Wedge errors, no torch/meta errors,
# fallback TF-IDF when SentenceTransformer fails)

import sys
import os
import json
import requests
from io import BytesIO
from PIL import Image
import streamlit as st

ROOT = os.path.dirname(__file__)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ---------------------------
# Safe Imports
# ---------------------------
try:
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Wedge, Circle
    MATPLOTLIB = True
except Exception:
    MATPLOTLIB = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_OK = True
except Exception:
    VADER_OK = False

try:
    from src.agents.query_agent import QueryAgent
    from src.agents.storage_agent import StorageAgent
except Exception:
    QueryAgent = None
    StorageAgent = None


# ---------------------------
# Fallback Sentiment if VADER not available
# ---------------------------
if not VADER_OK:
    class SentimentIntensityAnalyzer:
        def polarity_scores(self, text):
            text = text.lower()
            pos = sum(w in text for w in ["gain", "good", "great", "up"])
            neg = sum(w in text for w in ["loss", "bad", "crash", "down"])
            compound = (pos - neg) / 10
            neu = 1 - min(1, abs(compound))
            return {"pos": pos/10, "neg": neg/10, "neu": neu, "compound": compound}

sentiment = SentimentIntensityAnalyzer()


# ---------------------------
# Utilities
# ---------------------------
@st.cache_data
def load_lottie(url):
    try:
        r = requests.get(url, timeout=3)
        if r.status_code == 200:
            return r.json()
    except:
        return None
    return None

loading_animation = load_lottie(
    "https://assets9.lottiefiles.com/packages/lf20_qp1q7mct.json"
)


@st.cache_data
def get_company_logo(company_name):
    try:
        domain = company_name.replace(" ", "") + ".com"
        url = f"https://logo.clearbit.com/{domain}"
        r = requests.get(url, timeout=3)
        if r.status_code == 200:
            return Image.open(BytesIO(r.content))
    except:
        return None
    return None


# ---------------------------
# Visualization (safe)
# ---------------------------
def donut_chart(data):
    if not MATPLOTLIB:
        st.info("Matplotlib unavailable. Skipping chart.")
        return

    fig, ax = plt.subplots(figsize=(4, 4))
    vals = [data["pos"], data["neu"], data["neg"]]
    ax.pie(vals, wedgeprops=dict(width=0.4), labels=["Positive", "Neutral", "Negative"])
    ax.axis("equal")
    st.pyplot(fig)


def sentiment_line(values):
    if not MATPLOTLIB:
        st.info("Matplotlib unavailable.")
        return

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(values, marker="o")
    ax.set_ylim(-1, 1)
    st.pyplot(fig)


def risk_gauge(risk):
    if not MATPLOTLIB:
        st.info("Matplotlib unavailable.")
        return

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.add_patch(Circle((0,0), 1, color="white"))
    ax.add_patch(Wedge((0,0), 1, 0, 60, facecolor="#4CAF50"))
    ax.add_patch(Wedge((0,0), 1, 60, 120, facecolor="#FFC107"))
    ax.add_patch(Wedge((0,0), 1, 120, 180, facecolor="#F44336"))
    angle = risk * 180
    rad = np.deg2rad(angle)
    ax.plot([0, np.cos(rad)], [0, np.sin(rad)], lw=3, color="black")
    ax.axis("equal")
    ax.axis("off")
    st.pyplot(fig)


# ---------------------------
# Helpers
# ---------------------------
def analyze_sentiment(text):
    s = sentiment.polarity_scores(text)
    if s["compound"] >= 0.05:
        return "Positive", s
    elif s["compound"] <= -0.05:
        return "Negative", s
    return "Neutral", s


def aggregate_sentiments(results):
    if not results:
        return {"pos": 0, "neu": 0, "neg": 0}
    p = n = u = 0
    for r in results:
        s = sentiment.polarity_scores(r["content"])
        p += s["pos"]
        n += s["neg"]
        u += s["neu"]
    l = len(results)
    return {"pos": p/l, "neu": u/l, "neg": n/l}


# ---------------------------
# Init Agents
# ---------------------------
if QueryAgent is None or StorageAgent is None:
    st.error("Project agents not importable. Run app from project root.")
    st.stop()

query_agent = QueryAgent("data/storage.db")
store = StorageAgent("data/storage.db")


# ---------------------------
# UI
# ---------------------------
st.set_page_config(page_title="AI Financial News Intelligence", layout="wide")

page = st.sidebar.radio(
    "Navigation",
    ["🔍 Search News", "📚 All Articles", "🗂 Story Groups", "ℹ️ About"]
)

st.title("🔎 AI Financial News Intelligence")


# ---------------------------
# PAGE 1 – SEARCH
# ---------------------------
if page == "🔍 Search News":
    q = st.text_input("Enter search query:")
    if st.button("Run Search"):
        if not q.strip():
            st.warning("Enter a valid query.")
        else:
            res = query_agent.query(q)

            if not res["results"]:
                st.error("No articles found.")
                st.stop()

            st.subheader("Interpretation")
            st.json(res["interpretation"])

            st.subheader("Overall Sentiment")
            agg = aggregate_sentiments(res["results"])
            donut_chart(agg)

            st.subheader("Sentiment Timeline")
            values = [
                sentiment.polarity_scores(r["content"])["compound"]
                for r in res["results"]
            ]
            sentiment_line(values)

            st.subheader("Risk Gauge")
            r = agg["neg"] * 1 + agg["neu"] * 0.4 - agg["pos"] * 0.3
            r = max(0, min(1, r))
            risk_gauge(r)

            st.subheader("Articles")
            for a in res["results"]:
                st.write(f"### {a['title']}")
                st.write(a["content"])
                label, scr = analyze_sentiment(a["content"])
                st.write("Sentiment:", label)
                donut_chart(scr)


# ---------------------------
# PAGE 2 – ALL ARTICLES
# ---------------------------
elif page == "📚 All Articles":
    cur = store.conn.cursor()

    rows = cur.execute("SELECT raw FROM articles").fetchall()
    if not rows:
        st.warning("No articles found.")
    else:
        for r in rows:
            try:
                art = json.loads(r[0])
            except:
                art = r[0]
            st.write(art)


# ---------------------------
# PAGE 3 – STORY GROUPS
# ---------------------------
elif page == "🗂 Story Groups":
    stories = store.load_all_stories()
    if not stories:
        st.warning("No groups available.")
    else:
        for s in stories:
            st.write(s)


# ---------------------------
# PAGE 4 – ABOUT
# ---------------------------
else:
    st.write("Made for AI Hiring Hackathon 2025 — Hrishikesh Garje")
