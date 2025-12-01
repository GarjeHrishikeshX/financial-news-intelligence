# streamlit_app.py
import streamlit as st
import requests
from streamlit_lottie import st_lottie
import pandas as pd
import datetime
import math

# -------------------------
# Configuration
# -------------------------
st.set_page_config(page_title="AI Financial News Intelligence",
                   page_icon="💹", layout="wide")

# Update this to your deployed backend
BACKEND_URL = "https://financial-news-intelligence-1.onrender.com"

# -------------------------
# Helpers
# -------------------------
def safe_lottie_load(url: str, timeout: int = 5):
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None
    return None

def backend_post(path: str, payload: dict, timeout: int = 20):
    """
    Returns (ok:bool, status:int|None, parsed_json_or_text)
    """
    try:
        r = requests.post(f"{BACKEND_URL}{path}", json=payload, timeout=timeout)
    except Exception as e:
        return False, None, f"Request failed: {e}"
    if r.status_code != 200:
        return False, r.status_code, r.text
    try:
        return True, r.status_code, r.json()
    except Exception:
        return False, r.status_code, r.text

def backend_get(path: str, timeout: int = 20):
    try:
        r = requests.get(f"{BACKEND_URL}{path}", timeout=timeout)
    except Exception as e:
        return False, None, f"Request failed: {e}"
    if r.status_code != 200:
        return False, r.status_code, r.text
    try:
        return True, r.status_code, r.json()
    except Exception:
        return False, r.status_code, r.text

def simple_sentiment_score(text: str):
    """
    Lightweight sentiment: +1 for positive words, -1 for negative words.
    Not a replacement for a model, but fast and tiny for dashboards.
    """
    pos = {"gain","rise","up","surge","beat","profit","growth","positive","upgrade","record"}
    neg = {"fall","drop","loss","down","decline","weak","negative","downgrade","fraud","concern","losses"}
    t = text.lower()
    score = 0
    for w in pos:
        if w in t:
            score += 1
    for w in neg:
        if w in t:
            score -= 1
    # normalize roughly to [-1..1]
    if score == 0:
        return 0.0
    return max(-1.0, min(1.0, score / 5.0))

# -------------------------
# Load Lottie animations
# -------------------------
hero = safe_lottie_load("https://assets3.lottiefiles.com/packages/lf20_WamH3P.json")
search_anim = safe_lottie_load("https://assets8.lottiefiles.com/packages/lf20_tgm3ldwv.json")
finance_anim = safe_lottie_load("https://assets4.lottiefiles.com/packages/lf20_A1XbqB.json")

# -------------------------
# Header (Finance style)
# -------------------------
col1, col2 = st.columns([3,1])
with col1:
    st.markdown("<h1 style='margin:0; color:#0b6b3a;'>💹 Financial News Intelligence</h1>", unsafe_allow_html=True)
    st.markdown("<div style='color:#4b5862;'>Fast, lightweight insights for markets — TF-IDF fallback on Render</div>", unsafe_allow_html=True)
with col2:
    if hero:
        st_lottie(hero, height=120)

st.markdown("---")

# -------------------------
# Sidebar navigation & quick filters
# -------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Search", "Articles", "Finance Dashboard", "Health"])

# Optional: quick symbol filter (not used server-side here, but UI nicety)
symbol = st.sidebar.text_input("Company (optional)", placeholder="HDFC Bank")

# -------------------------
# SEARCH PAGE
# -------------------------
if page == "Search":
    colA, colB = st.columns([4,1])
    with colA:
        query = st.text_input("Search (natural language)", value=f"{symbol} news" if symbol else "")
    with colB:
        search_btn = st.button("Search", help="Query the backend for relevant news")

    if search_btn:
        if not query or not query.strip():
            st.warning("Please enter a query.")
        else:
            with st.spinner("Querying backend..."):
                ok, status, payload = backend_post("/query", {"query": query}, timeout=25)
                if not ok:
                    st.error(f"Backend returned error (status={status}). Raw response:")
                    st.code(str(payload))
                else:
                    data = payload
                    interpretation = data.get("interpretation", {})
                    results = data.get("results", [])

                    st.markdown("### 🔎 Interpretation")
                    st.json(interpretation)

                    st.markdown("### 📰 Results")
                    if not results:
                        st.info("No results found.")
                    else:
                        for r in results:
                            title = r.get("title","Untitled")
                            content = r.get("content","") or ""
                            score = r.get("score", 0.0)
                            explanation = r.get("explanation","")
                            # finance-styled card
                            color = "#27ae60" if score >= 0.5 else ("#f39c12" if score >= 0.25 else "#c0392b")
                            st.markdown(
                                f"""
                                <div style="
                                    padding:14px;
                                    border-radius:10px;
                                    background: linear-gradient(90deg, rgba(255,255,255,1), rgba(248,249,250,1));
                                    border-left:6px solid {color};
                                    box-shadow: 0 6px 18px rgba(11,11,11,0.03);
                                    margin-bottom:12px;
                                ">
                                    <h4 style="margin:0 0 6px 0;">{title}</h4>
                                    <p style="margin:0 0 8px 0; color:#333;">{content[:600]}{('...' if len(content)>600 else '')}</p>
                                    <div style="display:flex; align-items:center; gap:12px;">
                                        <div style="background:{color}; color:white; padding:6px 10px; border-radius:6px; font-weight:600;">
                                            Score {round(score,3)}
                                        </div>
                                        <div style="color:#555;">{explanation}</div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)

# -------------------------
# ARTICLES PAGE
# -------------------------
elif page == "Articles":
    st.header("All Articles")
    load_btn = st.button("Load articles from backend")
    if load_btn:
        with st.spinner("Loading articles..."):
            ok, status, payload = backend_get("/articles", timeout=20)
            if not ok:
                st.error(f"Failed to load (status={status}). Raw response:")
                st.code(str(payload))
            else:
                articles = payload
                if not articles:
                    st.info("No articles found.")
                else:
                    # Show compact list with sentiment and date
                    rows = []
                    for a in articles:
                        t = a.get("title","")
                        c = a.get("content","") or ""
                        d = a.get("date","")
                        try:
                            date_obj = datetime.datetime.fromisoformat(d).date() if d else None
                        except Exception:
                            date_obj = None
                        sent = simple_sentiment_score(t + " " + c)
                        rows.append({
                            "id": a.get("id"),
                            "title": t,
                            "date": str(date_obj) if date_obj else a.get("date",""),
                            "source": a.get("source",""),
                            "sentiment": sent
                        })
                    df = pd.DataFrame(rows)
                    st.dataframe(df.sort_values(by="date", ascending=False).reset_index(drop=True), use_container_width=True)

# -------------------------
# FINANCE DASHBOARD
# -------------------------
elif page == "Finance Dashboard":
    st.header("Finance Dashboard")
    col1, col2 = st.columns([2,1])
    with col2:
        if finance_anim:
            st_lottie(finance_anim, height=220)
    with col1:
        st.markdown("### Market Signals — article volume & simple sentiment over time")

    # fetch articles for timeline
    ok, status, payload = backend_get("/articles", timeout=20)
    if not ok:
        st.error(f"Failed to load timeline (status={status}). Raw response:")
        st.code(str(payload))
    else:
        articles = payload
        if not articles:
            st.info("No data to show.")
        else:
            # parse dates and compute daily counts and avg sentiment
            rows = []
            for a in articles:
                d = a.get("date") or ""
                try:
                    dt = datetime.datetime.fromisoformat(d).date()
                except Exception:
                    # fallback: try parse yyyy-mm-dd from start of string
                    try:
                        dt = datetime.date.fromisoformat(d.split("T")[0])
                    except Exception:
                        dt = None
                text = (a.get("title","") or "") + " " + (a.get("content","") or "")
                s = simple_sentiment_score(text)
                rows.append({"date": dt or datetime.date.today(), "sentiment": s})

            df = pd.DataFrame(rows)
            df_group = df.groupby("date").agg({"sentiment":["mean","count"]})
            df_group.columns = ["avg_sent", "count"]
            df_group = df_group.sort_index()

            # chart: counts
            st.markdown("#### Article Volume")
            st.line_chart(df_group["count"])

            # chart: sentiment
            st.markdown("#### Average Sentiment (simple)")
            st.line_chart(df_group["avg_sent"])

            # KPIs
            latest = df_group.iloc[-1]
            colk1, colk2, colk3 = st.columns(3)
            colk1.metric("Articles (latest day)", int(latest["count"]))
            colk2.metric("Avg sentiment (latest)", round(float(latest["avg_sent"]), 3))
            # simple momentum indicator
            if len(df_group) >= 2:
                prev = df_group.iloc[-2]["avg_sent"]
                delta = latest["avg_sent"] - prev
                colk3.metric("Sentiment change", f"{round(delta,3)}", delta=f"{round(delta,3)}")
            else:
                colk3.metric("Sentiment change", "0.0")

# -------------------------
# HEALTH PAGE
# -------------------------
elif page == "Health":
    st.header("System Health")
    ok, status, payload = backend_get("/", timeout=5)
    if ok:
        st.success("Backend reachable ✓")
        st.json(payload)
    else:
        st.error("Backend not reachable or returned non-JSON. Raw response:")
        st.code(str(payload))

    st.markdown("### Diagnostics")
    st.write("- Backend URL:", BACKEND_URL)
    st.write("- Embedding mode: TF-IDF fallback (lightweight)")
    st.write("- Use `/docs` on backend for API details")
