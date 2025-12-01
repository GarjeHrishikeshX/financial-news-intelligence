# streamlit_app.py
import streamlit as st
import requests
from streamlit_lottie import st_lottie
import json

# -------------------------
# Configuration
# -------------------------
st.set_page_config(page_title="AI Financial News Intelligence",
                   page_icon="📰", layout="wide")

# Replace with your Render backend URL
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

def backend_post_json(path: str, payload: dict, timeout: int = 20):
    """
    POST helper that returns (success, status_code, parsed_json_or_text).
    """
    try:
        resp = requests.post(f"{BACKEND_URL}{path}", json=payload, timeout=timeout)
    except Exception as e:
        return False, None, f"Request failed: {e}"
    if resp.status_code != 200:
        return False, resp.status_code, resp.text
    # try parse json
    try:
        return True, resp.status_code, resp.json()
    except Exception:
        return False, resp.status_code, resp.text

def backend_get_json(path: str, timeout: int = 20):
    try:
        resp = requests.get(f"{BACKEND_URL}{path}", timeout=timeout)
    except Exception as e:
        return False, None, f"Request failed: {e}"
    if resp.status_code != 200:
        return False, resp.status_code, resp.text
    try:
        return True, resp.status_code, resp.json()
    except Exception:
        return False, resp.status_code, resp.text

# -------------------------
# Lottie animations
# -------------------------
news_lottie = safe_lottie_load("https://assets3.lottiefiles.com/packages/lf20_WamH3P.json")
search_lottie = safe_lottie_load("https://assets8.lottiefiles.com/packages/lf20_tgm3ldwv.json")

# -------------------------
# Header
# -------------------------
col1, col2 = st.columns([2, 1])
with col1:
    st.title("📰 AI Financial News Intelligence Dashboard")
    st.write("Real-time analysis powered by NLP, TF-IDF, and knowledge agents.")
with col2:
    if news_lottie:
        st_lottie(news_lottie, height=140)

# -------------------------
# Sidebar navigation
# -------------------------
st.sidebar.header("🧭 Navigation")
page = st.sidebar.selectbox("Go to:", ["Search News", "Browse Articles", "System Health"])

# -------------------------
# Search Page
# -------------------------
if page == "Search News":
    st.subheader("🔍 Smart Financial Search")
    if search_lottie:
        st_lottie(search_lottie, height=180)

    query = st.text_input("Enter your query:", placeholder="Try: HDFC Bank news, RBI policy updates...")
    colA, colB = st.columns([3,1])
    with colB:
        search_clicked = st.button("Search")

    if search_clicked:
        if not query or not query.strip():
            st.warning("Please type a query.")
        else:
            with st.spinner("Analyzing your request..."):
                ok, status, payload = backend_post_json("/query", {"query": query}, timeout=30)
                if not ok:
                    st.error(f"Backend returned error (status={status}). Raw response below:")
                    st.code(str(payload))
                else:
                    data = payload
                    # safe guards
                    interpretation = data.get("interpretation", {})
                    results = data.get("results", [])

                    st.markdown("### 🧠 Query Understanding")
                    st.json(interpretation)

                    st.markdown("### 📄 Top Results")
                    if not results:
                        st.info("No matching articles found.")
                    else:
                        for item in results:
                            title = item.get("title", "Untitled")
                            content = item.get("content", "") or ""
                            score = item.get("score", 0.0)
                            explanation = item.get("explanation", "")

                            st.markdown(
                                f"""
                                <div style="
                                    padding: 15px;
                                    border-radius: 12px;
                                    background-color: #f7f7f9;
                                    box-shadow: 0 2px 6px rgba(0,0,0,0.06);
                                    margin-bottom: 12px;
                                ">
                                    <h4 style="margin-bottom:5px;">{title}</h4>
                                    <p style="margin-top:0;">{content[:600]}{('...' if len(content)>600 else '')}</p>
                                    <div style="display:flex; gap:8px; align-items:center;">
                                        <span style="
                                            background:#4CAF50;
                                            color:white;
                                            padding:4px 10px;
                                            border-radius:6px;
                                            font-size:13px;
                                        ">
                                            Score: {round(score,3)}
                                        </span>
                                        <span style="color:#555;font-size:13px">{explanation}</span>
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

# -------------------------
# Browse Articles Page
# -------------------------
elif page == "Browse Articles":
    st.subheader("📚 All Articles")
    with st.spinner("Loading articles..."):
        ok, status, payload = backend_get_json("/articles", timeout=20)
        if not ok:
            st.error(f"Failed to load articles (status={status}). Raw response below:")
            st.code(str(payload))
        else:
            articles = payload
            if not articles:
                st.info("No articles found.")
            for a in articles:
                title = a.get("title", "Untitled")
                content = a.get("content", "") or ""
                source = a.get("source", "")
                date = a.get("date", "")
                st.markdown(
                    f"""
                    <div style="
                        padding: 12px;
                        border-radius: 10px;
                        background-color: #fff;
                        border: 1px solid #e8e8e8;
                        margin-bottom: 10px;
                    ">
                        <h4 style="margin:0 0 6px 0;">{title}</h4>
                        <p style="margin:0 0 8px 0;">{content[:400]}{('...' if len(content)>400 else '')}</p>
                        <small style="color:#666;">
                            <b>Source:</b> {source} &nbsp; | &nbsp; <b>Date:</b> {date}
                        </small>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

# -------------------------
# System Health Page
# -------------------------
elif page == "System Health":
    st.subheader("🩺 System Status")
    ok, status, payload = backend_get_json("/", timeout=5)
    if not ok:
        st.error("Backend not reachable or returning non-JSON. See raw response below:")
        st.code(str(payload))
    else:
        st.success("Backend is healthy ✔")
        st.json(payload)

    st.markdown("### 🔧 Diagnostics")
    st.write("- Backend URL:", BACKEND_URL)
    st.write("- Embedding: TF-IDF fallback (lightweight)")
    st.write("- Pipeline: Ingest → Dedup → Entity → Impact → Query")
