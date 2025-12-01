# streamlit_app.py

import streamlit as st
import requests
from streamlit_lottie import st_lottie
import json

# ----------------------------------------------------------
# CONFIG
# ----------------------------------------------------------

st.set_page_config(
    page_title="AI Financial News Intelligence",
    page_icon="📰",
    layout="wide"
)

BACKEND_URL = "https://financial-news-intelligence-1.onrender.com"


# ----------------------------------------------------------
# LOTTIE ANIMATIONS
# ----------------------------------------------------------

def load_lottie(url: str):
    try:
        return requests.get(url).json()
    except:
        return None

news_lottie = load_lottie("https://assets3.lottiefiles.com/packages/lf20_WamH3P.json")
search_lottie = load_lottie("https://assets8.lottiefiles.com/packages/lf20_tgm3ldwv.json")


# ----------------------------------------------------------
# HEADER
# ----------------------------------------------------------

col1, col2 = st.columns([2, 1])
with col1:
    st.title("📰 AI Financial News Intelligence Dashboard")
    st.write("Real-time analysis powered by NLP, TF-IDF, and knowledge agents.")

with col2:
    st_lottie(news_lottie, height=140)


# ----------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------

st.sidebar.header("🧭 Navigation")

page = st.sidebar.selectbox(
    "Go to:",
    ["Search News", "Browse Articles", "System Health"]
)


# ----------------------------------------------------------
# SEARCH PAGE
# ----------------------------------------------------------

if page == "Search News":
    st.subheader("🔍 Smart Financial Search")
    st_lottie(search_lottie, height=180)

    query = st.text_input(
        "Enter a query:",
        placeholder="Try: HDFC Bank news, RBI policy updates, Banking sector analysis..."
    )

    if st.button("Search"):
        if not query.strip():
            st.warning("Please type a query.")
        else:
            with st.spinner("Analyzing your request..."):
                try:
                    response = requests.post(
                        f"{BACKEND_URL}/query",
                        json={"query": query},
                        timeout=30
                    )
                    data = response.json()

                    # Interpretation
                    st.markdown("### 🧠 Query Understanding")
                    st.json(data["interpretation"])

                    # Results
                    st.markdown("### 📄 Top Results")

                    for item in data["results"]:
                        with st.container():
                            st.markdown(
                                f"""
                                <div style="
                                    padding: 15px;
                                    border-radius: 12px;
                                    background-color: #f7f7f9;
                                    box-shadow: 0 2px 6px rgba(0,0,0,0.08);
                                    margin-bottom: 12px;
                                ">
                                    <h4 style="margin-bottom:5px;">{item['title']}</h4>
                                    <p>{item['content'][:400]}...</p>

                                    <span style="
                                        background:#4CAF50;
                                        color:white;
                                        padding:4px 10px;
                                        border-radius:6px;
                                        font-size:13px;
                                    ">
                                        Score: {item['score']}
                                    </span>

                                    <p style="font-size:13px;color:#555;margin-top:8px;">
                                        {item['explanation']}
                                    </p>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                except Exception as e:
                    st.error("Backend error.")
                    st.exception(e)


# ----------------------------------------------------------
# BROWSE ARTICLES PAGE
# ----------------------------------------------------------

elif page == "Browse Articles":
    st.subheader("📚 All Articles")

    with st.spinner("Loading articles..."):
        try:
            articles = requests.get(f"{BACKEND_URL}/articles", timeout=20).json()

            for a in articles:
                st.markdown(
                    f"""
                    <div style="
                        padding: 12px;
                        border-radius: 10px;
                        background-color: #fff;
                        border: 1px solid #e0e0e0;
                        margin-bottom: 10px;
                    ">
                        <h4>{a['title']}</h4>
                        <p>{a['content'][:300]}...</p>
                        <small>
                            <b>Source:</b> {a['source']} |
                            <b>Date:</b> {a['date']}
                        </small>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        except:
            st.error("Failed to fetch articles.")


# ----------------------------------------------------------
# SYSTEM HEALTH PAGE
# ----------------------------------------------------------

elif page == "System Health":
    st.subheader("🩺 System Status")

    try:
        r = requests.get(f"{BACKEND_URL}/", timeout=5).json()
        st.success("Backend is healthy ✔")
        st.json(r)
    except:
        st.error("Backend not reachable ❌")


    st.markdown("### 🔧 Diagnostics")

    st.write("- Backend URL:", BACKEND_URL)
    st.write("- Running using TF-IDF embeddings (Render-friendly)")
    st.write("- Multi-Agent Pipeline: Ingestion → Dedup → Entity → Impact → Query")
