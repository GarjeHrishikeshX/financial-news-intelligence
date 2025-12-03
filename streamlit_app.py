import streamlit as st
import requests
from streamlit_lottie import st_lottie

# ----------------------------------------------------------
# CONFIG (NO SECRETS, SIMPLE, FAST)
# ----------------------------------------------------------

st.set_page_config(
    page_title="AI Financial News Intelligence",
    page_icon="📰",
    layout="wide"
)

# 🔥 LOCAL BACKEND URL (WORKS INSTANTLY)
BACKEND_URL = "http://127.0.0.1:8000"


# ----------------------------------------------------------
# LOTTIE LOADER
# ----------------------------------------------------------

# ----------------------------------------------------------
# LOTTIE LOADER (SAFE)
# ----------------------------------------------------------

def load_lottie(url: str):
    try:
        r = requests.get(url)
        if r.status_code == 200:
            return r.json()
        return None
    except:
        return None

# WORKING LOTTIE LINK (verified now)
news_lottie = load_lottie("https://lottie.host/8afc652b-96fd-4f03-bb75-a554d968d39e/RzOjbzBeKL.json")



# ----------------------------------------------------------
# HEADER
# ----------------------------------------------------------

# ----------------------------------------------------------
# HEADER (SAFE LOTTIE)
# ----------------------------------------------------------

col1, col2 = st.columns([2, 1])
with col1:
    st.title("📰 AI Financial News Intelligence Dashboard")
    st.write("Real-time analysis powered by NLP + TF-IDF.")

with col2:
    if news_lottie:
        st_lottie(news_lottie, height=120)
    else:
        st.write("")  # fallback: empty space, no error

# ----------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------

st.sidebar.header("🔎 Navigation")

page = st.sidebar.selectbox(
    "Go to:",
    ["Search News", "Browse Articles", "System Health"]
)


# ----------------------------------------------------------
# SEARCH PAGE
# ----------------------------------------------------------

if page == "Search News":
    st.subheader("🔍 Search Financial News")

    query = st.text_input("Enter your query (example: HDFC Bank, Inflation, RBI Policy):")

    if st.button("Search"):
        if not query.strip():
            st.warning("Please type a query.")
        else:
            with st.spinner("Analyzing..."):
                try:
                    response = requests.post(
                        f"{BACKEND_URL}/query",
                        json={"query": query},
                        timeout=20
                    )

                    data = response.json()

                    if "error" in data:
                        st.error("Backend Error:")
                        st.code(data["traceback"])
                        st.stop()

                    # Show results
                    st.markdown("### 🧠 Interpretation")
                    st.markdown("### 🧠 Query Understanding")

                    if "interpretation" in data:
                      st.json(data["interpretation"])
                    else:
                        st.info("No interpretation returned by backend.")


                    st.markdown("### 📄 Top Search Results")

                    results = data.get("results", [])

                    if not results:
                          st.warning("No results returned by backend.")
                    else:
                     for item in results:
        # existing HTML block

                        st.markdown(
                            f"""
                            <div style="
                                padding: 15px;
                                border-radius: 12px;
                                background-color: #f7f7f9;
                                box-shadow: 0 2px 6px rgba(0,0,0,0.1);
                                margin-bottom: 12px;
                            ">
                                <h4>{item['title']}</h4>
                                <p>{item['content'][:300]}...</p>

                                <span style="
                                    background:#4CAF50;
                                    color:white;
                                    padding:4px 10px;
                                    border-radius:6px;
                                    font-size:13px;
                                ">Score: {item['score']}</span>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                except Exception as e:
                    st.error("Streamlit → Backend communication failed")
                    st.exception(e)


# ----------------------------------------------------------
# BROWSE ARTICLES
# ----------------------------------------------------------

elif page == "Browse Articles":
    st.subheader("📚 All Articles")

    with st.spinner("Loading..."):
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
                        <p>{a['content'][:250]}...</p>
                        <small><b>{a['source']}</b> — {a['date']}</small>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        except:
            st.error("Unable to fetch articles.")


# ----------------------------------------------------------
# SYSTEM HEALTH
# ----------------------------------------------------------

elif page == "System Health":
    st.subheader("🩺 System Status")

    try:
        r = requests.get(f"{BACKEND_URL}/", timeout=5).json()
        st.success("Backend running ✔")
        st.json(r)
    except:
        st.error("Backend not reachable ❌")

    st.info(f"Using backend: {BACKEND_URL}")
