# streamlit_app.py

import streamlit as st
import requests

st.set_page_config(
    page_title="AI Financial News Intelligence",
    page_icon="📰",
    layout="wide"
)

# ---------------------------------------------------
# BACKEND URL (Replace with your Render backend URL)
# ---------------------------------------------------

BACKEND_URL = "https://financial-news-intelligence-1.onrender.com"

st.title("📰 AI Financial News Intelligence Dashboard")
st.write("Search financial insights using NLP + multi-agent intelligence.")


# ---------------------------------------------------
# Search Box
# ---------------------------------------------------

query = st.text_input("Enter your query:", placeholder="Ex: HDFC Bank news")

if st.button("Search") and query.strip():
    with st.spinner("Fetching results..."):
        try:
            response = requests.post(
                f"{BACKEND_URL}/query",
                json={"query": query},
                timeout=20
            )
            data = response.json()

            st.subheader("🔍 Query Interpretation")
            st.json(data["interpretation"])

            st.subheader("📄 Results")
            results = data["results"]

            if not results:
                st.warning("No matching articles found.")
            else:
                for item in results:
                    st.markdown(f"### {item['title']}")
                    st.write(item["content"])
                    st.caption(f"Relevance Score: {item['score']}")
                    st.info(item["explanation"])
                    st.divider()

        except Exception as e:
            st.error("Error communicating with backend.")
            st.exception(e)


# ---------------------------------------------------
# Articles Browser
# ---------------------------------------------------

st.sidebar.title("All Articles")

if st.sidebar.button("Load Articles"):
    try:
        arts = requests.get(f"{BACKEND_URL}/articles", timeout=20).json()
        for a in arts:
            st.sidebar.markdown(f"**{a['id']}. {a['title']}**")
            st.sidebar.caption(a["source"])
    except:
        st.sidebar.error("Failed to load articles.")
