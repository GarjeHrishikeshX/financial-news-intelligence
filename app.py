# -------------------------------------------------------
# IMPORTS
# -------------------------------------------------------
import streamlit as st
import json
import requests
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from streamlit_lottie import st_lottie

from src.agents.query_agent import QueryAgent
from src.agents.storage_agent import StorageAgent
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# -------------------------------------------------------
# LOTTIE LOADER FUNCTION
# -------------------------------------------------------
def load_lottie(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

loading_animation = load_lottie(
    "https://assets9.lottiefiles.com/packages/lf20_qp1q7mct.json"
)


# -------------------------------------------------------
# AUTO-FETCH COMPANY LOGO
# -------------------------------------------------------
@st.cache_data
def get_company_logo(company_name):
    domain_guess = company_name.replace(" ", "") + ".com"
    url = f"https://logo.clearbit.com/{domain_guess}"

    try:
        response = requests.get(url, timeout=2)
        if response.status_code == 200:
            return Image.open(BytesIO(response.content))
        else:
            return None
    except:
        return None


# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------
st.set_page_config(
    page_title="AI Financial News Intelligence",
    page_icon="🔎",
    layout="wide"
)


# -------------------------------------------------------
# CUSTOM GOOGLE FONT
# -------------------------------------------------------
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif !important;
}
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------
# DARK MODE
# -------------------------------------------------------
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

def toggle_theme():
    st.session_state.dark_mode = not st.session_state.dark_mode

st.sidebar.checkbox("🌙 Dark Mode", value=st.session_state.dark_mode,
                    on_change=toggle_theme)
st.sidebar.markdown("---")


# -------------------------------------------------------
# DARK / LIGHT THEME CSS
# -------------------------------------------------------
if st.session_state.dark_mode:
    st.markdown("""
    <style>
        body, .stApp {
            background-color: #0d1117 !important;
            color: #f0f0f0 !important;
        }
        .glass-card {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.12);
            backdrop-filter: blur(16px);
            padding: 22px;
            border-radius: 16px;
            margin-bottom: 18px;
        }
    </style>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <style>
        .glass-card {
            background: rgba(255,255,255,0.7);
            border: 1px solid #d9d9d9;
            backdrop-filter: blur(14px);
            padding: 22px;
            border-radius: 16px;
            margin-bottom: 18px;
        }
    </style>
    """, unsafe_allow_html=True)


# -------------------------------------------------------
# PREMIUM HEADER + ICON CSS
# -------------------------------------------------------
st.markdown("""
<style>

.header {
    padding: 40px;
    border-radius: 16px;
    text-align: center;
    font-size: 48px;
    font-weight: 700;
    color: white;
    background: linear-gradient(120deg, #4c6ef5, #9b59b6, #f72585);
    background-size: 300% 300%;
    animation: gradientShift 6s ease infinite, fadeIn 2s ease-out;
    margin-bottom: 10px;
    box-shadow: 0 5px 25px rgba(0,0,0,0.4);
}

@keyframes gradientShift {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

@keyframes fadeIn {
    0% {opacity: 0; transform: translateY(-20px);}
    100% {opacity: 1; transform: translateY(0);}
}

.floating-icon {
    animation: float 3s ease-in-out infinite;
}

@keyframes float {
    0% {transform: translateY(0);}
    50% {transform: translateY(-14px);}
    100% {transform: translateY(0);}
}

.entity-chip {
    display: inline-block;
    padding: 6px 14px;
    border-radius: 20px;
    margin-right: 6px;
    margin-top: 6px;
    font-size: 13px;
    color: white;
    animation: glow 2s infinite alternate;
}

@keyframes glow {
    0% { box-shadow: 0 0 4px #9b59b6; }
    100% { box-shadow: 0 0 15px #4c6ef5; }
}

</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------
# SENTIMENT ENGINE
# -------------------------------------------------------
sentiment = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    score = sentiment.polarity_scores(text)
    if score['compound'] >= 0.05:
        return "Positive", score
    elif score['compound'] <= -0.05:
        return "Negative", score
    else:
        return "Neutral", score

def aggregate_sentiments(results, analyzer):
    if len(results) == 0:
        return {"pos": 0, "neu": 0, "neg": 0}

    total = {"pos": 0, "neu": 0, "neg": 0}

    for item in results:
        score = analyzer.polarity_scores(item["content"])
        total["pos"] += score["pos"]
        total["neu"] += score["neu"]
        total["neg"] += score["neg"]

    count = len(results)
    return {
        "pos": total["pos"] / count,
        "neu": total["neu"] / count,
        "neg": total["neg"] / count
    }



# -------------------------------------------------------
# DONUT CHART (ANIMATED)
# -------------------------------------------------------
def animated_donut_chart(sent_scores):
    labels = ["Positive", "Neutral", "Negative"]
    values = [sent_scores["pos"], sent_scores["neu"], sent_scores["neg"]]
    colors = ["#4CAF50", "#FFC107", "#F44336"]

    fig, ax = plt.subplots(figsize=(4, 4))
    wedges, _ = ax.pie(values, colors=colors, radius=1.0)

    center_color = "black" if st.session_state.dark_mode else "white"
    circle = plt.Circle((0, 0), 0.60, color=center_color)
    fig.gca().add_artist(circle)

    for w in wedges:
        w.set_theta2(w.theta1)

    for frame in range(1, 101):
        for i, w in enumerate(wedges):
            w.set_theta2(w.theta1 + (values[i] * 360 * frame / 100))
        plt.pause(0.01)

    ax.axis("equal")
    st.pyplot(fig)


# -------------------------------------------------------
# SENTIMENT TIMELINE
# -------------------------------------------------------
def sentiment_timeline(results):
    vals = []
    for item in results:
        score = sentiment.polarity_scores(item["content"])
        vals.append(score["compound"])
    return vals

def animated_sentiment_timeline(vals):
    fig, ax = plt.subplots(figsize=(6, 3))

    ax.set_ylim(-1, 1)
    ax.set_xlim(0, len(vals) - 1)
    ax.set_xlabel("Article Index")
    ax.set_ylabel("Sentiment (-1 to 1)")

    line, = ax.plot([], [], marker="o", linewidth=3)

    xs, ys = [], []

    for i in range(len(vals)):
        xs.append(i)
        ys.append(vals[i])
        line.set_data(xs, ys)
        plt.pause(0.12)

    st.pyplot(fig)


# -------------------------------------------------------
# RISK GAUGE METER
# -------------------------------------------------------
def calculate_risk(sent):
    risk = sent["neg"] * 1.0 + sent["neu"] * 0.4 - sent["pos"] * 0.3
    return max(0, min(1, risk))

def animated_risk_gauge(risk):
    angle = risk * 180

    fig, ax = plt.subplots(figsize=(5, 3))

    bg = "#0d1117" if st.session_state.dark_mode else "white"
    ax.add_patch(plt.Circle((0,0), 1, color=bg))

    ax.add_patch(plt.Wedge((0,0), 1, 0, 60, facecolor="#4CAF50"))
    ax.add_patch(plt.Wedge((0,0), 1, 60, 120, facecolor="#FFC107"))
    ax.add_patch(plt.Wedge((0,0), 1, 120, 180, facecolor="#F44336"))

    needle, = ax.plot([], [], lw=3,
                      color="white" if st.session_state.dark_mode else "black")

    ax.set_xlim(-1.2,1.2)
    ax.set_ylim(-0.2,1.2)
    ax.axis("off")

    for frame in range(1, 101):
        current = angle * (frame/100)
        rad = np.deg2rad(current)
        needle.set_data([0, np.cos(rad)], [0, np.sin(rad)])
        plt.pause(0.01)

    st.pyplot(fig)


# -------------------------------------------------------
# SECTOR SENTIMENT HEATMAP
# -------------------------------------------------------
def compute_sector_sentiment(results):
    data = {}
    for item in results:
        sectors = item.get("sectors", [])
        score = sentiment.polarity_scores(item["content"])
        for sec in sectors:
            if sec not in data:
                data[sec] = {"pos":0,"neu":0,"neg":0,"count":0}
            data[sec]["pos"] += score["pos"]
            data[sec]["neu"] += score["neu"]
            data[sec]["neg"] += score["neg"]
            data[sec]["count"] += 1

    final = {}
    for sec, v in data.items():
        c = v["count"]
        final[sec] = {
            "pos": v["pos"]/c,
            "neu": v["neu"]/c,
            "neg": v["neg"]/c,
        }
    return final

def sector_sentiment_heatmap(sector_data):
    if not sector_data:
        st.info("No sector data available.")
        return

    secs = list(sector_data.keys())
    vals = []

    for sec in secs:
        s = sector_data[sec]
        score = s["pos"]*1 + s["neu"]*0.5 - s["neg"]*1
        vals.append(score)

    vals = np.array(vals)

    fig, ax = plt.subplots(figsize=(6,3))
    heatmap = ax.imshow(vals.reshape(1,-1), cmap="RdYlGn", vmin=-1, vmax=1)

    ax.set_xticks(range(len(secs)))
    ax.set_xticklabels(secs, rotation=45, ha="right")
    ax.set_yticks([])

    plt.colorbar(heatmap)
    st.pyplot(fig)


# -------------------------------------------------------
# INITIALIZE AGENTS
# -------------------------------------------------------
query_agent = QueryAgent("data/storage.db")
store = StorageAgent("data/storage.db")


# -------------------------------------------------------
# SIDEBAR NAVIGATION
# -------------------------------------------------------
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio(
    "Choose a Page:",
    ["🔍 Search News", "📚 All Articles", "🗂 Story Groups", "ℹ️ About Project"]
)

st.sidebar.info("AI Hiring Hackathon 2025\nBy **Hrishikesh Garje**")
st.sidebar.markdown("---")


# -------------------------------------------------------
# HEADER (ANIMATED)
# -------------------------------------------------------
st.markdown("<div class='header'>🔎 AI Financial News Intelligence</div>",
            unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center;" class="floating-icon">
    <span style="font-size:60px;">📰</span>
</div>
""", unsafe_allow_html=True)


# -------------------------------------------------------
# PAGE 1 — SEARCH
# -------------------------------------------------------
if page == "🔍 Search News":
    st.subheader("✨ Smart Financial News Search Engine")

    query = st.text_input(
        "Type your query (e.g. 'RBI policy impact', 'HDFC Bank earnings', 'IT sector crash')",
        placeholder="Ask anything related to finance..."
    )

    search_btn = st.button("🚀 Run Search", use_container_width=True)

    if search_btn:
        if not query.strip():
            st.warning("⚠ Enter a query.")
        else:
            st_lottie(loading_animation, height=150)

            result = query_agent.query(query)
            # SAFETY CHECK: No results found
            if not result["results"]:
             st.error("❌ No articles found for this query. Try another keyword!")
             st.stop()


            st.success("Search Complete!")
            st.write("---")

            st.markdown("### 🧠 Query Understanding")
            st.json(result["interpretation"])
            st.write("---")

            # ENTITY CHIPS
            chips = ""

            for c in result["interpretation"].get("companies", []):
                chips += f"<span class='entity-chip' style='background:#4c6ef5;'>{c}</span> "

            for s in result["interpretation"].get("sectors", []):
                chips += f"<span class='entity-chip' style='background:#e67e22;'>{s}</span> "

            for r in result["interpretation"].get("regulators", []):
                chips += f"<span class='entity-chip' style='background:#d63031;'>{r}</span> "

            st.markdown("### 🔖 Extracted Entities")
            st.markdown(chips, unsafe_allow_html=True)
            st.write("---")

            # SENTIMENT DONUT
            st.markdown("### 📊 Overall Sentiment")
            overall = aggregate_sentiments(result["results"], sentiment)
            animated_donut_chart(overall)
            st.write("---")

            # SENTIMENT TIMELINE
            st.markdown("### 📈 Sentiment Timeline")
            timeline = sentiment_timeline(result["results"])
            animated_sentiment_timeline(timeline)
            st.write("---")

            # RISK GAUGE
            st.markdown("### 🚨 Market Risk Gauge")
            risk = calculate_risk(overall)
            animated_risk_gauge(risk)

            if risk < 0.33:
                st.success("🟢 Low Market Risk")
            elif risk < 0.66:
                st.warning("🟡 Moderate Market Risk")
            else:
                st.error("🔴 High Market Risk")
            st.write("---")

            # HEATMAP
            st.markdown("### 🔥 Sector-wise Sentiment Heatmap")
            sec_data = compute_sector_sentiment(result["results"])
            sector_sentiment_heatmap(sec_data)
            st.write("---")

            # ARTICLE RESULTS
            st.markdown("### 📰 News Articles")
            for item in result["results"]:

                sent_label, sent_score = analyze_sentiment(item["content"])

                logo = None
                comps = result["interpretation"].get("companies", [])
                if comps:
                    logo = get_company_logo(comps[0])

                st.markdown("<div class='glass-card'>", unsafe_allow_html=True)

                if logo:
                    st.image(logo, width=60)

                st.markdown(f"### 📰 {item['title']}")
                st.markdown(f"{item['content']}")
                st.markdown(
                    f"<b>Similarity:</b> {item['score']}<br>"
                    f"<b>Why:</b> {item['explanation']}<br>"
                    f"<b>Sentiment:</b> {sent_label}",
                    unsafe_allow_html=True
                )
                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown("#### 📊 Sentiment Breakdown")
                animated_donut_chart(sent_score)
                st.write("---")


# -------------------------------------------------------
# PAGE 2 — ALL ARTICLES
# -------------------------------------------------------
elif page == "📚 All Articles":
    st.subheader("📚 All Articles in Database")
    st.write("---")

    cur = store.conn.cursor()
    cur.execute("SELECT raw FROM articles")
    rows = cur.fetchall()

    if not rows:
        st.warning("No articles found.")
    else:
        for r in rows:
            art = eval(r[0])
            st.markdown(f"""
                <div class="glass-card">
                    <h3>{art['title']}</h3>
                    <p>{art['content']}</p>
                    <small>Source: {art['source']} • Date: {art['date']}</small>
                </div>
            """, unsafe_allow_html=True)


# -------------------------------------------------------
# PAGE 3 — STORY GROUPS
# -------------------------------------------------------
elif page == "🗂 Story Groups":
    st.subheader("🗂 Deduplicated Story Clusters")
    st.write("---")

    stories = store.load_all_stories()

    if not stories:
        st.warning("No grouped stories found.")
    else:
        for s in stories:
            rep = s["representative"]
            st.markdown(f"""
            <div class="glass-card">
                <h3>📘 Story #{s['story_id']}</h3>
                <p><b>Main Article:</b> {rep['title']}</p>
                <p><small>Articles Included: {s['article_ids']}</small></p>
            </div>
            """, unsafe_allow_html=True)


# -------------------------------------------------------
# PAGE 4 — ABOUT
# -------------------------------------------------------
elif page == "ℹ️ About Project":
    st.subheader("ℹ️ About This Project")
    st.write("---")

    st.markdown("""
    ### 🤖 AI Financial News Intelligence

    This platform uses a **multi-agent architecture** with advanced embeddings,
    clustering, semantic search, sentiment analysis, risk gauge, heatmaps
    and visual analytics.

    #### 🚀 Features:
    - AI-powered semantic search  
    - Entity extraction  
    - Sentiment Analyzer  
    - Market Risk Meter  
    - Sector Heatmaps  
    - Deduplicated story clusters  
    - Logo detection  
    - Animated financial dashboard  

    #### 🧠 Powered By:
    - LangChain / LangGraph  
    - FastAPI  
    - Streamlit  
    - ChromaDB  
    - VADER  
    - OpenAI Embeddings  
    """)

    st.info("Created for AI Hiring Hackathon — By **Hrishikesh Garje**")


# END OF FILE
