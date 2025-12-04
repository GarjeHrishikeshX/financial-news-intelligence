🔎 AI Financial News Intelligence

A Multi-Agent Financial News Understanding System
Built by: Hrishikesh Garje — DataSmith.ai Trainee

🚀 Overview

This project is an AI-powered financial news intelligence platform built using a multi-agent architecture, semantic embeddings, clustering, sentiment analysis, and interactive visual analytics.

The system analyzes financial news, extracts entities, groups related stories, evaluates sentiment, generates risk insights, and provides a complete visual dashboard for decision-making.

Designed for AI Hiring Hackathon 2025.

🧠 Key Features
1️⃣ Multi-Agent Architecture

News Ingestion Agent → Loads & preprocesses news

Deduplication Agent → Identifies similar news & removes duplicates

Entity Extraction Agent → Extracts companies, sectors, regulators

Stock Impact Agent → Computes sentiment & impact

Query Agent → Performs semantic search using embeddings

Storage Agent → Manages persistent storage in SQLite

2️⃣ Semantic Search Engine

Uses SentenceTransformer (all-MiniLM-L6-v2)

Retrieves relevant financial articles

Provides similarity scores & explanations

3️⃣ Real-Time Financial Dashboard (Streamlit)

Includes:
✔ Animated Donut Sentiment Chart
✔ Sentiment Timeline Plot
✔ Market Risk Gauge
✔ Sector Heatmap
✔ Entity Chips
✔ Company Logo Fetching
✔ Featured Article Cards
✔ Full article browser
✔ Story cluster visualizer

4️⃣ Sentiment Analysis (VADER)

Computes positive, neutral, negative scores

Aggregates overall story sentiment

Displays intuitive visuals

5️⃣ Market Risk Meter

A custom risk score:
risk = neg*1.0 + neu*0.4 – pos*0.3
Plotted as an animated gauge (0 = safe, 1 = risky)

6️⃣ Story Grouping / Clustering

Articles are grouped using embedding similarity

Helps the model recognize news narratives

📦 Tech Stack

Python 3.13

Sentence Transformers

FAISS

scikit-learn

spaCy

FastAPI

Streamlit

VADER Sentiment

SQLite Database

📁 Folder Structure

financial-news-intelligence/
│
├── app.py
├── requirements.txt
├── data/
│   ├── news.csv
│   └── storage.db
│
├── src/
│   ├── agents/
│   │   ├── news_ingestion.py
│   │   ├── dedup_agent.py
│   │   ├── entity_extraction_agent.py
│   │   ├── stock_impact_agent.py
│   │   ├── query_agent.py
│   │   └── storage_agent.py
│   │
│   └── api/
│       └── server.py


⚙️ Installation
1. Create virtual environment
python -m venv .venv
.venv\Scripts\activate

2. Install dependencies
pip install -r requirements.txt

3. Run ingestion
python src/agents/news_ingestion.py

4. Run FastAPI Backend
uvicorn src.api.server:app --reload

5. Run Streamlit Dashboard
streamlit run app.py

🖥️ Usage Instructions

Enter a financial query → e.g., HDFC results, RBI policy, IT sector crash.

System performs:

Query interpretation

Entity extraction

Semantic search

Sentiment scoring

Risk calculation

Visual dashboard presents insights.

Browse deduplicated story clusters.

🏆 Hackathon Requirements — Completed
Requirement	Status
Multi-agent financial news pipeline	✅ Done
Semantic search using embeddings	✅ Done
Sentiment scoring & polarity visualization	✅ Done
Stock/market risk estimation	✅ Done
Story clustering / deduplication	✅ Done
Full UI dashboard in Streamlit	✅ Done
FastAPI backend	✅ Done
Clean code & modular structure	✅ Done
Ready-to-deploy	✅ Done
👨‍💻 About the Developer

Hrishikesh Garje
Trainee — DataSmith.ai
Specialized in AI/ML & Intelligent Information Systems

This project was developed during hands-on training at DataSmith.ai as a demonstration of building real-world AI-powered financial intelligence products.
