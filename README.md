🔎 AI Financial News Intelligence
Multi-Agent Semantic Search • Sentiment Engine • Risk Analysis • Story Clustering

This project was built as part of the AI Hiring Hackathon 2025, using a modern multi-agent architecture to analyze, cluster, search, and visualize financial news with high accuracy.
It combines FastAPI, Streamlit, Sentence Transformers, FAISS, and custom NLP agents to create a full intelligence system.

🚀 Core Features
🔍 1. Semantic Financial News Search

Uses SentenceTransformer (MiniLM) embeddings

FAISS similarity search

Extracts entities (companies, sectors, regulators)

Provides detailed relevance score + explanations

🧠 2. Multi-Agent Pipeline
Agent	Purpose
News Ingestion Agent	Cleans + loads CSV data into DB
Dedup Agent	Removes duplicate financial news
Entity Extraction Agent	Extracts companies, sectors, regulators
Story Clustering Agent	Groups related articles into clusters
Query Agent	Performs semantic search + filtering
📊 3. Advanced Sentiment & Risk Analytics

VADER Sentiment

Animated donut chart

Sentiment timeline

Sector-wise heatmap

Market Risk Gauge (0–1 risk score)

📰 4. Story Groups

Clusters articles into meaningful "stories" using embeddings and similarity.

🎨 5. Premium Streamlit UI

Dark/light mode

Animated charts

Glassmorphic cards

Auto-fetched company logos

Real-time sentiment visuals

🧩 Project Architecture
financial-news-intelligence/
│
├── data/
│   ├── news.csv
│   ├── storage.db
│
├── src/
│   ├── agents/
│   │   ├── news_ingestion.py
│   │   ├── dedup_agent.py
│   │   ├── entity_extraction_agent.py
│   │   ├── stock_impact_agent.py
│   │   ├── query_agent.py
│   │   ├── storage_agent.py
│   │
│   └── api/
│       ├── server.py (FastAPI backend)
│
├── app.py (Streamlit Frontend)
├── requirements.txt
└── README.md

🛠️ Installation
Clone the project
git clone https://github.com/GarjeHrishikeshX/financial-news-intelligence
cd financial-news-intelligence

Create virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows

Install dependencies
pip install -r requirements.txt

📰 Run the Ingestion Pipeline
python src/agents/news_ingestion.py
python src/agents/dedup_agent.py
python src/agents/entity_extraction_agent.py
python src/agents/stock_impact_agent.py

⚡ Run FastAPI Backend
uvicorn src.api.server:app --reload --port 8000


Docs available at:
👉 http://127.0.0.1:8000/docs

🖥️ Run Streamlit UI
streamlit run app.py

📡 API Endpoints
POST /query

Semantic search endpoint.

Example:

{
  "query": "HDFC Bank quarterly earnings"
}

🏆 Why This Project Was Built

This solution was created for the AI Hiring Hackathon 2025 to demonstrate:

Multi-agent system design

Embedding-based search

Real-time NLP analytics

Backend + frontend integration

Clean, professional project structure

👨‍💻 Developer

Hrishikesh Garje
AI/ML Engineer • Data Science • GenAI Developer
GitHub: https://github.com/GarjeHrishikeshX

LinkedIn: https://www.linkedin.com/in/hrishikesh-garje-157a85327/
