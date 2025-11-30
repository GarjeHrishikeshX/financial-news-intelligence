📘 README.md — AI Financial News Intelligence System
📰 AI Financial News Intelligence System
Multi-Agent LLM + NLP Pipeline for Financial News Understanding

Built for Real-Time Impact Analysis • Semantic Search • News Deduplication • Entity Extraction • Query Intelligence

🚀 Project Overview

This project is an end-to-end financial news intelligence platform powered by a multi-agent architecture.

It can:

✔ Ingest financial news from multiple sources
✔ Deduplicate similar articles into unified “stories”
✔ Extract entities (companies, sectors, regulators)
✔ Identify impactful events (earnings, mergers, policy changes, etc.)
✔ Map news to relevant stocks or sectors
✔ Perform semantic search using vector embeddings
✔ Interpret natural-language user queries
✔ Serve results via FastAPI
✔ Provide a real-time UI via Streamlit

This system helps analysts, traders, fintech apps, and researchers access clean, deduped, structured, and queryable financial intelligence.

🧠 Tech Stack
Core Technologies

Python 3.10+

spaCy (NER)

Sentence Transformers (Semantic embeddings)

FAISS (optional) / In-DB vector storage

SQLite (lightweight storage)

FastAPI (Backend API)

Streamlit (Frontend UI)

AI / NLP

MiniLM sentence embeddings

Rule-based sector & regulator mapping

Entity-Impact graph model

Multi-agent design pattern

Project Architecture
financial-news-intelligence/
 ├── data/
 │    ├── storage.db             # SQLite + vector store
 │    ├── news.csv               # Raw dataset
 │
 ├── src/
 │    ├── agents/
 │    │    ├── news_ingestion.py
 │    │    ├── dedup_agent.py
 │    │    ├── entity_extraction_agent.py
 │    │    ├── stock_impact_agent.py
 │    │    ├── storage_agent.py
 │    │    ├── query_agent.py
 │    │
 │    ├── api/
 │         ├── server.py         # FastAPI backend
 │
 ├── tests/
 │    ├── test_query.py
 │
 ├── app.py                      # Streamlit UI
 ├── requirements.txt
 ├── README.md

🔑 Key Features
🔹 1. News Ingestion Agent

Reads CSV / API feeds and loads news into the database.

🔹 2. Deduplication Agent

Groups similar articles using embedding similarity.
Creates “story clusters”.

🔹 3. Entity Extraction Agent

Extract:

Companies

Regulators

Sectors
Using spaCy + rule-based dictionaries.

🔹 4. Stock Impact Analysis Agent

Classifies news into:

Earnings

Market movement

Policy change

Supply chain

Risk / fraud / regulatory crackdown

Maps article → stock.

🔹 5. Storage & Indexing Agent

Single-file SQLite database storing:

Articles

Entities

Impacts

Story groups

Vector embeddings

🔹 6. Query Processing Agent

Understands queries like:

“HDFC Bank updates”

“Banking sector analysis”

“RBI policy impact on markets”

Performs:

Semantic search

Entity-aware filtering

Sector expansion

Final ranking & explanation

🔹 7. FastAPI Server

Endpoints:

/query

/articles

/stories

/search/semantic/{text}

🔹 8. Streamlit UI

A beautiful, judge-friendly interface with:

Search bar

Results with explanations

Article browser

Story browse

📥 Installation
1. Clone the Project
git clone <your-repo-url>
cd financial-news-intelligence
2. Install Dependencies
pip install -r requirements.txt

Note on Core NLP/Embedding Packages (sentence-transformers + torch)
The project uses `sentence-transformers` for semantic embeddings which depends on a PyTorch backend.
On Windows, install PyTorch first and then sentence-transformers. The following examples install a CPU-only version
of PyTorch which works on most machines.

Powershell (CPU-only):
```
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install sentence-transformers
```

If you have a GPU and want to install CUDA-enabled PyTorch, use the official PyTorch installation instructions at https://pytorch.org/get-started/locally/ and then run `pip install sentence-transformers`.
3. Add Dataset

Place your CSV in:
data/news.csv
title,content,source,date
"HDFC Bank Q4 Results","HDFC Bank reports record profit...",Economic Times,2024-01-10

⚙️ Running the Pipeline
1. Run Ingestion + Dedup + Entity Extraction

(If you have script wrappers; otherwise run module-wise)
python -m src.agents.ingestion_agent
python -m src.agents.dedup_agent
python -m src.agents.entity_extraction_agent
python -m src.agents.stock_impact_agent

🌐 Start FastAPI Backend
uvicorn src.agents.utils.api.server:app --reload

Now open:

📌 API Docs: http://localhost:8000/docs
📌 Home: http://localhost:8000/

🖥 Start Streamlit UI
streamlit run app.py

Opens at:

📌 http://localhost:8501/

🧪 Testing

Run included tests:
python tests/test_query.py

🎯 Example Query Output
Query: "HDFC Bank news"

Interpretation:
{
  "query_type": "company",
  "companies": ["HDFC Bank"],
  "sectors": ["Banking"],
  "regulators": []
}

Results:
- Title: HDFC Bank Announces 15% Dividend
  Score: 0.92
  Explanation: Mentions companies: HDFC Bank | Sector: Banking
