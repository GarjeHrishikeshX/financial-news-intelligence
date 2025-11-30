# src/api/server.py

from fastapi import FastAPI
import json
from pydantic import BaseModel
from typing import List, Optional

from src.agents.news_ingestion import NewsIngestionAgent
from src.agents.dedup_agent import DeduplicationAgent
from src.agents.entity_extraction_agent import EntityExtractionAgent
from src.agents.stock_impact_agent import StockImpactAgent
from src.agents.storage_agent import StorageAgent
from src.agents.query_agent import QueryAgent


# -------------------------
# FastAPI App Initialization
# -------------------------

app = FastAPI(
    title="AI Financial News Intelligence API",
    description="Multi-Agent Financial News System (Hackathon Project)",
    version="1.0.0"
)

# -------------------------
# Request Models
# -------------------------

class QueryRequest(BaseModel):
    query: str


# -------------------------
# Initialize Agents (Global)
# -------------------------

store = StorageAgent("data/storage.db")
query_agent = QueryAgent("data/storage.db")


# -------------------------
# Endpoints
# -------------------------

@app.get("/")
def home():
    return {
        "message": "AI Financial News Intelligence System is running!",
        "endpoints": ["/query", "/articles", "/stories"]
    }


@app.post("/query")
def query_news(request: QueryRequest):
    """
    Accepts natural language queries:
    - "HDFC Bank news"
    - "Banking sector update"
    - "RBI policy change"
    """
    result = query_agent.query(request.query)
    return result


@app.get("/articles")
def list_articles():
    """
    Returns all stored articles (raw).
    """
    cur = store.conn.cursor()
    cur.execute("SELECT raw FROM articles")
    rows = cur.fetchall()
    return [json.loads(r[0]) for r in rows]


@app.get("/stories")
def list_stories():
    """
    Returns all deduplicated stories.
    """
    return store.load_all_stories()


@app.get("/article/{article_id}")
def get_single_article(article_id: int):
    """
    Returns details for a specific article.
    """
    art = store.get_article(article_id)
    ents = store.get_entities_for_article(article_id)
    imp = store.get_impacts_for_article(article_id)

    return {
        "article": art,
        "entities": ents,
        "impact": imp
    }


@app.get("/search/semantic/{text}")
def semantic_search(text: str):
    """
    Raw semantic similarity search (vector-based).
    """
    q_vec = query_agent.embedder.encode(text, convert_to_numpy=True)
    res = store.search_by_vector(q_vec, top_k=5, namespace="sent-emb")
    return res
