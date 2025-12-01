# src/agents/utils/api/server.py

from fastapi import FastAPI
from pydantic import BaseModel
import sqlite3
import json
from typing import List, Dict

from src.agents.query_agent import QueryAgent
from src.agents.storage_agent import StorageAgent

# --------------------------------------------------
# Initialize FastAPI
# --------------------------------------------------

app = FastAPI(
    title="AI Financial News Intelligence API",
    description="Lightweight Multi-Agent News Intelligence System",
    version="1.0.0"
)

# --------------------------------------------------
# Database + Agents
# --------------------------------------------------

DB_PATH = "data/storage.db"

store = StorageAgent(DB_PATH)
query_agent = QueryAgent(DB_PATH)


# --------------------------------------------------
# Models
# --------------------------------------------------

class QueryRequest(BaseModel):
    query: str


# --------------------------------------------------
# Endpoints
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "AI Financial News Intelligence System is running!",
        "endpoints": ["/query", "/articles", "/article/{id}", "/search/semantic/{text}"]
    }


# --------------------------------------------------
# MAIN QUERY ENDPOINT
# --------------------------------------------------

@app.post("/query")
def query_news(request: QueryRequest):
    result = query_agent.query(request.query)
    return result


# --------------------------------------------------
# LIST ARTICLES (DB SAFE VERSION)
# --------------------------------------------------

@app.get("/articles")
def list_articles():
    """
    Returns list of articles stored in SQLite.
    storage.db has: id, title, content, date, source
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, title, content, date, source FROM articles")
    rows = cur.fetchall()
    conn.close()

    out = []
    for r in rows:
        out.append({
            "id": r[0],
            "title": r[1],
            "content": r[2],
            "date": r[3],
            "source": r[4]
        })
    return out


# --------------------------------------------------
# GET SINGLE ARTICLE
# --------------------------------------------------

@app.get("/article/{article_id}")
def get_single_article(article_id: int):
    """
    Returns an article + its entities + impacts
    """
    art = store.get_article(article_id)

    if not art:
        return {"error": "Article not found"}

    ents = store.get_entities_for_article(article_id) or {}
    imp = store.get_impacts_for_article(article_id) or {}

    return {
        "article": art,
        "entities": ents,
        "impact": imp
    }


# --------------------------------------------------
# SEMANTIC SEARCH (TF-IDF VECTOR)
# --------------------------------------------------

@app.get("/search/semantic/{text}")
def semantic_search(text: str):
    """
    Performs TF-IDF semantic search.
    """
    vec = query_agent.embedder.encode(text, convert_to_numpy=True)
    res = store.search_by_vector(vec, top_k=10, namespace="sent-emb")
    return res
