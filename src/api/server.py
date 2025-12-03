# src/api/server.py
from fastapi import FastAPI
from pydantic import BaseModel
from src.agents.query_agent import QueryAgent
from src.agents.storage_agent import StorageAgent
import threading, traceback

app = FastAPI(title="AI Financial News Intelligence", version="1.0")

DB_PATH = "data/storage.db"
store = StorageAgent(DB_PATH)
query_agent = QueryAgent(DB_PATH)

class QueryRequest(BaseModel):
    query: str

@app.get("/")
def home():
    return {"message": "Backend running", "endpoints": ["/query","/articles","/article/{id}","/search/semantic/{text}"]}

@app.post("/query")
def query_news(req: QueryRequest):
    try:
        return query_agent.query(req.query)
    except Exception as e:
        tb = traceback.format_exc()
        return {"error": str(e), "traceback": tb}

@app.get("/articles")
def list_articles():
    conn = store.conn
    cur = conn.cursor()
    cur.execute("SELECT id,title,content,date,source FROM articles")
    rows = cur.fetchall()
    return [{"id":r[0],"title":r[1],"content":r[2],"date":r[3],"source":r[4]} for r in rows]

@app.get("/search/semantic/{text}")
def semantic_search(text: str):
    try:
        vec = query_agent.embedder.encode(text, convert_to_numpy=True)
        import numpy as np
        if isinstance(vec, np.ndarray) and vec.ndim == 2:
            vec = vec[0]
        res = store.search_by_vector(vec, top_k=10)
        return res
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}
