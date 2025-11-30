# src/agents/storage_agent.py
import sqlite3
import json
import numpy as np
from typing import List, Dict, Any, Optional
import os

def _ensure_dir_for_file(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

class StorageAgent:
    """
    Storage & Indexing Agent using SQLite for structured data and
    persisted embeddings as binary blobs. For similarity search we
    load embeddings to memory and compute cosine similarities.
    """

    def __init__(self, db_path: str = "data/storage.db"):
        _ensure_dir_for_file(db_path)
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_schema()

    def _init_schema(self):
        cur = self.conn.cursor()

        # articles table: raw articles
        cur.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY,
            title TEXT,
            content TEXT,
            date TEXT,
            source TEXT,
            raw JSON
        )
        """)

        # stories: deduplicated story groups
        cur.execute("""
        CREATE TABLE IF NOT EXISTS stories (
            story_id INTEGER PRIMARY KEY,
            representative_article_id INTEGER,
            article_ids TEXT, -- JSON list
            representative_json JSON
        )
        """)

        # entities: extracted entities per article
        cur.execute("""
        CREATE TABLE IF NOT EXISTS entities (
            id INTEGER PRIMARY KEY,
            article_id INTEGER,
            companies TEXT,
            sectors TEXT,
            regulators TEXT,
            people TEXT,
            events TEXT,
            raw JSON
        )
        """)

        # impacts: stock impact results per article
        cur.execute("""
        CREATE TABLE IF NOT EXISTS impacts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            article_id INTEGER,
            impacted_stocks JSON
        )
        """)

        # embeddings: store vectors as blob + shape meta; namespace allows multiple embedding types
        cur.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            article_id INTEGER,
            namespace TEXT,
            vector BLOB,
            dim INTEGER
        )
        """)

        self.conn.commit()

    # -------------------------
    # Article storage helpers
    # -------------------------
    def save_articles(self, articles: List[Dict[str, Any]]):
        """
        articles: list of dicts with keys: id, title, content, date, source
        """
        cur = self.conn.cursor()
        for a in articles:
            cur.execute(
                "INSERT OR REPLACE INTO articles (id, title, content, date, source, raw) VALUES (?, ?, ?, ?, ?, ?)",
                (int(a["id"]), a.get("title"), a.get("content"), str(a.get("date")), a.get("source"), json.dumps(a))
            )
        self.conn.commit()

    def get_article(self, article_id: int) -> Optional[Dict[str, Any]]:
        cur = self.conn.cursor()
        cur.execute("SELECT raw FROM articles WHERE id = ?", (article_id,))
        row = cur.fetchone()
        if not row:
            return None
        return json.loads(row[0])

    # -------------------------
    # Stories (dedup) helpers
    # -------------------------
    def save_stories(self, stories: List[Dict[str, Any]]):
        """
        stories: list of dicts with keys: story_id, representative, article_ids, members
        """
        cur = self.conn.cursor()
        for s in stories:
            rep = s["representative"]
            cur.execute(
                "INSERT OR REPLACE INTO stories (story_id, representative_article_id, article_ids, representative_json) VALUES (?, ?, ?, ?)",
                (int(s["story_id"]), int(rep["id"]), json.dumps(s["article_ids"]), json.dumps(rep))
            )
        self.conn.commit()

    def load_all_stories(self) -> List[Dict[str, Any]]:
        cur = self.conn.cursor()
        cur.execute("SELECT story_id, representative_json, article_ids FROM stories")
        rows = cur.fetchall()
        out = []
        for story_id, rep_json, article_ids in rows:
            out.append({
                "story_id": story_id,
                "representative": json.loads(rep_json),
                "article_ids": json.loads(article_ids)
            })
        return out

    # -------------------------
    # Entities helpers
    # -------------------------
    def save_entity(self, entity_obj: Dict[str, Any]):
        cur = self.conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO entities (id, article_id, companies, sectors, regulators, people, events, raw) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                int(entity_obj.get("id")),
                int(entity_obj.get("id")),
                json.dumps(entity_obj.get("companies", [])),
                json.dumps(entity_obj.get("sectors", [])),
                json.dumps(entity_obj.get("regulators", [])),
                json.dumps(entity_obj.get("people", [])),
                json.dumps(entity_obj.get("events", [])),
                json.dumps(entity_obj)
            )
        )
        self.conn.commit()

    def get_entities_for_article(self, article_id: int) -> Optional[Dict[str, Any]]:
        cur = self.conn.cursor()
        cur.execute("SELECT raw FROM entities WHERE article_id = ?", (article_id,))
        row = cur.fetchone()
        if not row:
            return None
        return json.loads(row[0])

    # -------------------------
    # Impacts helpers
    # -------------------------
    def save_impact(self, impact_obj: Dict[str, Any]):
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO impacts (article_id, impacted_stocks) VALUES (?, ?)",
            (int(impact_obj["article_id"]), json.dumps(impact_obj["impacted_stocks"]))
        )
        self.conn.commit()

    def get_impacts_for_article(self, article_id: int) -> List[Dict[str, Any]]:
        cur = self.conn.cursor()
        cur.execute("SELECT impacted_stocks FROM impacts WHERE article_id = ?", (article_id,))
        rows = cur.fetchall()
        return [json.loads(r[0]) for r in rows]

    # -------------------------
    # Embeddings helpers
    # -------------------------
    def save_embedding(self, article_id: int, vector: np.ndarray, namespace: str = "default"):
        """
        Persist a numpy vector as blob + dim.
        """
        vec = np.asarray(vector, dtype=np.float32)
        b = vec.tobytes()
        dim = vec.shape[0]
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO embeddings (article_id, namespace, vector, dim) VALUES (?, ?, ?, ?)",
            (int(article_id), namespace, sqlite3.Binary(b), int(dim))
        )
        self.conn.commit()

    def load_embeddings(self, namespace: str = "default") -> List[Dict[str, Any]]:
        """
        Return list of dicts: {article_id, vector(np.ndarray)}
        """
        cur = self.conn.cursor()
        cur.execute("SELECT article_id, vector, dim FROM embeddings WHERE namespace = ?", (namespace,))
        rows = cur.fetchall()
        out = []
        for aid, b, dim in rows:
            arr = np.frombuffer(b, dtype=np.float32)
            if dim and arr.size != dim:
                # safety reshape if flattened rows concatenated
                arr = arr[:dim]
            out.append({"article_id": int(aid), "vector": arr})
        return out

    # -------------------------
    # Simple in-memory vector search (cosine)
    # -------------------------
    def search_by_vector(self, query_vec: np.ndarray, top_k: int = 5, namespace: str = "default"):
        """
        Loads all embeddings for the namespace into memory, computes cosine similarity,
        and returns top_k article_ids sorted by similarity.
        """
        data = self.load_embeddings(namespace)
        if len(data) == 0:
            return []

        q = np.asarray(query_vec, dtype=np.float32)
        M = np.stack([d["vector"] for d in data], axis=0)  # shape (n, dim)
        # normalize
        q_norm = q / (np.linalg.norm(q) + 1e-12)
        M_norm = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
        sims = M_norm.dot(q_norm)
        idx = np.argsort(-sims)[:top_k]
        results = [{"article_id": int(data[i]["article_id"]), "score": float(sims[i])} for i in idx]
        return results

    def close(self):
        self.conn.close()
