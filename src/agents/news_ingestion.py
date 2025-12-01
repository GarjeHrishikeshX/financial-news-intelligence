# src/agents/news_ingestion.py
import os
import csv
import sqlite3
from typing import List, Dict, Any
import numpy as np

from src.agents.storage_agent import StorageAgent

# Lazy import fallback embedder
from src.agents.utils.fallback_embedder import get_fallback_embedder
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


class NewsIngestionAgent:
    """
    Lightweight ingestion for CSV / feeds. Builds embeddings using a safe fallback,
    then writes articles + embeddings to StorageAgent or directly to SQLite if StorageAgent lacks methods.
    """

    def __init__(self, db_path: str = "data/storage.db", csv_path: str = "data/news.csv"):
        self.csv_path = csv_path
        self.store = StorageAgent(db_path)
        self.db_path = db_path

        # prefer SBERT if available on heavy host, else fallback TF-IDF
        if SentenceTransformer is not None:
            try:
                self._model = SentenceTransformer("all-MiniLM-L6-v2")
                self._use_sbert = True
            except Exception:
                self._model = None
                self._use_sbert = False
        else:
            self._model = None
            self._use_sbert = False

        self._fallback = get_fallback_embedder()
        # Ensure DB tables exist (if we need to write directly)
        self._ensure_tables()

    def _ensure_tables(self):
        """Create articles & embeddings tables if they do not exist (direct SQLite fallback)."""
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            cur.execute("""
            CREATE TABLE IF NOT EXISTS articles (
                id INTEGER PRIMARY KEY,
                title TEXT,
                content TEXT,
                date TEXT,
                source TEXT
            )""")
            cur.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                article_id INTEGER PRIMARY KEY,
                namespace TEXT,
                dim INTEGER,
                dtype TEXT,
                data BLOB,
                FOREIGN KEY(article_id) REFERENCES articles(id)
            )""")
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[NewsIngestionAgent] Warning: could not ensure DB tables ({e})")

    def _text_for_embedding(self, row: Dict[str, Any]) -> str:
        title = row.get("title", "") or ""
        content = row.get("content", "") or ""
        return (title.strip() + " . " + content.strip()).strip()

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        if len(texts) == 0:
            return np.zeros((0, 0), dtype=np.float32)

        if self._use_sbert and self._model is not None:
            try:
                embs = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
                return np.asarray(embs, dtype=np.float32)
            except Exception:
                # fallback to TF-IDF
                pass

        # TF-IDF fallback (shared instance)
        embs = self._fallback.encode(texts, convert_to_numpy=True)
        return np.asarray(embs, dtype=np.float32)

    # ----------------------
    # Storage helper wrappers
    # ----------------------
    def _store_article_via_agent(self, article: Dict[str, Any]) -> int:
        """
        Try to use StorageAgent API to store/upsert article.
        Returns article_id (int) on success, or raises AttributeError if not supported.
        """
        # Try common method names (upsert_article, add_article, insert_article)
        for name in ("upsert_article", "add_article", "insert_article", "save_article"):
            fn = getattr(self.store, name, None)
            if callable(fn):
                try:
                    res = fn(article)
                    # If the storage method returns new id, return it; else assume article['id'] exists
                    if isinstance(res, int):
                        return int(res)
                    if article.get("id"):
                        return int(article["id"])
                    return int(article.get("id", 0))
                except Exception as e:
                    print(f"[NewsIngestionAgent] StorageAgent.{name} failed: {e}")
                    continue
        # If none found, raise to let caller do DB fallback
        raise AttributeError("StorageAgent has no article write helper methods")

    def _store_embedding_via_agent(self, article_id: int, vec: np.ndarray, namespace: str = "sent-emb"):
        """
        Try to call StorageAgent.store_embedding or similar.
        """
        # Try common method names
        for name in ("store_embedding", "save_embedding", "add_embedding", "upsert_embedding"):
            fn = getattr(self.store, name, None)
            if callable(fn):
                try:
                    fn(article_id, vec, namespace=namespace)
                    return True
                except Exception as e:
                    print(f"[NewsIngestionAgent] StorageAgent.{name} failed: {e}")
                    continue
        # If none available, signal False so caller saves directly
        return False

    # ---------------------
    # Direct SQLite fallback
    # ---------------------
    def _direct_upsert_article(self, article: Dict[str, Any]) -> int:
        """
        Insert or replace article directly into SQLite DB. Returns article_id.
        """
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        aid = article.get("id")
        if aid is None or aid == "" or aid == 0:
            # Insert new row and get id
            cur.execute("""
                INSERT INTO articles (title, content, date, source) VALUES (?, ?, ?, ?)
            """, (article.get("title", ""), article.get("content", ""), article.get("date", ""), article.get("source", "")))
            aid = cur.lastrowid
        else:
            # Upsert by primary key
            cur.execute("""
                INSERT INTO articles (id, title, content, date, source)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                  title=excluded.title,
                  content=excluded.content,
                  date=excluded.date,
                  source=excluded.source
            """, (int(aid), article.get("title", ""), article.get("content", ""), article.get("date", ""), article.get("source", "")))
        conn.commit()
        conn.close()
        return int(aid)

    def _direct_store_embedding(self, article_id: int, vec: np.ndarray, namespace: str = "sent-emb"):
        """
        Store embedding as blob with dtype + shape metadata.
        """
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        arr = np.asarray(vec)
        # Save dtype and shape so we can reconstruct later
        data = arr.tobytes()
        dtype = str(arr.dtype)
        dim = int(arr.size)
        cur.execute("""
            INSERT INTO embeddings (article_id, namespace, dim, dtype, data)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(article_id) DO UPDATE SET
              namespace=excluded.namespace,
              dim=excluded.dim,
              dtype=excluded.dtype,
              data=excluded.data
        """, (int(article_id), namespace, dim, dtype, sqlite3.Binary(data)))
        conn.commit()
        conn.close()

    # ---------------------
    # Main ingestion method
    # ---------------------
    def ingest_csv(self, commit: bool = True) -> int:
        """
        Ingest rows from CSV and store article and embedding.
        Returns number of rows ingested.
        """
        if not os.path.exists(self.csv_path):
            print(f"[NewsIngestionAgent] CSV not found: {self.csv_path}")
            return 0

        rows = []
        with open(self.csv_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for r in reader:
                # Ensure minimal fields and coerce id
                rid = r.get("id") or r.get("article_id") or None
                try:
                    rid = int(rid) if rid not in (None, "", "NULL") else None
                except Exception:
                    rid = None
                rows.append({
                    "id": rid,
                    "title": r.get("title", "") or "",
                    "content": r.get("content", "") or "",
                    "date": r.get("date", "") or "",
                    "source": r.get("source", "") or ""
                })

        if not rows:
            return 0

        texts = [self._text_for_embedding(r) for r in rows]
        embs = self._embed_texts(texts)

        ingested = 0
        for i, row in enumerate(rows):
            # try to store article via StorageAgent API
            try:
                article_id = self._store_article_via_agent(row)
            except AttributeError:
                # fallback: write directly to sqlite
                article_id = self._direct_upsert_article(row)

            # prepare vector (ensure 1D)
            vec = embs[i] if embs.shape[0] > i else np.zeros((self._fallback.vectorizer.max_features,), dtype=np.float32)

            # try to store embedding via StorageAgent
            ok = self._store_embedding_via_agent(article_id, vec, namespace="sent-emb")
            if not ok:
                # direct storage fallback
                self._direct_store_embedding(article_id, vec, namespace="sent-emb")

            ingested += 1

        if commit:
            print(f"[NewsIngestionAgent] Ingested {len(rows)} articles (stored {ingested}).")
        return ingested


if __name__ == "__main__":
    agent = NewsIngestionAgent()
    agent.ingest_csv()
