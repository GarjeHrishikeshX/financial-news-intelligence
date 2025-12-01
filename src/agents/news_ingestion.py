# src/agents/news_ingestion.py
import os
import csv
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
    then writes articles + embeddings to StorageAgent.
    """

    def __init__(self, db_path: str = "data/storage.db", csv_path: str = "data/news.csv"):
        self.csv_path = csv_path
        self.store = StorageAgent(db_path)
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

    def _text_for_embedding(self, row: Dict[str, Any]) -> str:
        title = row.get("title", "") or ""
        content = row.get("content", "") or ""
        return (title.strip() + " . " + content.strip()).strip()

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        if len(texts) == 0:
            return np.zeros((0, 0))

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
                # Ensure minimal fields
                rows.append({
                    "id": int(r.get("id", 0)),
                    "title": r.get("title", "") or "",
                    "content": r.get("content", "") or "",
                    "date": r.get("date", "") or "",
                    "source": r.get("source", "") or ""
                })

        if not rows:
            return 0

        texts = [self._text_for_embedding(r) for r in rows]
        embs = self._embed_texts(texts)

        # Store each article and its embedding
        for i, row in enumerate(rows):
            article_id = int(row["id"]) if row.get("id") else None
            # store article record in DB
            self.store.upsert_article(row)  # implement upsert_article in StorageAgent
            # store embedding - ensure your storage agent accepts numpy 1D arrays
            vec = embs[i] if embs.shape[0] > i else np.zeros((self._fallback.vectorizer.max_features,))
            self.store.store_embedding(article_id, vec, namespace="sent-emb")

        if commit:
            print(f"[NewsIngestionAgent] Ingested {len(rows)} articles.")
        return len(rows)


if __name__ == "__main__":
    agent = NewsIngestionAgent()
    agent.ingest_csv()
