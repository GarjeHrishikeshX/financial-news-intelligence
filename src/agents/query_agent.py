# query_agent.py — Final Stable Version (Torch-safe, CPU-only fallback)

import json
import sqlite3

# Try SentenceTransformer
USE_EMBEDDER = False
try:
    from sentence_transformers import SentenceTransformer
    USE_EMBEDDER = True
except:
    USE_EMBEDDER = False

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class QueryAgent:
    def __init__(self, db):
        self.db = db
        self.conn = sqlite3.connect(db, check_same_thread=False)

        self.articles = self._load_articles()

        if USE_EMBEDDER:
            try:
                self.embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
                self.use_embed = True
            except:
                self.use_embed = False
        else:
            self.use_embed = False

        if not self.use_embed:
            self.vectorizer = TfidfVectorizer(max_features=5000)
            contents = [a["content"] for a in self.articles]
            self.tfidf = self.vectorizer.fit_transform(contents)

    def _load_articles(self):
        cur = self.conn.cursor()
        rows = cur.execute("SELECT raw FROM articles").fetchall()
        out = []
        for r in rows:
            try:
                j = json.loads(r[0])
            except:
                j = {"title": "", "content": r[0]}
            out.append(j)
        return out

    def query(self, text, topk=10):
        if not self.articles:
            return {"results": [], "interpretation": {}}

        if self.use_embed:
            q_emb = self.embedder.encode([text])
            art_emb = self.embedder.encode([a["content"] for a in self.articles])
            sims = cosine_similarity(q_emb, art_emb)[0]
        else:
            qv = self.vectorizer.transform([text])
            sims = cosine_similarity(qv, self.tfidf)[0]

        idx = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:topk]

        results = []
        for i in idx:
            a = self.articles[i]
            results.append({
                "title": a["title"],
                "content": a["content"],
                "score": float(sims[i])
            })

        return {"results": results, "interpretation": {}}
