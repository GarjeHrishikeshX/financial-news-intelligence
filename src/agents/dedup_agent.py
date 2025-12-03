# src/agents/dedup_agent.py
from src.agents.storage_agent import StorageAgent
import numpy as np
from sklearn.cluster import AgglomerativeClustering

class DeduplicationAgent:
    def __init__(self, db_path="data/storage.db", threshold=0.85):
        self.store = StorageAgent(db_path)
        self.threshold = threshold

    def fetch_all_vectors(self):
        # load all articles via sqlite, then read vectors from DB
        conn = self.store.conn
        cur = conn.cursor()
        cur.execute("SELECT id FROM articles")
        rows = cur.fetchall()
        ids = [r[0] for r in rows]
        vecs = []
        valid_ids = []
        for aid in ids:
            cur.execute("SELECT vector FROM vectors WHERE article_id=?", (aid,))
            r = cur.fetchone()
            if r:
                import pickle
                v = pickle.loads(r[0])
                vecs.append(v)
                valid_ids.append(aid)
        if not vecs:
            return [], []
        return np.vstack(vecs), valid_ids

    def cluster(self):
        X, ids = self.fetch_all_vectors()
        if len(ids) == 0:
            print("[Dedup] No vectors to cluster.")
            return []
        # Agglomerative clustering based on cosine similarity approximated by distance
        # convert inner product similarity to distance: d = 1 - sim
        from sklearn.metrics.pairwise import cosine_similarity
        sim = cosine_similarity(X)
        # convert to distance
        dist = 1.0 - sim
        # Agglomerative with precomputed distance
        clustering = AgglomerativeClustering(
            n_clusters=None, affinity='precomputed', linkage='average', distance_threshold=1-self.threshold
        )
        labels = clustering.fit_predict(dist)
        groups = {}
        for lab, aid in zip(labels, ids):
            groups.setdefault(int(lab), []).append(aid)
        # Return groups list
        return groups

if __name__ == "__main__":
    d = DeduplicationAgent()
    groups = d.cluster()
    print("Groups:", groups)
