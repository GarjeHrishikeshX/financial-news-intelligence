# src/agents/dedup_agent.py
import numpy as np
from typing import List, Dict, Any
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from src.agents.utils.fallback_embedder import get_fallback_embedder
from difflib import SequenceMatcher
from typing import Optional

class DeduplicationAgent:
    """
    DeduplicationAgent groups semantically-similar articles into unique stories.
    Approach:
      - Build embeddings for each article (title + content)
      - Compute pairwise cosine similarity
      - Greedy clustering by threshold (union-find style / connected components)
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", similarity_threshold: float = 0.82):
        """
        Args:
            model_name: sentence-transformers model name
            similarity_threshold: cosine similarity threshold [0..1] to consider two articles duplicates
        """
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        print(f"[DeduplicationAgent] Loading model {model_name} ...")
        if SentenceTransformer is None:
            print("[DeduplicationAgent] sentence-transformers not found; falling back to difflib-based similarity (approximate).")
            # Use shared fallback embedder
            self.embedder = get_fallback_embedder()
            self.using_fallback = True
        else:
            self.embedder = SentenceTransformer(model_name)
            self.using_fallback = False
        print(f"[DeduplicationAgent] Model loaded.")

    def _text_for_embedding(self, article: Dict[str, Any]) -> str:
        # Combine title + content for embedding; you can expand with source/date etc.
        title = article.get("title", "") or ""
        content = article.get("content", "") or ""
        return title.strip() + " . " + content.strip()

    def compute_embeddings(self, articles: List[Dict[str, Any]]) -> Optional[np.ndarray]:
        if self.using_fallback:
            return None
        texts = [self._text_for_embedding(a) for a in articles]
        embeddings = self.embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embeddings

    def compute_similarity_matrix(self, articles: List[Dict[str, Any]], embeddings: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Returns a pairwise similarity matrix (n, n). Uses cosine similarity when embeddings are present,
        otherwise uses difflib.SequenceMatcher ratio on concatenated text.
        """
        n = len(articles)
        if n == 0:
            return np.zeros((0, 0))
        if embeddings is not None:
            return cosine_similarity(embeddings)

        # Fallback: compute pairwise text similarity
        texts = [self._text_for_embedding(a) for a in articles]
        sim = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    s = 1.0
                else:
                    s = SequenceMatcher(None, texts[i], texts[j]).ratio()
                sim[i, j] = s
                sim[j, i] = s
        return sim

    # ------------------------------------------------------------
    # Fallback encoder API for compatibility with SentenceTransformer
    # ------------------------------------------------------------
    def encode(self, texts, convert_to_numpy=True, show_progress_bar=False):
        """
        Lightweight encoder used when sentence-transformers isn't installed. Uses TF-IDF vectors.
        """
        if not isinstance(texts, list):
            texts = [texts]
        X = self.vectorizer.fit_transform(texts)
        arr = X.toarray()
        if convert_to_numpy:
            return np.asarray(arr, dtype=np.float32)
        return arr

    def cluster_by_threshold(self, embeddings: Optional[np.ndarray], articles: Optional[List[Dict[str, Any]]] = None) -> List[int]:
        """
        Greedy connected-components clustering using threshold on cosine similarity.
        Returns:
            labels: list of cluster labels per embedding (ints starting at 0)
        """
        # Compute or reuse similarity matrix
        if embeddings is not None:
            sim = cosine_similarity(embeddings)
            n = sim.shape[0]
        else:
            if articles is None:
                raise ValueError("articles must be provided when embeddings are None")
            sim = self.compute_similarity_matrix(articles)
            n = sim.shape[0]
        if n == 0:
            return []

        # Compute cosine similarity matrix

        # Build adjacency list where sim >= threshold (ignore self)
        visited = [False] * n
        labels = [-1] * n
        label = 0

        for i in range(n):
            if visited[i]:
                continue
            # BFS/DFS to collect connected component
            stack = [i]
            visited[i] = True
            labels[i] = label
            while stack:
                u = stack.pop()
                # neighbors where sim >= threshold
                neighbors = np.where(sim[u] >= self.similarity_threshold)[0]
                for v in neighbors:
                    if not visited[v]:
                        visited[v] = True
                        labels[v] = label
                        stack.append(int(v))
            label += 1

        return labels

    def deduplicate(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Main deduplication function.
        Input:
            articles - list of article dicts with at least 'id','title','content','date','source'
        Output:
            unique_stories - list of dicts:
                {
                  'story_id': int,
                  'representative': article_dict,
                  'article_ids': [list of article ids consolidated],
                  'members': [list of article dicts]
                }
        """
        if len(articles) == 0:
            return []

        embeddings = self.compute_embeddings(articles)
        labels = self.cluster_by_threshold(embeddings, articles=articles)

        # Group articles by label
        groups = {}
        for lbl, art in zip(labels, articles):
            groups.setdefault(lbl, []).append(art)

        unique_stories = []
        for story_idx, (lbl, members) in enumerate(groups.items()):
            # Choose representative article: longest content (heuristic)
            rep = max(members, key=lambda x: len(x.get("content", "") or "") + len(x.get("title", "") or ""))
            story = {
                "story_id": story_idx,
                "representative": rep,
                "article_ids": [int(m["id"]) for m in members],
                "members": members
            }
            unique_stories.append(story)

        # Sort by story_id
        unique_stories = sorted(unique_stories, key=lambda s: s["story_id"])
        return unique_stories
