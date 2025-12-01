# src/agents/dedup_agent.py
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher

# Use the lightweight TF-IDF fallback embedder we added
try:
    # If you later install sentence-transformers on a heavy host, this will still work:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

from src.agents.utils.fallback_embedder import get_fallback_embedder


class DeduplicationAgent:
    """
    DeduplicationAgent groups semantically-similar articles into unique stories.
    Approach:
      - Build embeddings for each article (title + content)
      - Compute pairwise cosine similarity (if using vector embeddings)
      - Otherwise use SequenceMatcher ratio as fallback
      - Greedy clustering by threshold (connected components)
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", similarity_threshold: float = 0.82):
        """
        Args:
            model_name: sentence-transformers model name (only used if available)
            similarity_threshold: cosine similarity threshold [0..1] to consider two articles duplicates
        """
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold

        # If sentence-transformers is installed and you want to use it on a heavy host, you can:
        if SentenceTransformer is not None:
            try:
                print(f"[DeduplicationAgent] Loading SentenceTransformer model {model_name} ...")
                self._sbert = SentenceTransformer(model_name)
                self._use_sentence_transformers = True
                print("[DeduplicationAgent] Loaded SentenceTransformer.")
            except Exception as e:
                print(f"[DeduplicationAgent] Failed to load SBERT ({e}); falling back to TF-IDF embedder.")
                self._sbert = None
                self._use_sentence_transformers = False
        else:
            self._sbert = None
            self._use_sentence_transformers = False

        # Shared lightweight TF-IDF embedder (safe for Render free tier)
        self._fallback = get_fallback_embedder()
        print(f"[DeduplicationAgent] Initialized. using_fallback={not self._use_sentence_transformers}")

    def _text_for_embedding(self, article: Dict[str, Any]) -> str:
        title = article.get("title", "") or ""
        content = article.get("content", "") or ""
        return (title.strip() + " . " + content.strip()).strip()

    def compute_embeddings(self, articles: List[Dict[str, Any]]) -> Optional[np.ndarray]:
        """
        Return embeddings as numpy array shape (n, dim) or None if not possible.
        Will prefer SentenceTransformer if available, else use the fallback TF-IDF embedder.
        """
        texts = [self._text_for_embedding(a) for a in articles]
        if len(texts) == 0:
            return None

        if self._use_sentence_transformers and self._sbert is not None:
            # SBERT returns numpy array
            try:
                embs = self._sbert.encode(texts, convert_to_numpy=True, show_progress_bar=False)
                return np.asarray(embs, dtype=np.float32)
            except Exception as e:
                print(f"[DeduplicationAgent] SBERT encode failed ({e}); falling back to TF-IDF.")
                # fall through to fallback

        # Use fallback TF-IDF embedder
        embedder = self._fallback
        # Fallback encode returns numpy array
        embs = embedder.encode(texts, convert_to_numpy=True)
        return np.asarray(embs, dtype=np.float32)

    def compute_similarity_matrix(self, articles: List[Dict[str, Any]], embeddings: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Returns a pairwise similarity matrix (n, n). Uses cosine similarity when embeddings are present,
        otherwise uses difflib.SequenceMatcher ratio on concatenated text.
        """
        n = len(articles)
        if n == 0:
            return np.zeros((0, 0))

        if embeddings is not None:
            # cosine_similarity handles 1D/2D; ensure 2D
            try:
                sim = cosine_similarity(embeddings)
                return sim
            except Exception as e:
                print(f"[DeduplicationAgent] cosine_similarity failed ({e}); falling back to SequenceMatcher.")

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

    def cluster_by_threshold(self, embeddings: Optional[np.ndarray], articles: Optional[List[Dict[str, Any]]] = None) -> List[int]:
        """
        Greedy connected-components clustering using threshold on similarity.
        Returns:
            labels: list of cluster labels per item (ints starting at 0)
        """
        if embeddings is None:
            if articles is None:
                raise ValueError("articles must be provided when embeddings is None")
            sim = self.compute_similarity_matrix(articles, embeddings=None)
        else:
            sim = self.compute_similarity_matrix(articles or [], embeddings=embeddings)

        n = sim.shape[0]
        if n == 0:
            return []

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
                neighbors = np.where(sim[u] >= self.similarity_threshold)[0]
                for v in neighbors:
                    if not visited[int(v)]:
                        visited[int(v)] = True
                        labels[int(v)] = label
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
        groups: Dict[int, List[Dict[str, Any]]] = {}
        for lbl, art in zip(labels, articles):
            groups.setdefault(int(lbl), []).append(art)

        unique_stories = []
        for story_idx, (lbl, members) in enumerate(sorted(groups.items(), key=lambda x: x[0])):
            # Choose representative article: longest content (heuristic)
            rep = max(members, key=lambda x: len(str(x.get("content", "") or "")) + len(str(x.get("title", "") or "")))
            story = {
                "story_id": story_idx,
                "representative": rep,
                "article_ids": [int(m.get("id")) for m in members],
                "members": members
            }
            unique_stories.append(story)

        return unique_stories
