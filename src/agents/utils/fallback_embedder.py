from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from typing import List

class FallbackEmbedder:
    """
    Lightweight TF-IDF embedder to replace heavy ML models.
    Works fully on Render free tier.
    """
    def __init__(self, max_features: int = 256):
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self._fitted = False

    def fit(self, texts: List[str]):
        if not isinstance(texts, list):
            texts = [texts]
        if len(texts) == 0:
            return None
        self.vectorizer.fit(texts)
        self._fitted = True

    def encode(self, texts, convert_to_numpy=True):
        if not isinstance(texts, list):
            texts = [texts]
        if not self._fitted:
            self.fit(texts)

        X = self.vectorizer.transform(texts)
        arr = X.toarray()

        if convert_to_numpy:
            return np.asarray(arr, dtype=np.float32)
        return arr


# GLOBAL SINGLETON (shared everywhere)
_global_fallback = FallbackEmbedder()

def get_fallback_embedder():
    """Returns global TF-IDF embedder instance."""
    return _global_fallback
