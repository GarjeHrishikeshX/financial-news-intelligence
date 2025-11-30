# Shared fallback TF-IDF embedder
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from typing import List

class FallbackEmbedder:
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
            # Fit on the texts provided if not fitted
            self.fit(texts)
        X = self.vectorizer.transform(texts)
        arr = X.toarray()
        if convert_to_numpy:
            return np.asarray(arr, dtype=np.float32)
        return arr

# A simple singleton instance that can be shared across modules
_global_fallback = FallbackEmbedder()

def get_fallback_embedder():
    return _global_fallback
