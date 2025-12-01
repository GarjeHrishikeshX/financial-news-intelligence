# src/agents/stock_impact_agent.py
from typing import List, Dict, Any
import numpy as np
from src.agents.entity_extraction_agent import EntityExtractionAgent
from src.agents.storage_agent import StorageAgent
from src.agents.utils.fallback_embedder import get_fallback_embedder

class StockImpactAgent:
    """
    Simple rule-based stock impact analysis. Keeps memory small.
    """

    def __init__(self, db_path: str = "data/storage.db"):
        self.store = StorageAgent(db_path)
        self.entity_agent = EntityExtractionAgent()
        # fallback embedder for any optional similarity needs
        self._embedder = get_fallback_embedder()

    def classify_impact(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return a lightweight impact dict for an article:
          { "impact_type": "earnings|merger|policy|other", "score": float }
        """
        text = (article.get("title", "") or "") + " . " + (article.get("content", "") or "")
        text_lower = text.lower()

        # simple keyword heuristics
        if "quarter" in text_lower or "q1" in text_lower or "q2" in text_lower or "earnings" in text_lower:
            return {"impact_type": "earnings", "score": 0.9}
        if "acquire" in text_lower or "merger" in text_lower or "acquisition" in text_lower:
            return {"impact_type": "merger", "score": 0.8}
        if "rbi" in text_lower or "reserve bank" in text_lower or "regulator" in text_lower:
            return {"impact_type": "policy", "score": 0.7}

        # fallback: small semantic cue using TF-IDF vector similarity to some templates
        templates = [
            "company earnings report",
            "merger and acquisition",
            "regulatory action"
        ]
        try:
            v1 = self._embedder.encode([text], convert_to_numpy=True)[0]
            v_templates = self._embedder.encode(templates, convert_to_numpy=True)
            sims = np.dot(v_templates, v1) / (np.linalg.norm(v_templates, axis=1) * (np.linalg.norm(v1) + 1e-12))
            top = int(np.argmax(sims))
            if sims[top] > 0.25:
                impact_map = {0: ("earnings", 0.55), 1: ("merger", 0.55), 2: ("policy", 0.55)}
                itype, score = impact_map[top]
                return {"impact_type": itype, "score": float(score + sims[top] * 0.4)}
        except Exception:
            pass

        return {"impact_type": "other", "score": 0.2}

    def process_and_store(self, article: Dict[str, Any]):
        impact = self.classify_impact(article)
        # store impact in DB (implement store.save_impact)
        self.store.save_impact(article.get("id"), impact)
        return impact
