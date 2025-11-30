# src/agents/query_agent.py

import numpy as np
from typing import Dict, Any, List
from sklearn.feature_extraction.text import TfidfVectorizer
from src.agents.utils.fallback_embedder import get_fallback_embedder
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

from src.agents.storage_agent import StorageAgent
from src.agents.entity_extraction_agent import EntityExtractionAgent


class QueryAgent:
    """
    Handles natural language queries and retrieves relevant news using:
      - Entity recognition in queries
      - Sector & company relationship expansion
      - Semantic vector similarity search
      - Structured filtering (companies, sectors, regulators)
    """

    def __init__(self, db_path="data/storage.db"):
        print("[QueryAgent] Initializing...")

        self.store = StorageAgent(db_path)
        self.entity_agent = EntityExtractionAgent()

        # Embedding model for query understanding; provide a lightweight fallback when not available
        if SentenceTransformer is not None:
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        else:
            print("[QueryAgent] sentence-transformers not found; using TF-IDF fallback embedder.")
            self.embedder = get_fallback_embedder()

        # Sector reverse lookup (company → sector mapping)
        self.company_to_sector = self.entity_agent.sector_map

        # Regulators
        self.regulators = self.entity_agent.regulators

        print("[QueryAgent] Ready.")

    # ------------------------------------------------------------
    # 1️⃣ Understand query type (company / sector / regulator / theme)
    # ------------------------------------------------------------
    def interpret_query(self, query: str) -> Dict[str, Any]:
        query_lower = query.lower()

        companies = []
        sectors = []
        regulators = []

        # Find companies via keywords
        for comp in self.entity_agent.company_keywords:
            if comp.lower() in query_lower:
                companies.append(comp)

        # Find regulators
        for reg in self.regulators:
            if reg.lower() in query_lower:
                regulators.append(reg)

        # Infer sector via keywords
        for comp in companies:
            sec = self.company_to_sector.get(comp)
            if sec:
                sectors.append(sec)

        # Generic sector queries ("banking news", "IT sector", etc.)
        for sec in set(self.company_to_sector.values()):
            if sec.lower() in query_lower:
                sectors.append(sec)

        # Remove duplicates
        companies = list(set(companies))
        sectors = list(set(sectors))
        regulators = list(set(regulators))

        # Determine query "intent"
        if companies:
            qtype = "company"
        elif sectors:
            qtype = "sector"
        elif regulators:
            qtype = "regulator"
        else:
            qtype = "theme"

        return {
            "query_type": qtype,
            "companies": companies,
            "sectors": sectors,
            "regulators": regulators
        }

    # ------------------------------------------------------------
    # 2️⃣ Expand query context
    # ------------------------------------------------------------
    def expand_context(self, info: Dict[str, Any]) -> Dict[str, Any]:
        # If user queries HDFC Bank → include Banking sector news
        if info["query_type"] == "company":
            companies = info["companies"]
            sectors = list(set([self.company_to_sector.get(c) for c in companies if c]))
            info["sectors"] = sectors

        return info

    # ------------------------------------------------------------
    # 3️⃣ Semantic Search + Filtering
    # ------------------------------------------------------------
    def semantic_search(self, query: str, search_k: int = 10) -> List[Dict[str, Any]]:
        q_vec = self.embedder.encode(query, convert_to_numpy=True)
        results = self.store.search_by_vector(q_vec, top_k=search_k, namespace="sent-emb")
        return results

    def structured_filter(
        self,
        semantic_results: List[Dict[str, Any]],
        info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        out = []
        companies = info["companies"]
        sectors = info["sectors"]
        regulators = info["regulators"]

        for r in semantic_results:
            art = self.store.get_article(r["article_id"])
            if not art:
                continue

            entities = self.store.get_entities_for_article(r["article_id"]) or {}
            art_comps = entities.get("companies", [])
            art_secs = entities.get("sectors", [])
            art_regs = entities.get("regulators", [])

            # Filtering logic
            ok = False

            # Company query → direct + sector news
            if info["query_type"] == "company":
                if any(c in art_comps for c in companies):
                    ok = True
                if any(s in art_secs for s in sectors):
                    ok = True

            # Sector query
            elif info["query_type"] == "sector":
                if any(s in art_secs for s in sectors):
                    ok = True

            # Regulator query
            elif info["query_type"] == "regulator":
                if any(rg in art_regs for rg in regulators):
                    ok = True

            # Theme query → accept high semantic scores only
            else:
                ok = r["score"] > 0.50

            if ok:
                out.append({
                    "article": art,
                    "entities": entities,
                    "score": r["score"]
                })

        # Sort by semantic score
        out = sorted(out, key=lambda x: -x["score"])
        return out

    # ------------------------------------------------------------
    # 4️⃣ Generate final explanation for each result
    # ------------------------------------------------------------
    def explain_result(self, item: Dict[str, Any], info: Dict[str, Any]) -> str:
        explanation = []

        if item["entities"].get("companies"):
            explanation.append(
                f"Mentions companies: {', '.join(item['entities']['companies'])}"
            )

        if item["entities"].get("sectors"):
            explanation.append(
                f"Sector relevance: {', '.join(item['entities']['sectors'])}"
            )

        if item["entities"].get("regulators"):
            explanation.append(
                f"Regulator involved: {', '.join(item['entities']['regulators'])}"
            )

        if not explanation:
            explanation.append("Relevant due to semantic similarity to your query.")

        return " | ".join(explanation)

    # ------------------------------------------------------------
    # 5️⃣ Main entry point
    # ------------------------------------------------------------
    def query(self, user_query: str) -> Dict[str, Any]:
        print("\n[QueryAgent] Processing query:", user_query)

        # Step 1: Understand query
        info = self.interpret_query(user_query)
        info = self.expand_context(info)

        # Step 2: Semantic search
        sem_results = self.semantic_search(user_query)

        # Step 3: Structured filter
        matches = self.structured_filter(sem_results, info)

        # Step 4: Generate explanation
        output = []
        for item in matches:
            output.append({
                "title": item["article"]["title"],
                "content": item["article"]["content"],
                "score": round(item["score"], 3),
                "explanation": self.explain_result(item, info)
            })

        return {
            "query": user_query,
            "interpretation": info,
            "results": output
        }
