# src/agents/stock_impact_agent.py
from src.agents.entity_extraction_agent import EntityExtractionAgent

class StockImpactAgent:
    def __init__(self):
        self.entity_agent = EntityExtractionAgent()

    def score_impact(self, article_text):
        ents = self.entity_agent.extract(article_text)
        score = 0.0
        details = []
        if ents["companies"]:
            score = max(score, 0.9)
            details.append("Direct company mention")
        if ents["sectors"]:
            score = max(score, 0.6)
            details.append("Sector-level impact")
        if ents["regulators"]:
            score = max(score, 0.8)
            details.append("Regulatory mention")
        return {"score": score, "reasons": details, "entities": ents}
