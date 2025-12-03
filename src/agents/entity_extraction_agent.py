# src/agents/entity_extraction_agent.py
import spacy
from typing import Dict, List

# small custom mapping; expand as needed
COMPANY_MAP = {
    "hdfc bank": ("HDFCBANK", "Banking"),
    "icici bank": ("ICICIBANK", "Banking"),
    "reliance": ("RELIANCE", "Conglomerate"),
    "tcs": ("TCS", "IT"),
}

REGULATORS = ["rbi", "sebi"]

class EntityExtractionAgent:
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except Exception:
            self.nlp = spacy.blank("en")
        self.company_map = COMPANY_MAP
        self.regulators = REGULATORS

    def extract(self, text: str) -> Dict[str, List[str]]:
        doc = self.nlp(text)
        companies = []
        regulators = []
        for ent in doc.ents:
            if ent.label_ in ("ORG","PERSON"):
                name = ent.text.lower()
                if name in self.company_map:
                    companies.append(self.company_map[name][0])
        # fallback keyword scanning
        txt = text.lower()
        for k, v in self.company_map.items():
            if k in txt and v[0] not in companies:
                companies.append(v[0])
        for reg in self.regulators:
            if reg in txt and reg.upper() not in regulators:
                regulators.append(reg.upper())
        sectors = list({self.company_map.get(k)[1] for k in self.company_map if k in txt})
        return {"companies": companies, "regulators": regulators, "sectors": sectors}
