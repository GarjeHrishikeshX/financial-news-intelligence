# src/agents/entity_extraction_agent.py

try:
    import spacy
except Exception:
    spacy = None
from typing import List, Dict, Any


class EntityExtractionAgent:
    def __init__(self):
        print("[EntityExtractionAgent] Loading spaCy model...")
        try:
            # Try using larger model if available
            if spacy is None:
                raise ImportError
            self.nlp = spacy.load("en_core_web_sm")
        except:
            # Fallback to blank model with built-in tokenizer
            if spacy is None:
                print("[WARNING] spaCy not installed. NER features will be disabled.")
                self.nlp = None
            else:
                print("[WARNING] en_core_web_sm not found. Using blank model.")
                self.nlp = spacy.blank("en")

        # -------------------------------
        # 📌 Domain-specific financial dictionary
        # -------------------------------

        self.company_keywords = [
            "HDFC Bank", "ICICI Bank", "Reliance Retail", "TCS", "Infosys", "Adani Ports",
            "Maruti Suzuki", "L&T", "Coal India", "Bajaj Finance", "Paytm", "Air India",
            "SpiceJet"
        ]

        self.regulators = [
            "RBI", "SEBI", "US Fed", "Federal Reserve"
        ]

        self.sector_map = {
            "HDFC Bank": "Banking",
            "ICICI Bank": "Banking",
            "Bajaj Finance": "Financial Services",
            "Reliance Retail": "Retail",
            "TCS": "IT Services",
            "Infosys": "IT Services",
            "Adani Ports": "Logistics",
            "L&T": "Infrastructure",
            "Coal India": "Mining",
            "Maruti Suzuki": "Automobile",
            "Paytm": "Fintech",
            "SpiceJet": "Aviation",
            "Air India": "Aviation"
        }

        print("[EntityExtractionAgent] Ready.")

    # ----------------------------------------------------------
    # Helper Functions
    # ----------------------------------------------------------

    def _find_companies(self, text: str) -> List[str]:
        found = []
        for name in self.company_keywords:
            if name.lower() in text.lower():
                found.append(name)
        return list(set(found))

    def _find_regulators(self, text: str) -> List[str]:
        found = []
        for reg in self.regulators:
            if reg.lower() in text.lower():
                found.append(reg)
        return list(set(found))

    def _infer_sectors(self, companies: List[str]) -> List[str]:
        sectors = []
        for comp in companies:
            sector = self.sector_map.get(comp)
            if sector:
                sectors.append(sector)
        return list(set(sectors))

    # ----------------------------------------------------------
    # Main extraction function
    # ----------------------------------------------------------

    def extract_entities(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """
        Input: article dict with fields title, content
        Output: extracted structured financial entities
        """
        text = article.get("title", "") + ". " + article.get("content", "")
        doc = None
        if self.nlp is not None:
            doc = self.nlp(text)

        companies = self._find_companies(text)
        regulators = self._find_regulators(text)
        sectors = self._infer_sectors(companies)

        # General purpose NER (names, events)
        people = []
        events = []

        if doc is not None:
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    people.append(ent.text)
                if ent.label_ in ["EVENT", "ORG"]:
                    events.append(ent.text)

        return {
            "id": article["id"],
            "companies": list(set(companies)),
            "sectors": list(set(sectors)),
            "regulators": list(set(regulators)),
            "people": list(set(people)),
            "events": list(set(events))
        }
