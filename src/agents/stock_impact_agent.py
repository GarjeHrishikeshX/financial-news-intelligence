# src/agents/stock_impact_agent.py

from typing import List, Dict, Any


class StockImpactAgent:
    """
    Maps extracted entities (companies, sectors, regulators)
    to impacted stock symbols with confidence scores.

    Scoring Rules:
      - Direct company mention → 1.0
      - Sector-level impact → 0.6–0.8
      - Regulatory event → 0.4–0.7
    """

    def __init__(self):
        print("[StockImpactAgent] Initializing...")

        # -------------------------------
        # Company → Stock Symbol Mapping
        # -------------------------------
        self.stock_symbols = {
            "HDFC Bank": "HDFCBANK",
            "ICICI Bank": "ICICIBANK",
            "TCS": "TCS",
            "Infosys": "INFY",
            "Reliance Retail": "RELIANCE",
            "Adani Ports": "ADANIPORTS",
            "Maruti Suzuki": "MARUTI",
            "L&T": "LT",
            "Coal India": "COALINDIA",
            "Bajaj Finance": "BAJFINANCE",
            "Paytm": "PAYTM",
            "Air India": "AIRINDIA",
            "SpiceJet": "SPICEJET"
        }

        # -------------------------------
        # Sector → Representative Stocks
        # (Used for sector-wide impact)
        # -------------------------------
        self.sector_to_stocks = {
            "Banking": ["HDFCBANK", "ICICIBANK"],
            "Financial Services": ["BAJFINANCE"],
            "Retail": ["RELIANCE"],
            "IT Services": ["TCS", "INFY"],
            "Logistics": ["ADANIPORTS"],
            "Infrastructure": ["LT"],
            "Mining": ["COALINDIA"],
            "Automobile": ["MARUTI"],
            "Fintech": ["PAYTM"],
            "Aviation": ["AIRINDIA", "SPICEJET"]
        }

        # Regulators trigger medium confidence impact
        self.regulator_confidence = {
            "RBI": 0.6,
            "SEBI": 0.5,
            "US Fed": 0.5,
            "Federal Reserve": 0.5
        }

        print("[StockImpactAgent] Ready.")

    # ---------------------------------------------------------
    # Main Impact Mapping Function
    # ---------------------------------------------------------

    def analyze_impact(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """
        Input:
            entities = {
               'id': 1,
               'companies': [...],
               'sectors': [...],
               'regulators': [...]
            }

        Output:
            {
              "article_id": 1,
              "impacted_stocks": [
                {"symbol": "HDFCBANK", "confidence": 1.0, "type": "direct"},
                {"symbol": "ICICIBANK", "confidence": 0.7, "type": "sector"},
                {"symbol": "HDFCBANK", "confidence": 0.6, "type": "regulatory"}
              ]
            }
        """

        article_id = entities["id"]
        companies = entities.get("companies", [])
        sectors = entities.get("sectors", [])
        regulators = entities.get("regulators", [])

        impacted = []

        # -------------------------------
        # 1️⃣ Direct Company Impact (1.0)
        # -------------------------------
        for comp in companies:
            if comp in self.stock_symbols:
                impacted.append({
                    "symbol": self.stock_symbols[comp],
                    "confidence": 1.0,
                    "type": "direct",
                    "company": comp
                })

        # -------------------------------
        # 2️⃣ Sector-Level Impact (0.6–0.8)
        # -------------------------------
        for sector in sectors:
            if sector in self.sector_to_stocks:
                for stock in self.sector_to_stocks[sector]:
                    impacted.append({
                        "symbol": stock,
                        "confidence": 0.7,  # midpoint confidence
                        "type": "sector",
                        "sector": sector
                    })

        # -------------------------------
        # 3️⃣ Regulator Impact (0.4–0.7)
        # -------------------------------
        for reg in regulators:
            conf = self.regulator_confidence.get(reg, 0.4)
            # Apply to all banking/financial stocks for simplicity
            for stock in ["HDFCBANK", "ICICIBANK", "BAJFINANCE"]:
                impacted.append({
                    "symbol": stock,
                    "confidence": conf,
                    "type": "regulatory",
                    "regulator": reg
                })

        # Remove duplicates while keeping highest confidence
        final_impacts = {}
        for imp in impacted:
            sym = imp["symbol"]
            if sym not in final_impacts or imp["confidence"] > final_impacts[sym]["confidence"]:
                final_impacts[sym] = imp

        return {
            "article_id": article_id,
            "impacted_stocks": list(final_impacts.values())
        }
