import pandas as pd
from src.agents.storage_agent import StorageAgent

class NewsIngestionAgent:
    def __init__(self, csv_path="data/news.csv", db_path="data/storage.db"):
        self.csv_path = csv_path
        self.store = StorageAgent(db_path)

    def ingest_csv(self):
        df = pd.read_csv(self.csv_path)
        count = 0

        for _, row in df.iterrows():
            article = {
                "title": str(row.get("title", "")),
                "content": str(row.get("content", "")),
                "source": str(row.get("source", "")),
                "date": str(row.get("date", ""))
            }

            # insert
            aid = self.store.insert_article(article)

            # verify
            _ = self.store.get_article(aid)

            count += 1

        print(f"[NewsIngestionAgent] Stored {count} articles.")

if __name__ == "__main__":
    agent = NewsIngestionAgent()
    agent.ingest_csv()
