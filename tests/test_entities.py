from src.agents.news_ingestion import NewsIngestionAgent
from src.agents.entity_extraction_agent import EntityExtractionAgent


def run():
    ing = NewsIngestionAgent()
    articles = ing.load_news("data/news.csv")

    ent_agent = EntityExtractionAgent()

    sample = articles[0]
    print("Sample article:", sample["title"])

    entities = ent_agent.extract_entities(sample)
    print("Extracted Entities:", entities)


if __name__ == "__main__":
    run()

