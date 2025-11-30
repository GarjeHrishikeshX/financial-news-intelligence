from src.agents.news_ingestion import NewsIngestionAgent
from src.agents.entity_extraction_agent import EntityExtractionAgent
from src.agents.stock_impact_agent import StockImpactAgent


def run():
    ing = NewsIngestionAgent()
    articles = ing.load_news("data/news.csv")

    ent = EntityExtractionAgent()
    stock_agent = StockImpactAgent()

    # Choose an article that mentions HDFC Bank and RBI
    sample_article = articles[0]   # HDFC Bank Announces 15% Dividend
    entities = ent.extract_entities(sample_article)
    impact = stock_agent.analyze_impact(entities)

    print("Article Title:", sample_article["title"])
    print("Extracted Entities:", entities)
    print("Stock Impact:", impact)


if __name__ == "__main__":
    run()

