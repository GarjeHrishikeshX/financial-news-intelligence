from src.agents.news_ingestion import NewsIngestionAgent

agent = NewsIngestionAgent()
articles = agent.load_news("data/news.csv")

print(articles[0])
print("Total articles:", len(articles))
