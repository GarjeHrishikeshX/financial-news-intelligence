# tests/test_dedup.py
from src.agents.news_ingestion import NewsIngestionAgent
from src.agents.dedup_agent import DeduplicationAgent

def run_test():
    ing = NewsIngestionAgent()
    articles = ing.load_news("data/news.csv")
    print("Loaded articles:", len(articles))

    deduper = DeduplicationAgent(similarity_threshold=0.82)  # tune threshold if needed
    stories = deduper.deduplicate(articles)

    print("Unique stories found:", len(stories))
    for s in stories:
        print("Story:", s["story_id"], "rep_id:", s["representative"]["id"], "members:", s["article_ids"])

if __name__ == "__main__":
    run_test()
