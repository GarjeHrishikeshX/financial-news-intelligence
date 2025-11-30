# tests/test_storage.py
from src.agents.news_ingestion import NewsIngestionAgent
from src.agents.dedup_agent import DeduplicationAgent
from src.agents.entity_extraction_agent import EntityExtractionAgent
from src.agents.stock_impact_agent import StockImpactAgent
from src.agents.storage_agent import StorageAgent
import numpy as np

def run():
    # Load articles
    ing = NewsIngestionAgent()
    articles = ing.load_news("data/news.csv")

    # Save articles
    store = StorageAgent("data/test_storage.db")
    store.save_articles(articles)
    print("Saved articles:", len(articles))

    # Deduplicate and save stories
    ded = DeduplicationAgent(similarity_threshold=0.82)
    stories = ded.deduplicate(articles)
    store.save_stories(stories)
    print("Saved stories:", len(stories))

    # Extract entities and impacts for first article and save
    ent_agent = EntityExtractionAgent()
    stock_agent = StockImpactAgent()

    sample = articles[0]
    entities = ent_agent.extract_entities(sample)
    store.save_entity(entities)

    impact = stock_agent.analyze_impact(entities)
    store.save_impact(impact)
    print("Saved entities & impacts for article:", sample["id"])

    # Save embeddings for first 5 articles using the dedup embedder
    embedder = ded.embedder
    texts = [a["title"] + " . " + a["content"] for a in articles[:5]]
    vecs = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    for a, v in zip(articles[:5], vecs):
        store.save_embedding(a["id"], v, namespace="sent-emb")

    # Query: search with first article embedding
    q = vecs[0]
    res = store.search_by_vector(q, top_k=3, namespace="sent-emb")
    print("Search results:", res)

    store.close()

if __name__ == "__main__":
    run()
