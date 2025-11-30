from src.agents.query_agent import QueryAgent

def run():
    qa = QueryAgent("data/storage.db")

    # Example queries:
    queries = [
        "HDFC Bank news",
        "Banking sector update",
        "RBI policy change",
        "Interest rate impact",
        "IT sector news",
        "Adani Ports update"
    ]

    for q in queries:
        res = qa.query(q)
        print("\n========== RESULT ==========")
        print("Query:", q)
        print("Interpretation:", res["interpretation"])
        for r in res["results"][:3]:
            print("\nTitle:", r["title"])
            print("Score:", r["score"])
            print("Explanation:", r["explanation"])


if __name__ == "__main__":
    run()
