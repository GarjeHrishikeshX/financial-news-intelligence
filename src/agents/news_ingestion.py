import pandas as pd
from datetime import datetime

class NewsIngestionAgent:
    def __init__(self):
        print("[NewsIngestionAgent] Initialized.")

    def load_news(self, filepath: str):
        """
        Loads financial news from a CSV file.

        Args:
            filepath (str): Path to CSV file.

        Returns:
            list[dict]: List of news articles as dictionaries.
        """
        try:
            df = pd.read_csv(filepath)

            # Convert date to datetime
            df['date'] = pd.to_datetime(df['date'])

            # Convert rows to dictionary format
            articles = []
            for _, row in df.iterrows():
                article = {
                    "id": int(row["id"]),
                    "title": row["title"],
                    "content": row["content"],
                    "date": row["date"],
                    "source": row["source"]
                }
                articles.append(article)

            print(f"[NewsIngestionAgent] Loaded {len(articles)} articles.")
            return articles

        except FileNotFoundError:
            print("❌ ERROR: The dataset file was not found.")
            return []
        except Exception as e:
            # Common CSV formatting errors (unquoted commas in titles) -> attempt a permissive fallback parser
            print(f"❌ ERROR while reading CSV with pandas: {e}")
            print("Attempting permissive fallback CSV parsing...")
            articles = []
            try:
                with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                    header = f.readline()  # skip header
                    for line in f:
                        # rsplit to get date and source (last two fields)
                        parts = line.rstrip('\n').rsplit(',', 2)
                        if len(parts) != 3:
                            # skip malformed line
                            continue
                        prefix, date, source = parts
                        # prefix contains id + ',' + title + maybe part of content
                        if ',' not in prefix:
                            continue
                        art_id, rest = prefix.split(',', 1)
                        # content should be quoted (start with ")
                        title = None
                        content = None
                        if ',"' in rest:
                            title, content_with_quote = rest.split(',"', 1)
                            # content_with_quote contains content possibly ending without closing quote if malformed
                            content = content_with_quote.rstrip('\n')
                            # strip ending quote if present
                            if content.endswith('"'):
                                content = content[:-1]
                        else:
                            # No explicit quoted content found — try to be tolerant
                            # Assume the rest begins with title and content is empty
                            title = rest
                            content = ""

                        article = {
                            "id": int(art_id),
                            "title": title.strip(),
                            "content": content.strip(),
                            "date": date.strip(),
                            "source": source.strip()
                        }
                        articles.append(article)
                print(f"[NewsIngestionAgent] Fallback parsed {len(articles)} articles.")
                return articles
            except Exception as e2:
                print(f"❌ ERROR in fallback parsing: {e2}")
                return []
