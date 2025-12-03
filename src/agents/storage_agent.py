import sqlite3
import json
import os

class StorageAgent:
    def __init__(self, db_path="data/storage.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._ensure_tables()

    def _ensure_tables(self):
        cur = self.conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                raw TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS stories (
                story_id INTEGER PRIMARY KEY AUTOINCREMENT,
                representative TEXT,
                article_ids TEXT
            )
        """)
        self.conn.commit()

    # -------------------------------
    # REQUIRED FOR news_ingestion.py
    # -------------------------------
    def insert_article(self, article: dict):
        """Insert article and return ID"""
        cur = self.conn.cursor()
        cur.execute("INSERT INTO articles (raw) VALUES (?)", (json.dumps(article),))
        self.conn.commit()
        return cur.lastrowid

    def get_article(self, article_id: int):
        """Fetch a single article"""
        cur = self.conn.cursor()
        row = cur.execute("SELECT raw FROM articles WHERE id=?", (article_id,)).fetchone()
        if not row:
            return None
        try:
            return json.loads(row[0])
        except:
            return row[0]

    def load_all_articles(self):
        cur = self.conn.cursor()
        rows = cur.execute("SELECT raw FROM articles").fetchall()
        all_articles = []
        for r in rows:
            try:
                all_articles.append(json.loads(r[0]))
            except:
                all_articles.append(r[0])
        return all_articles

    # story support
    def insert_story(self, representative, article_ids):
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO stories (representative, article_ids) VALUES (?,?)",
            (json.dumps(representative), json.dumps(article_ids))
        )
        self.conn.commit()

    def load_all_stories(self):
        cur = self.conn.cursor()
        rows = cur.execute("SELECT story_id, representative, article_ids FROM stories").fetchall()
        stories = []
        for s in rows:
            rep = json.loads(s[1]) if s[1] else {}
            ids = json.loads(s[2]) if s[2] else []
            stories.append({
                "story_id": s[0],
                "representative": rep,
                "article_ids": ids
            })
        return stories
