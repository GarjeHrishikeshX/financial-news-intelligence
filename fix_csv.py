import csv

clean_rows = []
with open("data/news.csv", "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    for row in reader:
        # Only accept rows with EXACT columns: 5
        if len(row) == 5:
            clean_rows.append(row)
        else:
            print("Bad row removed:", row)

with open("data/news_clean.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(clean_rows)

print("Cleaning DONE. Saved → data/news_clean.csv")
