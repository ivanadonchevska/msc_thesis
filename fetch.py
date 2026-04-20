import os
import json
import feedparser
import requests
import hashlib
from bs4 import BeautifulSoup
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

SITES = [
    {"name": "fakti", "rss": "https://fakti.bg/feed"},
    {"name": "24chasa", "rss": "https://24chasa.bg/rss"},
    {"name": "nova", "rss": "https://nova.bg/rss/latest"},
    {"name": "actualno", "rss": "https://www.actualno.com/rss"},
    {"name": "blitz", "rss": "https://blitz.bg/rss"},
]

DATA_DIR = "data/raw"


def parse_published_at(entry):
    try:
        dt = parsedate_to_datetime(entry.get("published", ""))
        return dt.astimezone(timezone.utc).isoformat()
    except Exception:
        return datetime.now(timezone.utc).isoformat()


def fetch_rss(site):
    feed = feedparser.parse(site["rss"])
    articles = []
    for entry in feed.entries:
        articles.append(
            {
                "source": site["name"],
                "title": entry.get("title", ""),
                "url": entry.get("link", ""),
                "published_at": parse_published_at(entry),
                "summary": entry.get("summary", ""),
                "full_text": "",
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            }
        )
    print(f"Fetched {len(articles)} articles from {site['name']}")
    return articles


HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; NewsBot/1.0)"}


def scrape_full_text(url):
    try:
        r = requests.get(url, timeout=10, headers=HEADERS)
        soup = BeautifulSoup(r.text, "html.parser")

        for tag in ["article", "main", ".article-body", ".content"]:
            content = soup.select_one(tag)
            if content:
                return content.get_text(separator=" ", strip=True)

        paragraphs = soup.find_all("p")
        return " ".join(p.get_text(strip=True) for p in paragraphs)

    except Exception as e:
        print(f"Failed to scrape {url}: {e}")
        return ""


def article_exists(article):
    date_dir = article["published_at"][:10]
    published_at = article["published_at"].replace(":", "-")
    filename = f"{article['source']}__{published_at}.json"
    filepath = os.path.join(DATA_DIR, date_dir, filename)
    return os.path.exists(filepath)


def make_filename(article):
    source = article["source"]
    published_at = article["published_at"].replace(":", "-")
    url_hash = hashlib.md5(article["url"].encode()).hexdigest()[:8]
    return f"{source}__{published_at}__{url_hash}.json"


def save_article(article):
    date_dir = article["published_at"][:10]
    dir_path = os.path.join(DATA_DIR, date_dir)
    os.makedirs(dir_path, exist_ok=True)

    filename = make_filename(article)
    filepath = os.path.join(dir_path, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(article, f, ensure_ascii=False, indent=2)

    print(f"Saved: {filename}")
    return filepath


def fetch_site(site):
    articles = fetch_rss(site)
    saved = 0
    for article in articles:
        if article_exists(article):
            continue
        article["full_text"] = scrape_full_text(article["url"])
        save_article(article)
        saved += 1
    print(f"{site['name']}: {saved} new articles saved\n")


def main():
    print(f"Starting fetch at {datetime.now(timezone.utc).isoformat()}")
    for site in SITES:
        try:
            fetch_site(site)
        except Exception as e:
            print(f"ERROR on {site['name']}: {e}")
    print("Done.")


if __name__ == "__main__":
    main()
