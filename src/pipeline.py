"""
pipeline.py

ETL pipeline for Bulgarian news articles.
Handles loading, cleaning, and incremental saving to parquet.
"""

import os

import pandas as pd

from loaders import load_raw_articles
from filters import (
    remove_backfill,
    drop_sources,
    drop_empty_full_text,
    deduplicate,
    drop_mojibake,
    drop_short_articles,
    drop_near_duplicates,
)
from source_cleaners import (
    clean_vesti,
    clean_monitor,
    clean_standartnews,
    clean_segabg,
    clean_bta,
    clean_actualno,
    clean_nova,
    clean_24chasa,
    clean_fakti,
    clean_global,
)

OUT_PATH = os.path.join(
    os.path.dirname(__file__), "../data/processed/articles_clean.parquet"
)
SEEN_URLS_PATH = os.path.join(
    os.path.dirname(__file__), "../data/processed/seen_urls.txt"
)


def _load_seen_urls() -> set:
    if os.path.exists(SEEN_URLS_PATH):
        with open(SEEN_URLS_PATH) as f:
            return set(f.read().splitlines())
    return set()


def _save_seen_urls(urls: set):
    os.makedirs(os.path.dirname(SEEN_URLS_PATH), exist_ok=True)
    with open(SEEN_URLS_PATH, "a") as f:
        for url in urls:
            f.write(url + "\n")


def run_pipeline():
    is_first_run = not os.path.exists(OUT_PATH)
    seen_urls = _load_seen_urls()

    # Load raw and keep only new records
    df = load_raw_articles()
    df = df[~df["url"].isin(seen_urls)].reset_index(drop=True)
    print(f"New articles to process: {len(df)}")

    # Mark all new URLs as seen before filtering
    _save_seen_urls(set(df["url"]))

    if df.empty:
        print("Nothing new to process.")
        return

    # Filters
    df = remove_backfill(df)
    if is_first_run:
        df = drop_sources(df, sources=["capital", "blitz", "dnevnik"])
    df = drop_empty_full_text(df)
    df = deduplicate(df)
    df = drop_near_duplicates(df)
    df = drop_mojibake(df)
    df = drop_short_articles(df, min_words=30)

    # Source-specific cleaning
    df = clean_vesti(df)
    df = clean_monitor(df)
    df = clean_standartnews(df)
    df = clean_segabg(df)
    df = clean_bta(df)
    df = clean_actualno(df)
    df = clean_nova(df)
    df = clean_24chasa(df)
    df = clean_fakti(df)

    # Global cleaning
    df = clean_global(df)
    df = drop_short_articles(df, min_words=30)
    df = drop_near_duplicates(df)

    df = df.drop(columns=["summary", "published_at"])

    # Append to existing parquet
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    if not is_first_run:
        existing = pd.read_parquet(OUT_PATH)
        df = pd.concat([existing, df], ignore_index=True)

    df.to_parquet(OUT_PATH, index=False)
    print(f"Saved {len(df)} total articles → {OUT_PATH}")


if __name__ == "__main__":
    run_pipeline()
