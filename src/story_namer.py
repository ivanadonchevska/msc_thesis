"""
story_namer.py

Pre-computes a story name for every (story_id, date) pair using per-day
centroid-based medoid selection: for each date, find the article in that
day's dots whose embedding is closest to the day's mean embedding.

Skips today's date — those names are computed dynamically in the app since
articles are still arriving. On the next pipeline run, today becomes
historical and its names get stored.

Output: data/processed_system/story_names.pkl
        dict { (story_id, date): title_string }
"""

import os
import pickle
from collections import defaultdict
from datetime import date as Date

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

PARQUET_PATH = os.path.join(
    os.path.dirname(__file__), "../data/processed/articles_clean.parquet"
)
EMBEDDINGS_PATH = os.path.join(
    os.path.dirname(__file__), "../data/processed_system/embeddings.npy"
)
DOTS_PATH = os.path.join(
    os.path.dirname(__file__), "../data/processed_system/dots_louvain.pkl"
)
NAMES_PATH = os.path.join(
    os.path.dirname(__file__), "../data/processed_system/story_names.pkl"
)


def run_story_namer():
    df = pd.read_parquet(PARQUET_PATH)
    embs_all = np.load(EMBEDDINGS_PATH)

    with open(DOTS_PATH, "rb") as f:
        all_dots = pickle.load(f)

    # load existing names — only compute what's missing
    if os.path.exists(NAMES_PATH):
        with open(NAMES_PATH, "rb") as f:
            names = pickle.load(f)
        print(f"Loaded {len(names):,} existing names. Computing new ones...")
    else:
        names = {}
        print("No existing names found. Computing from scratch...")

    today = Date.today()
    new_count = 0

    for sid, dots in all_dots.items():
        by_date = defaultdict(list)
        for dot in dots:
            dt = dot["effective_start"].date()
            if dt == today:
                continue  # skip today — handled dynamically in the app
            by_date[dt].extend(dot["indices"])

        for dt, indices in by_date.items():
            if (sid, dt) in names:
                continue
            embs = embs_all[indices]
            centroid = embs.mean(axis=0, keepdims=True)
            sims = cosine_similarity(centroid, embs)[0]
            best_idx = indices[int(sims.argmax())]
            names[(sid, dt)] = df.iloc[best_idx]["title"]
            new_count += 1

    os.makedirs(os.path.dirname(NAMES_PATH), exist_ok=True)
    with open(NAMES_PATH, "wb") as f:
        pickle.dump(names, f)
    print(f"Added {new_count:,} new names. Total: {len(names):,} → {NAMES_PATH}")


if __name__ == "__main__":
    run_story_namer()
