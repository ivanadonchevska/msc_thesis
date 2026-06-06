"""
dot_detector.py

Detects dots (sub-events) within each story using Louvain community detection.
Edges are weighted by cosine similarity * time decay. Articles too far apart
in time are not connected. Stories with no edges fall back to one dot per article.
"""

import os
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
import networkx as nx
from community import community_louvain
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
from tqdm import tqdm

MIN_SIMILARITY = 0.70
MAX_GAP_CAP_HOURS = 18.0
UPDATE_THRESHOLD_HOURS = 2.5
RANDOM_SEED = 42

PARQUET_PATH = os.path.join(
    os.path.dirname(__file__), "../data/processed/articles_clean.parquet"
)
EMBEDDINGS_PATH = os.path.join(
    os.path.dirname(__file__), "../data/processed_system/embeddings.npy"
)
STORIES_PATH = os.path.join(
    os.path.dirname(__file__), "../data/processed_system/stories.pkl"
)
DOTS_PATH = os.path.join(
    os.path.dirname(__file__), "../data/processed_system/dots_louvain.pkl"
)


def _effective_time(idx: int, df: pd.DataFrame) -> pd.Timestamp:
    row = df.iloc[idx]
    published = row["published_at_dt"]
    fetched = pd.to_datetime(row["fetched_at"], utc=True)
    gap = abs((fetched - published).total_seconds()) / 3600
    return fetched if gap > UPDATE_THRESHOLD_HOURS else published


def detect_dots(
    story_articles: list,
    df: pd.DataFrame,
    emb_index: dict,
    min_similarity: float = MIN_SIMILARITY,
    max_gap_cap_hours: float = MAX_GAP_CAP_HOURS,
    random_seed: int = RANDOM_SEED,
) -> list:
    n = len(story_articles)
    times = [_effective_time(i, df) for i in story_articles]

    # adaptive decay: median inter-article gap, floored at 1h
    sorted_times = sorted(times)
    if len(sorted_times) >= 2:
        gaps_h = [
            (sorted_times[i + 1] - sorted_times[i]).total_seconds() / 3600
            for i in range(len(sorted_times) - 1)
        ]
        decay_hours = max(float(np.median(gaps_h)), 1.0)
    else:
        decay_hours = 6.0

    embs = np.array([emb_index[i] for i in story_articles])
    sim_matrix = cos_sim(embs)

    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            sim = float(sim_matrix[i, j])
            if sim < min_similarity:
                continue
            time_gap = abs((times[i] - times[j]).total_seconds()) / 3600
            if time_gap >= max_gap_cap_hours:
                continue
            G.add_edge(i, j, weight=sim * np.exp(-time_gap / decay_hours))

    # no edges → each article is its own dot
    if G.number_of_edges() == 0:
        dot_list = []
        for idx in story_articles:
            pub = df.iloc[idx]["published_at_dt"]
            dot_list.append({
                "indices": [idx],
                "start": pub,
                "end": pub,
                "effective_start": _effective_time(idx, df),
                "size": 1,
                "sources": [df.iloc[idx]["source"]],
            })
        return sorted(dot_list, key=lambda x: x["effective_start"])

    partition = community_louvain.best_partition(G, random_state=random_seed)
    clusters = defaultdict(list)
    for local_idx, dot_id in partition.items():
        clusters[dot_id].append(story_articles[local_idx])

    dot_list = []
    for indices in clusters.values():
        dot_dates = [df.iloc[i]["published_at_dt"] for i in indices]
        eff_times = [_effective_time(i, df) for i in indices]
        dot_list.append({
            "indices": indices,
            "start": min(dot_dates),
            "end": max(dot_dates),
            "effective_start": min(eff_times),
            "size": len(indices),
            "sources": df.iloc[indices]["source"].tolist(),
        })

    dot_list.sort(key=lambda x: x["effective_start"])
    return dot_list


def run_dot_detector():
    df = pd.read_parquet(PARQUET_PATH)
    df["published_at_dt"] = pd.to_datetime(df["published_at_dt"], utc=True)

    embs_all = np.load(EMBEDDINGS_PATH)
    emb_index = {i: embs_all[i] for i in range(len(df))}

    with open(STORIES_PATH, "rb") as f:
        stories = pickle.load(f)
    print(f"Stories loaded: {len(stories):,}")

    all_dots = {}
    for sid, arts in tqdm(stories.items(), desc="Detecting dots"):
        if not arts:
            continue
        emb_idx = {i: emb_index[i] for i in arts}
        all_dots[sid] = detect_dots(arts, df, emb_idx)

    total_dots = sum(len(dots) for dots in all_dots.values())
    print(f"Total dots: {total_dots:,}  ({total_dots / len(all_dots):.1f} avg per story)")

    os.makedirs(os.path.dirname(DOTS_PATH), exist_ok=True)
    with open(DOTS_PATH, "wb") as f:
        pickle.dump(all_dots, f)
    print(f"Saved → {DOTS_PATH}")


if __name__ == "__main__":
    run_dot_detector()
