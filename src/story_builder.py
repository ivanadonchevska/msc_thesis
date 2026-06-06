"""
story_builder.py

Builds stories from cleaned articles using a sliding-window Louvain approach
(NewsLens + embeddings). Two steps:
  1. build_topics  — cluster articles within each time window using Louvain
  2. build_stories — link topic clusters across windows into persistent stories,
                     then merge near-duplicate stories via mean-embedding similarity
"""

import os
import pickle
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import networkx as nx
from community import community_louvain
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
from tqdm import tqdm

N_DAYS = 5
STEP = 2
MIN_SIMILARITY = 0.78
T3 = 0.8
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


def _build_windows(dates: pd.Series, n_days: int = N_DAYS, step: int = STEP):
    unique_days = sorted(dates.unique())
    return [
        (unique_days[i], unique_days[min(i + n_days - 1, len(unique_days) - 1)])
        for i in range(0, len(unique_days) - n_days + 1, step)
    ]


def build_topics(
    df: pd.DataFrame,
    emb_index: dict,
    dates: pd.Series,
    windows: list,
    min_similarity: float = MIN_SIMILARITY,
    random_seed: int = RANDOM_SEED,
) -> list:
    topics = []

    for ws, we in tqdm(windows, desc="Building topics"):
        mask = (dates >= ws) & (dates <= we)
        window_indices = list(df[mask].index)

        if len(window_indices) < 2:
            topics.append({})
            continue

        embs = np.array([emb_index[i] for i in window_indices])
        sim_matrix = cos_sim(embs)

        G = nx.Graph()
        G.add_nodes_from(range(len(window_indices)))
        for i in range(len(window_indices)):
            for j in range(i + 1, len(window_indices)):
                if sim_matrix[i, j] >= min_similarity:
                    G.add_edge(i, j, weight=float(sim_matrix[i, j]))

        partition = community_louvain.best_partition(G, random_state=random_seed)
        topic_map = {window_indices[local]: topic_id for local, topic_id in partition.items()}
        topics.append(topic_map)

    return topics


def build_stories(
    window_topics: list,
    emb_index: dict,
    T3: float = T3,
) -> dict:
    article_story = {}
    story_alias = {}
    story_last_win = {}
    next_id = [0]

    def resolve(sid):
        while story_alias.get(sid, sid) != sid:
            sid = story_alias[sid]
        return sid

    def create_story(arts, win_idx):
        sid = next_id[0]
        next_id[0] += 1
        for a in arts:
            article_story[a] = sid
        story_last_win[sid] = win_idx

    def assign_to_story(arts, sid, win_idx):
        for a in arts:
            article_story[a] = sid
        story_last_win[sid] = win_idx

    def merge_stories(sid_keep, sid_drop):
        sid_keep = resolve(sid_keep)
        sid_drop = resolve(sid_drop)
        if sid_keep != sid_drop:
            story_alias[sid_drop] = sid_keep

    # link topic clusters across windows
    for win_idx, partition in enumerate(tqdm(window_topics, desc="Linking stories")):
        clusters = defaultdict(list)
        for art_idx, cluster_id in partition.items():
            clusters[cluster_id].append(art_idx)

        for arts in clusters.values():
            prev = [resolve(article_story[a]) for a in arts if a in article_story]
            if not prev:
                create_story(arts, win_idx)
                continue
            vote = Counter(prev)
            top_sid, top_count = vote.most_common(1)[0]
            if top_count > len(arts) / 2:
                for other in list(vote.keys()):
                    if other != top_sid:
                        merge_stories(top_sid, other)
                assign_to_story(arts, top_sid, win_idx)
            else:
                create_story(arts, win_idx)

    # cross-gap merge: merge stories with similar mean embeddings that span nearby windows
    story_articles_tmp = defaultdict(list)
    for art_idx, sid in article_story.items():
        story_articles_tmp[resolve(sid)].append(art_idx)

    canonical_ids = sorted(story_articles_tmp.keys())
    mean_embs = np.array([
        np.mean([emb_index[a] for a in story_articles_tmp[sid]], axis=0)
        for sid in canonical_ids
    ])
    mean_embs = mean_embs / np.linalg.norm(mean_embs, axis=1, keepdims=True)
    sim_matrix = mean_embs @ mean_embs.T

    merged_count = 0
    for ri in range(len(canonical_ids)):
        for ci in range(ri + 1, len(canonical_ids)):
            if sim_matrix[ri, ci] >= T3:
                sid_a = resolve(canonical_ids[ri])
                sid_b = resolve(canonical_ids[ci])
                if sid_a == sid_b:
                    continue
                if abs(story_last_win.get(sid_a, -1) - story_last_win.get(sid_b, -1)) > 1:
                    merge_stories(sid_a, sid_b)
                    merged_count += 1

    story_articles = defaultdict(list)
    for art_idx, sid in article_story.items():
        story_articles[resolve(sid)].append(art_idx)
    story_articles = dict(story_articles)

    sizes = pd.Series({sid: len(arts) for sid, arts in story_articles.items()})
    print(f"Stories: {len(story_articles):,} | Cross-gap merges: {merged_count}")
    print(f"Singletons: {(sizes == 1).sum()} ({(sizes == 1).mean():.1%}) | Largest: {sizes.max()}")

    return story_articles


def run_story_builder():
    df = pd.read_parquet(PARQUET_PATH)
    df["published_at_dt"] = pd.to_datetime(df["published_at_dt"], utc=True)
    dates = df["published_at_dt"].dt.tz_localize(None).dt.normalize()

    embs_all = np.load(EMBEDDINGS_PATH)
    emb_index = {i: embs_all[i] for i in range(len(df))}
    print(f"Articles: {len(df):,}  Embeddings: {embs_all.shape}")

    windows = _build_windows(dates)
    print(f"Windows: {len(windows)}")

    topics = build_topics(df, emb_index, dates, windows)
    stories = build_stories(topics, emb_index)

    os.makedirs(os.path.dirname(STORIES_PATH), exist_ok=True)
    with open(STORIES_PATH, "wb") as f:
        pickle.dump(stories, f)
    print(f"Saved {len(stories):,} stories → {STORIES_PATH}")


if __name__ == "__main__":
    run_story_builder()
