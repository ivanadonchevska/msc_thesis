import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

COLLECTION_START = "2026-04-20"


def drop_mojibake(df: pd.DataFrame) -> pd.DataFrame:
    def _is_mojibake(text: str) -> bool:
        if not isinstance(text, str) or not text.strip():
            return False
        dash_ratio = (text.count("–") + text.count("—")) / max(len(text), 1)
        return dash_ratio > 0.05

    mask = df["full_text"].apply(_is_mojibake)
    print(f"Dropping {mask.sum()} mojibake articles")
    df = df[~mask].reset_index(drop=True)
    print(f"Remaining articles: {len(df)}")
    return df


def drop_short_articles(df: pd.DataFrame, min_words: int = 30) -> pd.DataFrame:
    df["word_count"] = df["full_text"].fillna("").apply(lambda x: len(x.split()))
    mask = df["word_count"] < min_words
    print(f"Dropping {mask.sum()} articles under {min_words} words")
    df = df[~mask].reset_index(drop=True)
    print(f"Remaining articles: {len(df)}")
    return df


def remove_backfill(
    df: pd.DataFrame,
    source: str = "segabg",
    collection_start: str = COLLECTION_START,
) -> pd.DataFrame:
    start = pd.Timestamp(collection_start, tz="UTC")
    mask = (df["source"] == source) & (df["published_at_dt"] < start)
    print(
        f"Dropping {mask.sum()} '{source}' articles published before {collection_start}"
    )
    df = df[~mask].reset_index(drop=True)
    print(f"Remaining articles: {len(df)}")
    return df


def drop_sources(df: pd.DataFrame, sources: list[str]) -> pd.DataFrame:
    mask = df["source"].isin(sources)
    print(f"Dropping {mask.sum()} articles from sources: {sources}")
    df = df[~mask].reset_index(drop=True)
    print(f"Remaining articles: {len(df)}")
    return df


def drop_empty_full_text(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["full_text"].isna() | (df["full_text"].str.strip() == "")
    print(f"Dropping {mask.sum()} articles with empty full_text")
    df = df[~mask].reset_index(drop=True)
    print(f"Remaining articles: {len(df)}")
    return df


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = (
        df.sort_values("fetched_at", ascending=False)
        .drop_duplicates(subset="url", keep="first")
        .reset_index(drop=True)
    )
    print(f"Dropped {before - len(df)} duplicate rows")
    print(f"Remaining articles: {len(df)}")
    return df


def drop_by_field(
    df: pd.DataFrame,
    patterns: list[str] | str,
    field: str = "full_text",
    source: str | None = None,
    match: str = "startswith",
) -> pd.DataFrame:
    patterns = [patterns] if isinstance(patterns, str) else patterns
    combined = "|".join(patterns)
    if match == "startswith":
        mask = df[field].str.startswith(tuple(patterns), na=False)
    else:
        mask = df[field].str.contains(combined, case=False, na=False)
    if source:
        mask = mask & (df["source"] == source)
    print(f"Dropping {mask.sum()} articles where {field} {match} {patterns}")
    df = df[~mask].reset_index(drop=True)
    print(f"Remaining articles: {len(df)}")
    return df


def drop_near_duplicates(
    df: pd.DataFrame,
    threshold: float = 0.95,
    window_days: int = 3,
) -> pd.DataFrame:
    df = df.copy()
    df["_window"] = (
        df["published_at_dt"].dt.tz_localize(None).dt.to_period(f"{window_days}D")
    )

    rows_to_drop = set()

    for (source, window), group in df.groupby(["source", "_window"]):
        if len(group) < 2:
            continue

        texts = (group["title"] + " " + group["full_text"].fillna("")).tolist()

        try:
            vectorizer = TfidfVectorizer(min_df=1, max_features=5000)
            tfidf_matrix = vectorizer.fit_transform(texts)
            sim_matrix = cosine_similarity(tfidf_matrix)
        except ValueError:
            continue

        rows = group.reset_index()
        n = len(rows)
        for i in range(n):
            for j in range(i + 1, n):
                if sim_matrix[i, j] >= threshold:
                    date_i = rows.iloc[i]["published_at_dt"]
                    date_j = rows.iloc[j]["published_at_dt"]
                    drop_idx = (
                        rows.iloc[i]["index"]
                        if date_i <= date_j
                        else rows.iloc[j]["index"]
                    )
                    rows_to_drop.add(drop_idx)

    df = df.drop(index=rows_to_drop).reset_index(drop=True)
    df = df.drop(columns=["_window"])

    print(
        f"Dropped {len(rows_to_drop)} near-duplicate articles (threshold={threshold})"
    )
    return df
