"""
embedder.py

Incremental article embedding using BAAI/bge-m3.
Text variant: title + lead (first 3 sentences of full_text).
Only embeds articles not yet in embeddings.npy, then appends.
"""

import os
import re

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

MODEL_ID = "BAAI/bge-m3"
BATCH_SIZE = 8

PARQUET_PATH = os.path.join(
    os.path.dirname(__file__), "../data/processed/articles_clean.parquet"
)
EMBEDDINGS_PATH = os.path.join(
    os.path.dirname(__file__), "../data/processed_system/embeddings.npy"
)


def _device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _extract_lead(text: str, n_sentences: int = 3) -> str:
    if not isinstance(text, str):
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(sentences[:n_sentences])


def _make_text(row) -> str:
    title = row["title"] if isinstance(row["title"], str) else ""
    lead = _extract_lead(row.get("full_text", ""))
    return f"{title}. {lead}".strip()


def run_embedder():
    df = pd.read_parquet(PARQUET_PATH)
    n_total = len(df)

    if os.path.exists(EMBEDDINGS_PATH):
        existing = np.load(EMBEDDINGS_PATH)
        n_existing = len(existing)
    else:
        existing = None
        n_existing = 0

    if n_existing >= n_total:
        print(f"Embeddings up to date ({n_existing} rows).")
        return

    n_new = n_total - n_existing
    print(f"Embedding {n_new} new articles ({n_existing} already embedded)...")

    new_rows = df.iloc[n_existing:]
    texts = [_make_text(row) for _, row in new_rows.iterrows()]

    device = _device()
    print(f"Device: {device}  Model: {MODEL_ID}")
    model = SentenceTransformer(MODEL_ID, device=device)

    new_embs = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=BATCH_SIZE,
    )

    os.makedirs(os.path.dirname(EMBEDDINGS_PATH), exist_ok=True)
    all_embs = np.vstack([existing, new_embs]) if existing is not None else new_embs
    np.save(EMBEDDINGS_PATH, all_embs)
    print(f"Saved {len(all_embs)} total embeddings → {EMBEDDINGS_PATH}")


if __name__ == "__main__":
    run_embedder()
