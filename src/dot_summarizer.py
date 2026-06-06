"""
dot_summarizer.py

Generates a 1-2 sentence Bulgarian summary for each dot using BgGPT via Ollama.
Singleton dots (size=1) use the article title directly — no LLM call needed.

Checkpoints every CHECKPOINT_EVERY dots so progress is not lost on interruption.

Output: data/processed_system/dot_summaries.pkl  — dict {(story_id, dot_idx): summary}
"""

import os
import pickle

import ollama
import pandas as pd

MODEL = "bggpt:latest"
CHECKPOINT_EVERY = 100

PARQUET_PATH = os.path.join(
    os.path.dirname(__file__), "../data/processed/articles_clean.parquet"
)
DOTS_PATH = os.path.join(
    os.path.dirname(__file__), "../data/processed_system/dots_louvain.pkl"
)
SUMMARIES_PATH = os.path.join(
    os.path.dirname(__file__), "../data/processed_system/dot_summaries.pkl"
)


def _build_prompt(dot: dict, df: pd.DataFrame) -> str:
    titles = "\n".join([f"- {df.iloc[i]['title']}" for i in dot["indices"]])
    return (
        "Ти си редактор на новинарски сайт. По-долу са заглавията на няколко новини "
        "за едно и също събитие, публикувани в един и същи ден от различни източници. "
        "Напиши 1-2 изречения на български, които обобщават какво се е случило. "
        "Бъди конкретен — включи имена, места и действия. "
        "Не добавяй въведение или обяснение, само резюмето.\n\n"
        f"Новини:\n{titles}"
    )


def _summarize(dot: dict, df: pd.DataFrame) -> str:
    if dot["size"] == 1:
        return df.iloc[dot["indices"][0]]["title"]
    prompt = _build_prompt(dot, df)
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"].strip()


def run_dot_summarizer():
    df = pd.read_parquet(PARQUET_PATH)

    with open(DOTS_PATH, "rb") as f:
        all_dots = pickle.load(f)

    # load checkpoint if exists
    if os.path.exists(SUMMARIES_PATH):
        with open(SUMMARIES_PATH, "rb") as f:
            summaries = pickle.load(f)
        print(f"Resuming from checkpoint: {len(summaries):,} summaries already done.")
    else:
        summaries = {}

    total_dots = sum(len(dots) for dots in all_dots.values())
    multi_dots = [
        (sid, idx, dot)
        for sid, dots in all_dots.items()
        for idx, dot in enumerate(dots)
        if (sid, idx) not in summaries
    ]

    print(f"Total dots: {total_dots:,} | Remaining: {len(multi_dots):,}")

    os.makedirs(os.path.dirname(SUMMARIES_PATH), exist_ok=True)

    for i, (sid, idx, dot) in enumerate(multi_dots):
        summaries[(sid, idx)] = _summarize(dot, df)

        if (i + 1) % CHECKPOINT_EVERY == 0:
            with open(SUMMARIES_PATH, "wb") as f:
                pickle.dump(summaries, f)
            print(f"  [{i+1}/{len(multi_dots)}] checkpoint saved")

    with open(SUMMARIES_PATH, "wb") as f:
        pickle.dump(summaries, f)
    print(f"Done. {len(summaries):,} summaries saved → {SUMMARIES_PATH}")


if __name__ == "__main__":
    run_dot_summarizer()
