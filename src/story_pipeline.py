"""
story_pipeline.py

Builds stories and dots from cleaned articles.
Run after pipeline.py (ETL) to produce the full story structure.

Steps:
  1. embedder        → data/processed_system/embeddings.npy
  2. story_builder   → data/processed_system/stories.pkl
  3. dot_detector    → data/processed_system/dots_louvain.pkl
  4. story_namer     → data/processed_system/story_names.pkl
  5. dot_summarizer  → data/processed_system/dot_summaries.pkl  (requires ollama + bggpt)
"""

from embedder import run_embedder
from story_builder import run_story_builder
from dot_detector import run_dot_detector
from story_namer import run_story_namer
from dot_summarizer import run_dot_summarizer


def run(skip_summaries: bool = False):
    print("=== Step 1: Embedder ===")
    run_embedder()

    print("\n=== Step 2: Story Builder ===")
    run_story_builder()

    print("\n=== Step 3: Dot Detector ===")
    run_dot_detector()

    print("\n=== Step 4: Story Namer ===")
    run_story_namer()

    if skip_summaries:
        print("\n=== Step 5: Dot Summarizer — SKIPPED ===")
    else:
        print("\n=== Step 5: Dot Summarizer ===")
        run_dot_summarizer()

    print("\nDone.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-summaries", action="store_true")
    args = parser.parse_args()
    run(skip_summaries=args.skip_summaries)
