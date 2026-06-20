# Bulgarian News Timeline

An end-to-end NLP system that continuously collects Bulgarian news articles from 11 RSS sources, clusters them into evolving stories, detects narrative turning points ("dots"), and presents the results as an interactive timeline in a Streamlit app.

Built as part of an MSc thesis.

---

## Project Structure

```
msc_thesis/
├── fetch.py                  # RSS fetcher (run by GitHub Actions every 2h)
├── requirements.txt
├── data/
│   ├── raw/                  # Raw JSON articles, one file per article (git-tracked)
│   ├── processed/            # Cleaned parquet + intermediate pipeline outputs (not git-tracked)
│   ├── processed_system/     # Final outputs consumed by the Streamlit app (not git-tracked)
│   ├── experiments/          # Embedding & text variant experiments (not git-tracked)
│   └── evaluation/           # Ground truth labels and sample parquets
├── notebooks/                # Research notebooks (01–12, see below)
├── src/                      # Production pipeline modules
│   ├── pipeline.py           # ETL: load → clean → save articles_clean.parquet
│   ├── story_pipeline.py     # NLP: embed → stories → dots → names → summaries
│   ├── loaders.py
│   ├── filters.py
│   ├── cleaners.py
│   ├── source_cleaners.py
│   ├── embedder.py
│   ├── story_builder.py
│   ├── dot_detector.py
│   ├── story_namer.py
│   └── dot_summarizer.py
└── streamlit_app/
    └── app.py                # Interactive timeline UI
```

---

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

For dot summarization (step 5 of the story pipeline), [Ollama](https://ollama.com) must be running locally with the BgGPT model installed. Download the `.gguf` file from [HuggingFace](https://huggingface.co), create a `Modelfile`:

```
FROM ./bggpt.gguf
```

Then register it with Ollama:

```bash
ollama create bggpt -f Modelfile
```

---

## Data Collection

Articles are collected automatically every 2 hours via a GitHub Actions workflow (`.github/workflows/fetch_rss.yml`). For each source it:
1. Parses the RSS feed with `feedparser`
2. Scrapes the full article text with `trafilatura`
3. Saves each new article as a JSON file under `data/raw/<date>/`

The 11 sources are: `fakti`, `24chasa`, `nova`, `actualno`, `economic`, `banker`, `bta`, `segabg`, `standartnews`, `monitor`, `vesti`.

To run the fetcher manually:

```bash
python fetch.py
```

---

## Production Pipeline

Large generated files are not tracked in git — run the two pipeline scripts to produce them locally.

**Step 1 — ETL (cleaning)**

```bash
cd src
python pipeline.py
```

Loads all raw JSON articles, applies source-specific and global cleaning, deduplicates, and saves to `data/processed/articles_clean.parquet`.

**Step 2 — NLP (stories, dots, names, summaries)**

```bash
cd src
python story_pipeline.py
# or, to skip LLM summarization:
python story_pipeline.py --skip-summaries
```

Runs five steps in sequence:

| Step | Module | Output |
|------|--------|--------|
| 1 | `embedder.py` | `data/processed_system/embeddings.npy` |
| 2 | `story_builder.py` | `data/processed_system/stories.pkl` |
| 3 | `dot_detector.py` | `data/processed_system/dots_louvain.pkl` |
| 4 | `story_namer.py` | `data/processed_system/story_names.pkl` |
| 5 | `dot_summarizer.py` | `data/processed_system/dot_summaries.pkl` |

---

## Streamlit App

```bash
streamlit run streamlit_app/app.py
```

Displays an interactive day-by-day timeline of active stories. Each row is a story; dots represent narrative events. Clicking a dot shows the contributing articles and an LLM-generated summary.

Requires the outputs of both pipeline steps above.

---

## Research Notebooks

The `notebooks/` folder documents the full research process:

| Notebook | Purpose |
|----------|---------|
| `01_data_exploration.ipynb` | Understand raw data structure, temporal coverage, duplicates |
| `02_data_cleaning.ipynb` | Source-specific and global text cleaning |
| `03_cleaning_with_llm.ipynb` | LLM-assisted cleaning experiment |
| `04_embedding_selection.ipynb` | Compare embedding models and text representations |
| `05_story_building_traditional.ipynb` | Keyword-based story clustering (TF-IDF + Louvain) |
| `06_story_building_modern.ipynb` | Embedding-based story clustering |
| `07_dots_building.ipynb` | Dot detection within stories |
| `08_story_naming.ipynb` | Automatic story naming |
| `09_dot_summaries.ipynb` | LLM dot summarization |
| `10_evaluation_labeling.ipynb` | Manual ground truth labeling |
| `11_parameter_tuning.ipynb` | Parameter sweep for clustering |
| `12_evaluation.ipynb` | Evaluation against ground truth |

---

## Evaluation

Ground truth annotations are stored in `data/evaluation/` across five story-size buckets (`2–5`, `5–10`, `10–20`, `20–50`, `50+` articles). Each bucket has a sample parquet and a corresponding JSON with manual dot labels. Evaluation is run in `12_evaluation.ipynb`.
