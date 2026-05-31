## Repository Structure

```
msc_thesis/
├── fetch.py                          # RSS fetcher — ingestion entry point
├── check_rss.py                      # utility — discovers RSS feed URLs for new sources
├── requirements.txt                  # Python dependencies
│
├── .github/
│   └── workflows/
│       └── fetch_rss.yml             # cron scheduler — triggers fetch.py every 2 hours
│
├── notebooks/
│   ├── 01_data_exploration.ipynb     # exploratory analysis of raw data
│   ├── 02_data_cleaning.ipynb        # cleaning analysis & pipeline decisions
│   └── 03_cleaning_with_llm.ipynb    # experiment: BgGPT-based cleaning vs rule-based pipeline
│
├── src/
│   ├── pipeline.py                   # ETL orchestrator
│   ├── loaders.py                    # loads raw JSON files
│   ├── filters.py                    # filtering functions
│   ├── cleaners.py                   # cleaning primitives
│   └── source_cleaners.py            # per-source cleaning
│
└── data/
    ├── raw/
    │   └── {date}/
    │       └── {source}__{date}__{hash}.json   # raw articles organised by date
    ├── processed/
    │   ├── articles_clean.parquet    # final cleaned dataset
    │   └── seen_urls.txt             # incremental processing tracker
    └── experiments/
        └── bggpt_cleaning_experiment.parquet  # LLM cleaning experiment results (300 articles)
```

---

## Component Interactions

```mermaid
flowchart LR
    subgraph ROOT["msc_thesis/"]
        FETCH_PY["fetch.py\nRSS fetcher — ingestion entry point"]
        REQ["requirements.txt\nPython dependencies"]

        subgraph GHA[".github/workflows/"]
            YML["fetch_rss.yml\ncron scheduler — triggers fetch.py every 2 hours"]
        end

        subgraph NB["notebooks/"]
            NB1["01_data_exploration.ipynb\nexploratory analysis of raw data"]
            NB2["02_data_cleaning.ipynb\ncleaning analysis & pipeline decisions"]
            NB3["03_cleaning_with_llm.ipynb\nBgGPT cleaning experiment"]
        end

        subgraph SRC["src/"]
            PIPE["pipeline.py\nETL orchestrator"]
            LOAD["loaders.py\nloads raw JSON files"]
            FILT["filters.py\nfiltering functions"]
            CLEAN["cleaners.py\ncleaning primitives"]
            SCLEAN["source_cleaners.py\nper-source cleaning"]
        end

        subgraph DATA["data/"]
            subgraph RAW["raw/"]
                RAWF["{date}/{source}__{date}__{hash}.json\nraw articles organised by date"]
            end
            subgraph PROC["processed/"]
                PARQ["articles_clean.parquet\nfinal cleaned dataset"]
                SEEN["seen_urls.txt\nincremental processing tracker"]
            end
            subgraph EXP["experiments/"]
                EXPF["bggpt_cleaning_experiment.parquet\nLLM cleaning experiment results"]
            end
        end
    end

    YML --> FETCH_PY
    FETCH_PY --> RAWF
    RAWF --> NB1
    NB1 --> NB2
    NB2 -- "informs pipeline decisions" --> SRC
    NB2 --> NB3
    PARQ --> NB3
    RAWF --> NB3
    NB3 --> EXPF
    RAWF --> LOAD
    LOAD --> PIPE
    FILT --> PIPE
    CLEAN --> PIPE
    SCLEAN --> PIPE
    SEEN --> PIPE
    PIPE --> PARQ
```
