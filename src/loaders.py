import json
import os

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../data/raw")


def load_raw_articles(data_dir: str = DATA_DIR) -> pd.DataFrame:
    records = []
    for date_folder in sorted(os.listdir(data_dir)):
        folder_path = os.path.join(data_dir, date_folder)
        if not os.path.isdir(folder_path):
            continue
        for filename in sorted(os.listdir(folder_path)):
            if filename.endswith(".json"):
                filepath = os.path.join(folder_path, filename)
                with open(filepath, encoding="utf-8") as f:
                    records.append(json.load(f))

    df = pd.DataFrame(records)
    df["published_at_dt"] = pd.to_datetime(df["published_at"], format="ISO8601", utc=True)

    print(f"Total articles loaded: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Date range: {df['published_at_dt'].min()} → {df['published_at_dt'].max()}")
    print(f"Sources:\n{df['source'].value_counts().to_string()}\n")

    return df
