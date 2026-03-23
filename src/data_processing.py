import os
import pandas as pd
import numpy as np
from transformers import AutoTokenizer

RAW_PATH = "data/raw/meta_All_Beauty.jsonl"
PROCESSED_PATH = "data/processed/products_clean.parquet"
URL = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/meta_categories/meta_All_Beauty.jsonl"


def download_raw_data(url: str, save_path: str) -> None:
    
    if os.path.exists(save_path):
        print(f"Raw data already exists at {save_path}. Skipping download.")
        return

    print("Downloading dataset...")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df = pd.read_json(url, lines=True)
    df.to_json(save_path, orient="records", lines=True)
    print(f"Saved raw data to {save_path}")


def clean_list_field(x) -> str:
    
    if isinstance(x, (np.ndarray, list)):
        return " ".join([str(item).strip() for item in x if item])
    elif pd.isna(x):
        return ""
    return str(x).strip()


def build_product_text(df: pd.DataFrame) -> pd.DataFrame:
    
    for col in ["categories", "features", "description"]:
        df[col] = df[col].apply(clean_list_field)

    # Title first — highest signal, must survive truncation
    df["product_text"] = (
        df["title"] + ". " +
        df["categories"] + ". " +
        df["features"] + ". " +
        df["description"]
    )

    df["product_text"] = (
        df["product_text"]
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    return df


def filter_low_quality(df: pd.DataFrame, min_length: int = 100) -> pd.DataFrame:
    
    df["text_length"] = df["product_text"].str.len()
    df = df[df["text_length"] > min_length].copy()
    print(f"Retained {len(df)} high-quality products after filtering.")
    return df


def profile_tokens(df: pd.DataFrame, sample_frac: float = 0.05) -> None:
    
    print("Profiling token lengths...")
    tokenizer = AutoTokenizer.from_pretrained(
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    sample = df["product_text"].sample(frac=sample_frac, random_state=42)
    token_counts = sample.apply(
        lambda x: len(tokenizer.encode(x, truncation=False))
    )
    truncation_rate = (token_counts > 256).mean() * 100
    print(f"Average tokens: {token_counts.mean():.1f}")
    print(f"Max tokens: {token_counts.max()}")
    print(f"Truncation rate: {truncation_rate:.2f}%")


def process(raw_path: str, save_path: str) -> pd.DataFrame:
    
    print("Loading raw data...")
    with open(raw_path, "r", encoding="utf-8") as f:
        df = pd.read_json(f, lines=True)

    cols = ["parent_asin", "title", "categories", "features", "description"]
    df = df[cols].dropna(subset=["title"]).copy()

    df = build_product_text(df)
    df = filter_low_quality(df)
    profile_tokens(df)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df[["parent_asin", "title", "product_text"]].to_parquet(save_path, index=False)
    print(f"Saved processed data to {save_path}")

    return df


if __name__ == "__main__":
    download_raw_data(URL, RAW_PATH)
    process(RAW_PATH, PROCESSED_PATH)