import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


PROCESSED_PATH = "data/processed/products_clean.parquet"
EMBEDDINGS_PATH = "artifacts/embeddings.npy"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def generate_embeddings(
    texts: list,
    model_name: str = MODEL_NAME,
    batch_size: int = 64,
    show_progress: bool = True
) -> np.ndarray:
    """
    Encode a list of texts into dense vectors using a sentence transformer.
    Returns a numpy array of shape (n_texts, embedding_dim).
    """
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    print(f"Generating embeddings for {len(texts)} products...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True   # L2 normalize — required for cosine similarity via dot product
    )

    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings


def save_embeddings(embeddings: np.ndarray, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, embeddings)
    print(f"Saved embeddings to {path}")


def load_embeddings(path: str) -> np.ndarray:
    embeddings = np.load(path)
    print(f"Loaded embeddings: {embeddings.shape}")
    return embeddings


if __name__ == "__main__":
    df = pd.read_parquet(PROCESSED_PATH)
    texts = df["product_text"].tolist()

    embeddings = generate_embeddings(texts)
    save_embeddings(embeddings, EMBEDDINGS_PATH)

    # Sanity checks
    print(f"\nSanity checks:")
    print(f"Shape: {embeddings.shape}")
    print(f"Expected: ({len(df)}, 384)")
    print(f"Sample vector norm: {np.linalg.norm(embeddings[0]):.4f}")  # Should be ~1.0
    print(f"Dtype: {embeddings.dtype}")
