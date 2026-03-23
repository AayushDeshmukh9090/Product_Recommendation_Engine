import os
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


PROCESSED_PATH = "data/processed/products_clean.parquet"
TFIDF_VECTORIZER_PATH = "artifacts/tfidf_vectorizer.pkl"
TFIDF_MATRIX_PATH = "artifacts/tfidf_matrix.pkl"


def build_tfidf_index(df: pd.DataFrame) -> tuple:
    """
    Fit TF-IDF vectorizer on product texts and transform all products.
    Returns the vectorizer and the sparse matrix.
    """
    print("Building TF-IDF index...")

    vectorizer = TfidfVectorizer(
        max_features=50000,   # Vocabulary cap — ignore extremely rare words
        ngram_range=(1, 2),   # Unigrams and bigrams: "dry skin" as one feature
        min_df=2,             # Ignore words appearing in fewer than 2 products
        sublinear_tf=True     # Apply log normalization to TF — dampens very high counts
    )

    tfidf_matrix = vectorizer.fit_transform(df["product_text"])
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")

    return vectorizer, tfidf_matrix


def save_tfidf_artifacts(vectorizer, tfidf_matrix) -> None:
    """Save vectorizer and matrix to disk."""
    os.makedirs("artifacts", exist_ok=True)

    with open(TFIDF_VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    with open(TFIDF_MATRIX_PATH, "wb") as f:
        pickle.dump(tfidf_matrix, f)

    print(f"Saved TF-IDF artifacts to artifacts/")


def load_tfidf_artifacts() -> tuple:
    """Load saved vectorizer and matrix from disk."""
    with open(TFIDF_VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)

    with open(TFIDF_MATRIX_PATH, "rb") as f:
        tfidf_matrix = pickle.load(f)

    return vectorizer, tfidf_matrix


def search_tfidf(
    query: str,
    vectorizer: TfidfVectorizer,
    tfidf_matrix,
    df: pd.DataFrame,
    top_k: int = 5
) -> pd.DataFrame:
    """
    Given a query string, return top_k most similar products using TF-IDF cosine similarity.
    """
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = df.iloc[top_indices][["parent_asin", "title"]].copy()
    results["score"] = similarities[top_indices]
    results = results[results["score"] > 0].reset_index(drop=True)

    return results


if __name__ == "__main__":
    df = pd.read_parquet(PROCESSED_PATH)
    vectorizer, tfidf_matrix = build_tfidf_index(df)
    save_tfidf_artifacts(vectorizer, tfidf_matrix)

    # Test with queries that will later expose TF-IDF's weaknesses
    test_queries = [
        "moisturizer for dry winter skin",
        "gentle cleanser for sensitive skin",
        "anti aging serum with retinol",
        "something for oily skin in summer"
    ]

    print("\n--- TF-IDF Retrieval Results ---")
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = search_tfidf(query, vectorizer, tfidf_matrix, df, top_k=5)
        print(results.to_string(index=False))