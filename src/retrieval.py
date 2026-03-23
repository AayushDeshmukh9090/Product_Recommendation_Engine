import numpy as np
import pandas as pd
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ── Paths ────────────────────────────────────────────────────────────────────
PROCESSED_PATH       = "data/processed/products_clean.parquet"
EMBEDDINGS_PATH      = "artifacts/embeddings.npy"
FAISS_INDEX_PATH     = "artifacts/faiss_index.bin"
TFIDF_VECTORIZER_PATH = "artifacts/tfidf_vectorizer.pkl"
TFIDF_MATRIX_PATH    = "artifacts/tfidf_matrix.pkl"
MODEL_NAME           = "sentence-transformers/all-MiniLM-L6-v2"


# ── Artifact Loader ───────────────────────────────────────────────────────────
class RecommendationEngine:
    """
    Unified interface for both TF-IDF and Semantic retrieval systems.
    Loads all artifacts once at initialization, serves queries instantly.
    """

    def __init__(self):
        print("Loading artifacts...")

        self.df = pd.read_parquet(PROCESSED_PATH)

        # Semantic search artifacts
        self.embeddings  = np.load(EMBEDDINGS_PATH)
        self.faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        self.model       = SentenceTransformer(MODEL_NAME)

        # TF-IDF artifacts
        with open(TFIDF_VECTORIZER_PATH, "rb") as f:
            self.vectorizer = pickle.load(f)
        with open(TFIDF_MATRIX_PATH, "rb") as f:
            self.tfidf_matrix = pickle.load(f)

        print(f"Engine ready. {len(self.df)} products loaded.")


    def recommend_semantic(self, query: str, top_k: int = 5) -> pd.DataFrame:
        """
        Return top_k products using semantic similarity (FAISS + embeddings).
        """
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        scores, indices = self.faiss_index.search(query_embedding, top_k)

        results = self.df.iloc[indices[0]][["parent_asin", "title"]].copy()
        results["score"] = scores[0]
        results["method"] = "semantic"

        return results.reset_index(drop=True)


    def recommend_tfidf(self, query: str, top_k: int = 5) -> pd.DataFrame:
        """
        Return top_k products using TF-IDF cosine similarity.
        """
        query_vec  = self.vectorizer.transform([query])
        scores     = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = self.df.iloc[top_indices][["parent_asin", "title"]].copy()
        results["score"]  = scores[top_indices]
        results["method"] = "tfidf"

        return results[results["score"] > 0].reset_index(drop=True)


    def recommend(
        self,
        query: str,
        method: str = "semantic",
        top_k: int = 5
    ) -> pd.DataFrame:
        """
        Unified entry point. method = 'semantic' or 'tfidf'.
        This is the only function your app.py needs to call.
        """
        if method == "semantic":
            return self.recommend_semantic(query, top_k)
        elif method == "tfidf":
            return self.recommend_tfidf(query, top_k)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'semantic' or 'tfidf'.")


# ── Quick Test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    engine = RecommendationEngine()

    test_queries = [
        "moisturizer for dry winter skin",
        "I need something hydrating for cold weather",
        "product for glowing skin before a party"
    ]

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        print("-- Semantic --")
        print(engine.recommend(query, method="semantic", top_k=3)[["title", "score"]].to_string(index=False))
        print("-- TF-IDF --")
        print(engine.recommend(query, method="tfidf", top_k=3)[["title", "score"]].to_string(index=False))