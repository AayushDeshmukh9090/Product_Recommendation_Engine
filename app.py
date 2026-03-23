import numpy as np
import pandas as pd
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr

# ── Paths ─────────────────────────────────────────────────────────────────────
PROCESSED_PATH         = "data/processed/products_clean.parquet"
EMBEDDINGS_PATH        = "artifacts/embeddings.npy"
FAISS_INDEX_PATH       = "artifacts/faiss_index.bin"
TFIDF_VECTORIZER_PATH  = "artifacts/tfidf_vectorizer.pkl"
TFIDF_MATRIX_PATH      = "artifacts/tfidf_matrix.pkl"
MODEL_NAME             = "sentence-transformers/all-MiniLM-L6-v2"


# ── Engine ────────────────────────────────────────────────────────────────────
class RecommendationEngine:
    def __init__(self):
        print("Loading artifacts...")
        self.df           = pd.read_parquet(PROCESSED_PATH)
        self.embeddings   = np.load(EMBEDDINGS_PATH)
        self.faiss_index  = faiss.read_index(FAISS_INDEX_PATH)
        self.model        = SentenceTransformer(MODEL_NAME)

        with open(TFIDF_VECTORIZER_PATH, "rb") as f:
            self.vectorizer = pickle.load(f)
        with open(TFIDF_MATRIX_PATH, "rb") as f:
            self.tfidf_matrix = pickle.load(f)

        print(f"Engine ready. {len(self.df)} products loaded.")

    def recommend_semantic(self, query: str, top_k: int = 5) -> pd.DataFrame:
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        scores, indices = self.faiss_index.search(query_embedding, top_k)
        results = self.df.iloc[indices[0]][["parent_asin", "title"]].copy()
        results["score"]  = scores[0]
        results["method"] = "semantic"
        return results.reset_index(drop=True)

    def recommend_tfidf(self, query: str, top_k: int = 5) -> pd.DataFrame:
        query_vec    = self.vectorizer.transform([query])
        scores       = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices  = np.argsort(scores)[::-1][:top_k]
        results      = self.df.iloc[top_indices][["parent_asin", "title"]].copy()
        results["score"]  = scores[top_indices]
        results["method"] = "tfidf"
        return results[results["score"] > 0].reset_index(drop=True)

    def recommend(self, query: str, method: str = "semantic", top_k: int = 5) -> pd.DataFrame:
        if method == "semantic":
            return self.recommend_semantic(query, top_k)
        elif method == "tfidf":
            return self.recommend_tfidf(query, top_k)
        else:
            raise ValueError(f"Unknown method: {method}")


engine = RecommendationEngine()


# ── Gradio UI ─────────────────────────────────────────────────────────────────
def format_results(results: pd.DataFrame) -> str:
    if results.empty:
        return "No results found."
    output = ""
    for i, row in results.iterrows():
        output += f"**{i+1}. {row['title']}**\n"
        output += f"Similarity Score: `{row['score']:.4f}`\n\n"
    return output.strip()


def search(query: str, method: str, top_k: int):
    if not query.strip():
        return "Please enter a search query."
    results = engine.recommend(query, method=method, top_k=int(top_k))
    return format_results(results)


with gr.Blocks(title="Product Recommendation Engine") as demo:
    gr.Markdown("""
    # Product Recommendation Engine
    ### Semantic Search vs TF-IDF Keyword Retrieval
    Search across **78,786 beauty products** using natural language.
    Compare how semantic search understands intent vs keyword matching.
    """)

    with gr.Row():
        with gr.Column(scale=3):
            query_input = gr.Textbox(
                label="Search Query",
                placeholder="e.g. something hydrating for cold weather...",
                lines=2
            )
        with gr.Column(scale=1):
            method_input = gr.Radio(
                choices=["semantic", "tfidf"],
                value="semantic",
                label="Retrieval Method"
            )
            topk_input = gr.Slider(
                minimum=1,
                maximum=10,
                value=5,
                step=1,
                label="Number of Results"
            )

    search_btn = gr.Button("Search", variant="primary")
    output = gr.Markdown(label="Results")

    gr.Examples(
        examples=[
            ["moisturizer for dry winter skin", "semantic", 5],
            ["I need something hydrating for cold weather", "semantic", 5],
            ["I need something hydrating for cold weather", "tfidf", 5],
            ["product for glowing skin before a party", "semantic", 5],
            ["product for glowing skin before a party", "tfidf", 5],
            ["gentle face wash for sensitive skin", "semantic", 5],
            ["help with fine lines around eyes", "semantic", 5],
        ],
        inputs=[query_input, method_input, topk_input],
        label="Try these — compare semantic vs TF-IDF on same query"
    )

    search_btn.click(
        fn=search,
        inputs=[query_input, method_input, topk_input],
        outputs=output
    )

demo.launch()