import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
from sentence_transformers import SentenceTransformer

# Load everything
df = pd.read_parquet("products_clean.parquet")
embeddings = np.load("embeddings.npy")
index = faiss.read_index("faiss_index.bin")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

####################

vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True
)
tfidf_matrix = vectorizer.fit_transform(df["product_text"])

###################

def retrieve_tfidf(query, top_k=10):
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = np.argsort(scores)[::-1][:top_k]
    results = df.iloc[top_indices][["parent_asin", "title", "product_text"]].copy()
    results["score"] = scores[top_indices]
    return results[results["score"] > 0].reset_index(drop=True)


def retrieve_semantic(query, top_k=10):
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    scores, indices = index.search(query_embedding, top_k)
    results = df.iloc[indices[0]][["parent_asin", "title", "product_text"]].copy()
    results["score"] = scores[0]
    return results.reset_index(drop=True)
  
  
  ############################
  
  # Query set with associated relevant keywords
# These are proxy relevance signals — products whose title contains
# these keywords are considered relevant for that query
eval_queries = [
    {
        "query": "moisturizer for dry winter skin",
        "relevant_keywords": ["winter", "cold weather", "dry skin relief", "intense moisture"]
    },
    {
        "query": "I need something hydrating for cold weather",
        "relevant_keywords": ["winter", "cold", "hydrating", "intense moisture"]
    },
    {
        "query": "face wash that won't irritate my skin",
        "relevant_keywords": ["sensitive", "gentle", "non irritating", "fragrance free"]
    },
    {
        "query": "help with fine lines around eyes",
        "relevant_keywords": ["eye cream", "eye serum", "fine lines", "eye area"]
    },
    {
        "query": "product for glowing skin before a party",
        "relevant_keywords": ["glow", "brightening", "radiance", "illuminating"]
    },
    {
        "query": "something for oily skin in summer",
        "relevant_keywords": ["oily skin", "oily", "oil free", "for oily"]
    }
]


def is_relevant(title: str, keywords: list) -> bool:
    title_lower = title.lower()
    return any(kw.lower() in title_lower for kw in keywords)


def precision_at_k(results: pd.DataFrame, keywords: list, k: int = 5) -> float:
    top_k = results.head(k)
    relevant = top_k["title"].apply(lambda t: is_relevant(t, keywords)).sum()
    return relevant / k


# Run evaluation
K = 5
print(f"{'Query':<45} {'TF-IDF P@'+str(K):<15} {'Semantic P@'+str(K):<15}")
print("-" * 75)

tfidf_scores = []
semantic_scores = []

for item in eval_queries:
    query = item["query"]
    keywords = item["relevant_keywords"]

    tfidf_results = retrieve_tfidf(query, top_k=K)
    semantic_results = retrieve_semantic(query, top_k=K)

    tfidf_p = precision_at_k(tfidf_results, keywords, K)
    semantic_p = precision_at_k(semantic_results, keywords, K)

    tfidf_scores.append(tfidf_p)
    semantic_scores.append(semantic_p)

    print(f"{query[:44]:<45} {tfidf_p:<15.2f} {semantic_p:<15.2f}")

print("-" * 75)
print(f"{'Mean Precision@K':<45} {np.mean(tfidf_scores):<15.2f} {np.mean(semantic_scores):<15.2f}")


########################

# Show side by side for the two hardest queries
hard_queries = [
    "I need something hydrating for cold weather",
    "product for glowing skin before a party"
]

for query in hard_queries:
    print(f"\n{'='*70}")
    print(f"Query: '{query}'")
    print(f"{'='*70}")

    tfidf_res = retrieve_tfidf(query, top_k=5)
    semantic_res = retrieve_semantic(query, top_k=5)

    print(f"\n{'TF-IDF Results':<5}")
    for i, row in tfidf_res.iterrows():
        print(f"  {i+1}. {row['title'][:80]}")

    print(f"\n{'Semantic Results':<5}")
    for i, row in semantic_res.iterrows():
        print(f"  {i+1}. {row['title'][:80]}")

