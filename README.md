# Product Recommendation Engine
### Semantic Search vs. Keyword Retrieval

---

## Problem Statement

Traditional keyword-based recommendation engines fail when users search in natural language. A query like *"I need something hydrating for cold weather"* should return skincare moisturizers not ear warmers. This project builds a semantic product recommendation engine that understands **intent**, not just keywords, using dense vector embeddings and approximate nearest neighbor search.

---

## Live Demo

[HuggingFace Spaces : Link after deployment](https://huggingface.co/spaces/Aayush814/Product_Recommendation_Engine)

---


## Architecture

```
Raw Product Data (Amazon Reviews 2023 — All Beauty)
            │
            ▼
┌─────────────────────────┐
│   Module 1              │
│   Data Processing       │  → Cleans, filters, constructs product_text
│   src/data_processing.py│    78,786 products retained
└───────────┬─────────────┘
            │
            ├──────────────────────────────────┐
            ▼                                  ▼
┌─────────────────────────┐      ┌─────────────────────────┐
│   Module 2              │      │   Module 3              │
│   TF-IDF Baseline       │      │   Embedding Generation  │
│   src/baseline.py       │      │   src/embeddings.py     │
│   Sparse vectors        │      │   Dense 384-dim vectors │
│   Vocabulary matching   │      │   all-MiniLM-L6-v2      │
└───────────┬─────────────┘      └───────────┬─────────────┘
            │                                │
            ▼                                ▼
┌─────────────────────────┐      ┌─────────────────────────┐
│   TF-IDF Matrix         │      │   Module 4              │
│   artifacts/            │      │   FAISS Index           │
│   tfidf_matrix.pkl      │      │   src/vector_store.py   │
│   tfidf_vectorizer.pkl  │      │   faiss_index.bin       │
└───────────┬─────────────┘      └───────────┬─────────────┘
            │                                │
            └──────────────┬─────────────────┘
                           ▼
            ┌─────────────────────────┐
            │   Module 5              │
            │   Evaluation            │
            │   src/evaluation.py     │
            │   Precision@K + Qual.   │
            └───────────┬─────────────┘
                        ▼
            ┌─────────────────────────┐
            │   Module 6              │
            │   Retrieval Interface   │
            │   src/retrieval.py      │
            └───────────┬─────────────┘
                        ▼
            ┌─────────────────────────┐
            │   Module 7              │
            │   Gradio UI             │
            │   app.py                │
            │   HuggingFace Spaces    │
            └─────────────────────────┘
```

---

## Tech Stack

| Tool | Role | Why |
|---|---|---|
| Pandas | Data processing | Standard, fast, typed with parquet |
| PyArrow / Parquet | Storage format | Columnar, compressed, type-preserving |
| Scikit-learn TF-IDF | Baseline retrieval | Classical IR baseline for comparison |
| Sentence-Transformers | Embedding generation | Production-grade semantic encoder, CPU-friendly |
| `all-MiniLM-L6-v2` | Embedding model | 384-dim, distilled BERT, fast inference |
| FAISS `IndexFlatIP` | Vector similarity search | Industry standard ANN search (Meta) |
| Gradio | UI layer | Fast, shareable, demonstrable in interviews |


---

## Dataset

**Amazon Reviews 2023 — All Beauty**
- Source: [McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)
- Raw products: 112,590
- After quality filtering (>100 chars): **78,786 products**
- Fields used: `title`, `features`, `description`, `categories`

---

## Data Processing

Each product is represented as a single concatenated text field:

```
Title: {title}. {categories}. {features}. {description}
```

**Key decisions:**
- Fields ordered by signal strength — title first, description last (handles 256-token truncation)
- Products with <100 character `product_text` dropped — bare titles produce weak embeddings
- 10.26% of products exceed the 256-token limit of `all-MiniLM-L6-v2` — truncated at inference, acceptable rate

---

## Evaluation Results

### Quantitative — Precision@5

| Query | TF-IDF P@5 | Semantic P@5 |
|---|---|---|
| moisturizer for dry winter skin | 0.20 | **0.40** |
| I need something hydrating for cold weather | 1.00 | 0.40 |
| face wash that won't irritate my skin | 0.40 | **0.60** |
| help with fine lines around eyes | 0.60 | **1.00** |
| product for glowing skin before a party | 1.00 | 1.00 |
| something for oily skin in summer | 1.00 | 0.60 |
| **Mean** | 0.70 | 0.67 |

> **Note on metric limitations:** Precision@K uses keyword proxy for relevance — this inherently favors TF-IDF since relevance is defined by vocabulary overlap. The qualitative comparison below tells a more accurate story.

---

### Qualitative — Where TF-IDF Fails

**Query: "I need something hydrating for cold weather"**

| Rank | TF-IDF | Semantic Search |
|---|---|---|
| 1 | Ear Warmer Headband (Winter) | Clinique Moisture Surge 72hr Hydrator |
| 2 | Cold Weather Ear Muffs | Tarte Rainforest H2O Hydrating Boost |
| 3 | Fleece Winter Headband | Live Clean Hydrating Conditioner |

TF-IDF matched "cold weather" literally and returned accessories. Semantic search understood the user wants a **skincare hydration product**.

---

**Query: "product for glowing skin before a party"**

| Rank | TF-IDF | Semantic Search |
|---|---|---|
| 1 | Generic Facial Oil | Skin Glow Natural Dark Spot Remover |
| 2 | Organic Facial Scrub | Everyday Beauty Glow Day Cream |
| 3 | Orange Essential Oil | Glow on 5th – The Perfect Canvas Mask |

TF-IDF returned unrelated products with scattered keyword matches. Semantic search returned products explicitly about **glowing skin**.

---

## Key Design Decisions

**Why embeddings over keyword search?**
A single dense vector captures cross-field semantic relationships. "Moisturizing" in features and "dry skin" in description are understood as related concepts — not independent keyword matches.

**Why normalize embeddings?**
L2 normalization reduces cosine similarity to a dot product (denominator = 1), enabling FAISS `IndexFlatIP` to use optimized BLAS matrix operations — critical at production scale.

**Why build a TF-IDF baseline first?**
To justify the complexity of semantic search. Without a baseline, there is no evidence the approach is better. The qualitative comparison proves semantic search handles natural language queries that TF-IDF fails on catastrophically.

**Why `all-MiniLM-L6-v2`?**
Distilled from a 768-dim model to 384 dimensions — 80% of the performance at 20% of the compute. Fast enough for CPU inference, small enough (~80MB) for free-tier deployment.

---

## Project Structure

```
product-recommendation-engine/
│
├── data/
│   ├── raw/                          # Downloaded once, gitignored
│   └── processed/
│       └── products_clean.parquet
│
├── src/
│   ├── data_processing.py
│   ├── baseline.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── evaluation.py
│   └── retrieval.py
│
├── artifacts/                        # Gitignored, pushed via Git LFS to HF Spaces
│   ├── tfidf_vectorizer.pkl
│   ├── tfidf_matrix.pkl
│   ├── faiss_index.bin
│   └── embeddings.npy
│
├── notebooks/
│   └── exploration.ipynb
│
├── app.py
├── requirements.txt
└── README.md
```

---

## How to Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/product-recommendation-engine
cd product-recommendation-engine

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download and process data
python src/data_processing.py

# 4. Build TF-IDF baseline
python src/baseline.py

# 5. Generate embeddings (GPU recommended, CPU works)
python src/embeddings.py

# 6. Build FAISS index
python src/vector_store.py

# 7. Launch Gradio UI
python app.py
```

---

## Requirements

```
pandas
numpy
pyarrow
scikit-learn
sentence-transformers
faiss-cpu
gradio
transformers
```

