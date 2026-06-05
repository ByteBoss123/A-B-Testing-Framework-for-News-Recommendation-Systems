"""
treatment_model.py — Semantic Recommender (Treatment Group)
Uses TF-IDF + SVD (LSA) as a lightweight semantic model.
In production: replace SVD with actual BERT sentence embeddings.
Represents the "after" — smarter semantic understanding of article content.

Real-world problem solved: 
Traditional keyword matching fails when two articles cover the same concept 
with different vocabulary. Semantic models capture latent meaning, improving 
recommendation relevance even when exact keywords don't match.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict
import time


class SemanticRecommender:
    """
    Latent Semantic Analysis (LSA) recommender — semantic treatment model.
    
    Upgrade over KeywordRecommender:
    - Captures latent topics, not just surface keywords
    - Handles vocabulary mismatch (synonym-aware)
    - Better cold-start via semantic category clustering
    
    Production upgrade path: swap SVD for BERT/sentence-transformers embeddings.
    """

    def __init__(self, top_k: int = 10, n_components: int = 150):
        self.top_k = top_k
        self.n_components = n_components
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words="english",
            ngram_range=(1, 3),
            sublinear_tf=True  # log normalization — better for news
        )
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.article_embeddings = None
        self.articles = None
        self.model_name = "SemanticLSA_v1"

    def fit(self, articles: pd.DataFrame):
        """Build semantic embedding space from articles."""
        self.articles = articles.reset_index(drop=True)

        # Richer text representation: title + tags + category
        corpus = (
            articles["title"] + " " +
            articles["tags"] + " " +
            articles["category"] + " " +
            articles["topic"]
        ).fillna("")

        tfidf_matrix = self.vectorizer.fit_transform(corpus)
        raw_embeddings = self.svd.fit_transform(tfidf_matrix)
        # L2 normalize for cosine similarity
        self.article_embeddings = normalize(raw_embeddings, norm="l2")

        explained_var = self.svd.explained_variance_ratio_.sum()
        print(f"[Treatment] Semantic index built: {self.article_embeddings.shape} | "
              f"Explained variance: {explained_var:.1%}")
        return self

    def _get_user_embedding(self, read_indices: List[int]) -> np.ndarray:
        """
        Compute user preference embedding from reading history.
        Uses weighted average — recent articles weighted higher (recency bias).
        """
        if not read_indices:
            return None

        weights = np.linspace(0.5, 1.0, len(read_indices))  # recency weighting
        weighted_vecs = self.article_embeddings[read_indices] * weights[:, np.newaxis]
        user_vec = weighted_vecs.mean(axis=0, keepdims=True)
        return normalize(user_vec, norm="l2")

    def recommend(self, user_history: Dict, n: int = 10) -> List[Dict]:
        """
        Semantic recommendation using LSA embedding similarity.
        Captures conceptual relatedness beyond keyword overlap.
        """
        start_time = time.time()

        read_ids = user_history.get("read_article_ids", [])
        read_mask = self.articles["article_id"].isin(read_ids)
        read_indices = self.articles[read_mask].index.tolist()

        if not read_indices:
            # Semantic cold start: use category preference if available
            preferred = user_history.get("preferred_categories", [])
            if preferred:
                candidates = self.articles[self.articles["category"].isin(preferred)]
                recs = candidates.nlargest(n, "engagement_score")
            else:
                recs = self.articles.nlargest(n, "engagement_score")
            latency = time.time() - start_time
            return self._format_results(recs, latency, strategy="semantic_cold_start")

        user_embedding = self._get_user_embedding(read_indices)
        similarities = cosine_similarity(user_embedding, self.article_embeddings).flatten()

        # Exclude already read articles
        similarities[read_indices] = -1

        # Diversity penalty: slightly penalize articles from over-represented categories
        top_category = self.articles.iloc[read_indices]["category"].mode()
        if not top_category.empty:
            dominant_cat = top_category.iloc[0]
            dominant_mask = self.articles["category"] == dominant_cat
            similarities[dominant_mask] *= 0.92  # 8% diversity nudge

        top_indices = similarities.argsort()[::-1][:n]
        recs = self.articles.iloc[top_indices].copy()
        recs["similarity_score"] = similarities[top_indices]

        latency = time.time() - start_time
        return self._format_results(recs, latency, strategy="semantic_similarity")

    def _format_results(self, recs: pd.DataFrame, latency: float, strategy: str) -> List[Dict]:
        results = []
        for rank, (_, row) in enumerate(recs.iterrows(), 1):
            results.append({
                "rank": rank,
                "article_id": row["article_id"],
                "title": row["title"],
                "category": row["category"],
                "engagement_score": row["engagement_score"],
                "similarity_score": float(row.get("similarity_score", 0.0)),
                "model": self.model_name,
                "strategy": strategy,
                "latency_ms": round(latency * 1000, 2),
            })
        return results

    def batch_recommend(self, users: pd.DataFrame, n: int = 10) -> pd.DataFrame:
        """Batch inference for all users."""
        all_recs = []
        for _, user in users.iterrows():
            recs = self.recommend(user.to_dict(), n=n)
            for rec in recs:
                rec["user_id"] = user["user_id"]
            all_recs.extend(recs)
        return pd.DataFrame(all_recs)


if __name__ == "__main__":
    import sys
    sys.path.append("..")
    from data.data_generator import load_dataset

    articles, users = load_dataset(n_articles=2000, n_users=100)
    model = SemanticRecommender()
    model.fit(articles)

    sample_user = users.iloc[0].to_dict()
    recs = model.recommend(sample_user, n=5)
    print("\nSample recommendations (Treatment):")
    for r in recs:
        print(f"  [{r['rank']}] {r['title'][:60]}... | score: {r['engagement_score']:.3f}")
