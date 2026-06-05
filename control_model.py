"""
control_model.py — Baseline Recommender (Control Group)
Keyword + category matching. Represents how most legacy news platforms work.
This is the "before" in the A/B test.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict
import time


class KeywordRecommender:
    """
    TF-IDF keyword-based article recommender.
    Control group: simulates legacy CNN recommendation engine.
    
    Real-world analog: Category + keyword matching without semantic understanding.
    Problem it solves: Fast but misses semantic relevance — 
    e.g., 'AI regulation' and 'machine learning policy' are related but keyword match fails.
    """

    def __init__(self, top_k: int = 10):
        self.top_k = top_k
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words="english",
            ngram_range=(1, 2)
        )
        self.article_vectors = None
        self.articles = None
        self.model_name = "KeywordTFIDF_v1"

    def fit(self, articles: pd.DataFrame):
        """Build TF-IDF index from article titles and tags."""
        self.articles = articles.reset_index(drop=True)
        corpus = (articles["title"] + " " + articles["tags"]).fillna("")
        self.article_vectors = self.vectorizer.fit_transform(corpus)
        print(f"[Control] TF-IDF index built: {self.article_vectors.shape}")
        return self

    def recommend(self, user_history: Dict, n: int = 10) -> List[Dict]:
        """
        Given user's reading history, recommend top-N articles.
        Strategy: Average TF-IDF vectors of read articles → find nearest neighbors.
        """
        start_time = time.time()

        read_ids = user_history.get("read_article_ids", [])
        read_mask = self.articles["article_id"].isin(read_ids)
        read_indices = self.articles[read_mask].index.tolist()

        if not read_indices:
            # Cold start: return top engaging articles
            recs = self.articles.nlargest(n, "engagement_score")
            latency = time.time() - start_time
            return self._format_results(recs, latency, strategy="cold_start")

        # Compute user profile vector
        user_vector = np.asarray(self.article_vectors[read_indices].mean(axis=0))
        similarities = cosine_similarity(user_vector, self.article_vectors).flatten()

        # Exclude already read
        similarities[read_indices] = -1

        top_indices = similarities.argsort()[::-1][:n]
        recs = self.articles.iloc[top_indices].copy()
        recs["similarity_score"] = similarities[top_indices]

        latency = time.time() - start_time
        return self._format_results(recs, latency, strategy="tfidf_similarity")

    def _format_results(self, recs: pd.DataFrame, latency: float, strategy: str) -> List[Dict]:
        results = []
        for rank, (_, row) in enumerate(recs.iterrows(), 1):
            results.append({
                "rank": rank,
                "article_id": row["article_id"],
                "title": row["title"],
                "category": row["category"],
                "engagement_score": row["engagement_score"],
                "similarity_score": row.get("similarity_score", 0.0),
                "model": self.model_name,
                "strategy": strategy,
                "latency_ms": round(latency * 1000, 2),
            })
        return results

    def batch_recommend(self, users: pd.DataFrame, n: int = 10) -> pd.DataFrame:
        """Batch inference for all users — used in experiment evaluation."""
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
    model = KeywordRecommender(top_k=10)
    model.fit(articles)

    sample_user = users.iloc[0].to_dict()
    recs = model.recommend(sample_user, n=5)
    print("\nSample recommendations (Control):")
    for r in recs:
        print(f"  [{r['rank']}] {r['title'][:60]}... | score: {r['engagement_score']:.3f}")
