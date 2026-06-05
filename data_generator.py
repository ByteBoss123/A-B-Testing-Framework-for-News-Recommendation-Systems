"""
data_generator.py
Simulates a real-world CNN-style news article dataset.
In production, replace with CC-News or All-The-News (Kaggle) dataset.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import hashlib

random.seed(42)
np.random.seed(42)

CATEGORIES = ["Politics", "Technology", "Sports", "Business", "Health", "Entertainment", "Science", "World"]

TOPICS = {
    "Politics": ["election", "congress", "president", "senate", "policy", "democracy", "vote", "legislation"],
    "Technology": ["AI", "machine learning", "startup", "software", "cybersecurity", "cloud", "data", "innovation"],
    "Sports": ["NBA", "NFL", "soccer", "Olympics", "championship", "athlete", "tournament", "team"],
    "Business": ["stocks", "market", "economy", "inflation", "earnings", "revenue", "CEO", "acquisition"],
    "Health": ["COVID", "vaccine", "mental health", "nutrition", "cancer", "FDA", "clinical trial", "wellness"],
    "Entertainment": ["movie", "music", "celebrity", "award", "streaming", "Netflix", "concert", "box office"],
    "Science": ["climate", "space", "research", "discovery", "NASA", "biology", "physics", "environment"],
    "World": ["war", "diplomacy", "United Nations", "trade", "refugee", "sanctions", "conflict", "peace"],
}

TEMPLATES = [
    "{topic} developments shake {category} landscape as experts weigh in",
    "Breaking: Major {topic} story emerges affecting millions worldwide",
    "Analysis: How {topic} is reshaping the future of {category}",
    "New {topic} findings challenge previous understanding in {category}",
    "Exclusive: Inside the {topic} controversy gripping {category} world",
    "Report: {topic} surge signals major shift in {category} sector",
    "Opinion: Why {topic} matters more than ever for {category}",
    "Investigation: The hidden {topic} crisis in {category}",
]


def generate_article(article_id: int) -> dict:
    category = random.choice(CATEGORIES)
    topic = random.choice(TOPICS[category])
    template = random.choice(TEMPLATES)
    title = template.format(topic=topic, category=category)

    # Simulate word count and engagement signals
    word_count = random.randint(300, 2500)
    days_ago = random.randint(0, 90)
    publish_date = datetime.now() - timedelta(days=days_ago)

    # Simulate engagement (ground truth for relevance)
    base_engagement = random.uniform(0.1, 0.9)
    category_boost = {"Technology": 0.15, "Politics": 0.1, "Health": 0.12}.get(category, 0.0)
    engagement_score = min(1.0, base_engagement + category_boost + random.uniform(-0.1, 0.1))

    return {
        "article_id": f"art_{article_id:05d}",
        "title": title,
        "category": category,
        "topic": topic,
        "word_count": word_count,
        "publish_date": publish_date.strftime("%Y-%m-%d"),
        "engagement_score": round(engagement_score, 4),
        "tags": f"{category.lower()},{topic.lower()}",
    }


def generate_user_history(user_id: int, articles: pd.DataFrame, n_interactions: int = 10) -> dict:
    """Simulate a user's reading history with category preferences."""
    preferred_categories = random.sample(CATEGORIES, k=random.randint(1, 3))
    preferred_articles = articles[articles["category"].isin(preferred_categories)]

    if len(preferred_articles) < n_interactions:
        preferred_articles = articles

    history = preferred_articles.sample(n=min(n_interactions, len(preferred_articles)))

    return {
        "user_id": f"user_{user_id:05d}",
        "preferred_categories": preferred_categories,
        "read_article_ids": history["article_id"].tolist(),
        "avg_engagement": round(history["engagement_score"].mean(), 4),
    }


def load_dataset(n_articles: int = 5000, n_users: int = 1000) -> tuple:
    """Generate full dataset. Replace with real dataset loader in production."""
    print(f"Generating {n_articles} articles and {n_users} user profiles...")

    articles = pd.DataFrame([generate_article(i) for i in range(n_articles)])
    users = [generate_user_history(i, articles) for i in range(n_users)]
    users_df = pd.DataFrame(users)

    print(f"Dataset ready: {len(articles)} articles, {len(users_df)} users")
    print(f"Category distribution:\n{articles['category'].value_counts()}\n")

    return articles, users_df


if __name__ == "__main__":
    articles, users = load_dataset()
    articles.to_csv("articles.csv", index=False)
    users.to_csv("users.csv", index=False)
    print("Saved to articles.csv and users.csv")
