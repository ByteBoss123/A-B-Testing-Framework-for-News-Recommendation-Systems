# News Recommendation A/B Testing Framework

> **Real-world problem:** A news platform publishes 500+ articles daily. How do you know if a new recommendation algorithm actually improves reader engagement — before deploying it to millions of users?

This framework provides the answer: a production-grade A/B testing pipeline that statistically validates recommendation algorithm changes using real news article data.

---

## Motivation

Every major news platform (CNN, NYT, BBC) faces this problem daily:

- Engineering builds a new recommendation model
- It looks better on offline metrics
- But does it **actually** improve reader engagement?
- How do you know with **statistical confidence** before rollout?

Without a rigorous testing framework, teams either:
1. Ship changes blindly and hope for the best
2. Run underpowered tests that can't detect real improvements
3. Miss regressions until they affect millions of readers

This framework solves that.

---

## Architecture

```
news_ab_framework/
├── data/
│   └── data_generator.py       # Simulated CNN-style news dataset (replace with CC-News)
├── models/
│   ├── control_model.py        # TF-IDF keyword recommender (baseline)
│   └── treatment_model.py      # LSA semantic recommender (treatment)
├── evaluation/
│   └── ab_experiment.py        # Core A/B engine: assignment, metrics, stats
├── tracking/
│   └── mlflow_tracker.py       # MLflow experiment logging + visualizations
└── run_experiment.py           # Single entry point
```

---

## Experiment Design

### User Assignment
- **Deterministic hash-based assignment** (user_id + experiment_id → MD5 → bucket)
- Same user always gets same group — no assignment drift between runs
- Mirrors production systems like Optimizely and Netflix's experimentation platform

### Models Compared

| | Control | Treatment |
|---|---|---|
| **Model** | TF-IDF Keyword Matching | LSA Semantic Similarity |
| **Approach** | Surface keyword overlap | Latent semantic topics |
| **Limitation** | Misses synonyms, context | Requires more compute |
| **Analog** | Legacy news recommendation | Modern semantic search |

### Metrics

| Metric | What it measures | CNN analog |
|---|---|---|
| `precision@k` | Relevant articles in top-K | Click quality |
| `ndcg@k` | Ranking quality (best articles first) | Engagement order |
| `simulated_ctr` | Position-weighted engagement | Click-through rate |
| `avg_engagement` | Mean relevance score | Time-on-page |
| `category_coverage` | Recommendation diversity | Content breadth |

### Statistical Testing
- **Welch's t-test** — parametric, handles unequal variances
- **Mann-Whitney U** — non-parametric, robust for skewed engagement distributions
- **Cohen's d** — effect size (how practically significant is the difference?)
- **95% Confidence Intervals** — range of true effect
- **Significance threshold:** α = 0.05

---

## Quick Start

```bash
# Install dependencies
pip install mlflow scikit-learn pandas numpy scipy matplotlib seaborn

# Run with defaults (3000 articles, 800 users)
python run_experiment.py

# Custom configuration
python run_experiment.py --n-articles 5000 --n-users 2000 --top-k 10 --split 0.5

# Skip MLflow tracking
python run_experiment.py --no-tracking

# View results in MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

---

## Sample Output

```
============================================================
EXPERIMENT RESULTS: news_ab_20260605
============================================================
Control (n=408): KeywordTFIDF_v1
Treatment (n=392): SemanticLSA_v1

Metric                    Control  Treatment    Lift%    p-value   Sig?
------------------------------------------------------------------------
precision_at_k             0.4262     0.4541     +6.8%     0.0060      ✓
ndcg_at_k                  0.8646     0.8666     +0.2%     0.6654      ✗
simulated_ctr              0.5414     0.5546     +2.4%     0.2234      ✗
avg_engagement             0.5381     0.5546     +3.1%     0.0225      ✓
category_coverage          0.1455     0.1448     -0.5%     0.6864      ✗

🏆 VERDICT: Treatment wins — significant improvement on 2 metrics
============================================================
```

---

## MLflow Tracking

Every experiment run is logged with:
- **Parameters:** model names, traffic split, top-K, sample sizes
- **Metrics:** all quality scores + statistical test results + lift percentages
- **Artifacts:** metric comparison chart, significance forest plot, distribution plots

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open http://localhost:5000
```

This answers: *"Which experiment run produced that precision lift 3 weeks ago?"*

---

## Production Upgrade Path

| Component | Current | Production |
|---|---|---|
| Dataset | Simulated 3K articles | CC-News / All-The-News (200K+) |
| Treatment model | LSA (TF-IDF + SVD) | BERT / sentence-transformers |
| User data | Simulated history | Real clickstream data |
| Metrics | Simulated engagement | Actual CTR, time-on-page, shares |
| Assignment | Hash-based | Feature flag system (LaunchDarkly) |
| Tracking | MLflow local | MLflow on managed infra |

---

## Key Engineering Decisions

**Why hash-based user assignment?**  
Ensures reproducibility. Running the same experiment twice produces identical group assignments, making results comparable across runs — critical for debugging and auditing.

**Why Mann-Whitney U over t-test?**  
News engagement scores are heavily right-skewed (most articles get low engagement, viral articles are outliers). Non-parametric tests are more robust under these conditions.

**Why LSA over BERT for treatment?**  
LSA runs on CPU without GPU dependencies, making the framework accessible without cloud infrastructure. The architecture is identical — swap `SemanticRecommender.fit()` with BERT embeddings for production.

---

## Resume Bullet

> Built production-grade A/B testing framework to validate news recommendation algorithms — compared TF-IDF keyword matching vs. semantic LSA similarity across 3,000 articles and 800 simulated users, measuring Precision@K, NDCG@K, and simulated CTR with Mann-Whitney U significance testing; integrated MLflow experiment tracking for reproducible model comparison.
