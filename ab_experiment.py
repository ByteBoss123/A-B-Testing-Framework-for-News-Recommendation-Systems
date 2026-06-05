"""
ab_experiment.py — Core A/B Testing Engine
The heart of the framework. Handles:
- User assignment to control/treatment
- Metric computation
- Statistical significance testing
- Result reporting

Real-world problem: Before deploying a new recommendation algorithm to CNN's 
millions of readers, engineers need statistical proof that it performs better — 
not just a gut feeling. This engine provides that proof.
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import mannwhitneyu, ttest_ind, chi2_contingency
from typing import Dict, Tuple, Optional
import hashlib
import json
from datetime import datetime


class ExperimentConfig:
    """Experiment configuration — mirrors production A/B test setup."""

    def __init__(
        self,
        experiment_id: str,
        control_name: str = "KeywordTFIDF_v1",
        treatment_name: str = "SemanticLSA_v1",
        traffic_split: float = 0.5,
        top_k: int = 10,
        significance_level: float = 0.05,
        min_effect_size: float = 0.05,  # Minimum detectable effect (5%)
    ):
        self.experiment_id = experiment_id
        self.control_name = control_name
        self.treatment_name = treatment_name
        self.traffic_split = traffic_split
        self.top_k = top_k
        self.significance_level = significance_level
        self.min_effect_size = min_effect_size
        self.created_at = datetime.now().isoformat()


class UserAssigner:
    """
    Deterministic user assignment to control/treatment.
    
    Uses hash-based assignment so:
    - Same user always gets same group (consistency)
    - No randomness drift between runs
    - Reproducible experiments
    
    This is how production A/B systems work (e.g., Optimizely, Netflix Experimentation).
    """

    def __init__(self, experiment_id: str, split: float = 0.5):
        self.experiment_id = experiment_id
        self.split = split

    def assign(self, user_id: str) -> str:
        """Hash user_id + experiment_id for stable, deterministic assignment."""
        hash_input = f"{self.experiment_id}:{user_id}".encode()
        hash_value = int(hashlib.md5(hash_input).hexdigest(), 16)
        bucket = (hash_value % 10000) / 10000.0
        return "treatment" if bucket < self.split else "control"

    def assign_batch(self, user_ids: pd.Series) -> pd.Series:
        return user_ids.apply(self.assign)


class MetricsComputer:
    """
    Computes recommendation quality metrics.
    
    Metrics mirror real CNN KPIs:
    - Precision@K: Of top-K recommendations, how many are truly relevant?
    - NDCG@K: Ranking quality — are the best articles ranked highest?
    - Coverage: Diversity of recommendation pool
    - Simulated CTR: Engagement-weighted click probability
    """

    def __init__(self, relevance_threshold: float = 0.6):
        self.relevance_threshold = relevance_threshold

    def precision_at_k(self, recommendations: pd.DataFrame, k: int = 10) -> float:
        """Proportion of top-K recommendations above relevance threshold."""
        top_k = recommendations[recommendations["rank"] <= k]
        if len(top_k) == 0:
            return 0.0
        relevant = (top_k["engagement_score"] >= self.relevance_threshold).sum()
        return relevant / len(top_k)

    def ndcg_at_k(self, recommendations: pd.DataFrame, k: int = 10) -> float:
        """
        Normalized Discounted Cumulative Gain.
        Rewards putting the most relevant articles at the top of the list.
        """
        top_k = recommendations[recommendations["rank"] <= k].sort_values("rank")
        if len(top_k) == 0:
            return 0.0

        relevances = top_k["engagement_score"].values

        # DCG
        dcg = relevances[0] + sum(
            rel / np.log2(i + 2) for i, rel in enumerate(relevances[1:], 1)
        )

        # Ideal DCG (perfect ranking)
        ideal = sorted(relevances, reverse=True)
        idcg = ideal[0] + sum(
            rel / np.log2(i + 2) for i, rel in enumerate(ideal[1:], 1)
        ) if len(ideal) > 1 else ideal[0]

        return dcg / idcg if idcg > 0 else 0.0

    def simulated_ctr(self, recommendations: pd.DataFrame, k: int = 5) -> float:
        """
        Simulated Click-Through Rate for top-K results.
        Models position bias: lower-ranked articles get fewer clicks.
        """
        top_k = recommendations[recommendations["rank"] <= k].sort_values("rank")
        if len(top_k) == 0:
            return 0.0

        position_weights = [1 / np.log2(rank + 1) for rank in range(1, len(top_k) + 1)]
        weighted_engagement = sum(
            w * e for w, e in zip(position_weights, top_k["engagement_score"])
        )
        return weighted_engagement / sum(position_weights)

    def category_coverage(self, recommendations: pd.DataFrame) -> float:
        """Diversity: fraction of available categories covered."""
        n_categories_total = 8  # CNN has 8 main categories
        n_covered = recommendations["category"].nunique()
        return n_covered / n_categories_total

    def compute_all(self, recommendations: pd.DataFrame, k: int = 10) -> Dict:
        return {
            "precision_at_k": round(self.precision_at_k(recommendations, k), 4),
            "ndcg_at_k": round(self.ndcg_at_k(recommendations, k), 4),
            "simulated_ctr": round(self.simulated_ctr(recommendations, k=5), 4),
            "category_coverage": round(self.category_coverage(recommendations), 4),
            "avg_engagement": round(recommendations["engagement_score"].mean(), 4),
            "avg_latency_ms": round(recommendations["latency_ms"].mean(), 2),
        }


class StatisticalTester:
    """
    Rigorous statistical testing for A/B experiment results.
    
    Uses both parametric (t-test) and non-parametric (Mann-Whitney U) tests.
    Non-parametric is more robust for skewed engagement distributions — 
    which is common in real news data (most articles get low engagement, few go viral).
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def test_metric(
        self,
        control_values: np.ndarray,
        treatment_values: np.ndarray,
        metric_name: str,
    ) -> Dict:
        """Run statistical tests and compute effect size."""

        # Welch's t-test (unequal variance)
        t_stat, p_value_ttest = ttest_ind(
            control_values, treatment_values, equal_var=False, alternative="two-sided"
        )

        # Mann-Whitney U (non-parametric, more robust)
        u_stat, p_value_mw = mannwhitneyu(
            control_values, treatment_values, alternative="two-sided"
        )

        # Cohen's d effect size
        pooled_std = np.sqrt(
            (np.std(control_values, ddof=1) ** 2 + np.std(treatment_values, ddof=1) ** 2) / 2
        )
        cohens_d = (np.mean(treatment_values) - np.mean(control_values)) / pooled_std if pooled_std > 0 else 0

        # Relative lift
        control_mean = np.mean(control_values)
        treatment_mean = np.mean(treatment_values)
        relative_lift = ((treatment_mean - control_mean) / control_mean * 100) if control_mean > 0 else 0

        # 95% confidence interval for the difference
        se = np.sqrt(np.var(treatment_values, ddof=1) / len(treatment_values) +
                     np.var(control_values, ddof=1) / len(control_values))
        diff = treatment_mean - control_mean
        ci_low = diff - 1.96 * se
        ci_high = diff + 1.96 * se

        significant = p_value_mw < self.alpha

        return {
            "metric": metric_name,
            "control_mean": round(control_mean, 4),
            "treatment_mean": round(treatment_mean, 4),
            "relative_lift_pct": round(relative_lift, 2),
            "p_value_ttest": round(p_value_ttest, 4),
            "p_value_mannwhitney": round(p_value_mw, 4),
            "cohens_d": round(cohens_d, 4),
            "ci_95_low": round(ci_low, 4),
            "ci_95_high": round(ci_high, 4),
            "significant": significant,
            "conclusion": (
                f"Treatment {'significantly' if significant else 'does NOT significantly'} "
                f"{'improves' if relative_lift > 0 else 'reduces'} {metric_name} "
                f"by {abs(relative_lift):.1f}% (p={p_value_mw:.4f})"
            ),
        }

    def full_report(
        self,
        control_metrics: pd.DataFrame,
        treatment_metrics: pd.DataFrame,
        metrics: list,
    ) -> pd.DataFrame:
        """Run tests across all metrics and return summary table."""
        results = []
        for metric in metrics:
            if metric in control_metrics.columns and metric in treatment_metrics.columns:
                result = self.test_metric(
                    control_metrics[metric].values,
                    treatment_metrics[metric].values,
                    metric,
                )
                results.append(result)
        return pd.DataFrame(results)


class ABExperiment:
    """
    Main experiment orchestrator.
    Ties together: assignment → inference → metrics → statistics → reporting.
    """

    def __init__(self, config: ExperimentConfig, control_model, treatment_model):
        self.config = config
        self.control_model = control_model
        self.treatment_model = treatment_model
        self.assigner = UserAssigner(config.experiment_id, config.traffic_split)
        self.metrics_computer = MetricsComputer()
        self.tester = StatisticalTester(alpha=config.significance_level)
        self.results_ = None

    def run(self, users: pd.DataFrame, articles: pd.DataFrame) -> Dict:
        """
        Full experiment run:
        1. Assign users to groups
        2. Get recommendations from each model
        3. Compute per-user metrics
        4. Run statistical tests
        5. Return structured results
        """
        print(f"\n{'='*60}")
        print(f"Running Experiment: {self.config.experiment_id}")
        print(f"{'='*60}")

        # Step 1: Assign users
        users = users.copy()
        users["group"] = self.assigner.assign_batch(users["user_id"])
        control_users = users[users["group"] == "control"]
        treatment_users = users[users["group"] == "treatment"]
        print(f"Users assigned — Control: {len(control_users)}, Treatment: {len(treatment_users)}")

        # Step 2: Run inference
        print("Running control model inference...")
        control_recs = self.control_model.batch_recommend(control_users, n=self.config.top_k)

        print("Running treatment model inference...")
        treatment_recs = self.treatment_model.batch_recommend(treatment_users, n=self.config.top_k)

        # Step 3: Compute per-user metrics
        print("Computing metrics...")
        control_metrics = self._compute_per_user_metrics(control_recs, control_users)
        treatment_metrics = self._compute_per_user_metrics(treatment_recs, treatment_users)

        # Step 4: Statistical tests
        print("Running statistical tests...")
        metric_cols = ["precision_at_k", "ndcg_at_k", "simulated_ctr", "avg_engagement", "category_coverage"]
        stats_report = self.tester.full_report(control_metrics, treatment_metrics, metric_cols)

        # Step 5: Package results
        self.results_ = {
            "experiment_id": self.config.experiment_id,
            "config": vars(self.config),
            "sample_sizes": {
                "control": len(control_users),
                "treatment": len(treatment_users),
            },
            "control_summary": control_metrics[metric_cols].mean().round(4).to_dict(),
            "treatment_summary": treatment_metrics[metric_cols].mean().round(4).to_dict(),
            "statistical_tests": stats_report.to_dict(orient="records"),
            "control_metrics_df": control_metrics,
            "treatment_metrics_df": treatment_metrics,
            "stats_df": stats_report,
            "winner": self._determine_winner(stats_report),
        }

        self._print_summary()
        return self.results_

    def _compute_per_user_metrics(self, recs: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
        """Compute metrics for each individual user."""
        user_metrics = []
        for user_id in users["user_id"]:
            user_recs = recs[recs["user_id"] == user_id]
            if len(user_recs) == 0:
                continue
            m = self.metrics_computer.compute_all(user_recs, k=self.config.top_k)
            m["user_id"] = user_id
            user_metrics.append(m)
        return pd.DataFrame(user_metrics)

    def _determine_winner(self, stats_report: pd.DataFrame) -> str:
        """Determine overall winner based on significant improvements."""
        significant = stats_report[stats_report["significant"]]
        if len(significant) == 0:
            return "No significant difference — keep control"
        positive = significant[significant["relative_lift_pct"] > 0]
        if len(positive) >= 2:
            return f"Treatment wins — significant improvement on {len(positive)} metrics"
        return "Inconclusive — mixed results"

    def _print_summary(self):
        """Print human-readable experiment summary."""
        r = self.results_
        print(f"\n{'='*60}")
        print(f"EXPERIMENT RESULTS: {r['experiment_id']}")
        print(f"{'='*60}")
        print(f"Control (n={r['sample_sizes']['control']}): {self.config.control_name}")
        print(f"Treatment (n={r['sample_sizes']['treatment']}): {self.config.treatment_name}")
        print(f"\n{'Metric':<25} {'Control':>10} {'Treatment':>10} {'Lift%':>8} {'p-value':>10} {'Sig?':>6}")
        print("-" * 75)
        for row in r["statistical_tests"]:
            sig = "✓" if row["significant"] else "✗"
            print(f"{row['metric']:<25} {row['control_mean']:>10.4f} {row['treatment_mean']:>10.4f} "
                  f"{row['relative_lift_pct']:>+8.1f}% {row['p_value_mannwhitney']:>10.4f} {sig:>6}")
        print(f"\n🏆 VERDICT: {r['winner']}")
        print(f"{'='*60}\n")
