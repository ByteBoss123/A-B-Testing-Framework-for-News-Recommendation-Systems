"""
run_experiment.py — Main Experiment Runner
Single entry point to run the full A/B test pipeline.

Usage:
    python run_experiment.py                    # Default run
    python run_experiment.py --n-articles 5000  # Larger dataset
    python run_experiment.py --n-users 2000     # More users
    python run_experiment.py --no-tracking      # Skip MLflow
"""

import sys
import os
import argparse
import json
import pandas as pd
from datetime import datetime

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.data_generator import load_dataset
from models.control_model import KeywordRecommender
from models.treatment_model import SemanticRecommender
from evaluation.ab_experiment import ABExperiment, ExperimentConfig
from tracking.mlflow_tracker import ExperimentTracker


def parse_args():
    parser = argparse.ArgumentParser(description="News Recommendation A/B Test Framework")
    parser.add_argument("--n-articles", type=int, default=3000, help="Number of articles to generate")
    parser.add_argument("--n-users", type=int, default=800, help="Number of users to simulate")
    parser.add_argument("--top-k", type=int, default=10, help="Top-K recommendations")
    parser.add_argument("--split", type=float, default=0.5, help="Traffic split (0.5 = 50/50)")
    parser.add_argument("--no-tracking", action="store_true", help="Skip MLflow tracking")
    parser.add_argument("--experiment-id", type=str, default=None, help="Custom experiment ID")
    return parser.parse_args()


def main():
    args = parse_args()

    experiment_id = args.experiment_id or f"news_ab_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║        News Recommendation A/B Testing Framework             ║
║        Solving: How do we validate a new recommendation      ║
║        algorithm before deploying to millions of readers?    ║
╚══════════════════════════════════════════════════════════════╝
Experiment: {experiment_id}
Articles:   {args.n_articles:,}
Users:      {args.n_users:,}
Top-K:      {args.top_k}
Split:      {args.split:.0%} / {1-args.split:.0%} (control/treatment)
    """)

    # Step 1: Load data
    articles, users = load_dataset(n_articles=args.n_articles, n_users=args.n_users)

    # Step 2: Train models
    print("Training models...")
    control_model = KeywordRecommender(top_k=args.top_k)
    control_model.fit(articles)

    treatment_model = SemanticRecommender(top_k=args.top_k)
    treatment_model.fit(articles)

    # Step 3: Configure experiment
    config = ExperimentConfig(
        experiment_id=experiment_id,
        control_name=control_model.model_name,
        treatment_name=treatment_model.model_name,
        traffic_split=args.split,
        top_k=args.top_k,
    )

    # Step 4: Run experiment
    experiment = ABExperiment(config, control_model, treatment_model)
    results = experiment.run(users, articles)

    # Step 5: Track with MLflow
    if not args.no_tracking:
        tracker = ExperimentTracker()
        run_id = tracker.log_experiment(results, config)
        print(f"\n[MLflow] Run ID: {run_id}")
        print(f"[MLflow] View results: mlflow ui --backend-store-uri sqlite:///mlflow.db\n")

    # Step 6: Save results JSON
    output = {
        "experiment_id": results["experiment_id"],
        "winner": results["winner"],
        "sample_sizes": results["sample_sizes"],
        "control_summary": results["control_summary"],
        "treatment_summary": results["treatment_summary"],
        "statistical_tests": results["statistical_tests"],
    }
    output_path = f"experiment_results_{experiment_id}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to: {output_path}")

    # Final summary
    print(f"\n{'🏆 FINAL VERDICT':}")
    print(f"{'='*50}")
    print(f"{results['winner']}")
    
    sig_improvements = [
        t for t in results["statistical_tests"]
        if t["significant"] and t["relative_lift_pct"] > 0
    ]
    if sig_improvements:
        print(f"\nSignificant improvements ({len(sig_improvements)}):")
        for t in sig_improvements:
            print(f"  • {t['metric']}: +{t['relative_lift_pct']:.1f}% lift (p={t['p_value_mannwhitney']:.4f})")

    return results


if __name__ == "__main__":
    main()
