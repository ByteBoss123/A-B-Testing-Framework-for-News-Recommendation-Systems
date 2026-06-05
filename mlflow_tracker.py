"""
mlflow_tracker.py — MLflow Experiment Tracking
Logs all experiment runs, parameters, metrics, and artifacts.

Real-world problem: 
"Which model config produced that 23% lift we saw 3 weeks ago?"
Without tracking, this question is unanswerable. MLflow makes every run reproducible.

This is what separates a research notebook from a production ML system.
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from datetime import datetime


class ExperimentTracker:
    """
    MLflow wrapper for the A/B testing framework.
    Logs: parameters, metrics, statistical results, and visualizations.
    """

    def __init__(self, tracking_uri: str = "sqlite:///mlflow.db", experiment_name: str = "NewsRecommendation_AB"):
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name
        print(f"[MLflow] Tracking at: {tracking_uri}")

    def log_experiment(self, results: dict, config) -> str:
        """Log a complete A/B experiment run to MLflow."""
        with mlflow.start_run(run_name=results["experiment_id"]) as run:
            run_id = run.info.run_id

            # Log config params
            mlflow.log_params({
                "experiment_id": config.experiment_id,
                "control_model": config.control_name,
                "treatment_model": config.treatment_name,
                "traffic_split": config.traffic_split,
                "top_k": config.top_k,
                "significance_level": config.significance_level,
                "n_control": results["sample_sizes"]["control"],
                "n_treatment": results["sample_sizes"]["treatment"],
            })

            # Log aggregate metrics
            for group, summary in [("control", results["control_summary"]),
                                    ("treatment", results["treatment_summary"])]:
                for metric, value in summary.items():
                    mlflow.log_metric(f"{group}_{metric}", value)

            # Log statistical test results as metrics
            for test in results["statistical_tests"]:
                metric = test["metric"]
                mlflow.log_metric(f"lift_pct_{metric}", test["relative_lift_pct"])
                mlflow.log_metric(f"pvalue_{metric}", test["p_value_mannwhitney"])
                mlflow.log_metric(f"cohens_d_{metric}", test["cohens_d"])

            # Log winner
            mlflow.log_param("winner", results["winner"])
            mlflow.set_tag("status", "completed")
            mlflow.set_tag("framework", "news_ab_v1")

            # Generate and log plots
            plots_dir = f"/tmp/ab_plots_{run_id[:8]}"
            os.makedirs(plots_dir, exist_ok=True)

            plot_paths = self._generate_plots(results, plots_dir)
            for path in plot_paths:
                mlflow.log_artifact(path)

            # Log stats CSV
            stats_path = f"{plots_dir}/statistical_results.csv"
            results["stats_df"].to_csv(stats_path, index=False)
            mlflow.log_artifact(stats_path)

            print(f"[MLflow] Run logged: {run_id}")
            print(f"[MLflow] View at: mlflow ui --backend-store-uri ./mlruns")
            return run_id

    def _generate_plots(self, results: dict, output_dir: str) -> list:
        """Generate experiment visualization plots."""
        paths = []

        # Plot 1: Metric comparison bar chart
        path = self._plot_metric_comparison(results, output_dir)
        paths.append(path)

        # Plot 2: Statistical significance forest plot
        path = self._plot_significance(results, output_dir)
        paths.append(path)

        # Plot 3: Distribution plots for key metrics
        path = self._plot_distributions(results, output_dir)
        paths.append(path)

        return paths

    def _plot_metric_comparison(self, results: dict, output_dir: str) -> str:
        """Bar chart comparing control vs treatment across all metrics."""
        metrics = list(results["control_summary"].keys())
        control_vals = list(results["control_summary"].values())
        treatment_vals = list(results["treatment_summary"].values())

        x = np.arange(len(metrics))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))
        bars1 = ax.bar(x - width/2, control_vals, width, label="Control (TF-IDF)", color="#2196F3", alpha=0.8)
        bars2 = ax.bar(x + width/2, treatment_vals, width, label="Treatment (Semantic)", color="#4CAF50", alpha=0.8)

        ax.set_xlabel("Metric")
        ax.set_ylabel("Score")
        ax.set_title("A/B Test: Control vs Treatment Recommendation Quality", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("_", "\n") for m in metrics], fontsize=9)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        # Add value labels
        for bar in bars1:
            ax.annotate(f"{bar.get_height():.3f}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
        for bar in bars2:
            ax.annotate(f"{bar.get_height():.3f}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

        plt.tight_layout()
        path = f"{output_dir}/metric_comparison.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        return path

    def _plot_significance(self, results: dict, output_dir: str) -> str:
        """Forest plot showing confidence intervals and significance."""
        stats = results["stats_df"]

        fig, ax = plt.subplots(figsize=(10, max(4, len(stats) * 1.2)))

        colors = ["#4CAF50" if sig else "#FF5722" for sig in stats["significant"]]
        y_pos = range(len(stats))

        for i, (_, row) in enumerate(stats.iterrows()):
            lift = row["relative_lift_pct"]
            ci_low_abs = abs(lift - row["ci_95_low"] * 100)
            ci_high_abs = abs(row["ci_95_high"] * 100 - lift)
            ax.barh(i, lift, color=colors[i], alpha=0.7, height=0.5)
            ax.errorbar(
                lift, i,
                xerr=[[max(0, ci_low_abs)], [max(0, ci_high_abs)]],
                fmt="none", color="black", capsize=5, linewidth=2
            )
            sig_label = f"p={row['p_value_mannwhitney']:.3f} {'✓' if row['significant'] else '✗'}"
            ax.text(lift + 0.5, i, sig_label, va="center", fontsize=9)

        ax.axvline(x=0, color="black", linestyle="--", alpha=0.5, label="No difference")
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(stats["metric"].tolist())
        ax.set_xlabel("Relative Lift % (Treatment vs Control)")
        ax.set_title("Statistical Significance: Treatment vs Control\n(Green = significant, Red = not significant)",
                     fontsize=12, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

        plt.tight_layout()
        path = f"{output_dir}/significance_plot.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        return path

    def _plot_distributions(self, results: dict, output_dir: str) -> str:
        """Distribution plots for key metrics."""
        control_df = results["control_metrics_df"]
        treatment_df = results["treatment_metrics_df"]

        metrics_to_plot = ["precision_at_k", "ndcg_at_k", "simulated_ctr"]
        available = [m for m in metrics_to_plot if m in control_df.columns]

        fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 5))
        if len(available) == 1:
            axes = [axes]

        for ax, metric in zip(axes, available):
            ax.hist(control_df[metric].dropna(), bins=30, alpha=0.6, label="Control", color="#2196F3", density=True)
            ax.hist(treatment_df[metric].dropna(), bins=30, alpha=0.6, label="Treatment", color="#4CAF50", density=True)
            ax.set_title(f"{metric.replace('_', ' ').title()}", fontweight="bold")
            ax.set_xlabel("Score")
            ax.set_ylabel("Density")
            ax.legend()
            ax.grid(alpha=0.3)

        plt.suptitle("Metric Distribution: Control vs Treatment", fontsize=13, fontweight="bold")
        plt.tight_layout()
        path = f"{output_dir}/distributions.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        return path
