"""
evaluate_models.py
==================
Phase 6: Comprehensive Model Evaluation & Metric Benchmarking
Calculates:
- Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Generates high-resolution ROC Curves, Precision-Recall Curves, and Confusion Matrices
- Benchmarks all candidate models and automatically selects the Best Model for Explainability & Serving
"""

import os
import sys
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, precision_recall_curve, average_precision_score
)

sys.path.append(str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger("ModelEvaluation")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def run_model_evaluation(
    models_dir: str = "models",
    reports_dir: str = "reports"
) -> Dict[str, Any]:
    """
    Evaluates all trained candidate models and generates benchmark visualizations.
    """
    models_path = Path(models_dir).resolve()
    reports_path = Path(reports_dir).resolve()
    figures_path = reports_path / "figures"
    figures_path.mkdir(parents=True, exist_ok=True)

    test_data_file = models_path / "test_data.pkl"
    if not test_data_file.exists():
        raise FileNotFoundError(f"Test data file not found at {test_data_file}. Run train_models.py first.")

    with open(test_data_file, "rb") as f:
        data_splits = pickle.load(f)

    X_test = data_splits["X_test"]
    y_test = data_splits["y_test"]

    model_files = {
        "Logistic Regression": models_path / "Logistic_Regression_pipeline.pkl",
        "Random Forest": models_path / "Random_Forest_pipeline.pkl",
        "Gradient Boosted Trees": models_path / "Gradient_Boosted_Trees_pipeline.pkl"
    }

    metrics_summary = []
    roc_data = {}
    pr_data = {}
    conf_matrices = {}

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    palette = ["#1f77b4", "#2ca02c", "#ff7f0e"]

    for idx, (model_name, model_file) in enumerate(model_files.items()):
        if not model_file.exists():
            logger.warning(f"Model file {model_file} not found. Skipping.")
            continue

        with open(model_file, "rb") as f:
            pipeline = pickle.load(f)

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)

        metrics_summary.append({
            "Model": model_name,
            "Accuracy": round(float(acc), 4),
            "Precision": round(float(prec), 4),
            "Recall": round(float(rec), 4),
            "F1_Score": round(float(f1), 4),
            "ROC_AUC": round(float(auc), 4),
            "PR_AUC": round(float(pr_auc), 4)
        })

        # ROC Curve data
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data[model_name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": auc}

        # PR Curve data
        precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
        pr_data[model_name] = {"precision": precision_vals.tolist(), "recall": recall_vals.tolist(), "pr_auc": pr_auc}

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        conf_matrices[model_name] = cm.tolist()

    metrics_df = pd.DataFrame(metrics_summary)
    # Sort by ROC_AUC descending
    metrics_df = metrics_df.sort_values(by="ROC_AUC", ascending=False).reset_index(drop=True)
    best_model_name = metrics_df.iloc[0]["Model"]

    logger.info("Model Comparison Table:")
    logger.info(metrics_df.to_string(index=False))
    logger.info(f"==> Best Performing Model Identified: {best_model_name} (ROC-AUC: {metrics_df.iloc[0]['ROC_AUC']})")

    # 1. Plot ROC Curves Comparison
    fig_roc, ax_roc = plt.subplots(figsize=(8, 6), dpi=300)
    for idx, (m_name, r_vals) in enumerate(roc_data.items()):
        ax_roc.plot(r_vals["fpr"], r_vals["tpr"], label=f"{m_name} (AUC = {r_vals['auc']:.4f})", linewidth=2.5, color=palette[idx])
    ax_roc.plot([0, 1], [0, 1], 'k--', label='Random Chance (AUC = 0.5000)', alpha=0.6)
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12, fontweight='bold')
    ax_roc.set_ylabel('True Positive Rate (Recall)', fontsize=12, fontweight='bold')
    ax_roc.set_title('Receiver Operating Characteristic (ROC) Benchmark Curves', fontsize=14, fontweight='bold', pad=15)
    ax_roc.legend(loc="lower right", frameon=True, fontsize=10)
    fig_roc.tight_layout()
    fig_roc.savefig(figures_path / "roc_curves_comparison.png")
    plt.close(fig_roc)

    # 2. Plot Precision-Recall Curves
    fig_pr, ax_pr = plt.subplots(figsize=(8, 6), dpi=300)
    for idx, (m_name, p_vals) in enumerate(pr_data.items()):
        ax_pr.plot(p_vals["recall"], p_vals["precision"], label=f"{m_name} (PR-AUC = {p_vals['pr_auc']:.4f})", linewidth=2.5, color=palette[idx])
    ax_pr.set_xlim([0.0, 1.0])
    ax_pr.set_ylim([0.0, 1.05])
    ax_pr.set_xlabel('Recall (True Positive Rate)', fontsize=12, fontweight='bold')
    ax_pr.set_ylabel('Precision (Positive Predictive Value)', fontsize=12, fontweight='bold')
    ax_pr.set_title('Precision-Recall Benchmark Curves', fontsize=14, fontweight='bold', pad=15)
    ax_pr.legend(loc="upper right", frameon=True, fontsize=10)
    fig_pr.tight_layout()
    fig_pr.savefig(figures_path / "precision_recall_curves.png")
    plt.close(fig_pr)

    # 3. Plot Confusion Matrices Subplots
    fig_cm, axes_cm = plt.subplots(1, 3, figsize=(18, 5), dpi=300)
    for idx, (m_name, cm_list) in enumerate(conf_matrices.items()):
        cm_arr = np.array(cm_list)
        sns.heatmap(cm_arr, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes_cm[idx],
                    xticklabels=["Retained", "Churned"], yticklabels=["Retained", "Churned"], annot_kws={"size": 13, "weight": "bold"})
        axes_cm[idx].set_title(f"{m_name}\nConfusion Matrix", fontsize=12, fontweight='bold')
        axes_cm[idx].set_xlabel("Predicted Label", fontsize=10)
        axes_cm[idx].set_ylabel("True Label", fontsize=10)
    fig_cm.tight_layout()
    fig_cm.savefig(figures_path / "confusion_matrices_comparison.png")
    plt.close(fig_cm)

    # Save Best Model metadata
    best_model_key = best_model_name.replace(" ", "_")
    best_model_metadata = {
        "best_model_name": best_model_name,
        "best_model_file": str(models_path / f"{best_model_key}_pipeline.pkl"),
        "metrics_summary": metrics_summary,
        "best_metrics": metrics_df.iloc[0].to_dict(),
        "roc_curve_figure": str(figures_path / "roc_curves_comparison.png"),
        "pr_curve_figure": str(figures_path / "precision_recall_curves.png"),
        "confusion_matrix_figure": str(figures_path / "confusion_matrices_comparison.png")
    }

    with open(models_path / "best_model_metadata.json", "w") as f:
        json.dump(best_model_metadata, f, indent=4)

    with open(reports_path / "model_evaluation_metrics.json", "w") as f:
        json.dump(best_model_metadata, f, indent=4)

    logger.info("Phase 6: Model Evaluation Completed Successfully.")
    return best_model_metadata


if __name__ == "__main__":
    run_model_evaluation()
