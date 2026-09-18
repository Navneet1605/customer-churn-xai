"""
explainability.py
=================
Phase 7: Explainable AI (XAI) using SHAP (SHapley Additive exPlanations)
Provides:
1. Global Interpretability:
   - SHAP Summary Beeswarm Plot
   - Feature Importance Bar Chart (Mean |SHAP|)
   - SHAP Dependence Plots
2. Local Interpretability:
   - 5 Diverse Customer Personas (High Risk, Moderate Risk, Loyal, Support-Deficient, Price-Sensitive)
   - Waterfall Plots
   - Force Plots
   - Natural Language Executive Explanation Engine
"""

import os
import sys
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap

sys.path.append(str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger("ExplainableAI")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def generate_natural_language_explanation(
    customer_data: Dict[str, Any],
    churn_prob: float,
    top_positive_drivers: List[Tuple[str, float]],
    top_negative_drivers: List[Tuple[str, float]]
) -> str:
    """
    Synthesizes human-readable executive reasoning based on SHAP contribution vectors.
    """
    risk_level = "High" if churn_prob >= 0.65 else ("Moderate" if churn_prob >= 0.35 else "Low")
    
    pos_desc = []
    for feat, val in top_positive_drivers[:3]:
        clean_feat = feat.replace("cat__", "").replace("num__", "").replace("_", " ").title()
        pos_desc.append(f"{clean_feat} (SHAP impact: +{val:.2f})")
        
    neg_desc = []
    for feat, val in top_negative_drivers[:2]:
        clean_feat = feat.replace("cat__", "").replace("num__", "").replace("_", " ").title()
        neg_desc.append(f"{clean_feat} (SHAP impact: {val:.2f})")

    explanation = f"Customer exhibits a **{risk_level} Churn Risk** with a predicted churn probability of **{churn_prob*100:.1f}%**.\n"
    
    if pos_desc:
        explanation += f"- **Primary Churn Accelerators:** {', '.join(pos_desc)}.\n"
    if neg_desc:
        explanation += f"- **Primary Retention Anchors:** {', '.join(neg_desc)}.\n"
        
    if risk_level == "High":
        explanation += "💡 **Recommended Action:** Immediate intervention required. Proactively offer a discounted annual contract lock-in, complimentary tech support, or customized loyalty incentive."
    elif risk_level == "Moderate":
        explanation += "💡 **Recommended Action:** Monitor usage velocity; propose value-add digital security or automated payment discounts."
    else:
        explanation += "💡 **Recommended Action:** High retention health. Consider upselling premium streaming tiers or family bundles."

    return explanation


def run_explainability(
    models_dir: str = "models",
    reports_dir: str = "reports"
) -> Dict[str, Any]:
    """
    Executes SHAP global and local explainability pipeline.
    """
    models_path = Path(models_dir).resolve()
    reports_path = Path(reports_dir).resolve()
    figures_path = reports_path / "figures"
    figures_path.mkdir(parents=True, exist_ok=True)

    # Load best model metadata
    meta_file = models_path / "best_model_metadata.json"
    if not meta_file.exists():
        raise FileNotFoundError(f"Model metadata not found at {meta_file}. Run evaluate_models.py first.")

    with open(meta_file, "r") as f:
        best_meta = json.load(f)

    best_model_path = Path(best_meta["best_model_file"])
    with open(best_model_path, "rb") as f:
        best_pipeline = pickle.load(f)

    # Load test split
    with open(models_path / "test_data.pkl", "rb") as f:
        data_splits = pickle.load(f)

    X_train = data_splits["X_train"]
    X_test = data_splits["X_test"]
    y_test = data_splits["y_test"]

    preprocessor = best_pipeline.named_steps["preprocessor"]
    classifier = best_pipeline.named_steps["classifier"]

    # Transform feature matrices
    X_train_trans = preprocessor.transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    # Retrieve feature names
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_cols = [c for c in X_train.columns if X_train[c].dtype == "object" or c == "SeniorCitizen"]
    num_cols = [c for c in X_train.columns if c not in cat_cols]
    
    cat_feature_names = list(cat_encoder.get_feature_names_out(cat_cols))
    all_feature_names = num_cols + cat_feature_names
    # Shorten feature names for clean plotting
    readable_feature_names = [
        f.replace("cat__", "").replace("num__", "").replace("_", " ").title()
        for f in all_feature_names
    ]

    logger.info(f"Computing SHAP values on {len(X_test)} test instances across {len(readable_feature_names)} features...")

    # Choose Explainer based on model architecture
    if hasattr(classifier, "feature_importances_"):
        # Tree-based model (Random Forest, Gradient Boosted Trees)
        explainer = shap.TreeExplainer(classifier)
        shap_values_raw = explainer.shap_values(X_test_trans)
        
        # Format handling for binary classifiers
        if isinstance(shap_values_raw, list) and len(shap_values_raw) == 2:
            shap_values = shap_values_raw[1]  # positive class
            expected_value = explainer.expected_value[1]
        elif isinstance(shap_values_raw, np.ndarray) and shap_values_raw.ndim == 3:
            shap_values = shap_values_raw[:, :, 1]
            expected_value = explainer.expected_value[1]
        else:
            shap_values = shap_values_raw
            expected_value = explainer.expected_value
            if isinstance(expected_value, np.ndarray):
                expected_value = expected_value[0]
    else:
        # Linear Model (Logistic Regression)
        explainer = shap.LinearExplainer(classifier, X_train_trans)
        shap_values = explainer.shap_values(X_test_trans)
        expected_value = explainer.expected_value

    # 1. Global Summary Beeswarm Plot
    fig_summary = plt.figure(figsize=(10, 8), dpi=300)
    shap.summary_plot(
        shap_values,
        X_test_trans,
        feature_names=readable_feature_names,
        max_display=15,
        show=False
    )
    plt.title("SHAP Global Feature Impact Summary (Beeswarm)", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    summary_plot_file = figures_path / "shap_summary_beeswarm.png"
    plt.savefig(summary_plot_file, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved SHAP summary beeswarm to {summary_plot_file}")

    # 2. Feature Importance Bar Chart (Mean |SHAP|)
    fig_bar = plt.figure(figsize=(10, 8), dpi=300)
    shap.summary_plot(
        shap_values,
        X_test_trans,
        feature_names=readable_feature_names,
        plot_type="bar",
        max_display=15,
        show=False
    )
    plt.title("SHAP Global Feature Importance (Mean |SHAP Value|)", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    bar_plot_file = figures_path / "shap_feature_importance_bar.png"
    plt.savefig(bar_plot_file, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved SHAP importance bar to {bar_plot_file}")

    # 3. Local Explainability for 5 Diverse Customer Personas
    y_probs = best_pipeline.predict_proba(X_test)[:, 1]
    
    # Identify 5 diverse customer indices:
    high_risk_idx = int(np.argmax(y_probs))
    low_risk_idx = int(np.argmin(y_probs))
    moderate_risk_idx = int(np.argmin(np.abs(y_probs - 0.50)))
    
    # Filter indices for specific archetypes
    test_df = X_test.copy().reset_index(drop=True)
    no_support_mask = (test_df["support_usage_indicator"] == 0) & (y_probs > 0.6)
    support_deficient_idx = int(test_df[no_support_mask].index[0]) if no_support_mask.any() else int(np.argsort(y_probs)[-5])

    senior_mask = (test_df["SeniorCitizen"] == 1) & (y_probs > 0.5)
    senior_idx = int(test_df[senior_mask].index[0]) if senior_mask.any() else int(np.argsort(y_probs)[-10])

    selected_profiles = [
        ("High_Risk_Persona", high_risk_idx, "High Risk Churn Archetype"),
        ("Moderate_Risk_Persona", moderate_risk_idx, "Moderate Vulnerability Archetype"),
        ("Loyal_Customer_Persona", low_risk_idx, "High Retention & Loyal Archetype"),
        ("Support_Deficient_Persona", support_deficient_idx, "Support-Deficient Tech Consumer"),
        ("Price_Sensitive_Senior_Persona", senior_idx, "Price-Sensitive Senior Subscriber")
    ]

    local_explanations = []

    for name, idx, title in selected_profiles:
        cust_row = test_df.iloc[idx].to_dict()
        cust_prob = float(y_probs[idx])
        cust_shap = shap_values[idx]

        # Rank features by positive and negative contributions
        paired = list(zip(readable_feature_names, cust_shap))
        pos_drivers = sorted([p for p in paired if p[1] > 0], key=lambda x: x[1], reverse=True)
        neg_drivers = sorted([p for p in paired if p[1] < 0], key=lambda x: x[1])

        # Generate Waterfall Plot for individual explanation
        fig_waterfall = plt.figure(figsize=(9, 6), dpi=300)
        shap_explanation_obj = shap.Explanation(
            values=cust_shap,
            base_values=float(expected_value),
            data=X_test_trans[idx],
            feature_names=readable_feature_names
        )
        shap.plots.waterfall(shap_explanation_obj, max_display=10, show=False)
        plt.title(f"Local SHAP Explanation: {title}", fontsize=13, fontweight="bold", pad=12)
        plt.tight_layout()
        waterfall_file = figures_path / f"shap_waterfall_{name}.png"
        plt.savefig(waterfall_file, bbox_inches='tight')
        plt.close()

        nl_text = generate_natural_language_explanation(
            cust_row,
            cust_prob,
            pos_drivers,
            neg_drivers
        )

        local_explanations.append({
            "persona_name": name,
            "persona_title": title,
            "customer_features": {k: (int(v) if isinstance(v, (np.integer, int)) else (float(v) if isinstance(v, (np.floating, float)) else str(v))) for k, v in cust_row.items()},
            "churn_probability": round(cust_prob, 4),
            "predicted_class": "Churn" if cust_prob >= 0.5 else "Retain",
            "top_positive_drivers": [(k, round(float(v), 4)) for k, v in pos_drivers[:5]],
            "top_negative_drivers": [(k, round(float(v), 4)) for k, v in neg_drivers[:5]],
            "natural_language_explanation": nl_text,
            "waterfall_plot_path": str(waterfall_file)
        })
        logger.info(f"Generated XAI Waterfall and Explanation for {title}")

    # Compute global feature importance ranking dictionary
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    feat_ranking = sorted(zip(readable_feature_names, mean_abs_shap), key=lambda x: x[1], reverse=True)
    global_importance_list = [{"feature": k, "mean_abs_shap": round(float(v), 4)} for k, v in feat_ranking[:20]]

    # Save XAI summary artifact
    xai_artifact = {
        "best_model": best_meta["best_model_name"],
        "num_features_analyzed": len(readable_feature_names),
        "global_importance_ranking": global_importance_list,
        "summary_plot_path": str(summary_plot_file),
        "feature_importance_bar_path": str(bar_plot_file),
        "local_explanations": local_explanations
    }

    with open(reports_path / "shap_explanations.json", "w") as f:
        json.dump(xai_artifact, f, indent=4)

    # Save serialized explainer for real-time live Streamlit inference
    explainer_export_path = models_path / "shap_explainer.pkl"
    with open(explainer_export_path, "wb") as f:
        pickle.dump({
            "explainer": explainer,
            "expected_value": expected_value,
            "readable_feature_names": readable_feature_names,
            "all_feature_names": all_feature_names,
            "preprocessor": preprocessor
        }, f)

    logger.info(f"Phase 7: Explainable AI Pipeline completed. Exported live explainer to {explainer_export_path}")
    return xai_artifact


if __name__ == "__main__":
    run_explainability()
