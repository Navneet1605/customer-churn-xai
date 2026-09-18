"""
feature_engineering.py
======================
Phase 4: Feature Engineering & Domain Enrichment using Apache Spark Transformations
Creates high-impact analytical & predictive features:
1. tenure_bucket (New: 0-12, Developing: 13-24, Stable: 25-48, Loyal: 49+)
2. avg_monthly_spend: TotalCharges / (tenure + 1)
3. contract_risk_score: High (3 for Month-to-month), Medium (2 for One year), Low (1 for Two year)
4. service_count: Total active bundled telecommunication services
5. support_usage_indicator: Flag for high-dependency support consumers
6. premium_customer_flag: Flag for top-tier spending subscribers (MonthlyCharges > $80)
7. long_term_customer_flag: Subscribed for over 3 years (tenure > 36 months)
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger("FeatureEngineering")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def calculate_service_count(row) -> int:
    """Calculates active telecommunication services subscribed by the user."""
    count = 0
    if str(row.get("PhoneService", "")).strip().lower() == "yes":
        count += 1
    if str(row.get("MultipleLines", "")).strip().lower() == "yes":
        count += 1
    if str(row.get("InternetService", "")).strip().lower() in ["dsl", "fiber optic"]:
        count += 1
    
    vas_services = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    for svc in vas_services:
        if str(row.get(svc, "")).strip().lower() == "yes":
            count += 1
    return count


def assign_tenure_bucket(tenure_val: float) -> str:
    """Classifies customer lifecycle stage based on subscription tenure."""
    if tenure_val <= 12:
        return "New"
    elif tenure_val <= 24:
        return "Developing"
    elif tenure_val <= 48:
        return "Stable"
    else:
        return "Loyal"


def get_contract_risk(contract_val: str) -> int:
    """Calculates contractual retention vulnerability score."""
    val = str(contract_val).strip().lower()
    if "month" in val:
        return 3
    elif "one" in val:
        return 2
    elif "two" in val:
        return 1
    return 2


def run_feature_engineering(
    input_path: str = "data/processed/preprocessed_telco.parquet",
    output_path: str = "data/processed/engineered_telco.parquet"
) -> Dict[str, Any]:
    """
    Executes Feature Engineering transformation pipeline on preprocessed data.
    """
    in_file = Path(input_path).resolve()
    if not in_file.exists():
        in_file = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv").resolve()

    logger.info(f"Starting Feature Engineering on: {in_file}")
    
    if str(in_file).endswith(".parquet"):
        df = pd.read_parquet(in_file)
    else:
        df = pd.read_csv(in_file)

    # 1. Tenure Buckets
    df["tenure_bucket"] = df["tenure"].apply(assign_tenure_bucket)

    # 2. Average Monthly Spend
    df["avg_monthly_spend"] = np.round(df["TotalCharges"] / (df["tenure"] + 1.0), 2)

    # 3. Contract Risk Score
    df["contract_risk_score"] = df["Contract"].apply(get_contract_risk)

    # 4. Total Subscribed Services Count
    df["service_count"] = df.apply(calculate_service_count, axis=1)

    # 5. Support Usage Indicator (TechSupport, OnlineSecurity, OnlineBackup)
    df["support_usage_indicator"] = df.apply(
        lambda r: 1 if (
            str(r.get("TechSupport", "")).strip().lower() == "yes" or
            str(r.get("OnlineSecurity", "")).strip().lower() == "yes" or
            str(r.get("OnlineBackup", "")).strip().lower() == "yes"
        ) else 0,
        axis=1
    )

    # 6. Premium Customer Flag (Monthly Spend > $80)
    df["premium_customer_flag"] = (df["MonthlyCharges"] > 80.0).astype(int)

    # 7. Long-Term Customer Flag (Tenure > 36 months)
    df["long_term_customer_flag"] = (df["tenure"] > 36).astype(int)

    # 8. Add Payment Risk Indicator (Electronic check is historically highest churn)
    df["electronic_check_risk"] = df["PaymentMethod"].apply(
        lambda x: 1 if "electronic check" in str(x).strip().lower() else 0
    )

    # Save to parquet
    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(str(out_file), index=False)

    logger.info(f"Engineered features successfully created. Total columns: {len(df.columns)}")
    logger.info(f"Saved engineered dataset to: {out_file}")

    engineered_summary = {
        "num_rows": len(df),
        "total_columns": len(df.columns),
        "new_features": [
            "tenure_bucket",
            "avg_monthly_spend",
            "contract_risk_score",
            "service_count",
            "support_usage_indicator",
            "premium_customer_flag",
            "long_term_customer_flag",
            "electronic_check_risk"
        ],
        "output_path": str(out_file)
    }

    return engineered_summary


if __name__ == "__main__":
    run_feature_engineering()
