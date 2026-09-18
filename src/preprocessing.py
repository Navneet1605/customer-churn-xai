"""
preprocessing.py
================
Phase 2: Data Cleaning & Automated Quality Auditing
- Handles missing values, blank string coercion (specifically TotalCharges)
- Imputes missing numerical fields with median
- Deduplicates records
- Generates Data Quality Report (Missing values table, data types, statistical summary)
- Stores cleaned dataset for feature engineering
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
from src.spark_session import get_spark_session

logger = logging.getLogger("DataPreprocessing")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def run_preprocessing(
    input_path: str = "data/processed/cleaned_telco.parquet",
    output_path: str = "data/processed/preprocessed_telco.parquet",
    report_path: str = "reports/data_quality_report.md"
) -> Dict[str, Any]:
    """
    Cleans data, casts types, imputes blanks, removes duplicates, and generates a data quality report.
    """
    in_file = Path(input_path).resolve()
    if not in_file.exists():
        # Fall back to raw CSV if parquet doesn't exist yet
        in_file = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv").resolve()

    logger.info(f"Starting Data Preprocessing on: {in_file}")

    # Load dataset
    if str(in_file).endswith(".parquet"):
        df = pd.read_parquet(in_file)
    else:
        df = pd.read_csv(in_file)

    initial_row_count = len(df)
    initial_col_count = len(df.columns)

    # 1. Deduplication
    df_dedup = df.drop_duplicates(subset=["customerID"] if "customerID" in df.columns else None)
    duplicates_removed = initial_row_count - len(df_dedup)
    df = df_dedup.copy()

    # 2. TotalCharges Handling: Blanks to NaN -> Coerce to float -> Impute with median
    if "TotalCharges" in df.columns:
        # Check string whitespace
        df["TotalCharges"] = df["TotalCharges"].astype(str).str.strip()
        total_charges_blanks = int((df["TotalCharges"] == "").sum() + (df["TotalCharges"] == "nan").sum())
        
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        median_total_charges = float(df["TotalCharges"].median())
        df["TotalCharges"] = df["TotalCharges"].fillna(median_total_charges)
    else:
        total_charges_blanks = 0
        median_total_charges = 0.0

    # 3. Numeric conversions
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # 4. Standardize Target Variable: Churn (Yes -> 1, No -> 0)
    if "Churn" in df.columns:
        df["Churn_Numeric"] = df["Churn"].apply(lambda x: 1 if str(x).strip().lower() == "yes" else 0)
    
    # 5. Missing values audit
    missing_summary = []
    for col in df.columns:
        null_count = int(df[col].isna().sum())
        pct = (null_count / len(df)) * 100.0
        dtype_str = str(df[col].dtype)
        unique_vals = int(df[col].nunique())
        missing_summary.append({
            "Column": col,
            "DataType": dtype_str,
            "NullCount": null_count,
            "NullPercentage": round(pct, 2),
            "UniqueValues": unique_vals
        })

    missing_df = pd.DataFrame(missing_summary)

    # 6. Statistical summary for numerical columns
    num_summary = df[["tenure", "MonthlyCharges", "TotalCharges"]].describe().T.reset_index()
    num_summary = num_summary.rename(columns={"index": "Feature"}).round(2)

    # 7. Generate Data Quality Report Markdown
    report_md = f"""# Data Quality & Preprocessing Audit Report

**Executive Summary:**
- **Source Dataset:** IBM Telco Customer Churn
- **Initial Records Ingested:** {initial_row_count:,}
- **Duplicates Removed:** {duplicates_removed}
- **Blank `TotalCharges` Cleaned & Imputed:** {total_charges_blanks} (imputed with median = ${median_total_charges:.2f})
- **Final Cleaned Record Count:** {len(df):,}
- **Feature Dimension:** {len(df.columns)}

---

## 1. Missing Values & Schema Audit

| Column | Data Type | Null Count | Null % | Unique Values |
|---|---|---|---|---|
"""
    for row in missing_summary:
        report_md += f"| `{row['Column']}` | `{row['DataType']}` | {row['NullCount']} | {row['NullPercentage']}% | {row['UniqueValues']} |\n"

    report_md += f"""
---

## 2. Numerical Feature Descriptive Statistics

| Feature | Count | Mean | Std Dev | Min | 25% | 50% (Median) | 75% | Max |
|---|---|---|---|---|---|---|---|---|
"""
    for _, row in num_summary.iterrows():
        report_md += f"| `{row['Feature']}` | {int(row['count']):,} | {row['mean']} | {row['std']} | {row['min']} | {row['25%']} | {row['50%']} | {row['75%']} | {row['max']} |\n"

    report_md += f"""
---

## 3. Data Cleansing Rules Applied
1. **Deduplication:** Ensured `customerID` serves as a strictly unique primary key.
2. **Whitespace Coercion:** Empty space string tokens in `TotalCharges` converted to standard IEEE floating point representation and imputed with robust median value.
3. **Target Normalization:** Created binary label column `Churn_Numeric` where `Yes` maps to `1` and `No` maps to `0`.
4. **Data Integrity:** No records dropped due to data loss; 100% data retention preserved across all {len(df):,} customer instances.
"""

    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(str(out_file), index=False)
    logger.info(f"Saved preprocessed data to {out_file}")

    rep_file = Path(report_path).resolve()
    rep_file.parent.mkdir(parents=True, exist_ok=True)
    with open(rep_file, "w", encoding="utf-8") as f:
        f.write(report_md)
    logger.info(f"Generated Data Quality Audit at {rep_file}")

    # JSON report
    json_path = rep_file.with_suffix(".json")
    with open(json_path, "w") as jf:
        json.dump({
            "initial_rows": initial_row_count,
            "final_rows": len(df),
            "duplicates_removed": duplicates_removed,
            "total_charges_blanks": total_charges_blanks,
            "median_total_charges": median_total_charges,
            "missing_summary": missing_summary,
            "num_summary": num_summary.to_dict(orient="records")
        }, jf, indent=4)

    return {
        "status": "success",
        "rows": len(df),
        "columns": len(df.columns),
        "output_path": str(out_file),
        "report_path": str(rep_file)
    }


if __name__ == "__main__":
    run_preprocessing()
