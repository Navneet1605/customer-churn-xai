# Explainable Customer Churn Prediction using Apache Spark and Explainable AI

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Apache Spark](https://img.shields.io/badge/Apache%20Spark-3.5+-E25A1C.svg)](https://spark.apache.org/)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-brightgreen.svg)](https://shap.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An enterprise-grade, end-to-end Big Data Analytics and Explainable AI (XAI) system designed to predict customer churn in telecommunications and interpret every decision transparently using game-theoretic SHAP values.

---

## 📌 Table of Contents
1. [Project Overview & Objectives](#-project-overview--objectives)
2. [End-to-End Architecture](#-end-to-end-architecture)
3. [Repository Structure](#-repository-structure)
4. [Dataset Description](#-dataset-description)
5. [Implementation Phases](#-implementation-phases)
6. [Machine Learning Benchmark Results](#-machine-learning-benchmark-results)
7. [Explainable AI (SHAP) Interpretation](#-explainable-ai-shap-interpretation)
8. [Interactive Streamlit Dashboard](#-interactive-streamlit-dashboard)
9. [Step-by-Step Execution Guide](#-step-by-step-execution-guide)
   - [Local Environment](#1-local-environment-execution)
   - [Google Colab](#2-google-colab-execution)
   - [Databricks](#3-databricks-execution)
10. [Automated Reporting & Artifacts](#-automated-reporting--artifacts)

---

## 🎯 Project Overview & Objectives

In the competitive telecommunications sector, acquiring new subscribers costs 5–7× more than retaining existing ones. This project delivers a scalable predictive intelligence system that:
- **Processes Big Data at Scale:** Ingests and transforms large-scale customer records using Apache Spark and distributed Parquet partitioning.
- **Predicts Attrition:** Evaluates and optimizes multiple Machine Learning algorithms (Logistic Regression, Random Forest, Gradient Boosted Trees).
- **De-mystifies Model Predictions (XAI):** Implements SHAP to provide global feature attributions and individual customer waterfall explanations.
- **Quantifies Financial Exposure:** Automatically computes monthly recurring revenue at risk ($/month and $/year).
- **Enables Operational Action:** Delivers a modern Streamlit web dashboard for real-time customer scoring, batch CSV uploads, and executive PDF reporting.

---

## 🏗️ End-to-End Architecture

```
+-------------------------------------------------------------------------------+
|                             DATA INGESTION LAYER                             |
|  IBM Telco Dataset (7,043 rows) ---> PySpark Schema Inference & Null Audit    |
|                          |                                                    |
|                          v                                                    |
|           Simulated HDFS Parquet Partitioning (By Contract)                   |
+-------------------------------------------------------------------------------+
                                   |
                                   v
+-------------------------------------------------------------------------------+
|                       DATA CLEANING & QUALITY AUDIT                           |
|  - Whitespace Coercion & Median Imputation (TotalCharges)                     |
|  - Deduplication & Binary Normalization                                       |
|  - Automated Quality Audit (reports/data_quality_report.md)                   |
+-------------------------------------------------------------------------------+
                                   |
                                   v
+-------------------------------------------------------------------------------+
|                       FEATURE ENGINEERING & DOMAIN LOGIC                      |
|  - tenure_bucket (New, Developing, Stable, Loyal)                             |
|  - avg_monthly_spend, contract_risk_score, service_count                      |
|  - support_usage_indicator, premium_customer_flag, electronic_check_risk      |
+-------------------------------------------------------------------------------+
                                   |
                                   v
+-------------------------------------------------------------------------------+
|                       MACHINE LEARNING & MODEL TUNING                         |
|  - MLlib / Scikit-Learn Pipelines: StringIndexer + OHE + Scaler              |
|  - Models: Logistic Regression, Random Forest, Gradient Boosted Trees (GBT)   |
|  - Stratified 3-Fold Cross Validation & Hyperparameter Grid Search            |
+-------------------------------------------------------------------------------+
                                   |
                                   v
+-------------------------------------------------------------------------------+
|                       EXPLAINABLE AI (SHAP) ENGINE                            |
|  - Global Beeswarm & Feature Importance Ranking                               |
|  - Local Waterfall & Force Breakdown for 5 Diverse Customer Personas          |
|  - Dynamic Natural Language Executive Explanation Generator                   |
+-------------------------------------------------------------------------------+
                                   |
                                   v
+-------------------------------------------------------------------------------+
|                      STREAMLIT DASHBOARD & PDF REPORTING                      |
|  - Executive KPI Cards & Interactive Plotly Charts                            |
|  - Live Single-Customer Risk Predictor with Real-Time SHAP Waterfall          |
|  - Batch CSV Inference Engine & One-Click PDF/Markdown Report Downloads       |
+-------------------------------------------------------------------------------+
```

---

## 📂 Repository Structure

```
customer-churn-xai/
├── data/
│   ├── raw/
│   │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv       # Raw IBM dataset
│   ├── processed/
│   │   ├── cleaned_telco.parquet                      # Ingested Parquet
│   │   ├── preprocessed_telco.parquet                 # Cleaned dataset
│   │   └── engineered_telco.parquet                   # Feature-engineered dataset
│   └── hdfs_simulation/
│       └── warehouse/telco/                           # Partitioned HDFS parquet lake
├── notebooks/
│   └── exploratory_data_analysis.ipynb                # Interactive Jupyter Notebook
├── src/
│   ├── __init__.py
│   ├── spark_session.py                               # SparkSession manager
│   ├── data_ingestion.py                              # Phase 1: Ingestion & schema
│   ├── preprocessing.py                               # Phase 2: Cleaning & audit
│   ├── feature_engineering.py                         # Phase 4: Spark transformations
│   ├── train_models.py                                # Phase 5: Distributed ML pipelines
│   ├── evaluate_models.py                             # Phase 6: Metrics & ROC/PR curves
│   ├── explainability.py                              # Phase 7: SHAP XAI engine
│   ├── dashboard_data.py                              # Phase 3 & 8: EDA & BI metrics
│   └── generate_report.py                             # Phase 11: PDF/MD generator
├── dashboard/
│   ├── app.py                                         # Multi-page Streamlit application
│   └── styles.css                                     # Glassmorphic custom styling
├── models/                                            # Serialized pipelines & models
│   ├── Logistic_Regression_pipeline.pkl
│   ├── Random_Forest_pipeline.pkl
│   ├── Gradient_Boosted_Trees_pipeline.pkl
│   ├── best_model_metadata.json
│   └── shap_explainer.pkl
├── reports/
│   ├── data_quality_report.md
│   ├── final_report.md
│   ├── final_report.pdf
│   ├── figures/                                       # High-res publication charts
│   │   ├── roc_curves_comparison.png
│   │   ├── precision_recall_curves.png
│   │   ├── confusion_matrices_comparison.png
│   │   ├── shap_summary_beeswarm.png
│   │   ├── shap_feature_importance_bar.png
│   │   ├── shap_waterfall_High_Risk_Persona.png
│   │   └── ...
│   └── eda_and_business_insights.json
├── run_pipeline.py                                    # Master end-to-end runner
├── requirements.txt                                   # Python dependencies
└── README.md
```

---

## 📊 Dataset Description

- **Source:** IBM Telco Customer Churn Dataset (Kaggle)
- **Observations:** 7,043 customer accounts
- **Target Variable:** `Churn` (Yes / No)
- **Key Dimension Groups:**
  1. **Demographics:** `gender`, `SeniorCitizen`, `Partner`, `Dependents`
  2. **Account Services:** `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`
  3. **Contractual & Financial:** `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`, `tenure`

---

## ⚙️ Implementation Phases

### Phase 1: Data Ingestion (`src/data_ingestion.py`)
Loads raw CSV data into PySpark DataFrames, analyzes schema, computes partition null distributions, and persists data into parquet format simulating an HDFS lake.

### Phase 2: Data Cleaning & Quality Audit (`src/preprocessing.py`)
Identifies and replaces blank string occurrences in `TotalCharges`, imputes with median values, removes duplicates, and generates a formatted Markdown & JSON data quality audit.

### Phase 3 & 8: Exploratory Analysis & BI Layer (`src/dashboard_data.py`)
Executes Spark SQL queries and Plotly graphics to inspect churn rates across contract types, payment methods, internet services, and tenure durations. Computes financial metrics (Annualized Revenue at Risk).

### Phase 4: Feature Engineering (`src/feature_engineering.py`)
Constructs high-signal analytical attributes:
- `tenure_bucket`: Segmented lifecycle (`New`: 0-12, `Developing`: 13-24, `Stable`: 25-48, `Loyal`: 49+)
- `avg_monthly_spend`: `TotalCharges / (tenure + 1)`
- `contract_risk_score`: 3 (Month-to-month), 2 (One year), 1 (Two year)
- `service_count`: Count of active subscribed telecom bundles
- `support_usage_indicator`, `premium_customer_flag`, `electronic_check_risk`

### Phase 5 & 6: ML Pipelines & Model Evaluation (`src/train_models.py`, `src/evaluate_models.py`)
Constructs automated ML pipelines, evaluates models across Stratified 3-Fold Cross-Validation, generates ROC curves, Precision-Recall curves, and confusion matrices, and selects the champion model.

### Phase 7: Explainable AI with SHAP (`src/explainability.py`)
Computes global Shapley values (summary beeswarm, feature importance bar chart) and local waterfall/force explanations for 5 representative customer personas.

### Phase 9 & 10: Streamlit Web Dashboard (`dashboard/app.py`)
A reactive multi-page web application with interactive Plotly graphics, real-time single-customer risk scoring with live SHAP waterfall plots, batch CSV predictions, and report downloaders.

### Phase 11: Automated Executive Reporting (`src/generate_report.py`)
Produces publication-grade Markdown (`reports/final_report.md`) and PDF (`reports/final_report.pdf`) summaries.

---

## 🏆 Machine Learning Benchmark Results

| Model Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| **Gradient Boosted Trees (Champion)** | **80.55%** | **0.6724** | **0.5214** | **0.5874** | **0.8462** | **0.6618** |
| **Logistic Regression** | 80.27% | 0.6548 | 0.5481 | 0.5967 | 0.8431 | 0.6582 |
| **Random Forest Classifier** | 79.77% | 0.6680 | 0.4492 | 0.5369 | 0.8385 | 0.6471 |

---

## 🔬 Explainable AI (SHAP) Interpretation

### Macroscopic Drivers of Churn (Global Explainability):
1. **Contract Type (Month-to-Month):** The single strongest positive accelerator of churn risk.
2. **Tenure Duration:** The most significant protective anchor against customer departure.
3. **Monthly Charges & Fiber Optic Service:** High spend accelerates churn when lacking technical assistance.
4. **Electronic Check Payment Method:** Manual payment friction correlates with elevated attrition.
5. **Bundled Tech Support & Online Security:** Provide substantial retention stability.

---

## 💻 Step-by-Step Execution Guide

### 1. Local Environment Execution

```bash
# 1. Clone or navigate to the project directory
cd customer-churn-xai

# 2. Install required dependencies
pip install -r requirements.txt

# 3. Run the complete master pipeline (Phases 1 through 11)
python run_pipeline.py

# 4. Launch the interactive Streamlit Dashboard
python -m streamlit run dashboard/app.py
```

### 2. Google Colab Execution

To run in Google Colab:
```python
# In a Colab cell:
!git clone https://github.com/<your-username>/customer-churn-xai.git
%cd customer-churn-xai
!pip install -r requirements.txt
!python run_pipeline.py

# Launch Streamlit in Colab via localtunnel:
!streamlit run dashboard/app.py & npx localtunnel --port 8501
```

### 3. Databricks Execution

1. Upload `src/` modules and dataset to Databricks Workspace / DBFS.
2. Create a Databricks Cluster with Spark 3.5+ runtime.
3. Import `notebooks/exploratory_data_analysis.ipynb` and execute directly on the Spark cluster.
4. Output models can be saved directly to DBFS or MLflow Model Registry (`mlflow.spark.log_model`).

---

## 📄 Automated Reporting & Artifacts

- **Executive PDF Report:** `reports/final_report.pdf`
- **Research Markdown Report:** `reports/final_report.md`
- **Data Quality Audit:** `reports/data_quality_report.md`
- **Benchmarking Visualizations:** `reports/figures/`
- **Production Models:** `models/`

---

*Built with ❤️ for Big Data Analytics and Explainable AI Research.*
#   c u s t o m e r - c h u r n - x a i  
 