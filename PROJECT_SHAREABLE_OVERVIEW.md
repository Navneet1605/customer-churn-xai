# ⚡ Explainable Customer Churn Prediction using Apache Spark & Explainable AI (XAI)
### *Comprehensive Project Guide & Sharing Document*

---

## 📌 1. Project At a Glance

| Property | Details |
|---|---|
| **Project Title** | Explainable Customer Churn Prediction using Apache Spark and Explainable AI |
| **Domain** | Telecommunications & Big Data Predictive Intelligence |
| **GitHub Repository** | [https://github.com/Navneet1605/customer-churn-xai](https://github.com/Navneet1605/customer-churn-xai) |
| **Tech Stack** | Python, Apache Spark (PySpark), Spark MLlib, Spark SQL, SHAP, Streamlit, Plotly, Scikit-Learn |
| **Dataset** | IBM Telco Customer Churn (7,043 Accounts, 21 Attributes) |
| **Champion Model** | **Gradient Boosted Trees (GBT)** with **ROC-AUC: 0.8465**, **Accuracy: 80.41%** |
| **Key Deliverable** | Real-Time Interactive Streamlit Web Application + Game-Theoretic SHAP Interpretability Engine |

---

## 🎯 2. What Is It & What Problem Does It Solve?

In the telecommunications industry, acquiring a new customer costs **5 to 7 times more** than retaining an existing subscriber. 

Traditional machine learning solutions often act as **"black boxes"**—they tell you *which* customer is going to churn, but cannot explain *why* or what specific action retention teams should take.

This project delivers an **end-to-end, enterprise-grade Big Data & Explainable AI (XAI) platform** that:
1. **Processes big data at scale** using distributed-style Parquet partitioning and Apache Spark workflows.
2. **Predicts churn risk** with high precision across multiple candidate models (Logistic Regression, Random Forest, Gradient Boosted Trees).
3. **Explains every decision transparently using SHAP (SHapley Additive exPlanations)**, giving both macroscopic company-wide trends and single-customer waterfall risk breakdowns.
4. **Calculates financial revenue at risk** ($/month and $/year) to help executive teams prioritize retention campaigns.
5. **Provides an interactive, modern web dashboard** for instant single-customer risk scoring, batch CSV uploads, and automated PDF report downloads.

---

## 🏗️ 3. End-to-End System Architecture

```
+-----------------------------------------------------------------------------------+
| 1. BIG DATA INGESTION & STORAGE                                                   |
|    - Raw Telco Dataset (7,043 rows x 21 columns)                                  |
|    - PySpark schema inference, null counts audit, and HDFS parquet partitioning   |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
| 2. DATA CLEANSING & PREPROCESSING                                                 |
|    - Coerced blank TotalCharges strings & imputed median values ($1,397.48)       |
|    - Deduplication & Binary encoding (Churn: 1=Yes, 0=No)                         |
|    - Automated Markdown & JSON quality audit reports                              |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
| 3. DISTRIBUTED FEATURE ENGINEERING                                                |
|    - tenure_bucket (New, Developing, Stable, Loyal)                               |
|    - avg_monthly_spend, contract_risk_score, service_count                        |
|    - support_usage_indicator, premium_customer_flag, electronic_check_risk        |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
| 4. MACHINE LEARNING & CROSS-VALIDATION PIPELINE                                   |
|    - ML Pipeline: StringIndexer -> OneHotEncoder -> VectorAssembler -> Scaler    |
|    - Algorithms: Logistic Regression, Random Forest, Gradient Boosted Trees       |
|    - 3-Fold Stratified Cross-Validation & Hyperparameter Grid Search              |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
| 5. EXPLAINABLE AI (SHAP) ENGINE                                                   |
|    - Global Interpretability: Summary Beeswarm Plot & Feature Importance Ranking  |
|    - Local Interpretability: Waterfall & Force Plots for 5 Customer Archetypes    |
|    - Automated Natural Language Executive Reasoning Engine                        |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
| 6. INTERACTIVE STREAMLIT WEB DASHBOARD & PDF EXPORT                               |
|    - Live single-customer risk scoring with dynamic SHAP waterfall charts         |
|    - Batch CSV drag-and-drop prediction engine with downloadable scored results   |
|    - Financial exposure metrics & downloadable Executive PDF Reports              |
+-----------------------------------------------------------------------------------+
```

---

## ⚙️ 4. How Does It Work (Under the Hood)?

### Step 1: Ingestion & Distributed HDFS Simulation (`src/data_ingestion.py`)
- Analyzes dataset dimensions, schema data types, and missing fields.
- Exports partitions to `data/hdfs_simulation/warehouse/telco/` partitioned by `Contract`.

### Step 2: Data Quality & Cleansing (`src/preprocessing.py`)
- Coerces corrupt string representations in `TotalCharges` into floating-point representation.
- Imputes missing values using the statistical median, retaining 100% of data integrity.

### Step 3: Domain-Specific Feature Engineering (`src/feature_engineering.py`)
- **`tenure_bucket`**: Groups customers into `New` (0–12 mos), `Developing` (13–24 mos), `Stable` (25–48 mos), and `Loyal` (48+ mos).
- **`avg_monthly_spend`**: `TotalCharges / (tenure + 1)`
- **`contract_risk_score`**: Quantifies contractual churn propensity (Month-to-month = 3, One year = 2, Two year = 1).
- **`service_count`**: Total number of subscribed services.
- **`support_usage_indicator`**: Whether the subscriber has Online Security, Tech Support, or Backup services.

### Step 4: Machine Learning & Benchmarking (`src/train_models.py`, `src/evaluate_models.py`)
- Evaluates 3 distinct models with Stratified 3-Fold Cross-Validation:

| Model Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| **Gradient Boosted Trees (Champion)** | **80.41%** | **0.6678** | **0.5214** | **0.5856** | **0.8465** | **0.6639** |
| **Logistic Regression** | 80.62% | 0.6678 | 0.5374 | 0.5956 | 0.8457 | 0.6566 |
| **Random Forest Classifier** | 80.06% | 0.6716 | 0.4866 | 0.5643 | 0.8436 | 0.6564 |

### Step 5: Explainable AI with SHAP (`src/explainability.py`)
- **What Drives Churn Globally?**
  1. *Month-to-Month Contract Status* is the single greatest accelerator of churn.
  2. *Tenure Duration* is the strongest protective retention anchor.
  3. *High Monthly Charges without Tech Support* increases departure velocity.
  4. *Electronic Check Payment* adds manual payment friction that correlates with high churn.
- **Local Persona Interpretability:** Generates individual waterfall charts for 5 diverse customer archetypes (High Risk, Moderate Risk, Loyal, Support-Deficient, Price-Sensitive Senior).

### Step 6: Streamlit Web Dashboard (`dashboard/app.py`)
- **Page 1: Executive Overview:** High-level KPI cards and architectural flow.
- **Page 2: Data Overview & Schema:** Raw dataset viewer, missing value audit, and HDFS explorer.
- **Page 3: Exploratory Analytics (EDA):** Interactive Plotly charts (churn donut, contract bar charts, tenure density, correlation heatmap).
- **Page 4: Model Performance & ROC:** Benchmark leaderboard, ROC curves, PR curves, and confusion matrices.
- **Page 5: Live XAI Churn Predictor:** Interactive form where you input customer traits, get an instant churn probability, dynamic gauge, live SHAP waterfall chart, and plain-English recommendations.
- **Page 6: Batch Prediction Engine:** Drag-and-drop any CSV file, get batch predictions with high-risk financial exposure, and download scored CSV results.
- **Page 7: Business Insights & Reports:** Download full Executive PDF and Markdown reports.

---

## 🚀 5. How to Access & Run the Project

### Option A: Running Locally (Fastest & Interactive)

#### 1. Clone the GitHub Repository:
```bash
git clone https://github.com/Navneet1605/customer-churn-xai.git
cd customer-churn-xai
```

#### 2. Install Dependencies:
```bash
pip install -r requirements.txt
```

#### 3. Run the Master Analytics Pipeline (Executes all 11 Phases):
```bash
python run_pipeline.py
```

#### 4. Launch the Interactive Web Dashboard:
```bash
python -m streamlit run dashboard/app.py
```
*The dashboard will automatically open in your browser at `http://localhost:8501`!*

---

### Option B: Running in Google Colab (Cloud)

Copy and paste this snippet into a new Google Colab notebook:
```python
# 1. Clone repository
!git clone https://github.com/Navneet1605/customer-churn-xai.git
%cd customer-churn-xai

# 2. Install requirements & execute pipeline
!pip install -r requirements.txt
!python run_pipeline.py

# 3. Launch Streamlit with localtunnel
!python -m streamlit run dashboard/app.py & npx localtunnel --port 8501
```

---

### Option C: Running on Databricks

1. Import the project files into your Databricks Workspace.
2. Attach to a cluster running **Apache Spark 3.5+**.
3. Open [`notebooks/exploratory_data_analysis.ipynb`](file:///C:/Users/navne/.gemini/antigravity-ide/scratch/customer-churn-xai/notebooks/exploratory_data_analysis.ipynb) and run all cells directly on the distributed Spark cluster.

---

## 📁 6. Repository File Directory

```
customer-churn-xai/
├── data/
│   ├── raw/WA_Fn-UseC_-Telco-Customer-Churn.csv   # Raw dataset (7,043 records)
│   ├── processed/                                 # Cleaned & feature-engineered Parquet
│   └── hdfs_simulation/warehouse/telco/           # Partitioned HDFS Parquet lake
├── notebooks/
│   └── exploratory_data_analysis.ipynb            # Jupyter / Colab / Databricks Notebook
├── src/
│   ├── spark_session.py                           # SparkSession manager
│   ├── data_ingestion.py                          # Phase 1: Ingestion & schema
│   ├── preprocessing.py                           # Phase 2: Cleansing & audit
│   ├── feature_engineering.py                     # Phase 4: Spark transformations
│   ├── train_models.py                            # Phase 5: Distributed ML pipelines
│   ├── evaluate_models.py                         # Phase 6: Metrics & ROC/PR curves
│   ├── explainability.py                          # Phase 7: SHAP XAI engine
│   ├── dashboard_data.py                          # Phase 3 & 8: EDA & Financial BI
│   └── generate_report.py                         # Phase 11: PDF & Markdown reports
├── dashboard/
│   ├── app.py                                     # Interactive 7-page Streamlit web app
│   └── styles.css                                 # Glassmorphic custom CSS styling
├── models/                                        # Trained model pipelines & SHAP explainers
├── reports/
│   ├── final_report.pdf                           # Publication-ready executive summary
│   ├── final_report.md                            # Comprehensive Markdown report
│   ├── data_quality_report.md                     # Data quality audit
│   └── figures/                                   # 14 High-res benchmark & SHAP charts
├── run_pipeline.py                                # End-to-end master runner
├── requirements.txt                               # Dependencies
├── README.md                                      # Documentation
└── PROJECT_SHAREABLE_OVERVIEW.md                  # This shareable guide
```

---

## 🔗 7. Key Links & References

- **GitHub Repository:** [https://github.com/Navneet1605/customer-churn-xai](https://github.com/Navneet1605/customer-churn-xai)
- **Executive PDF Report:** [`reports/final_report.pdf`](file:///C:/Users/navne/.gemini/antigravity-ide/scratch/customer-churn-xai/reports/final_report.pdf)
- **Interactive Jupyter Notebook:** [`notebooks/exploratory_data_analysis.ipynb`](file:///C:/Users/navne/.gemini/antigravity-ide/scratch/customer-churn-xai/notebooks/exploratory_data_analysis.ipynb)
- **Main Streamlit Dashboard App:** [`dashboard/app.py`](file:///C:/Users/navne/.gemini/antigravity-ide/scratch/customer-churn-xai/dashboard/app.py)

---
*Created by Navneet Singh for Big Data Analytics and Explainable AI Research.*
