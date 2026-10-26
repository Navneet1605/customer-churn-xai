# Explainable Customer Churn Prediction

An end-to-end Big Data Analytics and Explainable AI (XAI) system designed to predict customer churn in telecommunications and interpret every decision transparently using SHAP values.

## Architecture & Workflow

The system is built to process raw data, train a machine learning model, and provide a user-friendly dashboard for real-time inference.

1. **Data Ingestion & Processing:**
   - Raw data is ingested using Apache Spark.
   - Parquet partitioning is simulated for big data scalability.
   - Missing values are imputed and categorical variables are normalized.

2. **Feature Engineering:**
   - Domain-specific features are created, such as tenure buckets, average monthly spend, and contract risk scores.

3. **Machine Learning Pipeline:**
   - Distributed ML pipelines (Spark MLlib / Scikit-Learn) apply scaling and encoding.
   - A Gradient Boosted Trees (GBT) model is trained and tuned as the champion model.

4. **Explainable AI (SHAP):**
   - Global feature importance is calculated to understand macro churn drivers.
   - Local waterfall plots are generated to explain the precise risk factors for individual customers.

5. **Prescriptive AI & Counterfactual Optimization:**
   - Algorithmic Minimal Actionable Recourse solves for the lowest-cost intervention that flips high-risk accounts into secure subscribers.
   - Standardized enterprise action packages model Customer Lifetime Value (CLTV), preserved revenue, and return on retention spend (ROI).
   - Cohort Resource Allocation Matrix segments subscribers into high-touch, automated, cost-sensitive, and secure quadrants.

6. **Interactive Dashboard:**
   - A streamlined Streamlit web application with executive KPI cards.
   - Interactive "What-If" sandbox for live policy simulation, side-by-side pre/post risk gauges, and batch CSV scoring.

## How to Use

### Prerequisites
Ensure you have Python 3.11+ installed.

### 1. Install Dependencies
Install the required Python packages:
```bash
pip install -r requirements.txt
```

### 2. Run the End-to-End Pipeline
Execute the master script to ingest data, engineer features, train the model, and generate reports:
```bash
python run_pipeline.py
```

### 3. Launch the Dashboard
Start the interactive web application to view insights and run predictions:
```bash
python -m streamlit run app.py
```

## Repository Structure

- `app.py`: The main Streamlit web application.
- `run_pipeline.py`: Master script to execute the data and ML pipeline.
- `src/`: Contains core modules for data ingestion, preprocessing, training, and explainability.
- `models/`: Stores the serialized champion pipelines and SHAP explainers.
- `data/`: Contains raw CSVs and processed partitioned Parquet files.
- `reports/`: Contains generated analysis, benchmark metrics, and figures.