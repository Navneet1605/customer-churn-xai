"""
run_pipeline.py
===============
Master End-to-End Execution Pipeline
Orchestrates all 11 phases:
Phase 1: Big Data Ingestion (PySpark / HDFS Simulation)
Phase 2: Data Cleaning & Quality Audit
Phase 3: Exploratory Data Analysis & Visualizations
Phase 4: Feature Engineering & Domain Enrichment
Phase 5: MLlib Pipeline & Hyperparameter Tuning
Phase 6: Comprehensive Model Evaluation & Champion Selection
Phase 7: Explainable AI with SHAP (Global & Local Personas)
Phase 8: Business Intelligence & Revenue Exposure
Phase 9-10: Dashboard Data Prep & Dynamic Assets
Phase 11: Publication-Quality Markdown & PDF Reporting
"""

import sys
import time
import logging
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent))

from src.data_ingestion import run_data_ingestion
from src.preprocessing import run_preprocessing
from src.feature_engineering import run_feature_engineering
from src.train_models import run_model_training
from src.evaluate_models import run_model_evaluation
from src.explainability import run_explainability
from src.dashboard_data import run_eda_and_business_intelligence
from src.generate_report import run_report_generation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("PipelineOrchestrator")


def main():
    start_time = time.time()
    logger.info("=======================================================================")
    logger.info("  EXPLAINABLE CUSTOMER CHURN PREDICTION USING APACHE SPARK & XAI      ")
    logger.info("  Initiating Master Analytics & Machine Learning Pipeline...         ")
    logger.info("=======================================================================")

    # Phase 1: Ingestion
    logger.info("\n>>> PHASE 1: Executing Big Data Ingestion & Storage Workflow...")
    ingestion_meta = run_data_ingestion()

    # Phase 2: Preprocessing
    logger.info("\n>>> PHASE 2: Executing Data Preprocessing & Quality Audit...")
    preprocessing_meta = run_preprocessing()

    # Phase 4: Feature Engineering
    logger.info("\n>>> PHASE 4: Executing Distributed Feature Engineering...")
    feature_meta = run_feature_engineering()

    # Phase 3 & 8: EDA & Business Intelligence
    logger.info("\n>>> PHASE 3 & 8: Executing Exploratory Analytics & BI Engine...")
    eda_meta = run_eda_and_business_intelligence()

    # Phase 5: Model Training
    logger.info("\n>>> PHASE 5: Executing ML Pipelines & Model Tuning...")
    train_meta = run_model_training()

    # Phase 6: Model Evaluation
    logger.info("\n>>> PHASE 6: Executing Model Benchmarking & Metric Evaluation...")
    eval_meta = run_model_evaluation()

    # Phase 7: Explainable AI
    logger.info("\n>>> PHASE 7: Executing SHAP Interpretability & Persona Explanations...")
    xai_meta = run_explainability()

    # Phase 11: Final Report Generation
    logger.info("\n>>> PHASE 11: Generating Executive Research Reports (Markdown & PDF)...")
    reports_meta = run_report_generation()

    elapsed = time.time() - start_time
    logger.info("=======================================================================")
    logger.info(f"  PIPELINE EXECUTION COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS!  ")
    logger.info("  Generated Artifacts:")
    logger.info(f"   - Processed Parquet Data: data/processed/")
    logger.info(f"   - Trained Models: models/")
    logger.info(f"   - Executive Report (MD): {reports_meta.get('markdown_report')}")
    logger.info(f"   - Executive Report (PDF): {reports_meta.get('pdf_report')}")
    logger.info(f"   - Figures & Visualizations: reports/figures/")
    logger.info("  To launch the interactive dashboard, run:")
    logger.info("   streamlit run dashboard/app.py")
    logger.info("=======================================================================")


if __name__ == "__main__":
    main()
