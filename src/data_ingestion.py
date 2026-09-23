"""
data_ingestion.py
=================
Phase 1: Big Data Ingestion using Apache Spark
- Loads raw Telco Customer Churn dataset into Spark DataFrame
- Analyzes schema, data shape, and null count per partition
- Simulates HDFS distributed storage workflow (Parquet partitioning)
- Persists processed baseline to data/processed/cleaned_telco.parquet
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

# Add parent directory to path for module imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.spark_session import get_spark_session

logger = logging.getLogger("DataIngestion")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def run_data_ingestion(
    raw_csv_path: str = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv",
    output_parquet_path: str = "data/processed/cleaned_telco.parquet",
    hdfs_simulation_path: str = "data/hdfs_simulation/warehouse/telco/"
) -> Dict[str, Any]:
    """
    Executes Phase 1 data ingestion workflow.
    """
    raw_path = Path(raw_csv_path).resolve()
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data file not found at {raw_path}")

    logger.info(f"Starting Big Data Ingestion from: {raw_path}")
    
    spark_available = False
    spark = None
    try:
        spark = get_spark_session("TelcoDataIngestion")
        spark_available = (spark is not None)
    except Exception as e:
        logger.warning(f"Spark JVM session could not be started ({e}). Running with PyArrow / Pandas backend.")
        spark_available = False

    metadata: Dict[str, Any] = {
        "source_file": str(raw_path),
        "engine": "Apache Spark (PySpark)" if spark_available else "PyArrow / Pandas",
        "columns": [],
        "num_rows": 0,
        "num_cols": 0,
        "schema": {},
        "null_counts": {},
        "hdfs_storage_path": str(Path(hdfs_simulation_path).resolve()),
        "output_parquet_path": str(Path(output_parquet_path).resolve()),
    }

    if spark_available and spark is not None:
        # Load CSV into PySpark DataFrame
        df_spark = spark.read.csv(str(raw_path), header=True, inferSchema=True)
        
        num_rows = df_spark.count()
        num_cols = len(df_spark.columns)
        columns = df_spark.columns
        schema_dict = {field.name: str(field.dataType) for field in df_spark.schema.fields}
        
        # Calculate null counts using Spark SQL aggregation
        from pyspark.sql import functions as F
        null_exprs = [F.count(F.when(F.col(c).isNull() | (F.trim(F.col(c)) == ""), c)).alias(c) for c in columns]
        null_counts_row = df_spark.agg(*null_exprs).collect()[0].asDict()

        metadata["num_rows"] = num_rows
        metadata["num_cols"] = num_cols
        metadata["columns"] = columns
        metadata["schema"] = schema_dict
        metadata["null_counts"] = null_counts_row

        logger.info(f"Loaded Spark DataFrame with {num_rows} rows and {num_cols} columns.")
        logger.info(f"Schema Details: {schema_dict}")
        logger.info(f"Null Counts: {null_counts_row}")

        # Save to Processed Parquet (Simulate HDFS warehouse partitioning)
        out_path = Path(output_parquet_path).resolve()
        hdfs_path = Path(hdfs_simulation_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        hdfs_path.mkdir(parents=True, exist_ok=True)

        try:
            df_spark.write.mode("overwrite").parquet(str(out_path))
            df_spark.write.mode("overwrite").partitionBy("Contract").parquet(str(hdfs_path))
            logger.info(f"Exported Parquet partitions to HDFS simulation path: {hdfs_path}")
        except Exception as write_err:
            logger.warning(f"Native Spark Parquet write fallback: {write_err}")
            # Fallback export via pandas
            import shutil
            if out_path.exists() and out_path.is_dir():
                shutil.rmtree(out_path)
            pdf = df_spark.toPandas()
            pdf.to_parquet(str(out_path), index=False)
            pdf.to_parquet(str(hdfs_path / "telco_data.parquet"), index=False)

    else:
        # High-performance Pandas / PyArrow fallback
        import pandas as pd
        pdf = pd.read_csv(str(raw_path))
        num_rows, num_cols = pdf.shape
        columns = pdf.columns.tolist()
        schema_dict = {col: str(dtype) for col, dtype in pdf.dtypes.items()}
        
        # Count nulls and blanks
        null_counts = {}
        for col in columns:
            null_counts[col] = int(pdf[col].isna().sum() + (pdf[col].astype(str).str.strip() == '').sum())

        metadata["num_rows"] = num_rows
        metadata["num_cols"] = num_cols
        metadata["columns"] = columns
        metadata["schema"] = schema_dict
        metadata["null_counts"] = null_counts

        out_path = Path(output_parquet_path).resolve()
        hdfs_path = Path(hdfs_simulation_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        hdfs_path.mkdir(parents=True, exist_ok=True)

        pdf.to_parquet(str(out_path), index=False)
        # Partitioned Parquet simulation
        pdf.to_parquet(str(hdfs_path / "telco_partitioned.parquet"), partition_cols=["Contract"], index=False)
        logger.info(f"Ingested and saved {num_rows} rows to {out_path} and HDFS structure {hdfs_path}")

    # Write ingestion metadata summary
    meta_file = Path("reports/ingestion_metadata.json")
    meta_file.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_file, "w") as f:
        json.dump(metadata, f, indent=4)

    logger.info("Phase 1: Big Data Ingestion Completed Successfully.")
    return metadata


if __name__ == "__main__":
    run_data_ingestion()
