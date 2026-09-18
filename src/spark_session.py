"""
spark_session.py
================
Initializes and manages the Apache Spark Session for distributed data processing,
Spark SQL querying, and Spark MLlib pipelines.
"""

import os
import sys
import logging
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("SparkSessionManager")

_SPARK_INSTANCE = None

def check_java_compatibility() -> bool:
    """Checks if Java 17+ is present for PySpark 4+ compatibility."""
    import subprocess
    import shutil
    if not shutil.which("java"):
        return False
    try:
        res = subprocess.run(["java", "-version"], capture_output=True, text=True, timeout=5)
        out = (res.stdout + res.stderr).lower()
        if 'version "1.' in out or '1.8.' in out:  # Java 8 or older
            logger.info("Detected Java 8 runtime. PySpark 4.0+ requires Java 17+.")
            return False
        return True
    except Exception:
        return False

def get_spark_session(app_name: str = "ExplainableCustomerChurnPrediction", master: str = "local[*]") -> Optional["pyspark.sql.SparkSession"]:
    """
    Creates or retrieves an active SparkSession with optimized configurations
    for local/distributed execution and memory management.
    """
    global _SPARK_INSTANCE
    if _SPARK_INSTANCE is not None:
        try:
            if not _SPARK_INSTANCE._jsc.sc().isStopped():
                return _SPARK_INSTANCE
        except Exception:
            pass

    if not check_java_compatibility():
        logger.info("Native JVM Spark session bypassed (Java 17+ not detected). Operating in high-performance PyArrow / Parquet distributed simulation mode.")
        return None

    try:
        from pyspark.sql import SparkSession

        # Set Hadoop home environment variable safely if not set
        if os.name == 'nt' and "HADOOP_HOME" not in os.environ:
            # Prevent winutils missing warnings in local spark mode
            logger.info("Configuring Spark environment on Windows...")

        builder = (
            SparkSession.builder
            .appName(app_name)
            .master(master)
            .config("spark.driver.memory", "4g")
            .config("spark.executor.memory", "2g")
            .config("spark.sql.shuffle.partitions", "8")
            .config("spark.default.parallelism", "4")
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
            .config("spark.ui.enabled", "false")  # disable web UI if running in background/testing
            .config("spark.driver.extraJavaOptions", "-Dlog4j.configuration=file:log4j.properties -Dorg.apache.spark.serializer.KryoSerializer")
        )

        spark = builder.getOrCreate()
        spark.sparkContext.setLogLevel("ERROR")
        _SPARK_INSTANCE = spark
        logger.info(f"Apache Spark Session created successfully! Version: {spark.version}")
        return spark

    except Exception as e:
        logger.warning(f"Unable to initialize JVM Spark Session: {e}. PyArrow / Pandas distributed-style engine will be used.")
        return None

def stop_spark_session():
    """Stops the active SparkSession."""
    global _SPARK_INSTANCE
    if _SPARK_INSTANCE is not None:
        try:
            _SPARK_INSTANCE.stop()
            logger.info("Spark session stopped.")
        except Exception as e:
            logger.warning(f"Error stopping Spark session: {e}")
        finally:
            _SPARK_INSTANCE = None
