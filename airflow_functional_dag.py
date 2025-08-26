"""Simple Airflow DAG for Azure SQL to GCS extraction.

Functional, DRY DAG that follows the original logic.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from airflow import DAG
from airflow.operators.python import PythonOperator

from main import main as run_extraction_pipeline


class ExtractionPipelineError(Exception):
    """Exception for pipeline failures."""


def run_azure_sql_extraction() -> None:
    """Run the Azure SQL extraction pipeline in Airflow."""
    try:
        # Use the main pipeline function
        run_extraction_pipeline()
        print("✅ Azure SQL extraction completed successfully!")
        
    except SystemExit as e:
        if e.code != 0:
            raise ExtractionPipelineError(f"Pipeline failed with exit code: {e.code}")
        # Exit code 0 is success
        
    except Exception as e:
        raise ExtractionPipelineError(f"Pipeline execution failed: {e}") from e


# Simple DAG configuration
dag = DAG(
    "azure_sql_extraction",
    default_args={
        "owner": "data-engineering", 
        "depends_on_past": False,
        "start_date": datetime(2024, 1, 1, tzinfo=timezone.utc),
        "email_on_failure": True,
        "retries": 2,
        "retry_delay": timedelta(minutes=10),
    },
    description="Simple Azure SQL to GCS data extraction using DLT",
    schedule_interval="0 2 * * *",  # Daily at 2 AM
    catchup=False,
    tags=["dlt", "azure-sql", "gcs", "extraction"],
    max_active_runs=1,
)

# Single task that runs the extraction pipeline
extraction_task = PythonOperator(
    task_id="run_azure_sql_extraction",
    python_callable=run_azure_sql_extraction,
    dag=dag,
)
