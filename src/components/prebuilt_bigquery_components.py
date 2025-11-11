"""
Pre-built BigQuery Components
=============================
Centralized loader for Google-provided BigQuery components.

Author: MLOps Team
Maintained by: Data Engineering Team

This module imports pre-built components from the Google Cloud Pipeline
Components (GCPC) registry. This centralizes external dependencies
and makes pipeline definitions cleaner.

Components:
- bigquery_query_job_op: Runs a BigQuery SQL query job

Usage:
    from components.prebuilt_bigquery_components import bigquery_query_job_op

    bq_task = bigquery_query_job_op(
        project=project_id,
        location=region,
        query="SELECT * FROM my_dataset.my_table"
    )
"""
