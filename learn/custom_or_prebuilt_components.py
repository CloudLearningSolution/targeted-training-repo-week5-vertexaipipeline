"""
Machine Learning Operations Playbook Adoption Workshop – Phase 2:
Data Services Integration Architecture - Scavenger Hunt

File: custom_or_prebuilt_components.py

Purpose:
--------
This file provides scavenger hunt instructions for learners to explore
existing AWS-based labs (Lab 6.1 and Lab 6.2) and plan migration to
Vertex AI architecture. The focus is on Amazon S3 and Amazon Redshift
integration patterns, and how to prepare to convert them into Vertex AI
components using Kubeflow @component decorators.

Learners should use VSCode/PyCharm search to locate the TODO markers
listed below and record WHERE (line of code), WHAT (purpose), and WHY
(rationale for migration). This stage is planning-only: code remains in
its AWS state, but learners should envision how it will map to Vertex AI
components.

Target Vertex Architecture Structure:
-------------------------------------
├── src/
│   ├── components/
│   │   ├── __init__.py
│   │   │
│   │   ├── custom_data_quality_components.py      # ✅ Custom
│   │   ├── custom_training_components.py          # ✅ Custom
│   │   ├── custom_evaluation_components.py        # ✅ Custom
│   │   ├── custom_registry_components.py          # ✅ Custom
│   │   ├── custom_monitoring_components.py        # ✅ Custom
│   │   ├── custom_audit_components.py             # ✅ Custom
│   │   ├── custom_sysco_modelplaceholder_components.py        # ✅ Custom
│   │   │
│   │   └── prebuilt_bigquery_components.py        # ✅ Pre-built

Scavenger Hunt Instructions:
----------------------------

1. Lab 6.1 — Amazon S3 Integration with SageMaker Workflows
   - Search: "# TODO: Lab 6.1.1 - Line-by-Line Import Exploration"
     * WHERE: Top of model.py imports
     * WHAT: Identify boto3/joblib imports
     * WHY: These libraries enable artifact persistence in S3
     * Migration Planning: In Vertex AI, this logic would move into
       custom_training_components.py with @component decorators, using
       GCS (gs:// URIs) instead of S3.
   - Search: "# TODO: Lab 6.1.4 - S3 Data Loading Conversion"
     * WHERE: _s3_persist() function in model.py
     * WHAT: Inspect boto3.upload_file usage
     * WHY: Durable storage pattern in AWS
     * Migration Planning: Replace with GCS client logic inside a
       @component in custom_registry_components.py.

   AWS Information to Gather for Migration:
   - S3 bucket name (e.g., `my-ml-artifacts-bucket`)
   - Bucket region (e.g., `us-east-1`)
   - IAM role or access keys with `AmazonS3FullAccess`
   - Artifact paths (prefixes like `s3://bucket/models/`)
   - Current SageMaker registry integration points

   Equivalent in GCP:
   - GCS bucket name (e.g., `gs://my-ml-artifacts`)
   - GCP project ID and region
   - Service account with `Storage Admin` role
   - Artifact paths in GCS (prefixes like `gs://bucket/models/`)

2. Lab 6.2 — Amazon Redshift Data Pipeline and ML Integration
   - Search: "# TODO: Lab 6.2.1 - Data Access Pattern Conversion"
     * WHERE: ingest_model.py _read_from_redshift()
     * WHAT: Inspect select_sql_from_dict or pd.read_sql usage
     * WHY: Redshift → DataFrame conversion
     * Migration Planning: Equivalent logic would move into
       prebuilt_bigquery_components.py using BigQuery query components.
   - Search: "# TODO: Lab 6.2.4 - Data Movement and Performance Considerations"
     * WHERE: stage_table_to_s3() in ingest_model.py
     * WHAT: Inspect UNLOAD vs client-side upload patterns
     * WHY: Efficiency vs cost trade-offs in Redshift
     * Migration Planning: Replace with BigQuery export jobs inside
       prebuilt_bigquery_components.py or custom_data_quality_components.py.

   AWS Information to Gather for Migration:
   - Redshift cluster identifier (e.g., `redshift-cluster-1`)
   - Database name (e.g., `analytics_db`)
   - Schema names (e.g., `public`, `ml_features`)
   - User credentials or IAM role with Redshift access
   - Connection endpoint (host, port)
   - Common SQL queries used for ETL (COPY, UNLOAD, CTAS)

   Equivalent in GCP:
   - BigQuery dataset name (e.g., `ml_features_dataset`)
   - BigQuery table names (e.g., `training_data`, `evaluation_data`)
   - GCP project ID and region
   - Service account with `BigQuery Admin` role
   - SQL queries adapted to BigQuery syntax (SELECT, CREATE TABLE AS)

3. Planning Migration with Vertex Kubeflow @component Decorators
   - For S3 → GCS:
     * Wrap artifact persistence logic in @component functions inside
       custom_training_components.py and custom_registry_components.py.
     * Replace boto3 calls with google-cloud-storage client calls.
   - For Redshift → BigQuery:
     * Wrap ETL and query logic in @component functions inside
       prebuilt_bigquery_components.py.
     * Replace psycopg2/sqlalchemy calls with google-cloud-bigquery client
       or prebuilt BigQuery components.

Learner Deliverable:
--------------------
For each TODO marker found:
- Record WHERE: file name and line of code
- Record WHAT: the code pattern or component
- Record WHY: the rationale for its use in AWS
- Record Migration Plan: which Vertex component file it would map to
  (custom_* or prebuilt_bigquery_components.py) with @component decorator

This scavenger hunt prepares learners to design the component structure
shown above by understanding the migration path from AWS (S3, Redshift)
to Vertex AI (GCS, BigQuery, Pipeline components).
"""
