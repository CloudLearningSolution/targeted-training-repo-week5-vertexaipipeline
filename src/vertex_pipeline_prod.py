"""
Vertex AI KFP Pipeline for Production

This pipeline orchestrates the production workflow for the diabetes prediction model.
It includes the following steps:
- Queries diabetes data from BigQuery Feature Group view (splits via query logic).
- Trains a logistic regression model using scikit-learn.
- Evaluates the model on the test set and logs accuracy.
- Conditionally registers a new version of the model in Vertex AI Model Registry
  if accuracy meets the minimum threshold, supporting parent model versioning.
- Rejects the model if accuracy is below the required threshold.

All comments and documentation lines are kept <= 100 characters for .flake8.

===============================================================================
Lab 5.4: Vertex AI Pipeline Component Architecture Exploration
===============================================================================
This production pipeline demonstrates the same component architecture as the
development pipeline, but with production-specific configurations:
- Higher accuracy thresholds for quality control
- Production data sources and logging
- Model versioning support for production deployments
- Enhanced audit trail and monitoring

Understanding the similarities and differences between dev and prod pipelines
helps identify how the same component architecture adapts to different environments.

TODO: Lab 5.4.1 - Component Identification: Production pipeline with same component types as dev
TODO: Lab 5.4.2 - Purpose Recognition: Production-optimized configuration of components
TODO: Lab 5.4.3 - Architecture Understanding: Environment-specific pipeline architecture patterns

===============================================================================
Lab 5.5: Vertex AI Custom Components and Pre-built Components and Accelerator Templates
===============================================================================
This PRODUCTION pipeline demonstrates the same custom component strategy as the
development pipeline, with production-grade configurations and stricter quality gates.

ACCELERATOR TEMPLATE ARCHITECTURE (Production Environment):
==========================================================
1. INFRASTRUCTURE LAYER (Terraform):
   - Production Vertex AI resources with high availability
   - Separate service accounts for production isolation
   - Production GCS buckets with retention policies
   - Production BigQuery datasets and Feature Groups
   - Production-specific IAM and security controls
   - See: vertex_ai_infrastructure.tf (production workspace/environment)

2. PIPELINE LAYER (This File - Vertex AI Production):
   - Same components as dev (custom and pre-built mix)
   - Production configuration and parameters
   - Higher accuracy thresholds (0.75 vs 0.70)
   - Production data sources
   - Enhanced logging and audit trail
   - Model versioning for production lineage

3. ENTERPRISE LAYER (Production Best Practices):
   - Compliance and governance requirements
   - Production deployment workflows
   - Monitoring and alerting integration
   - Disaster recovery and backup strategies

COMPONENT STRATEGY - PRODUCTION CONSIDERATIONS:
==============================================
Production pipelines use the SAME COMPONENTS as development, but with:
- Stricter quality gates and thresholds
- Enhanced error handling and logging
- Production data sources and paths
- Model versioning support
- Integration with monitoring systems

This demonstrates COMPONENT REUSABILITY across environments - a key benefit
of the component architecture.

TODO: Lab 5.5.1 - Custom vs Pre-built: Same components as dev, production configuration
TODO: Lab 5.5.2 - Pre-built Alternatives: Same BigQuery integration as dev
TODO: Lab 5.5.3 - Accelerator Templates: Production-grade 3-layer architecture
TODO: Lab 5.5.4 - Environment Reusability: Same components, different configurations
TODO: Lab 5.5.5 - BigQuery Integration: Production uses same data architecture as dev
===============================================================================
"""

from kfp import dsl, components
from kfp.dsl import (
    component,
    pipeline,
    Input,
    Output,
    Model,
    Metrics
)
# TODO: Lab 5.5.5 - BigQuery Integration: Import BQTable artifact type for BigQuery table inputs
from google_cloud_pipeline_components.types import artifact_types

# TODO: Lab 5.4.1 - Component Identification: Pipeline metadata distinguishes prod from dev
# TODO: Lab 5.4.2 - Purpose Recognition: Separate pipeline names enable environment tracking
PIPELINE_NAME = "mlops-diabetes-prod-pipeline"
PIPELINE_DESCRIPTION = (
    "Production pipeline for diabetes prediction model on Vertex AI with BigQuery"
)

# TODO: Lab 5.4.1 - Component Identification: Shared base image ensures environment consistency
# TODO: Lab 5.4.3 - Architecture Understanding: Consistent component execution across environments
BASE_IMAGE = "python:3.9"
REQUIREMENTS_PATH = "src/requirements.txt"

# ==============================================================================
# PRE-BUILT COMPONENT: BigQuery Query Job
# ==============================================================================
# TODO: Lab 5.5.1 - Pre-built Component: Load BigQuery Query Job from Google Cloud registry
# TODO: Lab 5.5.2 - Pre-built vs Custom: Using Google-maintained component for data access
# TODO: Lab 5.5.5 - BigQuery Integration: Pre-built component executes SQL queries (production)
#
# TODO: Lab 5.6.3 — Caching and retry guidance
# INSPECT: Pre-built BigQuery Query Job component (same as dev pipeline).
# PRODUCTION CACHING GUIDANCE: DISABLE caching for production pipelines to ensure fresh data.
# NOTE: Even though component is idempotent, production should execute fresh queries.
# RETRY GUIDANCE: Built-in retries handle transient BigQuery API errors automatically.
#
# Load the BigQuery Query Job component (same as dev pipeline)
bigquery_query_job_op = components.load_component_from_url(
    'https://us-kfp.pkg.dev/ml-pipeline/google-cloud-registry/'
    'bigquery-query-job/sha256:'
    'd1cae80bc0de4e5b95b994739c8d0d7d42ce5a4cb17d3c9512eaed14540f6343'
)


# ==============================================================================
# Component: train_model_op (MODIFIED FOR BIGQUERY - PRODUCTION)
# ==============================================================================
# TODO: Lab 5.4.1 - Component Identification: Identical training component to dev pipeline
# TODO: Lab 5.4.2 - Purpose Recognition: Same training algorithm across environments
# TODO: Lab 5.4.3 - Architecture Understanding: Component reusability across pipeline variants
#
# TODO: Lab 5.5.1 - Custom Component Analysis: This is a CUSTOM COMPONENT (same as dev)
# TODO: Lab 5.5.4 - Component Reusability: Exact same component definition in dev and prod
# TODO: Lab 5.5.2 - Custom vs Pre-built Decision: Production uses identical custom training component
# TODO: Lab 5.5.5 - BigQuery Integration: Reads from BigQuery table via Python client (production)
#
# TODO: Lab 5.6.3 — Caching and retry guidance
# INSPECT: Training component is IDEMPOTENT (same as dev).
# PRODUCTION CACHING GUIDANCE: DISABLE caching - production pipelines should always execute fresh.
# PRODUCTION RETRY GUIDANCE: SAFE TO RETRY - no side effects, deterministic training.
# NOTE: Production environments typically disable caching for compliance and audit requirements.
#
# Trains a logistic regression model using the training data from BigQuery
# and saves the model artifact for downstream evaluation and registration.
@component(
    base_image=BASE_IMAGE,
    packages_to_install=[
        pkg.strip()
        for pkg in open(REQUIREMENTS_PATH)
        if pkg.strip() and not pkg.startswith("#")
    ],
)
def train_model_op(
    # TODO: Lab 5.5.5 - BigQuery Integration: Input uses BQTable artifact type for BigQuery tables
    train_data: Input[artifact_types.BQTable],
    output_model: Output[Model],
    reg_rate: float,
    # TODO: Lab 5.5.5 - BigQuery Integration: New parameters for BigQuery access (production)
    project_id: str,
    bq_location: str
):
    import pandas as pd
    import joblib
    from sklearn.linear_model import LogisticRegression
    from google.cloud import bigquery
    import logging
    import os
    import shutil
    import re

    FEATURE_COLUMNS = [
        "Pregnancies", "PlasmaGlucose", "DiastolicBloodPressure",
        "TricepsThickness", "SerumInsulin", "BMI", "DiabetesPedigree", "Age"
    ]

    # TODO: Lab 5.6.5 — Observability: metrics, logging, and dashboard links
    # INSPECT: Production logging setup (same pattern as dev, different prefix).
    # NOTE: Production logs are critical for compliance and audit trails.
    
    logging.basicConfig(level=logging.INFO)
    
    # TODO: Lab 5.5.5 - BigQuery Integration: Parse BigQuery table reference from artifact URI
    uri = train_data.uri
    
    # TODO: Lab 5.6.5 — Observability: metrics, logging, and dashboard links
    # INSPECT: Production logging with [PROD] prefix for environment identification.
    
    logging.info("[PROD] Parsing BigQuery table from URI: %s", uri)
    match = re.search(r'projects/([^/]+)/datasets/([^/]+)/tables/([^/]+)', uri)
    if not match:
        # TODO: Lab 5.6.7 — Failure modes and debugging tips
        # INSPECT: URI parsing error (same as dev, production context).
        # PRODUCTION DEBUGGING: Check upstream BigQuery Query Job success in production logs.
        
        raise ValueError(f"Could not parse BigQuery table reference from URI: {uri}")
    proj, dataset, table = match.groups()
    table_ref = f"{proj}.{dataset}.{table}"

    logging.info("[PROD] Reading training data from BigQuery: %s", table_ref)
    bq_client = bigquery.Client(project=project_id, location=bq_location)
    query = f"SELECT * FROM `{table_ref}`"
    train_df = bq_client.query(query).to_dataframe()
    logging.info("[PROD] Loaded %d training rows from BigQuery", len(train_df))
    
    X = train_df[FEATURE_COLUMNS]
    y = train_df["Diabetic"]

    # Train logistic regression model with regularization rate
    model = LogisticRegression(C=1 / reg_rate, solver="liblinear")
    model.fit(X, y)

    # Save model as model.joblib for Vertex AI compatibility
    model_path = os.path.join(os.path.dirname(output_model.path), "model.joblib")
    joblib.dump(model, model_path)
    shutil.copy(model_path, output_model.path)
    # TODO: Lab 5.4.2 - Purpose Recognition: Production logging for audit and compliance
    logging.info(
        "[PROD] Model trained and stored at: %s and copied to: %s",
        model_path,
        output_model.path
    )


# ==============================================================================
# Component: evaluate_model_op (MODIFIED FOR BIGQUERY - PRODUCTION)
# ==============================================================================
# TODO: Lab 5.4.1 - Component Identification: Identical evaluation component to dev pipeline
# TODO: Lab 5.4.2 - Purpose Recognition: Consistent evaluation logic across environments
# TODO: Lab 5.4.3 - Architecture Understanding: Component architecture supports shared evaluation
#
# TODO: Lab 5.5.1 - Custom Component Analysis: This is a CUSTOM COMPONENT (same as dev)
# TODO: Lab 5.5.4 - Component Reusability: Same evaluation logic in both environments
# TODO: Lab 5.5.2 - Custom vs Pre-built Decision: Production uses same custom evaluation
# TODO: Lab 5.5.5 - BigQuery Integration: Reads test data from BigQuery table (production)
#
# TODO: Lab 5.6.3 — Caching and retry guidance
# INSPECT: Evaluation component is IDEMPOTENT (same as dev).
# PRODUCTION CACHING GUIDANCE: DISABLE caching - production requires fresh evaluation.
# PRODUCTION RETRY GUIDANCE: SAFE TO RETRY - deterministic evaluation, no side effects.
#
# Evaluates the trained model on the test set from BigQuery, logs accuracy,
# and returns accuracy for conditional pipeline logic.
@component(
    base_image=BASE_IMAGE,
    packages_to_install=[
        pkg.strip()
        for pkg in open(REQUIREMENTS_PATH)
        if pkg.strip() and not pkg.startswith("#")
    ],
)
def evaluate_model_op(
    # TODO: Lab 5.4.1 - Component Identification: Multiple inputs create multi-dependency component
    # TODO: Lab 5.4.3 - Architecture Understanding: Component depends on two upstream components
    # TODO: Lab 5.5.5 - BigQuery Integration: Input uses BQTable artifact type for BigQuery tables
    test_data: Input[artifact_types.BQTable],
    model: Input[Model],
    metrics: Output[Metrics],
    min_accuracy: float,
    # TODO: Lab 5.5.5 - BigQuery Integration: New parameters for BigQuery access (production)
    project_id: str,
    bq_location: str
) -> float:
    import pandas as pd
    import joblib
    from sklearn.metrics import accuracy_score
    from google.cloud import bigquery
    import logging
    import re

    FEATURE_COLUMNS = [
        "Pregnancies", "PlasmaGlucose", "DiastolicBloodPressure",
        "TricepsThickness", "SerumInsulin", "BMI", "DiabetesPedigree", "Age"
    ]

    logging.basicConfig(level=logging.INFO)
    
    # TODO: Lab 5.5.5 - BigQuery Integration: Parse BigQuery table reference from artifact URI
    uri = test_data.uri
    logging.info("[PROD] Parsing BigQuery table from URI: %s", uri)
    match = re.search(r'projects/([^/]+)/datasets/([^/]+)/tables/([^/]+)', uri)
    if not match:
        raise ValueError(f"Could not parse BigQuery table reference from URI: {uri}")
    proj, dataset, table = match.groups()
    table_ref = f"{proj}.{dataset}.{table}"

    logging.info("[PROD] Reading test data from BigQuery: %s", table_ref)
    bq_client = bigquery.Client(project=project_id, location=bq_location)
    query = f"SELECT * FROM `{table_ref}`"
    test_df = bq_client.query(query).to_dataframe()
    logging.info("[PROD] Loaded %d test rows from BigQuery", len(test_df))
    
    model_artifact = joblib.load(model.path)

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["Diabetic"]
    predictions = model_artifact.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    
    # TODO: Lab 5.6.5 — Observability: metrics, logging, and dashboard links
    # INSPECT: metrics.log_metric() calls (same as dev).
    # PRODUCTION NOTE: Same metrics logged for dev/prod comparison and monitoring.
    # TASK: Note these metrics appear in production Vertex AI console for tracking.
    
    metrics.log_metric("accuracy", accuracy)
    metrics.log_metric("min_accuracy_threshold", min_accuracy)

    logging.info("[PROD] Accuracy = %.4f", accuracy)
    return accuracy


# ==============================================================================
# Component: model_approved_op
# ==============================================================================
# TODO: Lab 5.4.1 - Component Identification: Approval component for production audit trail
# TODO: Lab 5.4.2 - Purpose Recognition: Enhanced logging for production compliance
#
# TODO: Lab 5.6.3 — Caching and retry guidance
# INSPECT: Approval component is IDEMPOTENT (logging only).
# PRODUCTION CACHING GUIDANCE: Safe to cache but typically disabled in production.
# RETRY GUIDANCE: SAFE TO RETRY - no side effects.
#
# Logs approval message if model accuracy meets threshold. Used for audit trail.
@component(
    base_image=BASE_IMAGE
)
def model_approved_op(model_accuracy: float, model: Input[Model]):
    import logging
    logging.basicConfig(level=logging.INFO)
    logging.info(
        "[PROD] ✅ Model approved with accuracy: %.4f",
        model_accuracy
    )
    logging.info(
        "[PROD] Ready for registration from: %s",
        model.uri
    )


# ==============================================================================
# Component: register_model_op
# ==============================================================================
# TODO: Lab 5.4.1 - Component Identification: Model registration with versioning support
# TODO: Lab 5.4.2 - Purpose Recognition: Production registration requires parent model for versioning
# TODO: Lab 5.4.3 - Architecture Understanding: Model Registry integration for production lineage
#
# TODO: Lab 5.5.1 - Custom Component Analysis: This is a CUSTOM COMPONENT (same as dev)
# TODO: Lab 5.5.4 - Component Reusability: Same registration component with versioning emphasis
# TODO: Lab 5.5.2 - Custom vs Pre-built Decision: Production uses same custom registration
#
# TODO: Lab 5.6.3 — Caching and retry guidance
# INSPECT: Registration component is NOT IDEMPOTENT (creates model versions).
# PRODUCTION CACHING GUIDANCE: DO NOT CACHE - always executes fresh registration.
# PRODUCTION RETRY GUIDANCE: BE VERY CAREFUL - may create duplicate model versions.
# SIDE EFFECTS: Writes to Vertex AI Model Registry (external state change).
# PRODUCTION NOTE: Especially critical in production to avoid duplicate registrations.
#
# Registers the model in Vertex AI Model Registry if approved. Supports
# versioning under a parent model if provided.
@component(
    base_image=BASE_IMAGE,
    packages_to_install=[
        pkg.strip()
        for pkg in open(REQUIREMENTS_PATH)
        if pkg.strip() and not pkg.startswith("#")
    ],
)
def register_model_op(
    project_id: str,
    region: str,
    model_display_name: str,
    model_artifact: Input[Model],
    parent_model: str = ""
):
    from google.cloud import aiplatform
    import logging

    logging.basicConfig(level=logging.INFO)
    aiplatform.init(project=project_id, location=region)

    artifact_dir = model_artifact.uri.rsplit("/", 1)[0]
    upload_args = {
        "display_name": model_display_name,
        "artifact_uri": artifact_dir,
        "serving_container_image_uri": (
            "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-2:latest"
        ),
        "sync": True
    }

    # TODO: Lab 5.4.1 - Component Identification: Parent model parameter enables versioning
    # TODO: Lab 5.4.2 - Purpose Recognition: Production models registered as versions of parent model
    # TODO: Lab 5.4.3 - Architecture Understanding: Model versioning architecture in production
    # If parent_model is provided, register as a new version under parent model
    if parent_model:
        upload_args["parent_model"] = parent_model
        logging.info(
            "[PROD] Registering new version under parent model: %s",
            parent_model
        )

    model = aiplatform.Model.upload(**upload_args)
    logging.info(
        "[PROD] Model registered: %s",
        model.resource_name
    )


# ==============================================================================
# Component: model_rejected_op
# ==============================================================================
# TODO: Lab 5.4.1 - Component Identification: Rejection component with production error handling
# TODO: Lab 5.4.2 - Purpose Recognition: Production rejection enforces stricter quality gates
#
# TODO: Lab 5.6.3 — Caching and retry guidance
# INSPECT: Rejection component is IDEMPOTENT (logs and raises error).
# CACHING GUIDANCE: Safe to cache but typically disabled in production.
# RETRY GUIDANCE: DO NOT RETRY - intentional failure for quality gate.
# PURPOSE: Fails pipeline when production quality threshold not met (0.75 vs 0.70 dev).
#
# TODO: Lab 5.6.7 — Failure modes and debugging tips
# INSPECT: Production quality gate failure (stricter threshold than dev).
# PRODUCTION NOTE: min_accuracy=0.75 in prod vs 0.70 in dev - higher bar for production.
# DEBUGGING TIP: If production model rejected, compare to dev accuracy to diagnose issue.
# RESOLUTION: Improve model quality or adjust production threshold if appropriate.
#
# Logs rejection message and raises error if model accuracy is below threshold.
@component(
    base_image=BASE_IMAGE
)
def model_rejected_op(model_accuracy: float, min_accuracy: float):
    import logging
    logging.basicConfig(level=logging.INFO)
    logging.error(
        "[PROD] ❌ Model rejected. Accuracy %.4f < %.2f",
        model_accuracy,
        min_accuracy
    )
    # TODO: Lab 5.4.2 - Purpose Recognition: Production-specific error message
    raise ValueError(
        "Model accuracy does not meet minimum production threshold."
    )


# ==============================================================================
# Pipeline: prod_diabetes_pipeline (MODIFIED FOR BIGQUERY)
# ==============================================================================
# TODO: Lab 5.4.1 - Component Identification: Production pipeline with identical structure to dev
# TODO: Lab 5.4.2 - Purpose Recognition: Production configuration with stricter thresholds
# TODO: Lab 5.4.3 - Architecture Understanding: Same architecture, different environment configuration
# TODO: Lab 5.5.5 - BigQuery Integration: Pipeline now orchestrates BigQuery and custom components (production)
#
# Orchestrates all pipeline steps using BigQuery for data access. Registers
# model only if accuracy meets threshold, otherwise rejects. Supports parent
# model versioning.
@dsl.pipeline(name=PIPELINE_NAME, description=PIPELINE_DESCRIPTION)
def prod_diabetes_pipeline(
    project_id: str,
    region: str,
    model_display_name: str,
    # TODO: Lab 5.5.5 - BigQuery Integration: input_raw_data_gcs_uri replaced by bq_dataset and bq_view (production)
    bq_dataset: str,  # BigQuery dataset name
    bq_view: str,     # BigQuery view name
    reg_rate: float = 0.05,
    # TODO: Lab 5.4.1 - Component Identification: Higher min_accuracy default for production
    # TODO: Lab 5.4.2 - Purpose Recognition: Stricter quality gates in production environment
    # TODO: Lab 5.4.3 - Architecture Understanding: Environment-specific parameter defaults
    min_accuracy: float = 0.75,  # Production requires higher accuracy than dev (0.70)
    parent_model: str = ""
):
    # ==========================================================================
    # DATA PREPROCESSING - BigQuery Query Job Components
    # ==========================================================================
    # TODO: Lab 5.5.5 - BigQuery Integration: SQL-based data access and splitting (production)
    
    # Query training data from BigQuery view (80% split)
    train_query = f"""
    SELECT
      Pregnancies,
      PlasmaGlucose,
      DiastolicBloodPressure,
      TricepsThickness,
      SerumInsulin,
      BMI,
      DiabetesPedigree,
      Age,
      Diabetic
    FROM `{project_id}.{bq_dataset}.{bq_view}`
    WHERE MOD(ABS(FARM_FINGERPRINT(CAST(entity_id AS STRING))), 10) < 8
    """
    
    # TODO: Lab 5.5.1 - Pre-built Component Instantiation: BigQuery Query Job for training data
    # TODO: Lab 5.5.5 - BigQuery Integration: Component executes SQL query and outputs BQTable
    
    # TODO: Lab 5.6.4 — Parallelism and resource limits
    # INSPECT: BigQuery Query Job components (no .set_cpu_limit() or .set_memory_limit()).
    # PRODUCTION NOTE: Same parallelism as dev - bq_train_task and bq_test_task are INDEPENDENT.
    # OBSERVATION: These tasks execute IN PARALLEL in both dev and production environments.
    
    bq_train_task = bigquery_query_job_op(
        project=project_id,
        location=region,
        query=train_query
    )
    
    # Query test data from BigQuery view (20% split)
    test_query = f"""
    SELECT
      Pregnancies,
      PlasmaGlucose,
      DiastolicBloodPressure,
      TricepsThickness,
      SerumInsulin,
      BMI,
      DiabetesPedigree,
      Age,
      Diabetic
    FROM `{project_id}.{bq_dataset}.{bq_view}`
    WHERE MOD(ABS(FARM_FINGERPRINT(CAST(entity_id AS STRING))), 10) >= 8
    """
    
    # TODO: Lab 5.5.1 - Pre-built Component Instantiation: BigQuery Query Job for test data
    # TODO: Lab 5.5.5 - BigQuery Integration: Component executes SQL query and outputs BQTable
    bq_test_task = bigquery_query_job_op(
        project=project_id,
        location=region,
        query=test_query
    )

    # ==========================================================================
    # PIPELINE ORCHESTRATION - Production Component DAG
    # ==========================================================================
    # TODO: Lab 5.4.1 - Component Identification: Identical component orchestration to dev pipeline
    # TODO: Lab 5.4.2 - Purpose Recognition: Consistent pipeline architecture across environments
    # TODO: Lab 5.4.3 - Architecture Understanding: Pipeline architecture reusability
    
    # TODO: Lab 5.6.4 — Parallelism and resource limits
    # INSPECT: .set_cpu_limit("1") and .set_memory_limit("3840Mi") on train_task (same as dev).
    # PRODUCTION NOTE: Same resource limits as dev - adequate for current dataset size.
    # TASK: Record in orchestration-notes.md: train_model_op (PROD) | CPU: 1 | Memory: 3840Mi.
    # PRODUCTION GUIDANCE: Monitor production resource usage; scale if needed for larger datasets.
    
    # Train model on training data from BigQuery
    train_task = train_model_op(
        train_data=bq_train_task.outputs["destination_table"],
        reg_rate=reg_rate,
        project_id=project_id,
        bq_location=region
    ).set_cpu_limit("1").set_memory_limit("3840Mi")
    train_task.after(bq_train_task)

    # TODO: Lab 5.6.4 — Parallelism and resource limits
    # INSPECT: .set_cpu_limit("1") and .set_memory_limit("3840Mi") on eval_task (same as dev).
    # PRODUCTION NOTE: Same resource allocation pattern across environments.
    # TASK: Record in orchestration-notes.md: evaluate_model_op (PROD) | CPU: 1 | Memory: 3840Mi.
    
    # Evaluate model on test data from BigQuery
    eval_task = evaluate_model_op(
        model=train_task.outputs["output_model"],
        test_data=bq_test_task.outputs["destination_table"],
        min_accuracy=min_accuracy,
        project_id=project_id,
        bq_location=region
    ).set_cpu_limit("1").set_memory_limit("3840Mi")
    eval_task.after(train_task)

    # ==========================================================================
    # CONDITIONAL LOGIC - Production Quality Gates
    # ==========================================================================
    # TODO: Lab 5.4.1 - Component Identification: Same conditional structure as dev pipeline
    # TODO: Lab 5.4.2 - Purpose Recognition: Production quality gates enforce stricter standards
    # TODO: Lab 5.4.3 - Architecture Understanding: Conditional architecture pattern reuse
    
    # If accuracy >= min_accuracy, approve and register model
    with dsl.If(
        eval_task.outputs["Output"] >= min_accuracy,
        name="pass-accuracy-threshold"
    ):
        # TODO: Lab 5.6.4 — Parallelism and resource limits
        # INSPECT: Resource limits on approved_task (same as dev).
        # TASK: Record in orchestration-notes.md: model_approved_op (PROD) | CPU: 1 | Memory: 3840Mi.
        
        approved_task = model_approved_op(
            model_accuracy=eval_task.outputs["Output"],
            model=train_task.outputs["output_model"]
        ).set_cpu_limit("1").set_memory_limit("3840Mi")
        approved_task.after(eval_task)

        # TODO: Lab 5.4.2 - Purpose Recognition: Production registration includes parent model for versioning
        
        # TODO: Lab 5.6.4 — Parallelism and resource limits
        # INSPECT: Resource limits on register_task (same as dev).
        # TASK: Record in orchestration-notes.md: register_model_op (PROD) | CPU: 1 | Memory: 3840Mi.
        # PRODUCTION NOTE: Model registration especially critical in production for lineage tracking.
        
        register_task = register_model_op(
            project_id=project_id,
            region=region,
            model_display_name=model_display_name,
            model_artifact=train_task.outputs["output_model"],
            parent_model=parent_model
        ).set_cpu_limit("1").set_memory_limit("3840Mi")
        register_task.after(approved_task)

    # If accuracy < min_accuracy, reject model
    with dsl.If(
        eval_task.outputs["Output"] < min_accuracy,
        name="fail-accuracy-threshold"
    ):
        # TODO: Lab 5.4.1 - Component Identification: Model rejection component in failure branch
        # TODO: Lab 5.4.2 - Purpose Recognition: Rejection fails pipeline for inadequate models
        # TODO: Lab 5.4.3 - Architecture Understanding: Mutually exclusive branches based on condition
        
        # TODO: Lab 5.6.4 — Parallelism and resource limits
        # INSPECT: Resource limits on rejected_task (same as dev).
        # TASK: Record in orchestration-notes.md: model_rejected_op (PROD) | CPU: 1 | Memory: 3840Mi.
        # NOTE: Production rejection threshold is 0.75 (stricter than dev's 0.70).
        
        rejected_task = model_rejected_op(
            model_accuracy=eval_task.outputs["Output"],
            min_accuracy=min_accuracy
        ).set_cpu_limit("1").set_memory_limit("3840Mi")
        rejected_task.after(eval_task)


# ==============================================================================
# Lab 5.6 Production Pipeline Orchestration Summary
# ==============================================================================
# TODO: Lab 5.6.3 — Caching and retry guidance
# PRODUCTION CACHING AND RETRY SUMMARY:
# =====================================
# Component                | Cacheable? | Retry Safe? | Production Guidance
# ------------------------|------------|-------------|----------------------
# bigquery_query_job_op   | YES        | YES         | DISABLE cache in prod (fresh data)
# train_model_op          | YES        | YES         | DISABLE cache in prod (compliance)
# evaluate_model_op       | YES        | YES         | DISABLE cache in prod (fresh eval)
# model_approved_op       | YES        | YES         | DISABLE cache in prod (audit trail)
# register_model_op       | NO         | CAUTIOUS    | NEVER cache; careful with retries
# model_rejected_op       | YES        | NO          | Intentional failure (quality gate)
#
# PRODUCTION-SPECIFIC CACHING GUIDANCE:
# - ALWAYS disable caching for production pipelines (--enable-caching flag should NOT be used)
# - Reasons: Compliance requirements, audit trails, fresh data verification
# - Even idempotent components should execute fresh in production
# - Development can use caching for faster iteration; production never should
#
# PRODUCTION-SPECIFIC RETRY GUIDANCE:
# - Safe to retry: BigQuery queries, training, evaluation (idempotent operations)
# - BE CAREFUL: Model registration (may create duplicate versions)
# - DO NOT retry: Model rejection (intentional quality gate failure)
# - Production retries should be configured cautiously due to compliance implications
#
# TODO: Lab 5.6.4 — Parallelism and resource limits
# PRODUCTION RESOURCE LIMITS SUMMARY:
# ===================================
# Component                | CPU   | Memory  | Production Notes
# ------------------------|-------|---------|----------------------------------
# bigquery_query_job_op   | N/A   | N/A     | Managed by BigQuery (serverless)
# train_model_op          | 1     | 3840Mi  | Same as dev; monitor prod usage
# evaluate_model_op       | 1     | 3840Mi  | Same as dev; adequate for dataset
# model_approved_op       | 1     | 3840Mi  | Same as dev; lightweight
# register_model_op       | 1     | 3840Mi  | Same as dev; critical for versioning
# model_rejected_op       | 1     | 3840Mi  | Same as dev; stricter threshold (0.75)
#
# PRODUCTION PARALLELISM (SAME AS DEV):
# - bq_train_task and bq_test_task: PARALLEL (independent queries)
# - train_task depends on bq_train_task: SEQUENTIAL
# - eval_task depends on train_task and bq_test_task: WAITS FOR BOTH
# - Conditional branches (approved/rejected): MUTUALLY EXCLUSIVE
#
# PRODUCTION RESOURCE SIZING GUIDANCE:
# - Current limits adequate for current dataset size
# - Monitor production execution metrics in Vertex AI console
# - Scale up if production dataset grows significantly
# - Production environments should have resource monitoring alerts
#
# TODO: Lab 5.6.5 — Observability: metrics, logging, and dashboard links
# PRODUCTION OBSERVABILITY SUMMARY:
# =================================
# METRICS LOGGED (SAME AS DEV):
# - evaluate_model_op: "accuracy", "min_accuracy_threshold"
#
# LOGGING PATTERNS (PRODUCTION PREFIX):
# - All components use [PROD] prefix instead of [DEV]
# - train_model_op: 4 log statements (URI, loading, row count, storage)
# - evaluate_model_op: 4 log statements (URI, loading, row count, accuracy)
# - model_approved_op: 2 log statements (approval, model URI)
# - register_model_op: 2 log statements (versioning, registration)
# - model_rejected_op: 1 log statement (rejection error)
#
# PRODUCTION LOGGING IMPORTANCE:
# - Critical for compliance and audit requirements
# - Production logs retained longer than dev logs
# - [PROD] prefix enables filtering production logs
# - Model registration logs essential for production lineage tracking
#
# WHERE TO FIND PRODUCTION LOGS:
# - Vertex AI Console → Production Pipeline Run → Click component → Logs
# - Cloud Logging: filter by pipeline_job_id AND [PROD] prefix
# - Production dashboard URI: Logged by run_pipeline.py after submission
#
# PRODUCTION MONITORING BEST PRACTICES:
# - Set up alerts for production pipeline failures
# - Monitor accuracy trends over time
# - Track model registration success rates
# - Alert on quality gate rejections (accuracy < 0.75)
#
# TODO: Lab 5.6.7 — Failure modes and debugging tips
# PRODUCTION FAILURE MODES AND DEBUGGING:
# =======================================
# 1. BigQuery Query Job failures (PRODUCTION):
#    - Cause: Production view missing, IAM permissions, query syntax
#    - Production Impact: Pipeline fails immediately, no model trained
#    - Debug: Verify production BigQuery view exists, check service account permissions
#    - Resolution: Ensure Terraform deployed production BigQuery resources
#
# 2. URI parsing failures (train_model_op, evaluate_model_op):
#    - Cause: Upstream BigQuery component failed in production
#    - Production Impact: Training or evaluation cannot proceed
#    - Debug: Check production BigQuery Query Job logs, verify table creation
#    - Resolution: Fix upstream BigQuery issues, verify production data access
#
# 3. Model rejected (model_rejected_op) - PRODUCTION QUALITY GATE:
#    - Cause: Accuracy < 0.75 (STRICTER than dev's 0.70)
#    - Production Impact: Model NOT registered, pipeline fails (EXPECTED BEHAVIOR)
#    - Debug: Compare to dev accuracy, check production data quality
#    - Resolution: Improve model OR adjust threshold if appropriate
#    - NOTE: This is a QUALITY GATE, not a bug - ensures production quality standards
#
# 4. Model registration failures (register_model_op) - PRODUCTION CRITICAL:
#    - Cause: IAM permissions, parent model not found, artifact issues
#    - Production Impact: Approved model not registered, production deployment blocked
#    - Debug: Check service account Vertex AI User role, verify parent model exists
#    - Resolution: Verify Terraform IAM setup, check parent_model parameter
#    - Production Note: Registration failures are CRITICAL - require immediate attention
#
# 5. Resource exhaustion (OOM errors) - PRODUCTION SCALING:
#    - Cause: Production dataset larger than dev dataset
#    - Production Impact: Component fails during execution
#    - Debug: Check production logs for OOM, compare data volumes dev vs prod
#    - Resolution: Increase .set_memory_limit() for affected components
#    - Production Note: Monitor resource usage trends; plan for scaling
#
# PRODUCTION-SPECIFIC DEBUGGING WORKFLOW:
# 1. Check production pipeline dashboard (Vertex AI console)
# 2. Filter logs by [PROD] prefix to isolate production logs
# 3. Compare production vs dev accuracy if quality gate fails
# 4. Verify Terraform-provisioned production infrastructure exists
# 5. Check production service account IAM permissions
# 6. Verify production BigQuery view and data availability
# 7. Escalate production failures immediately (unlike dev failures)
#
# PRODUCTION BEST PRACTICES:
# - Never enable caching in production pipelines
# - Always use parent_model parameter for versioning
# - Monitor accuracy trends over time
# - Set up alerting for production pipeline failures
# - Maintain separate dev/prod environments with Terraform
# - Use labels to track production pipeline runs
# - Implement stricter quality gates (0.75 vs 0.70)
# - Ensure production logs retained for compliance
# - Test changes in dev before deploying to production
#
# PRODUCTION vs DEVELOPMENT COMPARISON:
# =====================================
# Aspect                  | Development        | Production
# -----------------------|-------------------|------------------
# min_accuracy threshold | 0.70              | 0.75 (stricter)
# Caching                | Can enable        | Always disabled
# Logging prefix         | [DEV]             | [PROD]
# Parent model           | Optional          | Typically set
# Resource limits        | Same              | Same (monitor usage)
# Quality gate impact    | Lower bar         | Higher bar
# Failure impact         | Low (iteration)   | High (critical)
# Monitoring             | Optional          | Required/alerts
# Compliance             | Relaxed           | Strict
#
# COMPONENT REUSABILITY DEMONSTRATION:
# - 100% of components identical between dev and prod
# - Only CONFIGURATION differs (parameters, thresholds, logging prefix)
# - Same BigQuery integration, same custom components
# - Demonstrates power of component architecture
# - Single codebase supports multiple environments
# ==============================================================================

# ==============================================================================
# Lab 5.4 Architecture Summary: Production vs Development Pipeline Architecture
# ==============================================================================
# This production pipeline demonstrates key architectural principles:
#
# 1. COMPONENT ARCHITECTURE REUSABILITY:
#    - Same component definitions as dev pipeline (BigQuery + custom components)
#    - Identical pipeline structure and dependencies
#    - Shared component interfaces and contracts
#    - Consistent DAG architecture across environments
#
# 2. ENVIRONMENT-SPECIFIC CONFIGURATION:
#    - Different pipeline names for environment tracking
#    - Higher accuracy threshold (0.75 vs 0.70)
#    - Production-specific logging and audit trail
#    - Model versioning support (parent_model parameter)
#    - Same BigQuery views (or separate prod views if needed)
#
# 3. ARCHITECTURAL BENEFITS:
#    - Single component codebase supports multiple environments
#    - Configuration-driven environment differences
#    - Consistent behavior across dev and prod
#    - Simplified maintenance and testing
#    - Clear separation of architecture vs configuration
#
# 4. PRODUCTION-SPECIFIC FEATURES:
#    - Stricter quality gates (higher accuracy threshold)
#    - Model versioning for production lineage
#    - Enhanced logging for compliance and audit
#    - Production-grade error handling
#
# 5. PROVEN WORKING IMPLEMENTATION:
#    - Uses artifact_types.BQTable for BigQuery table inputs
#    - Regex parsing for reliable table reference extraction
#    - Tested in production environment with real workloads
#    - Combines Google best practices with custom business logic
#
# TODO: Lab 5.4.3 - Architecture Understanding: This demonstrates how the same
#       component architecture adapts to different environments through configuration

# ==============================================================================
# Lab 5.5 Component Strategy Summary: Production Component Reusability
# ==============================================================================
# This PRODUCTION pipeline demonstrates the power of component reusability:
#
# 1. COMPONENT REUSE ACROSS ENVIRONMENTS:
#    Component              | Dev Pipeline | Prod Pipeline | Reused?
#    --------------------- | ------------ | ------------- | --------
#    BigQuery Query Job    | ✓            | ✓             | YES (100%)
#    train_model_op        | ✓            | ✓             | YES (100%)
#    evaluate_model_op     | ✓            | ✓             | YES (100%)
#    model_approved_op     | ✓            | ✓             | YES (100%)
#    register_model_op     | ✓            | ✓             | YES (100%)
#    model_rejected_op     | ✓            | ✓             | YES (100%)
#
# 2. CONFIGURATION DIFFERENCES ONLY:
#    Parameter             | Dev Value | Prod Value
#    --------------------- | --------- | ----------
#    min_accuracy          | 0.70      | 0.75
#    bq_dataset            | shared_bronze | shared_bronze (or prod dataset)
#    bq_view               | diabetes_features_view | (same or prod view)
#    logging_prefix        | [DEV]     | [PROD]
#    parent_model          | optional  | typically set
#
# 3. CUSTOM + PRE-BUILT COMPONENT BENEFITS DEMONSTRATED:
#    - 100% component reuse across environments
#    - Single codebase reduces maintenance
#    - Configuration-driven differences
#    - Consistent behavior guaranteed
#    - Pre-built components for standard operations (BigQuery Query Job)
#    - Custom components for business logic (training, evaluation)
#    - Simplified testing (test once, deploy everywhere)
#
# TODO: Lab 5.5.1 - Custom + Pre-built Mix: All components identical to dev pipeline
# TODO: Lab 5.5.2 - Pre-built Migration: BigQuery Query Job replaced custom preprocessing
# TODO: Lab 5.5.3 - Accelerator Templates: Production deployment via Terraform
# TODO: Lab 5.5.4 - Component Reusability: 100% component reuse demonstrates architecture benefit
# TODO: Lab 5.5.5 - BigQuery Integration: Production uses same architecture as dev with BQTable type
# ==============================================================================
