"""
Vertex AI KFP Pipeline for Development

This pipeline performs the following steps:
- Queries diabetes data from BigQuery Feature Group view (splits via query logic).
- Trains a logistic regression model in development mode.
- Evaluates the trained model on a test split.
- Conditionally registers the model in Vertex AI Model Registry if accuracy
  meets the minimum threshold.
- Rejects the model if accuracy is insufficient.

All comments and documentation lines are kept <= 100 characters for .flake8.

===============================================================================
Lab 5.4: Vertex AI Pipeline Component Architecture Exploration
===============================================================================
This pipeline demonstrates the complete Vertex AI Pipeline component architecture:
- KFP v2 component definitions using @component decorator
- Pipeline orchestration using @pipeline decorator
- Component dependencies and data flow
- Conditional logic using dsl.If

Understanding this pipeline helps identify Vertex AI Pipeline components,
their purposes, and how they work together in the pipeline architecture.

TODO: Lab 5.4.1 - Component Identification: KFP v2 pipeline with multiple component types
TODO: Lab 5.4.2 - Purpose Recognition: Complete ML workflow from data to model registration
TODO: Lab 5.4.3 - Architecture Understanding: Component-based pipeline architecture

===============================================================================
Lab 5.5: Vertex AI Custom Components and Pre-built Components and Accelerator Templates
===============================================================================
This pipeline demonstrates the strategic use of custom components and how they
compare to pre-built Google Cloud Pipeline Components. Understanding when to use
custom vs pre-built components is crucial for enterprise ML architecture.

ACCELERATOR TEMPLATE ARCHITECTURE (3-Layer Enterprise Model):
=============================================================
1. INFRASTRUCTURE LAYER (Terraform):
   - Vertex AI pipeline resources provisioned as code
   - Service accounts and IAM permissions
   - GCS buckets and datasets
   - BigQuery datasets and Feature Groups
   - Multi-environment deployment (dev/prod)
   - See: vertex_ai_infrastructure.tf for complete infrastructure

2. PIPELINE LAYER (This File - Vertex AI):
   - Custom components for business-specific logic
   - Pre-built components for standard operations (BigQuery Query Job)
   - Integration with Google Cloud services (BigQuery, Feature Groups)
   - MLOps workflows and orchestration

3. ENTERPRISE LAYER (Best Practices):
   - Industry-specific implementations
   - Governance and compliance patterns
   - Reusable solution frameworks
   - Production-ready templates

COMPONENT STRATEGY IN THIS PIPELINE:
====================================
This pipeline uses a MIX of custom and pre-built components:
- CUSTOM components for training and evaluation (business logic)
- PRE-BUILT BigQuery Query Job component for data preprocessing
- Feature Groups for feature governance and metadata
- Trade-offs between custom and pre-built approaches
- Integration patterns for enterprise accelerator templates

BIGQUERY AND FEATURE GROUP INTEGRATION:
=======================================
This pipeline demonstrates enterprise-grade BigQuery integration:
- BigQuery views as feature sources
- Pre-built BigQuery Query Job component for data access
- Feature Groups for feature registry and governance
- Serverless data preprocessing with BigQuery
- SQL-based train/test splitting using hash functions

TODO: Lab 5.5.1 - Custom vs Pre-built: Mix of custom and pre-built components
TODO: Lab 5.5.2 - Pre-built Alternatives: BigQuery Query Job for preprocessing
TODO: Lab 5.5.3 - Accelerator Templates: Understand 3-layer enterprise architecture
TODO: Lab 5.5.4 - BigQuery Integration: Pre-built component for data access
TODO: Lab 5.5.5 - Feature Groups: Registry and governance for features
===============================================================================
"""

from kfp import dsl, components
from kfp.dsl import (
    component,
    pipeline,
    Input,
    Output,
    Dataset,
    Model,
    Metrics
)
from google_cloud_pipeline_components.types import artifact_types

# TODO: Lab 5.4.1 - Component Identification: Pipeline metadata defines pipeline identity
# TODO: Lab 5.4.2 - Purpose Recognition: Pipeline name and description enable tracking and organization
PIPELINE_NAME = "mlops-diabetes-dev-pipeline"
PIPELINE_DESCRIPTION = (
    "Development pipeline for diabetes prediction model on Vertex AI with BigQuery"
)

# TODO: Lab 5.4.1 - Component Identification: Base image defines component execution environment
# TODO: Lab 5.4.2 - Purpose Recognition: Standardized Python environment ensures consistency
# TODO: Lab 5.4.3 - Architecture Understanding: Container images encapsulate component execution
BASE_IMAGE = "python:3.9"
REQUIREMENTS_PATH = "src/requirements.txt"

# ==============================================================================
# PRE-BUILT COMPONENT: BigQuery Query Job
# ==============================================================================
# TODO: Lab 5.5.1 - Pre-built Component: Load BigQuery Query Job from Google Cloud registry
# TODO: Lab 5.5.2 - Pre-built vs Custom: Using Google-maintained component for data access
# TODO: Lab 5.5.4 - BigQuery Integration: Pre-built component executes SQL queries
#
# PRE-BUILT COMPONENT BENEFITS:
# - Google-maintained and always up-to-date
# - Optimized for BigQuery operations
# - Built-in error handling and retries
# - Standardized interface across projects
# - No custom code needed for data access
#
# TODO: Lab 5.6.3 — Caching and retry guidance
# INSPECT: Pre-built BigQuery Query Job component has built-in retry logic.
# NOTE: This component is IDEMPOTENT (same query always returns same data at a point in time).
# CACHING GUIDANCE: Safe to cache if data is static; disable if data updates frequently.
# RETRY GUIDANCE: Automatically retries on transient BigQuery API errors.
#
# Load the BigQuery Query Job component from Google Cloud Pipeline Components
bigquery_query_job_op = components.load_component_from_url(
    'https://us-kfp.pkg.dev/ml-pipeline/google-cloud-registry/'
    'bigquery-query-job/sha256:'
    'd1cae80bc0de4e5b95b994739c8d0d7d42ce5a4cb17d3c9512eaed14540f6343'
)


# ==============================================================================
# Component: preprocess_data_op (DEPRECATED - COMMENTED OUT)
# ==============================================================================
# TODO: Lab 5.5.1 - Custom Component Analysis: This WAS a CUSTOM COMPONENT
# TODO: Lab 5.5.2 - Custom vs Pre-built Decision: REPLACED with BigQuery Query Job
# TODO: Lab 5.5.4 - BigQuery Migration: Preprocessing now done via BigQuery SQL
#
# MIGRATION NOTES:
# ===============
# This custom preprocessing component has been REPLACED by the BigQuery Query Job
# pre-built component. The old approach downloaded CSV from GCS and split in Python.
# The new approach queries BigQuery view and splits using SQL.
#
# OLD APPROACH (GCS-based - COMMENTED OUT):
# -----------------------------------------
# - Downloaded CSV from GCS (input_raw_data_gcs_uri parameter)
# - Used pandas to split 80/20 in Python
# - Wrote train/test CSVs to artifact storage
# - Required custom Python code for data handling
#
# NEW APPROACH (BigQuery-based - ACTIVE):
# ---------------------------------------
# - Queries BigQuery view (bq_dataset and bq_view parameters)
# - Uses SQL FARM_FINGERPRINT for deterministic 80/20 split
# - Two separate BigQuery Query Job component calls (train and test)
# - No custom Python code needed for data access
# - Serverless, scalable data processing
#
# BENEFITS OF MIGRATION:
# - Eliminates custom data handling code
# - Uses managed BigQuery infrastructure
# - SQL-based splitting is more transparent
# - Better integration with Feature Groups
# - Scalable to larger datasets
#
# OLD CODE (COMMENTED OUT - kept for reference and rollback):
# ============================================================
# @component(
#     base_image=BASE_IMAGE,
#     packages_to_install=[
#         pkg.strip()
#         for pkg in open("src/requirements.txt")
#         if pkg.strip() and not pkg.startswith("#")
#     ],
# )
# def preprocess_data_op(
#     # OLD PARAMETER: GCS URI to raw CSV file
#     # REPLACED BY: bq_dataset and bq_view in pipeline parameters
#     input_gcs_uri: str,
#     output_train_data: Output[Dataset],
#     output_test_data: Output[Dataset]
# ):
#     import pandas as pd
#     from google.cloud import storage
#     from urllib.parse import urlparse
#     import logging
#
#     logging.basicConfig(level=logging.INFO)
#     parsed = urlparse(input_gcs_uri)
#     bucket_name = parsed.netloc
#     blob_name = parsed.path.lstrip("/")
#
#     local_file = "diabetes_raw_dev.csv"
#     # Download file from GCS to local disk
#     storage.Client().bucket(bucket_name).blob(blob_name).download_to_filename(
#         local_file
#     )
#     df = pd.read_csv(local_file)
#
#     # Split data into train and test sets (80/20 split)
#     # NOW DONE IN SQL: WHERE MOD(ABS(FARM_FINGERPRINT(...)), 10) < 8
#     train_data = df.sample(frac=0.8, random_state=42)
#     test_data = df.drop(train_data.index)
#
#     # Save splits to output artifact paths
#     train_data.to_csv(output_train_data.path, index=False)
#     test_data.to_csv(output_test_data.path, index=False)
#
#     logging.info(
#         "[DEV] Preprocessed and split data saved to: %s, %s",
#         output_train_data.path,
#         output_test_data.path,
#     )


# ==============================================================================
# Component: train_model_op (MODIFIED FOR BIGQUERY)
# ==============================================================================
# TODO: Lab 5.4.1 - Component Identification: Model training component
# TODO: Lab 5.4.2 - Purpose Recognition: Trains ML model using training data
# TODO: Lab 5.4.3 - Architecture Understanding: Depends on BigQuery query component
#
# TODO: Lab 5.5.1 - Custom Component Analysis: This is a CUSTOM COMPONENT
# TODO: Lab 5.5.2 - Custom vs Pre-built Decision: Custom component chosen because:
#   - Implements specific scikit-learn LogisticRegression algorithm
#   - Custom regularization parameter handling
#   - Specific model serialization format (joblib)
#   - Direct control over training process
# TODO: Lab 5.5.4 - BigQuery Integration: Reads from BigQuery table via Python client
#
# TODO: Lab 5.6.3 — Caching and retry guidance
# INSPECT: This training component is IDEMPOTENT.
# CACHING GUIDANCE: SAFE TO CACHE - same training data + same reg_rate = same model.
# RETRY GUIDANCE: SAFE TO RETRY - training has no external side effects.
# IDEMPOTENCY: Same inputs always produce same outputs; no database writes or API calls.
#
# MODIFICATION NOTES:
# ==================
# CHANGED: Now reads from BigQuery table instead of CSV file
# - OLD: train_df = pd.read_csv(train_data.path)
# - NEW: Uses BigQuery Python client to read from table
# - Input is now a BigQuery table reference from BigQuery Query Job component
# - Added parameters: project_id and bq_location for BigQuery client
# - Uses URI parsing to extract table reference from artifact
#
# ALTERNATIVE: Pre-built Component Option (For Lab 5.5 Exploration)
# -----------------------------------------------------------------------------
# COMMENTED OUT - Pre-built Component Example for Lab 5.5 Learning:
#
# from google_cloud_pipeline_components.v1.custom_job import CustomTrainingJobOp
#
# # If using Vertex AI Training with container, could use pre-built component:
# train_task = CustomTrainingJobOp(
#     project=project_id,
#     location=region,
#     display_name="diabetes-training-job",
#     worker_pool_specs=[{
#         "machine_spec": {
#             "machine_type": "n1-standard-4",
#         },
#         "replica_count": 1,
#         "container_spec": {
#             "image_uri": "gcr.io/project-id/diabetes-trainer:latest",
#             "args": [
#                 f"--train-data={train_data_path}",
#                 f"--reg-rate={reg_rate}",
#             ],
#         },
#     }],
# )
#
# PRE-BUILT COMPONENT BENEFITS:
# - Managed Vertex AI Training infrastructure
# - Automatic scaling and resource management
# - Built-in experiment tracking
# - Integration with Vertex AI Experiments
#
# CUSTOM COMPONENT BENEFITS (Current Choice):
# - Simpler for lightweight training
# - No separate container image required
# - Training code embedded in pipeline
# - Faster iteration during development
# -----------------------------------------------------------------------------
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
    # TODO: Lab 5.4.1 - Component Identification: Input[artifact_types.BQTable] defines typed component input
    # TODO: Lab 5.4.2 - Purpose Recognition: Input references upstream BigQuery component output
    # TODO: Lab 5.4.3 - Architecture Understanding: Input creates dependency on BigQuery component
    train_data: Input[artifact_types.BQTable],
    # TODO: Lab 5.4.1 - Component Identification: Output[Model] defines typed model artifact
    # TODO: Lab 5.4.2 - Purpose Recognition: Model output enables downstream evaluation and registration
    output_model: Output[Model],
    # TODO: Lab 5.4.1 - Component Identification: reg_rate is component parameter
    # TODO: Lab 5.4.2 - Purpose Recognition: Parameters enable component configuration at runtime
    reg_rate: float,
    # TODO: Lab 5.5.4 - BigQuery Integration: New parameters for BigQuery access
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
    # INSPECT: logging.basicConfig() sets up component logging.
    # TASK: Count all logging.info() and logging.error() calls in this component (there are 4).
    # NOTE: Component logs appear in Vertex AI console under the component execution.
    
    logging.basicConfig(level=logging.INFO)
    
    # TODO: Lab 5.5.4 - BigQuery Integration: Read from BigQuery instead of CSV
    # OLD CODE (COMMENTED OUT):
    # train_df = pd.read_csv(train_data.path)
    #
    # NEW CODE (ACTIVE):
    # Extract BigQuery table information from BQTable artifact using URI parsing
    # The BigQuery Query Job component outputs a BQTable artifact with URI
    # URI format: https://www.googleapis.com/bigquery/v2/projects/{project}/datasets/{dataset}/tables/{table}
    
    # Parse table reference from URI using regex
    uri = train_data.uri
    
    # TODO: Lab 5.6.5 — Observability: metrics, logging, and dashboard links
    # INSPECT: Logging pattern for data source tracking.
    # NOTE: Logs the BigQuery table URI for debugging and audit purposes.
    
    logging.info("[DEV] Parsing BigQuery table from URI: %s", uri)
    match = re.search(r'projects/([^/]+)/datasets/([^/]+)/tables/([^/]+)', uri)
    if not match:
        # TODO: Lab 5.6.7 — Failure modes and debugging tips
        # INSPECT: Error raised when URI parsing fails.
        # DEBUGGING TIP: Check that upstream BigQuery Query Job component succeeded.
        # COMMON CAUSE: BigQuery component failure or incorrect artifact output.
        
        raise ValueError(f"Could not parse BigQuery table reference from URI: {uri}")
    
    proj, dataset, table = match.groups()
    table_ref = f"{proj}.{dataset}.{table}"
    
    # TODO: Lab 5.6.5 — Observability: metrics, logging, and dashboard links
    # INSPECT: Logging pattern for data loading operations.
    
    logging.info("[DEV] Reading training data from BigQuery: %s", table_ref)
    
    # TODO: Lab 5.4.2 - Purpose Recognition: Component reads input artifact from previous component
    # Create BigQuery client and query the table to get training data as DataFrame
    bq_client = bigquery.Client(project=project_id, location=bq_location)
    query = f"SELECT * FROM `{table_ref}`"
    train_df = bq_client.query(query).to_dataframe()
    
    # TODO: Lab 5.6.5 — Observability: metrics, logging, and dashboard links
    # INSPECT: Logging pattern for data shape verification.
    # NOTE: Logging row count helps verify data was loaded correctly.
    
    logging.info("[DEV] Loaded %d training rows from BigQuery", len(train_df))
    
    X = train_df[FEATURE_COLUMNS]
    y = train_df["Diabetic"]

    # Train logistic regression model with regularization rate
    model = LogisticRegression(C=1 / reg_rate, solver="liblinear")
    model.fit(X, y)

    # TODO: Lab 5.4.2 - Purpose Recognition: Model saved to artifact path for downstream use
    # TODO: Lab 5.4.3 - Architecture Understanding: Artifact storage enables stateless component execution
    # Save model as model.joblib for Vertex AI compatibility
    model_path = os.path.join(os.path.dirname(output_model.path), "model.joblib")
    joblib.dump(model, model_path)
    # Copy model to output_model.path for downstream use
    shutil.copy(model_path, output_model.path)
    
    # TODO: Lab 5.6.5 — Observability: metrics, logging, and dashboard links
    # INSPECT: Logging pattern for artifact storage confirmation.
    # NOTE: Logs model path for verification and debugging.
    
    logging.info(
        "[DEV] Model trained and stored at: %s and copied to: %s",
        model_path,
        output_model.path
    )


# ==============================================================================
# Component: evaluate_model_op (MODIFIED FOR BIGQUERY)
# ==============================================================================
# TODO: Lab 5.4.1 - Component Identification: Model evaluation component
# TODO: Lab 5.4.2 - Purpose Recognition: Evaluates model performance on test data
# TODO: Lab 5.4.3 - Architecture Understanding: Depends on both BigQuery and training components
# TODO: Lab 5.5.4 - BigQuery Integration: Reads test data from BigQuery table
#
# TODO: Lab 5.6.3 — Caching and retry guidance
# INSPECT: This evaluation component is IDEMPOTENT.
# CACHING GUIDANCE: SAFE TO CACHE - same test data + same model = same accuracy.
# RETRY GUIDANCE: SAFE TO RETRY - evaluation has no external side effects.
# IDEMPOTENCY: Reads data and model, computes metrics, logs results - no state changes.
#
# MODIFICATION NOTES:
# ==================
# CHANGED: Now reads from BigQuery table instead of CSV file
# - OLD: test_df = pd.read_csv(test_data.path)
# - NEW: Uses BigQuery Python client to read from table
# - Added parameters: project_id and bq_location for BigQuery client
# - Uses URI parsing to extract table reference from artifact
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
    test_data: Input[artifact_types.BQTable],
    model: Input[Model],
    # TODO: Lab 5.4.1 - Component Identification: Metrics output for tracking component
    # TODO: Lab 5.4.2 - Purpose Recognition: Metrics enable model performance monitoring
    metrics: Output[Metrics],
    min_accuracy: float,
    # TODO: Lab 5.5.4 - BigQuery Integration: New parameters for BigQuery access
    project_id: str,
    bq_location: str
) -> float:
    # TODO: Lab 5.4.1 - Component Identification: Float return value enables conditional logic
    # TODO: Lab 5.4.2 - Purpose Recognition: Return values can be used in pipeline conditions
    # TODO: Lab 5.4.3 - Architecture Understanding: Return values flow through pipeline DAG
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
    
    # TODO: Lab 5.5.4 - BigQuery Integration: Read from BigQuery instead of CSV
    # OLD CODE (COMMENTED OUT):
    # test_df = pd.read_csv(test_data.path)
    #
    # NEW CODE (ACTIVE):
    # Parse table reference from URI using regex
    uri = test_data.uri
    logging.info("[DEV] Parsing BigQuery table from URI: %s", uri)
    match = re.search(r'projects/([^/]+)/datasets/([^/]+)/tables/([^/]+)', uri)
    if not match:
        raise ValueError(f"Could not parse BigQuery table reference from URI: {uri}")
    
    proj, dataset, table = match.groups()
    table_ref = f"{proj}.{dataset}.{table}"
    
    logging.info("[DEV] Reading test data from BigQuery: %s", table_ref)
    
    # Create BigQuery client and query the table to get test data as DataFrame
    bq_client = bigquery.Client(project=project_id, location=bq_location)
    query = f"SELECT * FROM `{table_ref}`"
    test_df = bq_client.query(query).to_dataframe()
    
    logging.info("[DEV] Loaded %d test rows from BigQuery", len(test_df))
    
    model_artifact = joblib.load(model.path)

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["Diabetic"]
    predictions = model_artifact.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    
    # TODO: Lab 5.4.1 - Component Identification: metrics.log_metric() records performance metrics
    # TODO: Lab 5.4.2 - Purpose Recognition: Logged metrics appear in Vertex AI UI for tracking
    
    # TODO: Lab 5.6.5 — Observability: metrics, logging, and dashboard links
    # INSPECT: metrics.log_metric() calls record performance metrics.
    # TASK: List ALL metrics logged here: "accuracy" and "min_accuracy_threshold".
    # NOTE: These metrics appear in Vertex AI console under the evaluate_model_op component.
    # LOCATION: Vertex AI Console → Pipeline Run → Click evaluate_model_op → Metrics tab.
    
    metrics.log_metric("accuracy", accuracy)
    metrics.log_metric("min_accuracy_threshold", min_accuracy)

    # TODO: Lab 5.6.5 — Observability: metrics, logging, and dashboard links
    # INSPECT: Logging pattern for key evaluation results.
    
    logging.info("[DEV] Accuracy = %.4f", accuracy)
    return accuracy


# ==============================================================================
# Component: model_approved_op
# ==============================================================================
# TODO: Lab 5.4.1 - Component Identification: Model approval logging component
# TODO: Lab 5.4.2 - Purpose Recognition: Conditional component executed only if accuracy threshold met
# TODO: Lab 5.4.3 - Architecture Understanding: Component within conditional branch of pipeline
#
# TODO: Lab 5.6.3 — Caching and retry guidance
# INSPECT: This approval component is IDEMPOTENT (only logs information).
# CACHING GUIDANCE: SAFE TO CACHE - same inputs produce same log output.
# RETRY GUIDANCE: SAFE TO RETRY - only logs information, no side effects.
#
# Logs approval message if model accuracy meets threshold.
@component(
    base_image=BASE_IMAGE
)
def model_approved_op(model_accuracy: float, model: Input[Model]):
    import logging
    logging.basicConfig(level=logging.INFO)
    logging.info(
        "[DEV] ✅ Model approved with accuracy: %.4f",
        model_accuracy
    )
    logging.info(
        "[DEV] Ready for registration from: %s",
        model.uri
    )


# ==============================================================================
# Component: register_model_op
# ==============================================================================
# TODO: Lab 5.4.1 - Component Identification: Model registration component
# TODO: Lab 5.4.2 - Purpose Recognition: Registers approved model to Vertex AI Model Registry
# TODO: Lab 5.4.3 - Architecture Understanding: Integration with Vertex AI Model Registry service
#
# TODO: Lab 5.5.1 - Custom Component Analysis: This is a CUSTOM COMPONENT
# TODO: Lab 5.5.2 - Custom vs Pre-built Decision: Custom component chosen because:
#   - Requires custom model upload logic
#   - Handles versioning with parent model parameter
#   - Specific serving container configuration
#   - Integration with custom model artifacts
#
# TODO: Lab 5.6.3 — Caching and retry guidance
# INSPECT: This registration component is NOT IDEMPOTENT (has side effects).
# CACHING GUIDANCE: DO NOT CACHE - creates new model version in Vertex AI each time.
# RETRY GUIDANCE: BE CAREFUL WITH RETRIES - may create duplicate model versions.
# SIDE EFFECTS: Writes to Vertex AI Model Registry (external state change).
# RECOMMENDATION: Disable caching for this component; use retries cautiously.
#
# ALTERNATIVE: Pre-built Component Option (For Lab 5.5 Exploration)
# TODO: Lab 5.5.2 - Pre-built Alternative: Could use ModelUploadOp pre-built component
# -----------------------------------------------------------------------------
# COMMENTED OUT - Pre-built Component Example for Lab 5.5 Learning:
#
# from google_cloud_pipeline_components.v1.model import ModelUploadOp
#
# # Pre-built component for model registration:
# register_task = ModelUploadOp(
#     project=project_id,
#     location=region,
#     display_name=model_display_name,
#     artifact_uri=model_artifact.uri,
#     serving_container_image_uri=(
#         "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-2:latest"
#     ),
#     parent_model=parent_model,  # For versioning
# )
#
# PRE-BUILT COMPONENT BENEFITS:
# - Standardized model upload interface
# - Consistent with other Vertex AI components
# - Google-maintained and tested
# - Built-in validation and error handling
# - Better integration with Vertex AI features
#
# CUSTOM COMPONENT BENEFITS (Current Choice):
# - More flexible artifact handling
# - Custom logging for development visibility
# - Direct control over upload parameters
# - Easier to debug during development
# -----------------------------------------------------------------------------
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
    # TODO: Lab 5.4.1 - Component Identification: Component parameters for Vertex AI integration
    # TODO: Lab 5.4.2 - Purpose Recognition: Parameters configure model registration in Vertex AI
    project_id: str,
    region: str,
    model_display_name: str,
    model_artifact: Input[Model],
    parent_model: str = ""
):
    from google.cloud import aiplatform
    import logging

    logging.basicConfig(level=logging.INFO)
    # TODO: Lab 5.4.2 - Purpose Recognition: aiplatform.init() connects component to Vertex AI
    # TODO: Lab 5.4.3 - Architecture Understanding: Components can interact with GCP services
    aiplatform.init(project=project_id, location=region)

    artifact_dir = model_artifact.uri.rsplit("/", 1)[0]
    upload_args = {
        "display_name": model_display_name,
        "artifact_uri": artifact_dir,
        # TODO: Lab 5.4.1 - Component Identification: Serving container for model deployment
        # TODO: Lab 5.4.2 - Purpose Recognition: Container enables model serving in Vertex AI
        "serving_container_image_uri": (
            "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-2:latest"
        ),
        "sync": True
    }

    # TODO: Lab 5.4.2 - Purpose Recognition: Parent model enables model versioning
    # TODO: Lab 5.4.3 - Architecture Understanding: Model Registry supports version lineage
    if parent_model:
        upload_args["parent_model"] = parent_model
        logging.info(
            "[DEV] Registering new version under parent model: %s",
            parent_model
        )

    model = aiplatform.Model.upload(**upload_args)
    logging.info(
        "[DEV] Model registered: %s",
        model.resource_name
    )


# ==============================================================================
# Component: model_rejected_op
# ==============================================================================
# TODO: Lab 5.4.1 - Component Identification: Model rejection component
# TODO: Lab 5.4.2 - Purpose Recognition: Conditional component executed only if accuracy below threshold
# TODO: Lab 5.4.3 - Architecture Understanding: Component in alternate conditional branch
#
# TODO: Lab 5.6.3 — Caching and retry guidance
# INSPECT: This rejection component is IDEMPOTENT (logs error and raises exception).
# CACHING GUIDANCE: SAFE TO CACHE - same inputs produce same error.
# RETRY GUIDANCE: DO NOT RETRY - intentional failure when accuracy is insufficient.
# PURPOSE: Raises ValueError to fail pipeline when model quality is below threshold.
#
# TODO: Lab 5.6.7 — Failure modes and debugging tips
# INSPECT: This component intentionally FAILS the pipeline with ValueError.
# DEBUGGING TIP: If you see this error, model accuracy is below min_accuracy threshold.
# RESOLUTION: Check training data quality, adjust reg_rate, or lower min_accuracy threshold.
# NOTE: This is a QUALITY GATE - not a bug, but an expected failure for low-quality models.
#
# Logs rejection message and raises error if model accuracy is below threshold.
@component(
    base_image=BASE_IMAGE
)
def model_rejected_op(model_accuracy: float, min_accuracy: float):
    import logging
    logging.basicConfig(level=logging.INFO)
    logging.error(
        "[DEV] ❌ Model rejected. Accuracy %.4f < %.2f",
        model_accuracy,
        min_accuracy
    )
    # TODO: Lab 5.4.2 - Purpose Recognition: Raising error fails pipeline execution
    # TODO: Lab 5.4.3 - Architecture Understanding: Component failures propagate through pipeline
    raise ValueError(
        "Model accuracy does not meet minimum development threshold."
    )


# ==============================================================================
# Pipeline: dev_diabetes_pipeline (MODIFIED FOR BIGQUERY)
# ==============================================================================
# TODO: Lab 5.4.1 - Component Identification: @dsl.pipeline decorator defines pipeline function
# TODO: Lab 5.4.2 - Purpose Recognition: Pipeline function orchestrates component execution
# TODO: Lab 5.4.3 - Architecture Understanding: Pipeline defines component dependencies and data flow
# TODO: Lab 5.5.4 - BigQuery Integration: Pipeline now orchestrates BigQuery and custom components
#
# Orchestrates all pipeline steps using BigQuery for data access. Registers
# model only if accuracy meets threshold, otherwise rejects.
@dsl.pipeline(name=PIPELINE_NAME, description=PIPELINE_DESCRIPTION)
def dev_diabetes_pipeline(
    # TODO: Lab 5.4.1 - Component Identification: Pipeline parameters configure entire workflow
    # TODO: Lab 5.4.2 - Purpose Recognition: Pipeline-level parameters flow to individual components
    # TODO: Lab 5.4.3 - Architecture Understanding: Parameters enable dynamic pipeline configuration
    project_id: str,
    region: str,
    model_display_name: str,
    # OLD PARAMETER (GCS-based - COMMENTED OUT):
    # input_raw_data_gcs_uri: str,  # GCS URI to raw CSV - NO LONGER USED
    # TODO: Lab 5.5.4 - BigQuery Integration: input_raw_data_gcs_uri replaced by bq_dataset and bq_view
    # NEW PARAMETERS (BigQuery-based - ACTIVE):
    bq_dataset: str,  # BigQuery dataset name (e.g., "shared_bronze")
    bq_view: str,     # BigQuery view name (e.g., "diabetes_features_view")
    reg_rate: float = 0.05,
    min_accuracy: float = 0.70,
    parent_model: str = ""
):
    # ==========================================================================
    # DATA PREPROCESSING - BigQuery Query Job Components (REPLACES preprocess_data_op)
    # ==========================================================================
    # TODO: Lab 5.4.1 - Component Identification: BigQuery Query Job components replace custom preprocessing
    # TODO: Lab 5.4.2 - Purpose Recognition: Pre-built components for scalable data access
    # TODO: Lab 5.4.3 - Architecture Understanding: Two query tasks create train/test split
    # TODO: Lab 5.5.1 - Pre-built Component: Using Google Cloud Pipeline Components
    # TODO: Lab 5.5.2 - Pre-built Benefits: No custom code, managed infrastructure, standardized
    # TODO: Lab 5.5.4 - BigQuery Integration: SQL-based data access and splitting
    # TODO: Lab 5.5.5 - Feature Groups: Queries reference Feature Group-backed view
    
    # OLD PREPROCESSING CODE (COMMENTED OUT - GCS-based):
    # ====================================================
    # preprocess_task = preprocess_data_op(
    #     input_gcs_uri=input_raw_data_gcs_uri
    # ).set_cpu_limit("1").set_memory_limit("3840Mi")
    #
    # This custom component:
    # - Downloaded CSV from GCS
    # - Split data 80/20 using pandas
    # - Wrote train/test CSVs to artifact storage
    # - Limited scalability
    #
    # REPLACED BY: BigQuery Query Job components (see below)
    # ======================================================
    
    # NEW PREPROCESSING CODE (ACTIVE - BigQuery-based):
    # ==================================================
    # Query training data from BigQuery view (80% split using hash function)
    # Uses FARM_FINGERPRINT for deterministic, reproducible splitting
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
    # TODO: Lab 5.5.4 - BigQuery Integration: Component executes SQL query and outputs table
    
    # TODO: Lab 5.6.4 — Parallelism and resource limits
    # INSPECT: BigQuery Query Job components do NOT have .set_cpu_limit() or .set_memory_limit().
    # NOTE: Pre-built BigQuery components use managed resources; resource limits not applicable.
    # PARALLELISM: bq_train_task and bq_test_task are INDEPENDENT (no .after() dependency).
    # OBSERVATION: These two tasks can execute IN PARALLEL since they have no dependencies.
    # TASK: Mark bq_train_task and bq_test_task as parallelizable in orchestration-notes.md.
    
    bq_train_task = bigquery_query_job_op(
        project=project_id,
        location=region,
        query=train_query
    )
    
    # Query test data from BigQuery view (20% split using hash function)
    test_query = f"""SELECT
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
    # TODO: Lab 5.5.4 - BigQuery Integration: Component executes SQL query and outputs table
    bq_test_task = bigquery_query_job_op(
        project=project_id,
        location=region,
        query=test_query
    )

    # ==========================================================================
    # MODEL TRAINING - Custom Component with BigQuery Input
    # ==========================================================================
    # TODO: Lab 5.4.1 - Component Identification: Training component depends on BigQuery query
    # TODO: Lab 5.4.2 - Purpose Recognition: Component input changed from CSV to BigQuery table
    # TODO: Lab 5.4.3 - Architecture Understanding: Dependencies updated for BigQuery integration
    # TODO: Lab 5.5.4 - BigQuery Integration: Training reads from BigQuery query output
    
    # TODO: Lab 5.6.4 — Parallelism and resource limits
    # INSPECT: .set_cpu_limit("1") and .set_memory_limit("3840Mi") calls on train_task.
    # TASK: Record in orchestration-notes.md: train_model_op | CPU: 1 | Memory: 3840Mi.
    # NOTE: 3840Mi = 3.75 GB memory allocation for training component.
    # RESOURCE GUIDANCE: Increase limits if training fails with OOM (out of memory) errors.
    # PARALLELISM: train_task depends on bq_train_task (cannot run in parallel with it).
    
    # Train model on training data from BigQuery
    train_task = train_model_op(
        # OLD INPUT (COMMENTED OUT - GCS-based):
        # train_data=preprocess_task.outputs["output_train_data"],
        # NEW INPUT (ACTIVE - BigQuery-based):
        train_data=bq_train_task.outputs["destination_table"],
        reg_rate=reg_rate,
        # TODO: Lab 5.5.4 - BigQuery Integration: Pass project and location for BigQuery client
        project_id=project_id,
        bq_location=region
    ).set_cpu_limit("1").set_memory_limit("3840Mi")
    # TODO: Lab 5.4.1 - Component Identification: .after() explicitly defines execution dependency
    # TODO: Lab 5.4.3 - Architecture Understanding: Explicit dependencies augment implicit data dependencies
    # OLD DEPENDENCY (COMMENTED OUT):
    # train_task.after(preprocess_task)
    # NEW DEPENDENCY (ACTIVE):
    train_task.after(bq_train_task)

    # ==========================================================================
    # MODEL EVALUATION - Custom Component with BigQuery Input
    # ==========================================================================
    # TODO: Lab 5.4.1 - Component Identification: Multiple output references create multi-dependencies
    # TODO: Lab 5.4.3 - Architecture Understanding: Component depends on multiple upstream components
    # TODO: Lab 5.5.4 - BigQuery Integration: Evaluation reads from BigQuery query output
    
    # TODO: Lab 5.6.4 — Parallelism and resource limits
    # INSPECT: .set_cpu_limit("1") and .set_memory_limit("3840Mi") calls on eval_task.
    # TASK: Record in orchestration-notes.md: evaluate_model_op | CPU: 1 | Memory: 3840Mi.
    # PARALLELISM: eval_task depends on train_task and bq_test_task (waits for both).
    # OBSERVATION: eval_task CANNOT run until both dependencies complete.
    
    # Evaluate model on test data from BigQuery
    eval_task = evaluate_model_op(
        model=train_task.outputs["output_model"],
        # OLD INPUT (COMMENTED OUT - GCS-based):
        # test_data=preprocess_task.outputs["output_test_data"],
        # NEW INPUT (ACTIVE - BigQuery-based):
        test_data=bq_test_task.outputs["destination_table"],
        min_accuracy=min_accuracy,
        # TODO: Lab 5.5.4 - BigQuery Integration: Pass project and location for BigQuery client
        project_id=project_id,
        bq_location=region
    ).set_cpu_limit("1").set_memory_limit("3840Mi")
    eval_task.after(train_task)

    # ==========================================================================
    # CONDITIONAL LOGIC - Pipeline Branching
    # ==========================================================================
    # TODO: Lab 5.4.1 - Component Identification: dsl.If creates conditional pipeline branch
    # TODO: Lab 5.4.2 - Purpose Recognition: Conditional logic enables quality gates in pipelines
    # TODO: Lab 5.4.3 - Architecture Understanding: Branches are DAG subgraphs conditionally executed
    
    # If accuracy >= min_accuracy, approve and register model
    with dsl.If(
        # TODO: Lab 5.4.1 - Component Identification: Component outputs used in conditional expressions
        # TODO: Lab 5.4.2 - Purpose Recognition: Evaluation output determines pipeline branch execution
        eval_task.outputs["Output"] >= min_accuracy,
        name="pass-accuracy-threshold"
    ):
        # TODO: Lab 5.4.3 - Architecture Understanding: Components in conditional block execute only if condition true
        
        # TODO: Lab 5.6.4 — Parallelism and resource limits
        # INSPECT: .set_cpu_limit("1") and .set_memory_limit("3840Mi") calls on approved_task.
        # TASK: Record in orchestration-notes.md: model_approved_op | CPU: 1 | Memory: 3840Mi.
        # NOTE: Lightweight logging component; 1 CPU and 3840Mi is more than sufficient.
        
        approved_task = model_approved_op(
            model_accuracy=eval_task.outputs["Output"],
            model=train_task.outputs["output_model"]
        ).set_cpu_limit("1").set_memory_limit("3840Mi")
        approved_task.after(eval_task)

        # TODO: Lab 5.4.1 - Component Identification: Model registration component in approval branch
        # TODO: Lab 5.4.2 - Purpose Recognition: Registration occurs only for approved models
        
        # TODO: Lab 5.6.4 — Parallelism and resource limits
        # INSPECT: .set_cpu_limit("1") and .set_memory_limit("3840Mi") calls on register_task.
        # TASK: Record in orchestration-notes.md: register_model_op | CPU: 1 | Memory: 3840Mi.
        # NOTE: Model upload component; resources adequate for model registration.
        # PARALLELISM: register_task depends on approved_task (sequential within conditional branch).
        
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
        # INSPECT: .set_cpu_limit("1") and .set_memory_limit("3840Mi") calls on rejected_task.
        # TASK: Record in orchestration-notes.md: model_rejected_op | CPU: 1 | Memory: 3840Mi.
        # PARALLELISM: Only ONE conditional branch executes (approved OR rejected, never both).
        
        rejected_task = model_rejected_op(
            model_accuracy=eval_task.outputs["Output"],
            min_accuracy=min_accuracy
        ).set_cpu_limit("1").set_memory_limit("3840Mi")
        rejected_task.after(eval_task)


# ==============================================================================
# Lab 5.4 Architecture Summary: Development Pipeline Component Architecture
# ==============================================================================
# This pipeline demonstrates the complete Vertex AI Pipeline component architecture:
#
# 1. COMPONENT TYPES IDENTIFIED:
#    - Data processing components (BigQuery Query Job - PRE-BUILT)
#    - Model training components (train_model_op - CUSTOM)
#    - Model evaluation components (evaluate_model_op - CUSTOM)
#    - Conditional logic components (model_approved_op, model_rejected_op - CUSTOM)
#    - Integration components (register_model_op - CUSTOM)
#
# 2. COMPONENT PURPOSES:
#    - Query data from BigQuery with SQL-based splitting
#    - Train ML models with configurable parameters
#    - Evaluate model performance against thresholds
#    - Implement quality gates through conditional logic
#    - Integrate with Vertex AI services (Model Registry)
#
# 3. PIPELINE ARCHITECTURE:
#    - Components connected through typed inputs/outputs
#    - Dependencies define execution order (DAG structure)
#    - Conditional branching enables quality control
#    - Resource specifications ensure scalability
#    - Artifact storage enables stateless components
#    - Pipeline parameters enable dynamic configuration
#
# TODO: Lab 5.4.3 - Architecture Understanding: This pipeline demonstrates
#       complete component-based architecture for ML workflows in Vertex AI
# ==============================================================================

# ==============================================================================
# Lab 5.5 Component Strategy Summary: Custom vs Pre-built Components
# ==============================================================================
# This pipeline demonstrates strategic use of BOTH custom and pre-built components:
#
# 1. COMPONENTS USED IN THIS PIPELINE:
#    Component              | Type       | Why?
#    --------------------- | ---------- | ------------------------------------------
#    BigQuery Query Job    | PRE-BUILT  | Google-managed, optimized for BigQuery
#    train_model_op        | CUSTOM     | Business-specific sklearn algorithm
#    evaluate_model_op     | CUSTOM     | Custom metrics and threshold evaluation
#    model_approved_op     | CUSTOM     | Custom logging and approval workflow
#    register_model_op     | CUSTOM     | Custom versioning and upload logic
#    model_rejected_op     | CUSTOM     | Custom error handling and rejection
#
# 2. DECISION CRITERIA APPLIED:
#    PRE-BUILT (BigQuery Query Job):
#    - Standard data access operation
#    - Google-maintained and optimized
#    - No business-specific logic needed
#    - Built-in error handling and retries
#    - Eliminates custom code maintenance
#
#    CUSTOM (Training, Evaluation, Registration):
#    - Business-specific algorithms and logic
#    - Custom metrics and quality gates
#    - Organization-specific workflows
#    - Full control over implementation
#
# 3. MIGRATION BENEFITS:
#    BEFORE (All Custom):
#    - Custom preprocessing component for CSV handling
#    - Required maintenance of data access code
#    - Limited scalability for large datasets
#
#    AFTER (Mixed Custom/Pre-built):
#    - Pre-built component for data access
#    - Reduced code maintenance burden
#    - Serverless, scalable BigQuery processing
#    - Custom components only where needed
#
# 4. BIGQUERY AND FEATURE GROUP INTEGRATION:
#    
#    BIGQUERY ARCHITECTURE:
#    ----------------------
#    Layer                  | Component
#    --------------------- | ------------------------------------------
#    Data Source           | GCS CSV → BigQuery source table
#    Feature Engineering   | BigQuery view (column selection, filtering)
#    Feature Governance    | Vertex AI Feature Group (metadata, lineage)
#    Data Access           | BigQuery Query Job component (SQL queries)
#    ML Training           | Custom components (read from BigQuery)
#    
#    BIGQUERY VIEW STRUCTURE:
#    - Source: BigQuery table loaded from GCS CSV
#    - View columns: 11 total (entity_id, feature_timestamp, + 9 features)
#    - Feature Group requirements: entity_id, feature_timestamp
#    - Pipeline usage: SELECT only 9 feature columns (ignores metadata)
#    - Train/test split: SQL FARM_FINGERPRINT hash function
#    
#    TRAIN/TEST SPLITTING METHODOLOGY:
#    SQL-based deterministic splitting:
#    - Training: WHERE MOD(ABS(FARM_FINGERPRINT(CAST(entity_id AS STRING))), 10) < 8
#    - Test: WHERE MOD(ABS(FARM_FINGERPRINT(CAST(entity_id AS STRING))), 10) >= 8
#    - Result: Consistent 80/20 split across pipeline runs
#    - Benefits: Reproducible, no random seed needed, scales to any data size
#    
#    FEATURE GROUP INTEGRATION:
#    - Feature Group: Registered in Vertex AI Feature Registry
#    - Purpose: Metadata, lineage tracking, governance
#    - Usage: Not directly queried by pipeline (queries view directly)
#    - Benefits: Feature discovery, versioning, audit trail
#    
#    MIGRATION FROM GCS TO BIGQUERY:
#    BEFORE (GCS-based):
#    - Downloaded CSV from GCS
#    - Python pandas splitting
#    - Custom preprocessing component
#    - Limited scalability
#    
#    AFTER (BigQuery-based):
#    - SQL queries against BigQuery view
#    - SQL-based hash splitting
#    - Pre-built BigQuery Query Job component
#    - Serverless, unlimited scalability
#    - No data duplication (view references source table)
#    
#    BENEFITS:
#    - Serverless: No infrastructure management
#    - Scalable: Handles datasets of any size
#    - Governed: Feature Groups provide metadata and lineage
#    - Efficient: No data duplication (views reference source)
#    - Transparent: SQL-based transformations are readable
#    - Reusable: Same view can serve multiple pipelines
#
# 5. ACCELERATOR TEMPLATE INTEGRATION:
#    This pipeline fits into the 3-layer enterprise architecture:
#    
#    INFRASTRUCTURE (Terraform) → vertex_ai_infrastructure.tf
#    ├── Provisions: Pipelines, Endpoints, Buckets, Service Accounts
#    ├── Provisions: BigQuery datasets, Feature Groups
#    ├── Manages: Multi-environment deployment (dev/prod)
#    └── Enables: Reproducible infrastructure as code
#    
#    PIPELINE (This File) → vertex_pipeline_dev.py
#    ├── Pre-built Components: BigQuery data access
#    ├── Custom Components: Business-specific logic
#    └── Orchestration: ML workflow from data to deployment
#    
#    ENTERPRISE (Best Practices) → Industry-specific implementations
#    ├── Governance patterns
#    ├── Compliance requirements
#    └── Reusable solution frameworks
#
# TODO: Lab 5.5.1 - Component Mix: Strategic use of both custom and pre-built
# TODO: Lab 5.5.2 - Pre-built Migration: Replaced custom preprocessing with BigQuery
# TODO: Lab 5.5.3 - Accelerator Templates: This pipeline deployed via Terraform infrastructure
# TODO: Lab 5.5.4 - BigQuery Integration: Complete data access migration to BigQuery
# TODO: Lab 5.5.5 - Feature Groups: Governance and metadata layer for features
# ==============================================================================

# ==============================================================================
# Lab 5.6 Pipeline Orchestration Summary
# ==============================================================================
# TODO: Lab 5.6.3 — Caching and retry guidance
# CACHING AND RETRY SUMMARY FOR THIS PIPELINE:
# ============================================
# Component                | Cacheable? | Retry Safe? | Reason
# ------------------------|------------|-------------|---------------------------
# bigquery_query_job_op   | YES        | YES         | Idempotent query, built-in retries
# train_model_op          | YES        | YES         | Deterministic training, no side effects
# evaluate_model_op       | YES        | YES         | Deterministic evaluation, no side effects
# model_approved_op       | YES        | YES         | Logging only, no side effects
# register_model_op       | NO         | CAUTIOUS    | Creates model version (side effect)
# model_rejected_op       | YES        | NO          | Intentional failure (quality gate)
#
# IDEMPOTENCY GUIDANCE:
# - Idempotent components: Same inputs → same outputs, no external state changes
# - Non-idempotent components: Have side effects (write to external systems)
# - Enable caching for idempotent components in dev pipelines
# - Disable caching for components with side effects or in production pipelines
#
# TODO: Lab 5.6.4 — Parallelism and resource limits
# RESOURCE LIMITS SUMMARY FOR THIS PIPELINE:
# ==========================================
# Component                | CPU   | Memory  | Notes
# ------------------------|-------|---------|--------------------------------
# bigquery_query_job_op   | N/A   | N/A     | Managed by BigQuery (serverless)
# train_model_op          | 1     | 3840Mi  | Adequate for small dataset training
# evaluate_model_op       | 1     | 3840Mi  | Adequate for model evaluation
# model_approved_op       | 1     | 3840Mi  | Lightweight logging (over-provisioned)
# register_model_op       | 1     | 3840Mi  | Adequate for model upload
# model_rejected_op       | 1     | 3840Mi  | Lightweight logging (over-provisioned)
#
# PARALLELISM OPPORTUNITIES:
# - bq_train_task and bq_test_task: PARALLEL (independent queries)
# - train_task depends on bq_train_task: SEQUENTIAL
# - eval_task depends on train_task and bq_test_task: WAITS FOR BOTH
# - Conditional branches (approved/rejected): MUTUALLY EXCLUSIVE (only one executes)
#
# RESOURCE SIZING GUIDANCE:
# - Increase CPU/memory if components fail with OOM or timeout errors
# - Decrease for lightweight components to optimize costs
# - 3840Mi (3.75 GB) is adequate for this small dataset
# - For larger datasets, scale up train_model_op and evaluate_model_op resources
#
# TODO: Lab 5.6.5 — Observability: metrics, logging, and dashboard links
# OBSERVABILITY SUMMARY FOR THIS PIPELINE:
# ========================================
# METRICS LOGGED:
# - evaluate_model_op: "accuracy", "min_accuracy_threshold"
#
# LOGGING PATTERNS (logging.info/error):
# - train_model_op: 4 log statements (URI parsing, data loading, row count, model storage)
# - evaluate_model_op: 4 log statements (URI parsing, data loading, row count, accuracy)
# - model_approved_op: 2 log statements (approval message, model URI)
# - register_model_op: 2 log statements (versioning info, registration confirmation)
# - model_rejected_op: 1 log statement (rejection error)
#
# WHERE TO FIND LOGS:
# - Vertex AI Console → Pipeline Run → Click component → Logs tab
# - Cloud Logging: filter by pipeline_job_id for all component logs
# - Dashboard URI: Logged by run_pipeline.py after submission
#
# TODO: Lab 5.6.7 — Failure modes and debugging tips
# COMMON FAILURE MODES IN THIS PIPELINE:
# ======================================
# 1. BigQuery Query Job failures:
#    - Cause: Invalid SQL syntax, missing table/view, insufficient IAM permissions
#    - Debug: Check BigQuery query syntax, verify view exists, check service account permissions
#
# 2. URI parsing failures (train_model_op, evaluate_model_op):
#    - Cause: Upstream BigQuery component failed or output format changed
#    - Debug: Verify BigQuery Query Job succeeded, check artifact URI format
#
# 3. Model rejected (model_rejected_op):
#    - Cause: Accuracy below min_accuracy threshold (EXPECTED BEHAVIOR, not a bug)
#    - Debug: Check training data quality, adjust reg_rate, review evaluation metrics
#    - Resolution: This is a QUALITY GATE - improve model or adjust threshold
#
# 4. Model registration failures (register_model_op):
#    - Cause: Insufficient IAM permissions, invalid serving container, model artifact issues
#    - Debug: Check service account has Vertex AI User role, verify artifact URI, check logs
#
# 5. Resource exhaustion (OOM errors):
#    - Cause: Insufficient memory allocation for component
#    - Debug: Check component logs for OOM errors, increase .set_memory_limit()
#
# BEST PRACTICES FOR THIS PIPELINE:
# - Use small components (single responsibility principle)
# - Enable logging for all major operations
# - Use idempotent components for cacheability
# - Apply appropriate resource limits
# - Use labels for cost attribution and filtering
# - Implement quality gates with conditional logic
# ==============================================================================
