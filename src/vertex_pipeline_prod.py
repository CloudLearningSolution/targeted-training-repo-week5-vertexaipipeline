"""
Vertex AI KFP Pipeline for Production

This pipeline orchestrates the production workflow for the diabetes prediction model.
It includes the following steps:
- Preprocesses diabetes data from GCS (splits into train/test sets).
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
"""

from kfp import dsl
from kfp.dsl import (
    component,
    pipeline,
    Input,
    Output,
    Dataset,
    Model,
    Metrics
)

# TODO: Lab 5.4.1 - Component Identification: Pipeline metadata distinguishes prod from dev
# TODO: Lab 5.4.2 - Purpose Recognition: Separate pipeline names enable environment tracking
PIPELINE_NAME = "mlops-diabetes-prod-pipeline"
PIPELINE_DESCRIPTION = (
    "Production pipeline for diabetes prediction model on Vertex AI"
)

# TODO: Lab 5.4.1 - Component Identification: Shared base image ensures environment consistency
# TODO: Lab 5.4.3 - Architecture Understanding: Consistent component execution across environments
BASE_IMAGE = "python:3.9"
REQUIREMENTS_PATH = "src/requirements.txt"


# ==============================================================================
# Component: preprocess_data_op
# ==============================================================================
# TODO: Lab 5.4.1 - Component Identification: Identical component definition to dev pipeline
# TODO: Lab 5.4.2 - Purpose Recognition: Reusable components across environments
# TODO: Lab 5.4.3 - Architecture Understanding: Component architecture enables environment reuse
#
# Downloads raw CSV data from GCS, splits into train/test sets, and saves
# outputs for downstream steps. Used for production data preprocessing.
@component(
    base_image=BASE_IMAGE,
    packages_to_install=[
        pkg.strip()
        for pkg in open("src/requirements.txt")
        if pkg.strip() and not pkg.startswith("#")
    ],
)
def preprocess_data_op(
    input_gcs_uri: str,
    output_train_data: Output[Dataset],
    output_test_data: Output[Dataset]
):
    import pandas as pd
    from google.cloud import storage
    from urllib.parse import urlparse
    import logging

    logging.basicConfig(level=logging.INFO)
    parsed = urlparse(input_gcs_uri)
    bucket_name = parsed.netloc
    blob_name = parsed.path.lstrip("/")

    # TODO: Lab 5.4.2 - Purpose Recognition: Production-specific file naming for audit trail
    local_file = "diabetes_raw_prod.csv"
    # Download file from GCS to local disk for processing
    storage.Client().bucket(bucket_name).blob(blob_name).download_to_filename(
        local_file
    )
    df = pd.read_csv(local_file)

    # Split data into train and test sets (80/20 split)
    train_data = df.sample(frac=0.8, random_state=42)
    test_data = df.drop(train_data.index)

    # Save splits to output artifact paths for downstream pipeline steps
    train_data.to_csv(output_train_data.path, index=False)
    test_data.to_csv(output_test_data.path, index=False)

    # TODO: Lab 5.4.2 - Purpose Recognition: Production logging with environment prefix
    logging.info(
        "[PROD] Preprocessed and split data saved to: %s, %s",
        output_train_data.path,
        output_test_data.path,
    )


# ==============================================================================
# Component: train_model_op
# ==============================================================================
# TODO: Lab 5.4.1 - Component Identification: Identical training component to dev pipeline
# TODO: Lab 5.4.2 - Purpose Recognition: Same training algorithm across environments
# TODO: Lab 5.4.3 - Architecture Understanding: Component reusability across pipeline variants
#
# Trains a logistic regression model using the training data and saves the
# model artifact for downstream evaluation and registration.
@component(
    base_image=BASE_IMAGE,
    packages_to_install=[
        pkg.strip()
        for pkg in open("src/requirements.txt")
        if pkg.strip() and not pkg.startswith("#")
    ],
)
def train_model_op(
    train_data: Input[Dataset],
    output_model: Output[Model],
    reg_rate: float
):
    import pandas as pd
    import joblib
    from sklearn.linear_model import LogisticRegression
    import logging
    import os
    import shutil

    FEATURE_COLUMNS = [
        "Pregnancies", "PlasmaGlucose", "DiastolicBloodPressure",
        "TricepsThickness", "SerumInsulin", "BMI", "DiabetesPedigree", "Age"
    ]

    logging.basicConfig(level=logging.INFO)
    train_df = pd.read_csv(train_data.path)
    X = train_df[FEATURE_COLUMNS]
    y = train_df["Diabetic"]

    # Train logistic regression model with regularization rate
    model = LogisticRegression(C=1 / reg_rate, solver="liblinear")
    model.fit(X, y)

    # Save model as model.joblib for Vertex AI compatibility
    model_path = os.path.join(os.path.dirname(output_model.path), "model.joblib")
    joblib.dump(model, model_path)
    # Copy model to output_model.path for downstream use
    shutil.copy(model_path, output_model.path)
    # TODO: Lab 5.4.2 - Purpose Recognition: Production logging for audit and compliance
    logging.info(
        "[PROD] Model trained and stored at: %s and copied to: %s",
        model_path,
        output_model.path
    )


# ==============================================================================
# Component: evaluate_model_op
# ==============================================================================
# TODO: Lab 5.4.1 - Component Identification: Identical evaluation component to dev pipeline
# TODO: Lab 5.4.2 - Purpose Recognition: Consistent evaluation logic across environments
# TODO: Lab 5.4.3 - Architecture Understanding: Component architecture supports shared evaluation
#
# Evaluates the trained model on the test set, logs accuracy, and returns
# accuracy for conditional pipeline logic.
@component(
    base_image=BASE_IMAGE,
    packages_to_install=[
        pkg.strip()
        for pkg in open("src/requirements.txt")
        if pkg.strip() and not pkg.startswith("#")
    ],
)
def evaluate_model_op(
    test_data: Input[Dataset],
    model: Input[Model],
    metrics: Output[Metrics],
    min_accuracy: float
) -> float:
    import pandas as pd
    import joblib
    from sklearn.metrics import accuracy_score
    import logging

    FEATURE_COLUMNS = [
        "Pregnancies", "PlasmaGlucose", "DiastolicBloodPressure",
        "TricepsThickness", "SerumInsulin", "BMI", "DiabetesPedigree", "Age"
    ]

    logging.basicConfig(level=logging.INFO)
    test_df = pd.read_csv(test_data.path)
    model_artifact = joblib.load(model.path)

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["Diabetic"]
    predictions = model_artifact.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    metrics.log_metric("accuracy", accuracy)
    metrics.log_metric("min_accuracy_threshold", min_accuracy)

    logging.info(
        "[PROD] Accuracy = %.4f",
        accuracy
    )
    return accuracy


# ==============================================================================
# Component: model_approved_op
# ==============================================================================
# TODO: Lab 5.4.1 - Component Identification: Approval component for production audit trail
# TODO: Lab 5.4.2 - Purpose Recognition: Enhanced logging for production compliance
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
# Registers the model in Vertex AI Model Registry if approved. Supports
# versioning under a parent model if provided.
@component(
    base_image=BASE_IMAGE,
    packages_to_install=[
        pkg.strip()
        for pkg in open("src/requirements.txt")
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
# Pipeline: prod_diabetes_pipeline
# ==============================================================================
# TODO: Lab 5.4.1 - Component Identification: Production pipeline with identical structure to dev
# TODO: Lab 5.4.2 - Purpose Recognition: Production configuration with stricter thresholds
# TODO: Lab 5.4.3 - Architecture Understanding: Same architecture, different environment configuration
#
# Orchestrates all pipeline steps. Registers model only if accuracy meets
# threshold, otherwise rejects. Supports parent model versioning.
@pipeline(name=PIPELINE_NAME, description=PIPELINE_DESCRIPTION)
def prod_diabetes_pipeline(
    project_id: str,
    region: str,
    model_display_name: str,
    input_raw_data_gcs_uri: str,
    reg_rate: float = 0.05,
    # TODO: Lab 5.4.1 - Component Identification: Higher min_accuracy default for production
    # TODO: Lab 5.4.2 - Purpose Recognition: Stricter quality gates in production environment
    # TODO: Lab 5.4.3 - Architecture Understanding: Environment-specific parameter defaults
    min_accuracy: float = 0.75,  # Production requires higher accuracy than dev (0.70)
    parent_model: str = ""
):
    # ==========================================================================
    # PIPELINE ORCHESTRATION - Production Component DAG
    # ==========================================================================
    # TODO: Lab 5.4.1 - Component Identification: Identical component orchestration to dev pipeline
    # TODO: Lab 5.4.2 - Purpose Recognition: Consistent pipeline architecture across environments
    # TODO: Lab 5.4.3 - Architecture Understanding: Pipeline architecture reusability
    
    # Preprocess raw data from GCS
    preprocess_task = preprocess_data_op(
        input_gcs_uri=input_raw_data_gcs_uri
    ).set_cpu_limit("1").set_memory_limit("3840Mi")

    # Train model on training data
    train_task = train_model_op(
        train_data=preprocess_task.outputs["output_train_data"],
        reg_rate=reg_rate
    ).set_cpu_limit("1").set_memory_limit("3840Mi")
    train_task.after(preprocess_task)

    # Evaluate model on test data
    eval_task = evaluate_model_op(
        model=train_task.outputs["output_model"],
        test_data=preprocess_task.outputs["output_test_data"],
        min_accuracy=min_accuracy
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
        approved_task = model_approved_op(
            model_accuracy=eval_task.outputs["Output"],
            model=train_task.outputs["output_model"]
        ).set_cpu_limit("1").set_memory_limit("3840Mi")
        approved_task.after(eval_task)

        # TODO: Lab 5.4.2 - Purpose Recognition: Production registration includes parent model for versioning
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
        rejected_task = model_rejected_op(
            model_accuracy=eval_task.outputs["Output"],
            min_accuracy=min_accuracy
        ).set_cpu_limit("1").set_memory_limit("3840Mi")
        rejected_task.after(eval_task)


# ==============================================================================
# Lab 5.4 Architecture Summary: Production vs Development Pipeline Architecture
# ==============================================================================
# This production pipeline demonstrates key architectural principles:
#
# 1. COMPONENT ARCHITECTURE REUSABILITY:
#    - Same component definitions as dev pipeline (preprocess, train, evaluate)
#    - Identical pipeline structure and dependencies
#    - Shared component interfaces and contracts
#    - Consistent DAG architecture across environments
#
# 2. ENVIRONMENT-SPECIFIC CONFIGURATION:
#    - Different pipeline names for environment tracking
#    - Higher accuracy threshold (0.75 vs 0.70)
#    - Production-specific logging and audit trail
#    - Model versioning support (parent_model parameter)
#    - Production data sources and storage paths
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
# TODO: Lab 5.4.3 - Architecture Understanding: This demonstrates how the same
#       component architecture adapts to different environments through configuration
# ==============================================================================
