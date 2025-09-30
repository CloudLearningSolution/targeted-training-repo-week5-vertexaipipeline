"""
Vertex AI KFP Pipeline for Development

This pipeline performs the following steps:
- Preprocesses diabetes data from GCS.
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

# TODO: Lab 5.4.1 - Component Identification: Pipeline metadata defines pipeline identity
# TODO: Lab 5.4.2 - Purpose Recognition: Pipeline name and description enable tracking and organization
PIPELINE_NAME = "mlops-diabetes-dev-pipeline"
PIPELINE_DESCRIPTION = (
    "Development pipeline for diabetes prediction model on Vertex AI"
)

# TODO: Lab 5.4.1 - Component Identification: Base image defines component execution environment
# TODO: Lab 5.4.2 - Purpose Recognition: Standardized Python environment ensures consistency
# TODO: Lab 5.4.3 - Architecture Understanding: Container images encapsulate component execution
BASE_IMAGE = "python:3.9"
REQUIREMENTS_PATH = "src/requirements.txt"


# ==============================================================================
# Component: preprocess_data_op
# ==============================================================================
# TODO: Lab 5.4.1 - Component Identification: Data preprocessing component
# TODO: Lab 5.4.2 - Purpose Recognition: Transforms raw data into train/test datasets
# TODO: Lab 5.4.3 - Architecture Understanding: First component in pipeline DAG
#
# Downloads raw CSV data from GCS, splits into train/test sets, and saves
# outputs for downstream steps.
@component(
    # TODO: Lab 5.4.1 - Component Identification: @component decorator defines KFP component
    # TODO: Lab 5.4.2 - Purpose Recognition: Decorator converts Python function to pipeline component
    # TODO: Lab 5.4.3 - Architecture Understanding: Components are containerized execution units
    base_image=BASE_IMAGE,
    packages_to_install=[
        pkg.strip()
        for pkg in open("src/requirements.txt")
        if pkg.strip() and not pkg.startswith("#")
    ],
)
def preprocess_data_op(
    # TODO: Lab 5.4.1 - Component Identification: input_gcs_uri is component input parameter
    # TODO: Lab 5.4.2 - Purpose Recognition: GCS URI enables cloud-native data access
    input_gcs_uri: str,
    # TODO: Lab 5.4.1 - Component Identification: Output[Dataset] defines typed component output
    # TODO: Lab 5.4.2 - Purpose Recognition: Typed outputs enable type-safe component connections
    # TODO: Lab 5.4.3 - Architecture Understanding: Outputs create dependencies in pipeline DAG
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

    local_file = "diabetes_raw_dev.csv"
    # TODO: Lab 5.4.2 - Purpose Recognition: GCS client downloads data into component container
    # Download file from GCS to local disk
    storage.Client().bucket(bucket_name).blob(blob_name).download_to_filename(
        local_file
    )
    df = pd.read_csv(local_file)

    # Split data into train and test sets (80/20 split)
    train_data = df.sample(frac=0.8, random_state=42)
    test_data = df.drop(train_data.index)

    # TODO: Lab 5.4.1 - Component Identification: .path attribute provides artifact storage location
    # TODO: Lab 5.4.2 - Purpose Recognition: Artifacts enable data passing between components
    # TODO: Lab 5.4.3 - Architecture Understanding: KFP manages artifact storage and retrieval
    # Save splits to output artifact paths
    train_data.to_csv(output_train_data.path, index=False)
    test_data.to_csv(output_test_data.path, index=False)

    logging.info(
        "[DEV] Preprocessed and split data saved to: %s, %s",
        output_train_data.path,
        output_test_data.path,
    )


# ==============================================================================
# Component: train_model_op
# ==============================================================================
# TODO: Lab 5.4.1 - Component Identification: Model training component
# TODO: Lab 5.4.2 - Purpose Recognition: Trains ML model using training data
# TODO: Lab 5.4.3 - Architecture Understanding: Depends on preprocessing component output
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
    # TODO: Lab 5.4.1 - Component Identification: Input[Dataset] defines typed component input
    # TODO: Lab 5.4.2 - Purpose Recognition: Input references upstream component output
    # TODO: Lab 5.4.3 - Architecture Understanding: Input creates dependency on preprocessing component
    train_data: Input[Dataset],
    # TODO: Lab 5.4.1 - Component Identification: Output[Model] defines typed model artifact
    # TODO: Lab 5.4.2 - Purpose Recognition: Model output enables downstream evaluation and registration
    output_model: Output[Model],
    # TODO: Lab 5.4.1 - Component Identification: reg_rate is component parameter
    # TODO: Lab 5.4.2 - Purpose Recognition: Parameters enable component configuration at runtime
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
    # TODO: Lab 5.4.2 - Purpose Recognition: Component reads input artifact from previous component
    train_df = pd.read_csv(train_data.path)
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
    logging.info(
        "[DEV] Model trained and stored at: %s and copied to: %s",
        model_path,
        output_model.path
    )


# ==============================================================================
# Component: evaluate_model_op
# ==============================================================================
# TODO: Lab 5.4.1 - Component Identification: Model evaluation component
# TODO: Lab 5.4.2 - Purpose Recognition: Evaluates model performance on test data
# TODO: Lab 5.4.3 - Architecture Understanding: Depends on both preprocessing and training components
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
    # TODO: Lab 5.4.1 - Component Identification: Multiple inputs create multi-dependency component
    # TODO: Lab 5.4.3 - Architecture Understanding: Component depends on two upstream components
    test_data: Input[Dataset],
    model: Input[Model],
    # TODO: Lab 5.4.1 - Component Identification: Metrics output for tracking component
    # TODO: Lab 5.4.2 - Purpose Recognition: Metrics enable model performance monitoring
    metrics: Output[Metrics],
    min_accuracy: float
) -> float:
    # TODO: Lab 5.4.1 - Component Identification: Float return value enables conditional logic
    # TODO: Lab 5.4.2 - Purpose Recognition: Return values can be used in pipeline conditions
    # TODO: Lab 5.4.3 - Architecture Understanding: Return values flow through pipeline DAG
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
    # TODO: Lab 5.4.1 - Component Identification: metrics.log_metric() records performance metrics
    # TODO: Lab 5.4.2 - Purpose Recognition: Logged metrics appear in Vertex AI UI for tracking
    metrics.log_metric("accuracy", accuracy)
    metrics.log_metric("min_accuracy_threshold", min_accuracy)

    logging.info(
        "[DEV] Accuracy = %.4f",
        accuracy
    )
    return accuracy


# ==============================================================================
# Component: model_approved_op
# ==============================================================================
# TODO: Lab 5.4.1 - Component Identification: Model approval logging component
# TODO: Lab 5.4.2 - Purpose Recognition: Conditional component executed only if accuracy threshold met
# TODO: Lab 5.4.3 - Architecture Understanding: Component within conditional branch of pipeline
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
# Pipeline: dev_diabetes_pipeline
# ==============================================================================
# TODO: Lab 5.4.1 - Component Identification: @dsl.pipeline decorator defines pipeline function
# TODO: Lab 5.4.2 - Purpose Recognition: Pipeline function orchestrates component execution
# TODO: Lab 5.4.3 - Architecture Understanding: Pipeline defines component dependencies and data flow
#
# Orchestrates all pipeline steps. Registers model only if accuracy meets
# threshold, otherwise rejects.
@dsl.pipeline(name=PIPELINE_NAME, description=PIPELINE_DESCRIPTION)
def dev_diabetes_pipeline(
    # TODO: Lab 5.4.1 - Component Identification: Pipeline parameters configure entire workflow
    # TODO: Lab 5.4.2 - Purpose Recognition: Pipeline-level parameters flow to individual components
    # TODO: Lab 5.4.3 - Architecture Understanding: Parameters enable dynamic pipeline configuration
    project_id: str,
    region: str,
    model_display_name: str,
    input_raw_data_gcs_uri: str,
    reg_rate: float = 0.05,
    min_accuracy: float = 0.70,
    parent_model: str = ""
):
    # ==========================================================================
    # COMPONENT INSTANTIATION - Creating Pipeline DAG
    # ==========================================================================
    # TODO: Lab 5.4.1 - Component Identification: Component instantiation creates pipeline tasks
    # TODO: Lab 5.4.2 - Purpose Recognition: Tasks are instances of components with specific parameters
    # TODO: Lab 5.4.3 - Architecture Understanding: Task creation builds pipeline execution graph
    
    # Preprocess raw data from GCS
    preprocess_task = preprocess_data_op(
        input_gcs_uri=input_raw_data_gcs_uri
    ).set_cpu_limit("1").set_memory_limit("3840Mi")
    # TODO: Lab 5.4.1 - Component Identification: set_cpu_limit() and set_memory_limit() configure resources
    # TODO: Lab 5.4.2 - Purpose Recognition: Resource limits ensure predictable component execution
    # TODO: Lab 5.4.3 - Architecture Understanding: Resource specifications apply to containerized components

    # Train model on training data
    train_task = train_model_op(
        # TODO: Lab 5.4.1 - Component Identification: .outputs["output_train_data"] references upstream output
        # TODO: Lab 5.4.2 - Purpose Recognition: Output references create component dependencies
        # TODO: Lab 5.4.3 - Architecture Understanding: Dependencies define pipeline execution order
        train_data=preprocess_task.outputs["output_train_data"],
        reg_rate=reg_rate
    ).set_cpu_limit("1").set_memory_limit("3840Mi")
    # TODO: Lab 5.4.1 - Component Identification: .after() explicitly defines execution dependency
    # TODO: Lab 5.4.3 - Architecture Understanding: Explicit dependencies augment implicit data dependencies
    train_task.after(preprocess_task)

    # Evaluate model on test data
    eval_task = evaluate_model_op(
        # TODO: Lab 5.4.1 - Component Identification: Multiple output references create multi-dependencies
        # TODO: Lab 5.4.3 - Architecture Understanding: Component depends on multiple upstream components
        model=train_task.outputs["output_model"],
        test_data=preprocess_task.outputs["output_test_data"],
        min_accuracy=min_accuracy
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
        approved_task = model_approved_op(
            model_accuracy=eval_task.outputs["Output"],
            model=train_task.outputs["output_model"]
        ).set_cpu_limit("1").set_memory_limit("3840Mi")
        approved_task.after(eval_task)

        # TODO: Lab 5.4.1 - Component Identification: Model registration component in approval branch
        # TODO: Lab 5.4.2 - Purpose Recognition: Registration occurs only for approved models
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
#    - Data processing components (preprocess_data_op)
#    - Model training components (train_model_op)
#    - Model evaluation components (evaluate_model_op)
#    - Conditional logic components (model_approved_op, model_rejected_op)
#    - Integration components (register_model_op)
#
# 2. COMPONENT PURPOSES:
#    - Transform raw data into ML-ready datasets
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
