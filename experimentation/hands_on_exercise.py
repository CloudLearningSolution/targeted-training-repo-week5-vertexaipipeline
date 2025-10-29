"""
Hands-On Coding Exercise: Convert train.py to Vertex AI Pipeline
===============================================================

OBJECTIVE: Complete the missing component by converting the 
old_model.py to a Vertex AI pipeline component.

INSTRUCTIONS:
1. Review the original train.py file
2. Study the train_to_vertex_ai_conversion.py file for patterns
3. Complete the missing component below
4. Test your understanding of function-to-component conversion

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
from google_cloud_pipeline_components.types import artifact_types

# Pipeline Configuration
PIPELINE_NAME = "diabetes-classification-exercise-pipeline"
BASE_IMAGE = "python:3.9"

# Pre-built BigQuery component (already completed for you)
bigquery_query_job_op = components.load_component_from_url(
    'https://us-kfp.pkg.dev/ml-pipeline/google-cloud-registry/'
    'bigquery-query-job/sha256:'
    'd1cae80bc0de4e5b95b994739c8d0d7d42ce5a4cb17d3c9512eaed14540f6343'
)

# Evaluation component (already completed for you)
@component(
    base_image=BASE_IMAGE,
    packages_to_install=[
        "google-cloud-bigquery",
        "scikit-learn",
        "joblib", 
        "pandas"
    ]
)
def evaluate_model_op(
    test_data: Input[artifact_types.BQTable],
    model: Input[Model],
    metrics: Output[Metrics],
    min_accuracy: float,
    project_id: str,
    bq_location: str
) -> float:
    import re, logging, joblib
    import pandas as pd
    from sklearn.metrics import accuracy_score
    from google.cloud import bigquery

    logging.basicConfig(level=logging.INFO)
    
    uri = test_data.uri
    match = re.search(r'projects/([^/]+)/datasets/([^/]+)/tables/([^/]+)', uri)
    if not match:
        raise ValueError(f"Could not parse BQ table from URI: {uri}")
    proj, dataset, table = match.groups()
    table_ref = f"{proj}.{dataset}.{table}"

    bq_client = bigquery.Client(project=project_id, location=bq_location)
    query = f"SELECT * FROM `{table_ref}`"
    test_df = bq_client.query(query).to_dataframe()

    model_obj = joblib.load(model.path)
    
    FEATURE_COLUMNS = ["Pregnancies","PlasmaGlucose","DiastolicBloodPressure",
                       "TricepsThickness","SerumInsulin","BMI","DiabetesPedigree","Age"]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["Diabetic"]
    
    preds = model_obj.predict(X_test)
    accuracy = accuracy_score(y_test, preds)

    metrics.log_metric("accuracy", accuracy)
    metrics.log_metric("test_samples", len(test_df))
    
    return accuracy

# =============================================================================
# YOUR CODING Exam:
# =============================================================================
# HINT: The original function signature was:
# def train_model(reg_rate, X_train, X_test, y_train, y_test):

# YOUR CODE HERE - Replace this comment block with your component: 
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


# Model approval components (already completed for you)
@component(base_image=BASE_IMAGE)
def model_approved_op(model_accuracy: float, model_name: str):
    import logging
    logging.basicConfig(level=logging.INFO)
    logging.info("✅ Model '%s' approved with accuracy: %.4f", model_name, model_accuracy)

@component(base_image=BASE_IMAGE)
def model_rejected_op(model_accuracy: float, min_accuracy: float):
    import logging
    logging.basicConfig(level=logging.ERROR)
    logging.error("❌ Model rejected. Accuracy %.4f < %.2f", model_accuracy, min_accuracy)

@component(
    base_image=BASE_IMAGE,
    packages_to_install=["google-cloud-aiplatform"]
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
        "serving_container_image_uri": "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-2:latest",
        "sync": True
    }
    
    if parent_model:
        upload_args["parent_model"] = parent_model
    
    model = aiplatform.Model.upload(**upload_args)
    logging.info("Model registered: %s", model.resource_name)

# =============================================================================
# MAIN PIPELINE DEFINITION (already completed for you)
# =============================================================================
@dsl.pipeline(
    name=PIPELINE_NAME,
    description="Exercise pipeline for train.py to Vertex AI conversion"
)
def diabetes_training_pipeline(
    project_id: str,
    region: str = "us-central1",
    model_display_name: str = "diabetes-classification-model",
    bq_dataset: str = "shared_bronze",
    bq_view: str = "diabetes_features_view",
    reg_rate: float = 0.01,
    min_accuracy: float = 0.70,
    parent_model: str = ""
):
    train_query = f"""
    SELECT Pregnancies, PlasmaGlucose, DiastolicBloodPressure, TricepsThickness,
           SerumInsulin, BMI, DiabetesPedigree, Age, Diabetic
    FROM `{project_id}.{bq_dataset}.{bq_view}`
    WHERE MOD(ABS(FARM_FINGERPRINT(CAST(CONCAT(Pregnancies, PlasmaGlucose) AS STRING))), 10) < 8
    """
    
    test_query = f"""
    SELECT Pregnancies, PlasmaGlucose, DiastolicBloodPressure, TricepsThickness,
           SerumInsulin, BMI, DiabetesPedigree, Age, Diabetic
    FROM `{project_id}.{bq_dataset}.{bq_view}`
    WHERE MOD(ABS(FARM_FINGERPRINT(CAST(CONCAT(Pregnancies, PlasmaGlucose) AS STRING))), 10) >= 8
    """
    
    bq_train_task = bigquery_query_job_op(
        project=project_id, 
        location=region, 
        query=train_query
    )
    
    bq_test_task = bigquery_query_job_op(
        project=project_id, 
        location=region, 
        query=test_query
    )
    
    # TODO: Uncomment and complete the train_task once you implement train_model_op
    """
    train_task = train_model_op(
        train_data=bq_train_task.outputs["destination_table"],
        reg_rate=reg_rate,
        project_id=project_id,
        bq_location=region
    ).set_cpu_limit("1").set_memory_limit("3840Mi")
    train_task.after(bq_train_task)
    """
    
    # TODO: Uncomment the evaluation task once train_task is implemented
    """
    eval_task = evaluate_model_op(
        test_data=bq_test_task.outputs["destination_table"],
        model=train_task.outputs["output_model"],
        min_accuracy=min_accuracy,
        project_id=project_id,
        bq_location=region
    ).set_cpu_limit("1").set_memory_limit("3840Mi")
    eval_task.after(train_task)
    
    with dsl.If(eval_task.outputs["Output"] >= min_accuracy, name="pass-accuracy-threshold"):
        approved_task = model_approved_op(
            model_accuracy=eval_task.outputs["Output"],
            model_name=model_display_name
        )
        approved_task.after(eval_task)
        
        register_task = register_model_op(
            project_id=project_id,
            region=region,
            model_display_name=model_display_name,
            model_artifact=train_task.outputs["output_model"],
            parent_model=parent_model
        )
        register_task.after(approved_task)

    with dsl.If(eval_task.outputs["Output"] < min_accuracy, name="fail-accuracy-threshold"):
        rejected_task = model_rejected_op(
            model_accuracy=eval_task.outputs["Output"],
            min_accuracy=min_accuracy
        )
        rejected_task.after(eval_task)
    """

# =============================================================================
# (optional - for advanced users)
# =============================================================================
if __name__ == "__main__":
