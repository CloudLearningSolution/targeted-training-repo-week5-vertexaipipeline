"""
Advanced Hands-On Exercise: BigQuery Integration for Vertex AI Pipeline
======================================================================

TODO: Complete all missing BigQuery-related components and configurations
TODO: Integrate data loading, querying, and artifact handling
TODO: Ensure proper pipeline task connections and data flow
TODO: Reference train.py and train_to_vertex_ai_conversion.py for patterns

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
 

PIPELINE_NAME = "diabetes-classification-bigquery-exercise"
BASE_IMAGE = "python:3.9"

@component(
    base_image=BASE_IMAGE,
    packages_to_install=[
        "google-cloud-bigquery",
        "scikit-learn", 
        "joblib",
        "pandas"
    ]
)
def train_model_op(
    train_data: Input[artifact_types.BQTable],
    output_model: Output[Model],
    metrics: Output[Metrics],
    reg_rate: float,
    project_id: str,
    bq_location: str
) -> float:
    import re, os, shutil, joblib, logging
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from google.cloud import bigquery

    logging.basicConfig(level=logging.INFO)
    logging.info("[CONVERSION] Starting model training component")

    
    uri = train_data.uri
    match = re.search(r'projects/([^/]+)/datasets/([^/]+)/tables/([^/]+)', uri)
    if not match:
        raise ValueError(f"Could not parse BQ table from URI: {uri}")
    proj, dataset, table = match.groups()
    table_ref = f"{proj}.{dataset}.{table}"
    
    ##ADDED BQCOMPONENT
    bq_client = bigquery.Client(project=project_id, location=bq_location)
    query = f"SELECT * FROM `{table_ref}`"
    train_df = bq_client.query(query).to_dataframe()
    logging.info("[CONVERSION] Loaded %d training rows from BigQuery", len(train_df))

    
    FEATURE_COLUMNS = ["Pregnancies","PlasmaGlucose","DiastolicBloodPressure",
                       "TricepsThickness","SerumInsulin","BMI","DiabetesPedigree","Age"]
    X = train_df[FEATURE_COLUMNS]
    y = train_df["Diabetic"]
    
    model = LogisticRegression(C=1 / reg_rate, solver="liblinear")
    model.fit(X, y)
    
    training_accuracy = model.score(X, y)
    
    model_path = os.path.join(os.path.dirname(output_model.path), "model.joblib")
    joblib.dump(model, model_path)
    shutil.copy(model_path, output_model.path)
    
    metrics.log_metric("training_accuracy", training_accuracy)
    metrics.log_metric("regularization_rate", reg_rate)
    metrics.log_metric("training_samples", len(train_df))
    
    return training_accuracy

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

    # Added Load test data from BigQuery
    bq_client = bigquery.Client(project=project_id, location=bq_location)
    query = f"SELECT * FROM `{table_ref}`"
    test_df = bq_client.query(query).to_dataframe()
    logging.info("[CONVERSION] Loaded %d test rows from BigQuery", len(test_df))

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

@component(base_image=BASE_IMAGE)
def model_approved_op(model_accuracy: float, model_name: str):
    import logging
    logging.basicConfig(level=logging.INFO)
    logging.info("Model approved with accuracy: %.4f", model_accuracy)

@component(base_image=BASE_IMAGE)
def model_rejected_op(model_accuracy: float, min_accuracy: float):
    import logging
    logging.basicConfig(level=logging.ERROR)
    logging.error("Model rejected. Accuracy %.4f < %.2f", model_accuracy, min_accuracy)

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

@dsl.pipeline(
    name=PIPELINE_NAME,
    description="BigQuery integration exercise pipeline"
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
    
    train_task = train_model_op(
        train_data=,
        reg_rate=reg_rate,
        project_id=project_id,
        bq_location=region
    ).set_cpu_limit("1").set_memory_limit("3840Mi")
    train_task.after(bq_train_task)
    
    eval_task = evaluate_model_op(
        test_data=,
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

if __name__ == "__main__":
    from kfp import compiler
    
    compiler.Compiler().compile(
        pipeline_func=diabetes_training_pipeline,
        package_path="diabetes_bigquery_exercise.json"
    )
    print("Pipeline compiled successfully!")
