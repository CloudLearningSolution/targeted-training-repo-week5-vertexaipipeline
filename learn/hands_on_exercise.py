"""
Ready?

Review train.py and train_to_vertex_ai_conversion.py files for patterns.

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

PIPELINE_NAME = "diabetes-classification-exercise-pipeline"
PIPELINE_DESCRIPTION = "Conversion pipeline from train.py to Vertex AI with BigQuery integration"

BASE_IMAGE = "python:3.9"

# Load pre-built component for data pull
bigquery_query_job_op = components.load_component_from_url(
    'https://us-kfp.pkg.dev/ml-pipeline/google-cloud-registry/'
    'bigquery-query-job/sha256:'
    'd1cae80bc0de4e5b95b994739c8d0d7d42ce5a4cb17d3c9512eaed14540f6343'
)

# Model training
@component(
    base_image=BASE_IMAGE,
    packages_to_install=[
        'google-cloud-bigquery',
        'scikit-learn',
        'joblib',
        'pandas',
    ]
)
def train_model_op(
    train_data: Input[artifact_types.BQTable],
    output_model: Output[Model],
    metrics: Output[Metrics],
    reg_rate: float,
    project_id: str,
    bq_location: str,
) -> float:

    import re, os, shutil, joblib, logging
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from google.cloud import bigquery

    logging.basicConfig(level=logging.INFO)
    logging.info("[CONVERSION] Starting model training component")

    # Get train data table reference
    uri = train_data.uri
    logging.info("[CONVERSION] Parsing BQ URI: %s", uri)
    match = re.search(r'projects/([^/]+)/datasets/([^/]+)/tables/([^/]+)', uri)
    if not match:
        raise ValueError(f"Could not parse BQ table from URI: {uri}")
    proj, dataset, table = match.groups()
    table_ref = f"{proj}.{dataset}.{table}"

    # Pull training data from table_ref
    bq_client = bigquery.Client(project=project_id, location=bq_location)
    query = f"SELECT * FROM `{table_ref}`"
    train_df = bq_client.query(query).to_dataframe()
    logging.info("[CONVERSION] Loaded %d training rows from BigQuery", len(train_df))

    # Get input and target
    FEATURE_COLUMNS = ["Pregnancies","PlasmaGlucose","DiastolicBloodPressure",
                       "TricepsThickness","SerumInsulin","BMI","DiabetesPedigree","Age"]
    X = train_df[FEATURE_COLUMNS]
    y = train_df["Diabetic"]

    # Fit a model
    model = LogisticRegression(C=1 / reg_rate, solver="liblinear")
    model.fit(X, y)

    # Get training accuracy
    training_accuracy = model.score(X, y)
    logging.info("[CONVERSION] Training accuracy: %.4f", training_accuracy)

    # Save model
    model_path = os.path.join(os.path.dirname(output_model.path), "model.joblib")
    joblib.dump(model, model_path)
    shutil.copy(model_path, output_model.path)  # Move model to the detination

    # Log the metrics
    metrics.log_metric("training_accuracy", training_accuracy)
    metrics.log_metric("regularization_rate", reg_rate)
    metrics.log_metric("training_samples", len(train_df))

    logging.info("[CONVERSION] Model stored at %s", output_model.path)

    return training_accuracy

# Model evaluation
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
    from sklearn.metrics import accuracy_score, classification_report
    from google.cloud import bigquery

    logging.basicConfig(level=logging.INFO)
    logging.info("[CONVERSION] Starting model evaluation component")

    # Get input data table ref
    uri = test_data.uri
    match = re.search(r'projects/([^/]+)/datasets/([^/]+)/tables/([^/]+)', uri)
    if not match:
        raise ValueError(f"Could not parse BQ table from URI: {uri}")
    proj, dataset, table = match.groups()
    table_ref = f"{proj}.{dataset}.{table}"

    # Pull test data
    bq_client = bigquery.Client(project=project_id, location=bq_location)
    query = f"SELECT * FROM `{table_ref}`"
    test_df = bq_client.query(query).to_dataframe()
    logging.info("[CONVERSION] Loaded %d test rows from BigQuery", len(test_df))

    # Do model predictions
    model_obj = joblib.load(model.path)
    FEATURE_COLUMNS = ["Pregnancies","PlasmaGlucose","DiastolicBloodPressure",
                       "TricepsThickness","SerumInsulin","BMI","DiabetesPedigree","Age"]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["Diabetic"]
    preds = model_obj.predict(X_test)
    accuracy = accuracy_score(y_test, preds)

    # Log the metrics
    metrics.log_metric("accuracy", accuracy)
    metrics.log_metric("min_accuracy_threshold", min_accuracy)
    metrics.log_metric("test_samples", len(test_df))
    metrics.log_metric("passes_threshold", float(accuracy >= min_accuracy))

    report = classification_report(y_test, preds, output_dict=True)
    metrics.log_metric("precision_class_0", report['0']['precision'])
    metrics.log_metric("recall_class_0", report['0']['recall'])
    metrics.log_metric("precision_class_1", report['1']['precision'])
    metrics.log_metric("recall_class_1", report['1']['recall'])

    logging.info("[CONVERSION] Accuracy = %.4f", accuracy)
    logging.info("[CONVERSION] Threshold check: %.4f >= %.2f = %s",
                 accuracy, min_accuracy, accuracy >= min_accuracy)

    return accuracy


# Model approval logic
@component(base_image=BASE_IMAGE)
def model_approved_op(
    model_accuracy: float,
    model_name: str
):

    import logging

    logging.basicConfig(level=logging.INFO)
    logging.info("[CONVERSION] ✅ Model '%s' approved with accuracy: %.4f",
                 model_name, model_accuracy)
    logging.info("[CONVERSION] Model ready for registration and deployment")


# Model rejection logic
@component(
    base_image=BASE_IMAGE
)
def model_rejected_op(
    model_accuracy: float,
    min_accuracy: float
):

    import logging
    logging.basicConfig(level=logging.ERROR)
    logging.error("[CONVERSION] ❌ Model rejected. Accuracy %.4f < %.2f",
                  model_accuracy, min_accuracy)
    logging.error("[CONVERSION] Model will not be registered. Review training approach.")


# Model rejistry
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
    logging.info("[CONVERSION] Starting model registration")

    # Initialize Vertex AI client
    aiplatform.init(project=project_id, location=region)

    # Get model registry path
    artifact_dir = model_artifact.uri.rsplit("/", 1)[0]

    # Logs to upload to vertex ai
    upload_args = {
        "display_name": model_display_name,
        "artifact_uri": artifact_dir,
        "serving_container_image_uri": "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-2:latest",
        "labels": {
            "converted_from": "train_py",
            "framework": "scikit_learn",
            "use_case": "diabetes_classification"
        },
        "sync": True
    }
    
    if parent_model:
        upload_args["parent_model"] = parent_model

    # Upload model to model registry
    model = aiplatform.Model.upload(**upload_args)
    logging.info("[CONVERSION] Model registered: %s", model.resource_name)



@dsl.pipeline(
    name=PIPELINE_NAME,
    description=PIPELINE_DESCRIPTION
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
        train_data=bq_train_task.outputs["destination_table"],
        reg_rate=reg_rate,
        project_id=project_id,
        bq_location=region
    ).set_cpu_limit("1").set_memory_limit("3840Mi")
    train_task.after(bq_train_task)
    
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

if __name__ == "__main__":

    from kfp.v2 import compiler
    from google.cloud import aiplatform
    import argparse

    parser = argparse.ArgumentParser(description="Compile and run diabetes classification pipeline")
    parser.add_argument("--project-id", required=True, help="Google Cloud project ID")
    parser.add_argument("--region", default="us-central1", help="Google Cloud region")
    parser.add_argument("--pipeline-root", required=True, help="GCS bucket for pipeline artifacts")
    parser.add_argument("--dataset-location", required=True, help="BigQuery table reference (dataset.table)")
    parser.add_argument("--compile-only", action="store_true", help="Only compile pipeline, don't run")

    args = parser.parse_args()

    pipeline_spec_path = "diabetes_pipeline.json"
    compiler.Compiler().compile(
        pipeline_func=diabetes_training_pipeline,
        package_path=pipeline_spec_path
    )
    print(f"Pipeline compiled to: {pipeline_spec_path}")

    if not args.compile_only:

        # Initialize Vertex AI and submit pipeline
        aiplatform.init(project=args.project_id, location=args.region)

        job = aiplatform.PipelineJob(
            display_name="diabetes-classification-training",
            template_path=pipeline_spec_path,
            pipeline_root=args.pipeline_root,
            parameter_values={
                "project_id": args.project_id,
                "region": args.region,
                "model_display_name": "diabetes-classification-model",
                "dataset_location": args.dataset_location,  # BigQuery table reference
                "reg_rate": 0.01,  # Same default as original train.py
                "min_accuracy_threshold": 0.70
            }
        )

        print("Submitting pipeline to Vertex AI...")
        job.submit()
        print(f"Pipeline job submitted. Monitor at: {job._dashboard_uri()}")
        print(f"Using BigQuery table: {args.dataset_location}")
        print("Note: CSV data has been migrated to BigQuery for better performance and scalability")
