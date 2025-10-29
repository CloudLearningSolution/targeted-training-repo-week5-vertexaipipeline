"""
Vertex AI KFP Pipeline - Train.py to Vertex AI Conversion
=========================================================

This pipeline demonstrates the complete conversion from a standalone ML training 
script (train.py) to a production-ready Vertex AI Kubeflow Pipeline. It showcases:
- Migration from local CSV files to BigQuery tables
- Transformation from sequential script execution to distributed pipeline components
- Evolution from MLflow to Vertex AI Model Registry
- Implementation of automated quality gates and conditional model registration

All comments and documentation lines are kept <= 100 characters for .flake8.

===============================================================================
Lab 5.7: General Machine Learning train.py to Vertex AI Pipeline Conversion Workshop
===============================================================================
This conversion pipeline demonstrates the transformation patterns from train.py:
- Walk through EACH LINE of train.py and see its Vertex AI equivalent
- Understand HIGH-LEVEL conversion patterns before diving into details
- Explore how standalone script structure becomes distributed pipeline architecture
- See how local execution transforms into cloud-native orchestration

WORKSHOP OBJECTIVE: Line-by-line exploration to understand the overall conversion approach.
Attendees should identify each train.py code line and find its pipeline equivalent.

TODO: Lab 5.7.1 - Line-by-Line Import Exploration: Find each train.py import and its pipeline equivalent
TODO: Lab 5.7.2 - High-Level Architecture: Compare script execution flow vs pipeline DAG structure
TODO: Lab 5.7.3 - Parameter Handling Evolution: Explore argparse lines → pipeline parameter declarations

===============================================================================
Lab 5.8: Component Mapping and Functionality Translation
===============================================================================
This section demonstrates FUNCTION-BY-FUNCTION conversion from train.py:

WORKSHOP OBJECTIVE: Component-by-component deep dive into train.py functions.
Attendees should map each train.py function to its corresponding pipeline component
and understand the line-by-line translation process.

CONVERSION MAPPING PROCESS:
1. Identify train.py function (e.g., get_csvs_df, split_data, train_model)
2. Find the equivalent Vertex AI component
3. Map function parameters to component inputs/outputs
4. Understand data flow transformation (files → artifacts)
5. Learn error handling and logging evolution

TODO: Lab 5.8.1 - Function Mapping: Map train.py functions to pipeline components
TODO: Lab 5.8.2 - Data Flow Translation: Understand CSV DataFrames → BigQuery artifacts
TODO: Lab 5.8.3 - Parameter Evolution: Function args → component parameters

===============================================================================
"""

# ==============================================================================
# IMPORT SECTION CONVERSION
# ==============================================================================
# TODO: Lab 5.7.1 - Line-by-Line Import Exploration: Find each train.py import and its pipeline equivalent
# INSTRUCTIONS: Look at each import below and find where it maps to in the Vertex AI imports
# EXPLORE: Why some imports disappear, transform, or get replaced by cloud services

# TODO: Lab 5.7.1.1 - ANSWER: Original train.py imports (WHAT: These are the standalone script imports)
# WHY: Train.py needed these for local file handling, argument parsing, and MLflow tracking
# Original train.py imports:
# import argparse          # TODO: Lab 5.7.1.1a - WHERE: Used for --training_data and --reg_rate parameters
# import glob              # TODO: Lab 5.7.1.1b - WHERE: Used in get_csvs_df() for finding CSV files  
# import os                # TODO: Lab 5.7.1.1c - WHERE: Used in get_csvs_df() for path validation
# import pandas as pd      # TODO: Lab 5.7.1.1d - WHERE: Used for DataFrame operations and CSV reading
# from sklearn.linear_model import LogisticRegression  # TODO: Lab 5.7.1.1e - WHERE: Used in train_model() function
# from sklearn.model_selection import train_test_split # TODO: Lab 5.7.1.1f - WHERE: Used in split_data() function
# import mlflow            # TODO: Lab 5.7.1.1g - WHERE: Used in main() for experiment tracking
# import mlflow.sklearn    # TODO: Lab 5.7.1.1h - WHERE: Used in main() for model registration
#
# TODO: Lab 5.7.1.2 - ANSWER: Vertex AI Kubeflow Pipeline imports (WHAT: These are the cloud-native equivalents)
# WHY: Pipeline needs orchestration framework, cloud artifacts, and distributed execution
from kfp import dsl, components  # TODO: Lab 5.7.1.2a - WHERE: Replaces script execution with pipeline orchestration
from kfp.dsl import (            # TODO: Lab 5.7.1.2b - WHERE: Replaces function definitions with component decorators
    component,                   # TODO: Lab 5.7.1.2c - WHAT: @component decorator replaces def functions
    pipeline,                    # TODO: Lab 5.7.1.2d - WHAT: @pipeline decorator replaces if __name__ == "__main__"
    Input,                       # TODO: Lab 5.7.1.2e - WHAT: Input[Model] replaces function parameters
    Output,                      # TODO: Lab 5.7.1.2f - WHAT: Output[Model] replaces return statements
    Model,                       # TODO: Lab 5.7.1.2g - WHAT: Cloud artifact replaces Python objects
    Metrics                      # TODO: Lab 5.7.1.2h - WHAT: Structured metrics replace print() statements
)
# TODO: Lab 5.7.1.2i - ANSWER: BigQuery Integration (WHAT: Replaces CSV file handling)
# WHY: BigQuery provides scalable data access vs local file system limitations
from google_cloud_pipeline_components.types import artifact_types  # TODO: Lab 5.7.1.2j - WHERE: BQTable replaces glob + pandas

# TODO: Lab 5.7.2 - High-Level Architecture Exploration: How script metadata becomes pipeline configuration
# INSTRUCTIONS: Compare how train.py script name and execution becomes pipeline definition

# TODO: Lab 5.7.2.1 - ANSWER: Pipeline Metadata (WHAT: Pipeline identification and description)
# WHERE: Replaces the script filename and purpose
# WHY: Pipelines need explicit names for tracking and management in cloud environments
PIPELINE_NAME = "diabetes-classification-conversion-pipeline"  # TODO: Lab 5.7.2.1a - WHAT: Replaces "train.py" filename
PIPELINE_DESCRIPTION = (  # TODO: Lab 5.7.2.1b - WHAT: Explicit description vs inferring from script name
    "Conversion pipeline from train.py to Vertex AI with BigQuery integration"
)

# TODO: Lab 5.7.2.2 - ANSWER: Execution Environment (WHAT: Container-based execution vs local Python)  
# WHERE: Replaces local Python environment with containerized execution
# WHY: Distributed pipeline needs consistent, reproducible execution environments
BASE_IMAGE = "python:3.9"                # TODO: Lab 5.7.2.2a - WHAT: Container image vs local Python installation
REQUIREMENTS_PATH = "src/requirements.txt" # TODO: Lab 5.7.2.2b - WHAT: Explicit dependencies vs local pip installs

# ==============================================================================
# DATA LOADING SECTION CONVERSION
# ==============================================================================
# TODO: Lab 5.8.1 - Function Mapping Exploration: How get_csvs_df() becomes BigQuery component
# INSTRUCTIONS: Compare the train.py get_csvs_df function with the BigQuery component line by line

# TODO: Lab 5.8.1.1 - ANSWER: Original train.py data loading function (WHAT: Local CSV file handling)
# WHERE: This function handles local file system operations and DataFrame creation
# WHY: Train.py needed to read multiple CSV files from a local directory
# Original train.py data loading function:
# def get_csvs_df(path):                                          # TODO: Lab 5.8.1.1a - WHERE: Function takes file path parameter
#     # Debugging: Print the path before checking existence        
#     print(f"DEBUG: Checking existence of path -> {path}")       # TODO: Lab 5.8.1.1b - WHERE: Debug logging with print()
#     if not os.path.exists(path):                                # TODO: Lab 5.8.1.1c - WHERE: Local file system validation
#         raise RuntimeError(f"Cannot use non-existent path provided: {path}")
#     csv_files = glob.glob(f"{path}/*.csv")                      # TODO: Lab 5.8.1.1d - WHERE: File pattern matching for CSV files
#     if not csv_files:                                           # TODO: Lab 5.8.1.1e - WHERE: Validation that CSV files exist
#         raise RuntimeError(f"No CSV files found in provided data path: {path}")
#     return pd.concat((pd.read_csv(f) for f in csv_files), sort=False)  # TODO: Lab 5.8.1.1f - WHERE: Pandas DataFrame concatenation
#
# TODO: Lab 5.8.1.2 - ANSWER: Vertex AI BigQuery Component (WHAT: Cloud-native data access)
# WHERE: Pre-built component replaces custom function with enterprise data access
# WHY: BigQuery provides scalable, managed data access vs local file limitations
# PRE-BUILT COMPONENT: BigQuery Query Job (Replaces CSV Loading)
# ==============================================================================
bigquery_query_job_op = components.load_component_from_url(  # TODO: Lab 5.8.1.2a - WHERE: Pre-built component vs custom function
    'https://us-kfp.pkg.dev/ml-pipeline/google-cloud-registry/'  # TODO: Lab 5.8.1.2b - WHAT: Google-maintained component registry
    'bigquery-query-job/sha256:'                                 # TODO: Lab 5.8.1.2c - WHY: Version-controlled, tested components
    'd1cae80bc0de4e5b95b994739c8d0d7d42ce5a4cb17d3c9512eaed14540f6343'  # TODO: Lab 5.8.1.2d - WHERE: Specific component version hash
)

# TODO: Lab 5.8.1.3 - COMPARISON SUMMARY: Function vs Component transformation
# WHAT CHANGED: get_csvs_df(path) → bigquery_query_job_op(project, location, query)
# WHERE: Local file access → Cloud data warehouse access  
# WHY: Scalability, performance, enterprise data governance vs local file limitations

# ==============================================================================
# TRAINING FUNCTION CONVERSION
# ==============================================================================
# TODO: Lab 5.8.1 - Function Mapping Deep Dive: How train_model() becomes train_model_op component
# INSTRUCTIONS: Compare the train.py train_model function with the train_model_op component function signature and body

# TODO: Lab 5.8.1.1 - ANSWER: Original train.py train_model function (WHAT: Core ML training logic)
# WHERE: This function contains the main training algorithm and evaluation
# WHY: Encapsulates LogisticRegression training with accuracy calculation
# Original train.py train_model function:
# def train_model(reg_rate, X_train, X_test, y_train, y_test):    # TODO: Lab 5.8.1.1a - WHERE: Function signature with direct data parameters
#     # Train model                                               
#     model = LogisticRegression(C=1 / reg_rate, solver="liblinear")  # TODO: Lab 5.8.1.1b - WHERE: Sklearn model initialization
#     model.fit(X_train, y_train)                                # TODO: Lab 5.8.1.1c - WHERE: Model training on DataFrames
#     # Evaluate model                                           
#     accuracy = model.score(X_test, y_test)                     # TODO: Lab 5.8.1.1d - WHERE: Model evaluation on test data
#     print(f"Model accuracy: {accuracy}")                       # TODO: Lab 5.8.1.1e - WHERE: Simple print statement for logging
#     return model                                                # TODO: Lab 5.8.1.1f - WHERE: Return Python object directly
#
# TODO: Lab 5.8.1.2 - ANSWER: Component Transformation (WHAT: Function becomes distributed component)
# WHERE: @component decorator and cloud-native inputs/outputs
# WHY: Enable distributed execution with artifact management
# Component: train_model_op (Converted from train.py train_model function)
# ==============================================================================
@component(                                                      # TODO: Lab 5.8.1.2a - WHERE: @component decorator replaces def
    base_image=BASE_IMAGE,                                       # TODO: Lab 5.8.1.2b - WHAT: Container execution vs local Python
    packages_to_install=[                                        # TODO: Lab 5.8.1.2c - WHERE: Explicit dependencies vs local imports
        "google-cloud-bigquery",                                 # TODO: Lab 5.8.1.2d - WHAT: Cloud data access vs pandas
        "scikit-learn",                                          # TODO: Lab 5.8.1.2e - WHERE: Same sklearn but containerized
        "joblib",                                                # TODO: Lab 5.8.1.2f - WHAT: Model serialization for artifacts
        "pandas"                                                 # TODO: Lab 5.8.1.2g - WHERE: DataFrame operations still needed
    ]
)
def train_model_op(                                              # TODO: Lab 5.8.1.2h - WHERE: Component function signature
    train_data: Input[artifact_types.BQTable],                   # TODO: Lab 5.8.1.2i - WHAT: BQTable artifact vs X_train DataFrame
    output_model: Output[Model],                                 # TODO: Lab 5.8.1.2j - WHAT: Output[Model] vs return statement
    metrics: Output[Metrics],                                    # TODO: Lab 5.8.1.2k - WHAT: Structured metrics vs print()
    reg_rate: float,                                             # TODO: Lab 5.8.1.2l - WHERE: Same parameter, different data flow
    project_id: str,                                             # TODO: Lab 5.8.1.2m - WHAT: Cloud context vs local execution
    bq_location: str                                             # TODO: Lab 5.8.1.2n - WHAT: Regional data access parameter
) -> float:                                                      # TODO: Lab 5.8.1.2o - WHERE: Return type for pipeline decisions
    """
    Train logistic regression model using BigQuery training data.
    Converted from train_model function in original train.py.
    
    Args:
        train_data: BigQuery table containing training data
        output_model: Output model artifact for pipeline consumption
        metrics: Training metrics for monitoring and evaluation
        reg_rate: Regularization rate (inverse of C parameter)
        project_id: Google Cloud project ID
        bq_location: BigQuery location/region
        
    Returns:
        float: Training accuracy for pipeline decision making
    """
    import re, os, shutil, joblib, logging
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from google.cloud import bigquery

    logging.basicConfig(level=logging.INFO)
    logging.info("[CONVERSION] Starting model training component")
    
    # TODO: Lab 5.8.2 - Data Source Translation: DataFrame input → BigQuery table parsing
    # EXPLORE: How train.py receives DataFrames vs pipeline components receive artifact URIs
    # TRANSLATE: Direct DataFrame access → URI parsing + BigQuery client
    uri = train_data.uri
    logging.info("[CONVERSION] Parsing BQ URI: %s", uri)
    match = re.search(r'projects/([^/]+)/datasets/([^/]+)/tables/([^/]+)', uri)
    if not match:
        raise ValueError(f"Could not parse BQ table from URI: {uri}")
    proj, dataset, table = match.groups()
    table_ref = f"{proj}.{dataset}.{table}"
    
    # TODO: Lab 5.8.3 - Data Loading Evolution: pandas.read_csv() → BigQuery client
    # EXPLORE: train.py loads data from memory vs component loads from BigQuery
    # UNDERSTAND: Same end result (DataFrame) but different data source
    bq_client = bigquery.Client(project=project_id, location=bq_location)
    query = f"SELECT * FROM `{table_ref}`"
    train_df = bq_client.query(query).to_dataframe()
    logging.info("[CONVERSION] Loaded %d training rows from BigQuery", len(train_df))
    
    # TODO: Lab 5.8.4 - Algorithm Consistency Exploration: Same sklearn code in both versions
    # INSTRUCTIONS: Find these exact same lines in train.py train_model function
    # WHAT STAYS THE SAME: Core ML algorithm logic remains identical
    
    # TODO: Lab 5.8.4.1 - ANSWER: Feature Engineering Consistency (WHAT: Same feature selection)
    # WHERE: Find these exact column names in train.py split_data function
    # WHY: Feature consistency ensures model behavior remains the same
    FEATURE_COLUMNS = ["Pregnancies","PlasmaGlucose","DiastolicBloodPressure",     # TODO: Lab 5.8.4.1a - WHERE: Same features as train.py
                       "TricepsThickness","SerumInsulin","BMI","DiabetesPedigree","Age"]  # TODO: Lab 5.8.4.1b - WHAT: Identical feature list
    X = train_df[FEATURE_COLUMNS]                                                  # TODO: Lab 5.8.4.1c - WHERE: Same DataFrame indexing as train.py
    y = train_df["Diabetic"]                                                       # TODO: Lab 5.8.4.1d - WHERE: Same target column as train.py
    
    # TODO: Lab 5.8.5 - Line-by-Line Algorithm Mapping: Identical sklearn training code
    # INSTRUCTIONS: Find these EXACT lines in train.py train_model function
    # WHAT'S IDENTICAL: Model initialization and training calls are 100% the same
    
    # TODO: Lab 5.8.5.1 - ANSWER: Find these exact lines in train.py train_model function:
    # model = LogisticRegression(C=1 / reg_rate, solver="liblinear")              # TODO: Lab 5.8.5.1a - WHERE: Exact same model initialization  
    # model.fit(X_train, y_train)                                                 # TODO: Lab 5.8.5.1b - WHERE: Exact same fit() call
    model = LogisticRegression(C=1 / reg_rate, solver="liblinear")                # TODO: Lab 5.8.5.1c - WHAT: Identical algorithm parameters
    model.fit(X, y)                                                               # TODO: Lab 5.8.5.1d - WHAT: Same training call (X,y vs X_train,y_train)
    
    # TODO: Lab 5.8.5.2 - ANSWER: Accuracy Calculation (WHAT: Same evaluation method)  
    # WHERE: Find model.score() call in train.py train_model function
    # WHY: Consistent evaluation ensures comparable results
    training_accuracy = model.score(X, y)                                         # TODO: Lab 5.8.5.2a - WHERE: Same score() method as train.py
    logging.info("[CONVERSION] Training accuracy: %.4f", training_accuracy)      # TODO: Lab 5.8.5.2b - WHAT: Enhanced logging vs simple print()
    
    # TODO: Lab 5.8.6 - Model Persistence Translation: return model → artifact serialization
    # INSTRUCTIONS: Compare how train.py returns model vs component saves artifact
    # WHAT CHANGES: Direct return vs cloud artifact storage
    
    # TODO: Lab 5.8.6.1 - ANSWER: Original train.py model return (WHAT: Direct Python object return)
    # WHERE: Find "return model" statement in train.py train_model function  
    # WHY: Functions return objects directly for immediate use
    
    # TODO: Lab 5.8.6.2 - ANSWER: Pipeline artifact storage (WHAT: Persistent cloud storage)
    # WHERE: joblib serialization and artifact path management
    # WHY: Distributed components need persistent, shareable model storage
    model_path = os.path.join(os.path.dirname(output_model.path), "model.joblib") # TODO: Lab 5.8.6.2a - WHAT: Artifact path generation
    joblib.dump(model, model_path)                                                # TODO: Lab 5.8.6.2b - WHAT: Model serialization for storage
    shutil.copy(model_path, output_model.path)                                    # TODO: Lab 5.8.6.2c - WHAT: Artifact path compliance
    
    # TODO: Lab 5.8.7 - Logging Evolution: print() statements → structured metrics
    # INSTRUCTIONS: Compare train.py print() vs pipeline metrics
    # WHAT IMPROVES: Simple console output vs structured, queryable metrics
    
    # TODO: Lab 5.8.7.1 - ANSWER: Original train.py logging (WHAT: Simple print statement)
    # WHERE: Find print(f"Model accuracy: {accuracy}") in train.py train_model function
    # WHY: Quick console output for immediate feedback
    
    # TODO: Lab 5.8.7.2 - ANSWER: Pipeline structured metrics (WHAT: Cloud-native observability)  
    # WHERE: metrics.log_metric() calls for dashboard and monitoring integration
    # WHY: Enterprise monitoring, alerting, and historical tracking capabilities
    metrics.log_metric("training_accuracy", training_accuracy)                    # TODO: Lab 5.8.7.2a - WHAT: Structured accuracy metric
    metrics.log_metric("regularization_rate", reg_rate)                           # TODO: Lab 5.8.7.2b - WHAT: Parameter tracking for experiments
    metrics.log_metric("training_samples", len(train_df))                         # TODO: Lab 5.8.7.2c - WHAT: Data volume tracking for monitoring
    
    logging.info("[CONVERSION] Model stored at %s", output_model.path)
    return training_accuracy

# ==============================================================================
# DATA SPLITTING AND EVALUATION CONVERSION  
# ==============================================================================
# Original train.py split_data function:
# def split_data(df):
#     """
#     Splits the dataset into training and testing sets.
#     Assumes the target column is named 'Diabetic'.
#     """
#     if 'Diabetic' not in df.columns:
#         raise RuntimeError("The dataset must contain a 'Diabetic' column.")
#     columns = [
#         'Pregnancies', 'PlasmaGlucose', 'DiastolicBloodPressure', 'TricepsThickness',
#         'SerumInsulin', 'BMI', 'DiabetesPedigree', 'Age'
#     ]
#     X = df[columns].values
#     y = df['Diabetic'].values
#     return train_test_split(X, y, test_size=0.2, random_state=42)
#
# Original train.py evaluation (embedded in train_model):
# accuracy = model.score(X_test, y_test)
# print(f"Model accuracy: {accuracy}")
#
# Component: evaluate_model_op (Extracted from train.py train_model function)
# ==============================================================================
# TODO: Lab 5.8.1 - Function Extraction Exploration: Evaluation logic separated from training
# EXPLORE: In train.py, evaluation is embedded in train_model function
# FIND: Lines like "accuracy = model.score(X_test, y_test)" in train.py
# UNDERSTAND: Why pipeline design separates training and evaluation into different components
# TRANSLATE: Embedded evaluation → standalone evaluation component
#
# Vertex AI component separation:
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
    """
    Evaluate trained model on BigQuery test dataset.
    Extracted and enhanced from train_model function evaluation logic.
    
    Args:
        test_data: BigQuery table containing test data
        model: Trained model artifact from training component
        metrics: Evaluation metrics for monitoring and decision making
        min_accuracy: Minimum accuracy threshold for quality gate
        project_id: Google Cloud project ID
        bq_location: BigQuery location/region
        
    Returns:
        float: Test accuracy for conditional pipeline logic
    """
    import re, logging, joblib
    import pandas as pd
    from sklearn.metrics import accuracy_score, classification_report
    from google.cloud import bigquery

    logging.basicConfig(level=logging.INFO)
    logging.info("[CONVERSION] Starting model evaluation component")
    
    # TODO: Lab 5.8.2 - Data Splitting Translation: split_data() → SQL-based splitting
    # EXPLORE: train.py uses train_test_split() function for splitting
    # COMPARE: Pipeline uses separate BigQuery queries for train/test data
    # UNDERSTAND: Why SQL-based splitting replaces sklearn splitting
    uri = test_data.uri
    match = re.search(r'projects/([^/]+)/datasets/([^/]+)/tables/([^/]+)', uri)
    if not match:
        raise ValueError(f"Could not parse BQ table from URI: {uri}")
    proj, dataset, table = match.groups()
    table_ref = f"{proj}.{dataset}.{table}"

    # Load test data from BigQuery
    bq_client = bigquery.Client(project=project_id, location=bq_location)
    query = f"SELECT * FROM `{table_ref}`"
    test_df = bq_client.query(query).to_dataframe()
    logging.info("[CONVERSION] Loaded %d test rows from BigQuery", len(test_df))

    # TODO: Lab 5.8.3 - Model Loading Translation: Direct object → artifact loading
    # EXPLORE: train.py passes model object directly to evaluation code
    # COMPARE: Pipeline loads model from artifact using joblib.load()
    # UNDERSTAND: How model persistence enables distributed execution
    model_obj = joblib.load(model.path)
    
    # TODO: Lab 5.8.4 - Feature Consistency Mapping: Same feature columns across components
    # FIND: Same FEATURE_COLUMNS list in train.py split_data function
    # UNDERSTAND: Consistency in feature engineering across pipeline components
    FEATURE_COLUMNS = ["Pregnancies","PlasmaGlucose","DiastolicBloodPressure",
                       "TricepsThickness","SerumInsulin","BMI","DiabetesPedigree","Age"]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["Diabetic"]
    
    # TODO: Lab 5.8.5 - Evaluation Logic Mapping: Find equivalent lines in train.py
    # FIND: model.score(X_test, y_test) in train.py train_model function
    # COMPARE: Enhanced evaluation with multiple metrics vs simple accuracy
    preds = model_obj.predict(X_test)
    accuracy = accuracy_score(y_test, preds)

    # TODO: Lab 5.8.6 - Metrics Enhancement: print() → structured logging
    # COMPARE: train.py print(f"Model accuracy: {accuracy}") vs comprehensive metrics
    # UNDERSTAND: Enhanced observability and monitoring capabilities
    metrics.log_metric("accuracy", accuracy)
    metrics.log_metric("min_accuracy_threshold", min_accuracy)
    metrics.log_metric("test_samples", len(test_df))
    metrics.log_metric("passes_threshold", float(accuracy >= min_accuracy))
    
    # Enhanced metrics beyond original train.py
    report = classification_report(y_test, preds, output_dict=True)
    metrics.log_metric("precision_class_0", report['0']['precision'])
    metrics.log_metric("recall_class_0", report['0']['recall'])
    metrics.log_metric("precision_class_1", report['1']['precision'])
    metrics.log_metric("recall_class_1", report['1']['recall'])
    
    logging.info("[CONVERSION] Accuracy = %.4f", accuracy)
    logging.info("[CONVERSION] Threshold check: %.4f >= %.2f = %s", 
                 accuracy, min_accuracy, accuracy >= min_accuracy)
    
    return accuracy

# ==============================================================================
# Component: model_approved_op (Enhancement not in original train.py)
# ==============================================================================
# TODO: Lab 5.9.1 - Pipeline Enhancement: Automated approval logic addition
# TODO: Lab 5.9.2 - Quality Gates: Structured approval vs always-register
@component(base_image=BASE_IMAGE)
def model_approved_op(model_accuracy: float, model_name: str):
    """
    Handle model approval notification.
    Enhancement not present in original train.py - demonstrates pipeline benefits.
    """
    import logging
    logging.basicConfig(level=logging.INFO)
    logging.info("[CONVERSION] ✅ Model '%s' approved with accuracy: %.4f", 
                 model_name, model_accuracy)
    logging.info("[CONVERSION] Model ready for registration and deployment")

# ==============================================================================
# MODEL REGISTRATION CONVERSION
# ==============================================================================
# Original train.py MLflow model registration:
# # Explicitly register the model in Azure ML's model registry
# run_id = run.info.run_id
# mlflow.register_model(f"runs:/{run_id}/model", "diabetes-classification-prod")
# print(f"Model registered from run {run_id}")
#
# Component: register_model_op (Replaces MLflow registration)
# ==============================================================================
# TODO: Lab 5.8.1 - MLflow to Vertex AI Registry Translation: Line-by-line mapping
# FIND: These lines in train.py main() function:
# run_id = run.info.run_id
# mlflow.register_model(f"runs:/{run_id}/model", "diabetes-classification-prod")
# print(f"Model registered from run {run_id}")
# TRANSLATE: MLflow registration → Vertex AI Model Registry
# UNDERSTAND: Enhanced metadata and cloud-native model management
#
# Vertex AI replacement:
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
    """
    Register model in Vertex AI Model Registry.
    Replaces MLflow model registration from original train.py.
    
    Args:
        project_id: Google Cloud project ID
        region: Google Cloud region
        model_display_name: Display name for registered model
        model_artifact: Trained model artifact to register
        parent_model: Optional parent model for versioning
    """
    from google.cloud import aiplatform
    import logging
    
    logging.basicConfig(level=logging.INFO)
    logging.info("[CONVERSION] Starting model registration")
    
    # Initialize Vertex AI client
    aiplatform.init(project=project_id, location=region)
    
    # TODO: Lab 5.8.2 - Registration Enhancement: Simple MLflow → Rich Vertex AI metadata
    # COMPARE: train.py simple model name vs enhanced model metadata
    # EXPLORE: Labels, serving containers, and versioning capabilities
    artifact_dir = model_artifact.uri.rsplit("/", 1)[0]
    
    # TODO: Lab 5.8.3 - Metadata Evolution: Enhanced model information vs basic registration
    # COMPARE: MLflow simple name vs Vertex AI rich metadata
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
        logging.info("[CONVERSION] Registering as version under parent: %s", parent_model)
    
    model = aiplatform.Model.upload(**upload_args)
    logging.info("[CONVERSION] Model registered: %s", model.resource_name)

# ==============================================================================
# Component: model_rejected_op (Enhancement not in original train.py)
# ==============================================================================
# TODO: Lab 5.9.1 - Quality Gates: Automated rejection vs manual intervention
@component(base_image=BASE_IMAGE)
def model_rejected_op(model_accuracy: float, min_accuracy: float):
    """
    Handle model rejection due to poor performance.
    Enhancement not present in original train.py - demonstrates automated quality gates.
    """
    import logging
    logging.basicConfig(level=logging.ERROR)
    logging.error("[CONVERSION] ❌ Model rejected. Accuracy %.4f < %.2f", 
                  model_accuracy, min_accuracy)
    logging.error("[CONVERSION] Model will not be registered. Review training approach.")

# ==============================================================================
# MAIN EXECUTION AND PARAMETER HANDLING CONVERSION
# ==============================================================================
# Original train.py argument parsing:
# def parse_args():
#     # Setup argument parser
#     parser = argparse.ArgumentParser()
#     # Add arguments
#     parser.add_argument("--training_data", dest='training_data',
#                         type=str, required=True)
#     parser.add_argument("--reg_rate", dest='reg_rate',
#                         type=float, default=0.01)
#     # Parse args
#     args = parser.parse_args()
#     # Return args
#     return args
#
# Original train.py main function:
# def main(args):
#     # Enable autologging
#     mlflow.autolog()
#     # Start an MLflow run
#     with mlflow.start_run() as run:
#         # Debugging: Print the training data path before loading
#         print(f"DEBUG: training_data path received -> {args.training_data}")
#         # Read data
#         df = get_csvs_df(args.training_data)
#         # Split data
#         X_train, X_test, y_train, y_test = split_data(df)
#         # Train model and get model object
#         model = train_model(args.reg_rate, X_train, X_test, y_train, y_test)
#         # Explicitly register the model in Azure ML's model registry
#         run_id = run.info.run_id
#         mlflow.register_model(f"runs:/{run_id}/model", "diabetes-classification-prod")
#         print(f"Model registered from run {run_id}")
#
# Original train.py execution:
# if __name__ == "__main__":
#     # Add space in logs
#     print("\n\n")
#     print("*" * 60)
#     # Parse args
#     args = parse_args()
#     # Run main function
#     main(args)
#     # Add space in logs
#     print("*" * 60)
#     print("\n\n")
#
# MAIN PIPELINE DEFINITION
# ==============================================================================
# TODO: Lab 5.7.1 - Workflow Structure Exploration: main() function → @dsl.pipeline
# EXPLORE: How train.py main() function becomes pipeline definition
# COMPARE: Sequential function calls vs DAG task dependencies
# UNDERSTAND: Parameter flow from argparse → pipeline parameters

# TODO: Lab 5.7.2 - Execution Model Translation: if __name__ == "__main__" → pipeline orchestration
# FIND: train.py execution block starting with if __name__ == "__main__"
# COMPARE: Direct script execution vs distributed pipeline execution
# UNDERSTAND: How single-process script becomes multi-container pipeline
#
# Vertex AI pipeline orchestration:
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
    reg_rate: float = 0.01,  # Same default as original train.py
    min_accuracy: float = 0.70,
    parent_model: str = ""
):
    """
    Main pipeline function orchestrating diabetes classification workflow.
    Converted from sequential train.py execution to distributed pipeline.
    
    Args:
        project_id: Google Cloud project ID
        region: Google Cloud region for pipeline execution
        model_display_name: Name for registered model in Vertex AI Model Registry
        bq_dataset: BigQuery dataset containing diabetes data
        bq_view: BigQuery view with diabetes features (migrated from CSV)
        reg_rate: Regularization rate for LogisticRegression (same as train.py)
        min_accuracy: Minimum accuracy threshold for model approval
        parent_model: Parent model for versioning (MLops enhancement)
    """
    
    # TODO: Lab 5.7.3 - Data Flow Exploration: train.py sequential calls → pipeline DAG
    # FIND: In train.py main(), these sequential calls:
    # df = get_csvs_df(args.training_data)
    # X_train, X_test, y_train, y_test = split_data(df)
    # model = train_model(args.reg_rate, X_train, X_test, y_train, y_test)
    # COMPARE: Sequential calls vs parallel BigQuery queries + dependent tasks
    # UNDERSTAND: How data flows through pipeline artifacts vs Python variables
    
    # Training data query (replaces get_csvs_df + train_test_split logic)
    train_query = f"""
    SELECT Pregnancies, PlasmaGlucose, DiastolicBloodPressure, TricepsThickness,
           SerumInsulin, BMI, DiabetesPedigree, Age, Diabetic
    FROM `{project_id}.{bq_dataset}.{bq_view}`
    WHERE MOD(ABS(FARM_FINGERPRINT(CAST(CONCAT(Pregnancies, PlasmaGlucose) AS STRING))), 10) < 8
    """
    
    # Test data query (equivalent to test split)
    test_query = f"""
    SELECT Pregnancies, PlasmaGlucose, DiastolicBloodPressure, TricepsThickness,
           SerumInsulin, BMI, DiabetesPedigree, Age, Diabetic
    FROM `{project_id}.{bq_dataset}.{bq_view}`
    WHERE MOD(ABS(FARM_FINGERPRINT(CAST(CONCAT(Pregnancies, PlasmaGlucose) AS STRING))), 10) >= 8
    """
    
    # TODO: Lab 5.7.4 - Task Creation Exploration: Function calls → component instantiation
    # COMPARE: get_csvs_df(args.training_data) vs bigquery_query_job_op(...)
    # UNDERSTAND: How function calls become distributed tasks
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
    
    # TODO: Lab 5.7.5 - Dependency Management: Sequential execution → explicit dependencies
    # FIND: train.py automatic sequential execution via variable passing
    # COMPARE: model = train_model(...) vs train_task.after(bq_train_task)
    # UNDERSTAND: Explicit task dependencies vs implicit execution order
    train_task = train_model_op(
        train_data=bq_train_task.outputs["destination_table"],
        reg_rate=reg_rate,
        project_id=project_id,
        bq_location=region
    ).set_cpu_limit("1").set_memory_limit("3840Mi")
    train_task.after(bq_train_task)
    
    # Model evaluation task (extracted from train_model function)
    eval_task = evaluate_model_op(
        test_data=bq_test_task.outputs["destination_table"],
        model=train_task.outputs["output_model"],
        min_accuracy=min_accuracy,
        project_id=project_id,
        bq_location=region
    ).set_cpu_limit("1").set_memory_limit("3840Mi")
    eval_task.after(train_task)
    
    # TODO: Lab 5.9.1 - Quality Gate Enhancement: Always-register → conditional logic
    # EXPLORE: train.py always registers model vs pipeline conditional registration
    # UNDERSTAND: Automated quality gates and pipeline control flow
    with dsl.If(eval_task.outputs["Output"] >= min_accuracy, name="pass-accuracy-threshold"):
        # Model approval notification
        approved_task = model_approved_op(
            model_accuracy=eval_task.outputs["Output"],
            model_name=model_display_name
        )
        approved_task.after(eval_task)
        
        # Model registration (replaces MLflow registration)
        register_task = register_model_op(
            project_id=project_id,
            region=region,
            model_display_name=model_display_name,
            model_artifact=train_task.outputs["output_model"],
            parent_model=parent_model
        )
        register_task.after(approved_task)

    with dsl.If(eval_task.outputs["Output"] < min_accuracy, name="fail-accuracy-threshold"):
        # Model rejection handling (enhancement over original)
        rejected_task = model_rejected_op(
            model_accuracy=eval_task.outputs["Output"],
            min_accuracy=min_accuracy
        )
        rejected_task.after(eval_task)

# ==============================================================================
# CONVERSION SUMMARY
# ==============================================================================
# TODO: Lab 5.9.1 - Conversion Review: Key transformation patterns summary
"""
CONVERSION SUMMARY - train.py → Vertex AI Pipeline:

1. EXECUTION MODEL TRANSFORMATION:
   Original: if __name__ == "__main__": main(args)
   Converted: @dsl.pipeline with distributed component execution

2. DATA LOADING EVOLUTION:
   Original: get_csvs_df() with glob.glob() and pd.concat()
   Converted: BigQuery Query Job component with SQL queries

3. DATA SPLITTING MODERNIZATION:
   Original: train_test_split(X, y, test_size=0.2, random_state=42)
   Converted: SQL FARM_FINGERPRINT for deterministic splitting

4. FUNCTION TO COMPONENT TRANSFORMATION:
   Original: def train_model() → return model
   Converted: @component def train_model_op() → Output[Model]

5. MODEL REGISTRY EVOLUTION:
   Original: mlflow.register_model()
   Converted: aiplatform.Model.upload() with enhanced metadata

6. PARAMETER HANDLING MODERNIZATION:
   Original: argparse with command-line arguments
   Converted: Pipeline parameters with type hints

7. ERROR HANDLING ENHANCEMENT:
   Original: Simple print statements and exceptions
   Converted: Structured logging and conditional pipeline logic

8. SCALABILITY IMPROVEMENT:
   Original: Single-machine limitations
   Converted: Distributed, auto-scaling cloud execution

9. MONITORING ADVANCEMENT:
   Original: Print statements and MLflow UI
   Converted: Vertex AI Metrics and Cloud Logging integration

10. QUALITY GATES ADDITION:
    Original: Always register model regardless of performance
    Converted: Conditional registration based on accuracy thresholds

Lab 5.7 demonstrates how traditional ML scripts evolve into production-ready 
MLOps pipelines with minimal algorithmic changes but significant architectural 
improvements for enterprise deployment.

Lab 5.8 showcases component mapping patterns that maintain functional equivalence 
while adding cloud-native scalability and automated quality assurance.

Lab 5.9 validates that the converted pipeline maintains the same machine learning 
logic while providing enhanced monitoring, error handling, and deployment automation.
"""

    # ----------------------------
    # Model Training Component (Using BigQuery Data)
    # ----------------------------
    # Original train.py model training:
    # def train_model(reg_rate, X_train, X_test, y_train, y_test):
    #     model = LogisticRegression(C=1 / reg_rate, solver="liblinear")
    #     model.fit(X_train, y_train)
    #     accuracy = model.score(X_test, y_test)
    #     print(f"Model accuracy: {accuracy}")
    #     return model
    #
    # Vertex AI component transformation (using BigQuery artifacts):
    @component(
        base_image="python:3.9",
        packages_to_install=["google-cloud-bigquery", "scikit-learn", "joblib", "pandas"]
    )
    def train_model_component(
        train_data: Input[artifact_types.BQTable],
        trained_model: Output[Model],
        training_metrics: Output[Metrics],
        reg_rate: float,
        project_id: str,
        bq_location: str
    ) -> float:
        """
        Train logistic regression model using BigQuery training data.
        Converted from train_model function in original train.py.
        
        # TODO: Lab 5.9.1 - ANSWER: Pipeline Enhancement (WHAT: Automated quality gates)
        # WHERE: This conditional logic is not present in original train.py
        # WHY: Pipelines enable sophisticated automation and quality control
        
        # TODO: Lab 5.9.2 - ANSWER: Vertex AI Metrics Integration (WHAT: Enterprise monitoring)
        # WHERE: Replaces simple MLflow autologging from train.py
        # WHY: Cloud-native observability and alerting capabilities
        """
        import re, os, shutil, joblib, logging
        import pandas as pd
        from sklearn.linear_model import LogisticRegression
        from google.cloud import bigquery

        logging.basicConfig(level=logging.INFO)
        
        # Parse BigQuery table from artifact URI (same pattern as conversion example)
        uri = train_data.uri
        logging.info("Parsing BQ URI: %s", uri)
        match = re.search(r'projects/([^/]+)/datasets/([^/]+)/tables/([^/]+)', uri)
        if not match:
                    # TODO: Lab 5.9.X - ANSWER: BigQuery URI parsing for robust pipeline execution
            raise ValueError(f"Could not parse BQ table from URI: {uri}")
        proj, dataset, table = match.groups()
        table_ref = f"{proj}.{dataset}.{table}"
        
        # Load training data from BigQuery
        bq_client = bigquery.Client(project=project_id, location=bq_location)
        query = f"SELECT * FROM `{table_ref}`"
        train_df = bq_client.query(query).to_dataframe()
        logging.info("Loaded %d training rows from BigQuery", len(train_df))
        
                # TODO: Lab 5.9.X - ANSWER: Same feature extraction as original train.py
        FEATURE_COLUMNS = ["Pregnancies","PlasmaGlucose","DiastolicBloodPressure",
                           "TricepsThickness","SerumInsulin","BMI","DiabetesPedigree","Age"]
        X = train_df[FEATURE_COLUMNS]
        y = train_df["Diabetic"]
        
        # Train model with same configuration as original
        model = LogisticRegression(C=1 / reg_rate, solver="liblinear")
        model.fit(X, y)
        
        # Calculate training accuracy for initial assessment
        train_accuracy = model.score(X, y)
        logging.info(f"Training accuracy: {train_accuracy:.4f}")
        
                # TODO: Lab 5.9.X - ANSWER: How model return values become saved artifacts
        model_path = os.path.join(os.path.dirname(trained_model.path), "model.joblib")
        joblib.dump(model, model_path)
        shutil.copy(model_path, trained_model.path)
        
                # TODO: Lab 5.9.X - ANSWER: Vertex AI Metrics vs MLflow tracking
        training_metrics.log_metric("training_accuracy", train_accuracy)
        training_metrics.log_metric("regularization_rate", reg_rate)
        training_metrics.log_metric("training_samples", len(train_df))
        
        logging.info("Model stored at %s and copied to %s", model_path, trained_model.path)
        return train_accuracy

    train_task = train_model_component(
        train_data=bq_train_task.outputs["destination_table"],
        reg_rate=reg_rate,
        project_id=project_id,
        bq_location=region
    ).set_cpu_limit("1").set_memory_limit("3840Mi")
    train_task.after(bq_train_task)

    # ----------------------------
    # Model Evaluation Component (Using BigQuery Test Data)
    # ----------------------------
    # Original train.py evaluation (embedded in train_model):
    # accuracy = model.score(X_test, y_test)
    # print(f"Model accuracy: {accuracy}")
    #
    # Vertex AI component transformation (using BigQuery test data):
    @component(
        base_image="python:3.9",
        packages_to_install=["google-cloud-bigquery", "scikit-learn", "joblib", "pandas"]
    )
    def evaluate_model_component(
        test_data: Input[artifact_types.BQTable],
        trained_model: Input[Model],
        evaluation_metrics: Output[Metrics],
        min_accuracy_threshold: float,
        project_id: str,
        bq_location: str
    ) -> float:
        """
        Evaluate trained model on BigQuery test dataset.
        Extracted and enhanced from train_model function evaluation logic.
        
        Args:
            test_data: BigQuery table containing test data
            model: Trained model artifact from training component
            metrics: Evaluation metrics for monitoring and decision making
            min_accuracy: Minimum accuracy threshold for quality gate
            project_id: Google Cloud project ID
            bq_location: BigQuery location/region
            
        Returns:
            float: Test accuracy for conditional pipeline logic
        """
        import re, logging, joblib
        import pandas as pd
        from sklearn.metrics import accuracy_score, classification_report
        from google.cloud import bigquery

        logging.basicConfig(level=logging.INFO)
        
        # TODO: Lab 5.8.2.1 - ANSWER: URI Parsing Consistency (WHAT: Same pattern as training component)
        # WHERE: Same regex parsing logic used in train_model_op component 
        # WHY: Consistent artifact handling across all pipeline components
        uri = test_data.uri
        match = re.search(r'projects/([^/]+)/datasets/([^/]+)/tables/([^/]+)', uri)
        if not match:
            # TODO: Lab 5.8.2.2 - ANSWER: Error Handling (WHAT: Consistent error messages)
            # WHERE: Same error handling pattern as training component
            # WHY: Uniform error handling improves debugging and reliability
            raise ValueError(f"Could not parse BQ table from URI: {uri}")
        proj, dataset, table = match.groups()
        table_ref = f"{proj}.{dataset}.{table}"

        # Load test data from BigQuery
        bq_client = bigquery.Client(project=project_id, location=bq_location)
        query = f"SELECT * FROM `{table_ref}`"
        test_df = bq_client.query(query).to_dataframe()
        logging.info("Loaded %d test rows from BigQuery", len(test_df))

        # Load trained model
        model_obj = joblib.load(trained_model.path)
        
        # TODO: Lab 5.8.3.1 - ANSWER: Feature Consistency (WHAT: Identical feature columns)
        # WHERE: Same FEATURE_COLUMNS list as used in train_model_op component
        # WHY: Consistent features ensure model predictions work correctly
        FEATURE_COLUMNS = ["Pregnancies","PlasmaGlucose","DiastolicBloodPressure",  # TODO: Lab 5.8.3.1a - WHERE: Same as training component
                           "TricepsThickness","SerumInsulin","BMI","DiabetesPedigree","Age"]  # TODO: Lab 5.8.3.1b - WHAT: Exact feature match
        X_test = test_df[FEATURE_COLUMNS]                                          # TODO: Lab 5.8.3.1c - WHERE: Same DataFrame indexing
        y_test = test_df["Diabetic"]                                               # TODO: Lab 5.8.3.1d - WHERE: Same target column
        
        # Generate predictions and calculate metrics
        preds = model_obj.predict(X_test)
        accuracy = accuracy_score(y_test, preds)

        # TODO: Lab 5.8.4.1 - ANSWER: Enhanced Metrics (WHAT: Structured metrics vs print statements)
        # WHERE: Comprehensive logging vs train.py simple print(f"Model accuracy: {accuracy}")
        # WHY: Enterprise monitoring and alerting integration
        evaluation_metrics.log_metric("accuracy", accuracy)                       # TODO: Lab 5.8.4.1a - WHAT: Structured accuracy metric
        evaluation_metrics.log_metric("min_accuracy_threshold", min_accuracy_threshold)  # TODO: Lab 5.8.4.1b - WHAT: Threshold tracking
        evaluation_metrics.log_metric("test_samples", len(test_df))               # TODO: Lab 5.8.4.1c - WHAT: Data volume monitoring
        evaluation_metrics.log_metric("passes_threshold", float(accuracy >= min_accuracy_threshold))
        
        # Log detailed classification metrics
        report = classification_report(y_test, preds, output_dict=True)
        evaluation_metrics.log_metric("precision_class_0", report['0']['precision'])
        evaluation_metrics.log_metric("recall_class_0", report['0']['recall'])
        evaluation_metrics.log_metric("precision_class_1", report['1']['precision'])
        evaluation_metrics.log_metric("recall_class_1", report['1']['recall'])
        
        logging.info("Accuracy = %.4f", accuracy)
        logging.info(f"Accuracy threshold: {min_accuracy_threshold:.2f}")
        logging.info(f"Threshold met: {accuracy >= min_accuracy_threshold}")
        
        return accuracy

    eval_task = evaluate_model_component(
        test_data=bq_test_task.outputs["destination_table"],
        trained_model=train_task.outputs["trained_model"],
        min_accuracy_threshold=min_accuracy_threshold,
        project_id=project_id,
        bq_location=region
    ).set_cpu_limit("1").set_memory_limit("3840Mi")
    eval_task.after(train_task)

    # ----------------------------
    # Model Registration Component
    # ----------------------------
    # Original train.py model registration:
    # run_id = run.info.run_id
    # mlflow.register_model(f"runs:/{run_id}/model", "diabetes-classification-prod")
    # print(f"Model registered from run {run_id}")
    #
    # Vertex AI component transformation:
    @component(
        base_image="python:3.9",
        packages_to_install=["google-cloud-aiplatform"]
    )
    def register_model_component(
        project_id: str,
        region: str,
        model_display_name: str,
        trained_model: Input[Model],
        model_accuracy: float
    ):
        """
        Register model in Vertex AI Model Registry.
        Replaces MLflow model registration from original train.py.
        
        # TODO: Lab 5.9.X - ANSWER: MLflow vs Vertex AI Model Registry patterns
        # TODO: Lab 5.9.X - ANSWER: How local model files become cloud-managed artifacts
        """
        from google.cloud import aiplatform
        import logging
        
        logging.basicConfig(level=logging.INFO)
        
        # Initialize Vertex AI client
        aiplatform.init(project=project_id, location=region)
        
                # TODO: Lab 5.9.X - ANSWER: How pipeline artifacts integrate with model registry
        # Extract model artifact directory for registration
        model_artifact_uri = trained_model.uri.rsplit("/", 1)[0]
        
        # Register model with enhanced metadata (improvement over original)
        model = aiplatform.Model.upload(
            display_name=f"{model_display_name}-{int(model_accuracy * 10000)}",  # Include accuracy in name
            artifact_uri=model_artifact_uri,
            serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-2:latest",
            labels={
                "model_type": "logistic_regression",
                "use_case": "diabetes_classification",
                "framework": "scikit_learn",
                "accuracy": str(int(model_accuracy * 10000))  # Store accuracy as label
            },
            sync=True
        )
        
        logging.info(f"Model registered: {model.resource_name}")
        logging.info(f"Model display name: {model.display_name}")
        logging.info(f"Model URI: {model_artifact_uri}")

    # ----------------------------
    # Conditional Model Approval
    # ----------------------------
    # Original train.py: No conditional logic (always registers)
    # Vertex AI enhancement: Add quality gate for automated MLOps
            # TODO: Lab 5.9.X - ANSWER: Adding automated approval logic absent from original script
            # TODO: Lab 5.9.X - ANSWER: Learn dsl.If for conditional execution
    
    with dsl.If(eval_task.outputs["Output"] >= min_accuracy_threshold, name="model-approval-gate"):
                # TODO: Lab 5.9.X - ANSWER: Only register models meeting quality thresholds
        register_task = register_model_component(
            project_id=project_id,
            region=region,
            model_display_name=model_display_name,
            trained_model=train_task.outputs["trained_model"],
            model_accuracy=eval_task.outputs["Output"]
        )
        register_task.after(eval_task)
        
        @component(base_image="python:3.9")
        def model_approved_notification(model_accuracy: float, model_name: str):
            """
            Notify about successful model approval and registration.
            Enhancement not present in original train.py.
            """
            import logging
            logging.basicConfig(level=logging.INFO)
            logging.info(f"✅ Model '{model_name}' approved and registered with accuracy: {model_accuracy:.4f}")
            logging.info("Model is ready for deployment to production endpoints.")
        
        approval_task = model_approved_notification(
            model_accuracy=eval_task.outputs["Output"],
            model_name=model_display_name
        )
        approval_task.after(register_task)

    with dsl.If(eval_task.outputs["Output"] < min_accuracy_threshold, name="model-rejection-gate"):
        @component(base_image="python:3.9")
        def model_rejected_notification(model_accuracy: float, threshold: float):
            """
            Handle model rejection due to poor performance.
            Enhancement not present in original train.py.
            
            # TODO: Lab 5.9.X - ANSWER: Graceful failure vs script crashes
            """
            import logging
            logging.basicConfig(level=logging.ERROR)
            logging.error(f"❌ Model rejected. Accuracy {model_accuracy:.4f} below threshold {threshold:.2f}")
            logging.error("Model will not be registered. Review training data and hyperparameters.")
            # Note: Using logging.error instead of raising exception for graceful pipeline completion
        
        rejection_task = model_rejected_notification(
            model_accuracy=eval_task.outputs["Output"],
            threshold=min_accuracy_threshold
        )
        rejection_task.after(eval_task)

            # TODO: Lab 5.9.X - ANSWER: How main() function becomes pipeline orchestration
            # TODO: Lab 5.9.X - ANSWER: Local script execution vs distributed pipeline execution
    # Teaching note: Original train.py runs sequentially in single process; pipeline runs distributed across Vertex AI

# ----------------------------
# Pipeline Compilation and Execution
# ----------------------------
# Original train.py execution:
# if __name__ == "__main__":
#     print("\n\n" + "*" * 60)
#     args = parse_args()
#     main(args)
#     print("*" * 60 + "\n\n")
#
# Vertex AI pipeline compilation and execution:
if __name__ == "__main__":
    """
    Pipeline compilation and submission.
    Replaces direct script execution from original train.py.
    
    # TODO: Lab 5.9.X - ANSWER: Local script vs cloud pipeline submission
    # TODO: Lab 5.9.X - ANSWER: Learn how @dsl.pipeline becomes executable artifact
    """
    from kfp.v2 import compiler
    from google.cloud import aiplatform
    import argparse
    
            # TODO: Lab 5.9.X - ANSWER: Command line args vs pipeline parameters
    parser = argparse.ArgumentParser(description="Compile and run diabetes classification pipeline")
    parser.add_argument("--project-id", required=True, help="Google Cloud project ID")
    parser.add_argument("--region", default="us-central1", help="Google Cloud region")
    parser.add_argument("--pipeline-root", required=True, help="GCS bucket for pipeline artifacts")
    parser.add_argument("--dataset-location", required=True, help="BigQuery table reference (dataset.table)")
    parser.add_argument("--compile-only", action="store_true", help="Only compile pipeline, don't run")
    
    args = parser.parse_args()
    
    # Compile pipeline to JSON
    pipeline_spec_path = "diabetes_pipeline.json"
    compiler.Compiler().compile(
        pipeline_func=diabetes_training_pipeline,
        package_path=pipeline_spec_path
    )
    print(f"Pipeline compiled to: {pipeline_spec_path}")
    
    if not args.compile_only:
                # TODO: Lab 5.9.X - ANSWER: How compiled pipelines execute in Vertex AI
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

        # TODO: Lab 5.9.X - ANSWER: Key transformations from standalone script to cloud pipeline
"""
CONVERSION SUMMARY - Key Transformations:

1. EXECUTION MODEL:
   - Original: Single-process local execution
   - Vertex AI: Distributed container-based execution

2. DATA HANDLING:
   - Original: Local CSV files with glob patterns
   - Vertex AI: BigQuery tables with SQL-based data processing
   - Migration: CSV files have been migrated to BigQuery for better performance

3. DATA SPLITTING:
   - Original: sklearn train_test_split with random_state=42
   - Vertex AI: SQL FARM_FINGERPRINT for deterministic 80/20 splits

4. PARAMETER MANAGEMENT:
   - Original: Command-line arguments with argparse
   - Vertex AI: Pipeline parameters with type hints

5. MODEL PERSISTENCE:
   - Original: MLflow tracking and registry
   - Vertex AI: Vertex AI Model Registry with metadata

6. ERROR HANDLING:
   - Original: Simple exceptions and prints
   - Vertex AI: Structured logging and graceful failures

7. WORKFLOW ORCHESTRATION:
   - Original: Sequential function calls
   - Vertex AI: Component-based pipeline with dependencies

8. SCALABILITY:
   - Original: Single machine limitations
   - Vertex AI: Automatic scaling and resource management

9. DATA ARCHITECTURE:
   - Original: File-based data loading
   - Vertex AI: BigQuery-native data processing with pre-built components

10. MONITORING:
    - Original: Print statements and MLflow UI
    - Vertex AI: Cloud Logging and Vertex AI console

# TODO: Lab 5.9.X - ANSWER: Deploy pipeline, create recurring runs, set up monitoring alerts
# TODO: Lab 5.9.X - ANSWER: Explore hyperparameter tuning, custom training containers, model serving
# TODO: Lab 5.9.X - ANSWER: Learn query optimization and data partitioning strategies
"""

# End of train_to_vertex_ai_conversion.py
