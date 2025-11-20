"""
Vertex AI KFP Pipeline for Development

This pipeline performs the following steps:
- Queries data from BigQuery Feature Group view (splits via
  query logic).
- Validates data quality (completeness checks) on training data.
- Trains a sysco model in development mode.
- Evaluates the trained model on a test split.
- Conditionally registers the model in Vertex AI Model Registry if
  accuracy meets the minimum threshold.
- Rejects the model if accuracy is insufficient.

All comments and documentation lines are kept <= 100 characters for
.flake8.
"""

from kfp import dsl
from kfp.dsl import pipeline

# Import custom components from the specific modules within the
# components package.
from .components.custom_data_quality_components import (
    data_quality_check_op
)
from .components.custom_training_components import train_model_op
from .components.custom_evaluation_components import (
    evaluate_model_op,
    model_approved_op,
    model_rejected_op,
)
from .components.custom_registry_components import register_model_op

# Import pre-built components from the specific module within the
# components package.
from .components.prebuilt_bigquery_components import (
    bigquery_query_job_op
)

PIPELINE_NAME = "vertex-template-sysco-model-name"
PIPELINE_DESCRIPTION = (
    "Development template pipeline for Sysco model on Vertex AI "
    "with BigQuery"
)
