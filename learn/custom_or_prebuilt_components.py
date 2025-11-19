"""
Ready?

Review model.py and ingest_model.py in the AWS repo and train_to_vertex_ai_conversion.py files from this repo for conversion patterns.

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
