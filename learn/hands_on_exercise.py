"""
Hands-On Coding Exercise:
===============================================================

Complete challenge.

Review train_to_vertex_ai_conversion.py files for patterns.

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
