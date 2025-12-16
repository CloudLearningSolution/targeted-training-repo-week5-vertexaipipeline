"""
Custom Registry Components
==========================
Custom-built components for model registration and versioning.

Author: MLOps Team
Maintained by: ML Engineering Team

Components:
- register_model_op: Registers a model to Vertex AI Model Registry

Usage:
    from components.custom_registry_components import register_model_op

    register_task = register_model_op(
        project_id=project_id,
        region=region,
        model_display_name="diabetes-model",
        model_artifact=train_task.outputs["output_model"],
        parent_model="projects/123/locations/us-east1/models/456",
        env_prefix="[PROD]"
    )
"""
     * WHAT: Inspect boto3.upload_file usage
     * WHY: Durable storage pattern in AWS
     * Migration Planning: Replace with GCS client logic inside a
