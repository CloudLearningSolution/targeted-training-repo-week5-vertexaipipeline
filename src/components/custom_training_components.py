"""
Custom Training Components
=========================
Custom-built components for model training.

Author: MLOps Team
Maintained by: ML Engineering Team

Components:
- train_model_op: Train logistic regression model from BigQuery data

Usage:
    from components.custom_training_components import train_model_op

    train_task = train_model_op(
        train_data=bq_train_task.outputs["destination_table"],
        feature_columns=['Age', 'BMI', 'PlasmaGlucose'],  # <-- Passed as parameter
        target_column='Diabetic',
        reg_rate=0.05,
        project_id=project_id,
        bq_location=region,
        env_prefix="[DEV]"
    )
"""
# Planning DEMO
# S3 logic
# S3 information
# Libraries lab 6.1.1 in the model.py

