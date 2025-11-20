"""
Custom Data Quality Components
==============================
Custom-built components for data validation and quality checks.

Author: MLOps Team
Maintained by: Data Quality Team

Components:
- data_quality_check_op: Comprehensive data quality validation (completeness only - Phase 1)

Future Components (Phase 2+):
- schema_validation_op: Schema and type validation
- drift_detection_op: Statistical drift detection

Usage:
    from components.custom_data_quality_components import data_quality_check_op

    quality_task = data_quality_check_op(
        input_data=bq_train_task.outputs["destination_table"],
        project_id=project_id,
        bq_location=region,
        required_columns=['Age', 'BMI', 'Diabetic'],  # <-- Passed as parameter
        completeness_threshold=0.95,
        fail_on_completeness=True,
        min_row_count=100,
        env_prefix="[DEV]"
    )

Integration:
    Works with BigQuery Query Job component outputs (Input[artifact_types.BQTable])
"""
