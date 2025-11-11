"""
Vertex AI Pipeline Template Components Library
=====================================
Reusable template components, domain-organized components for MLOps Capabilty.

Naming Convention:
- custom_*_components.py → Custom-built components (maintained by MLOps & AI Governance team)
- prebuilt_*_components.py → Google pre-built component imports

Current Implementation Status:
✅ Phase 1A: Data Quality (Completeness only)
✅ Phase 1B: Training, Evaluation, Registry, BigQuery components
⏳ Phase 2: Monitoring, Audit, Experiment components (future)

Import Patterns:
    # Pattern 1: Import specific components
    from components import (
        data_quality_check_op,
        train_model_op,
        evaluate_model_op,
        bigquery_query_job_op
    )

    # Pattern 2: Import from specific domain
    from components.custom_data_quality_components import data_quality_check_op
    from components.custom_training_components import train_model_op
    from components.prebuilt_bigquery_components import bigquery_query_job_op

Used in Pipelines:
    # In vertex_pipeline_dev.py or vertex_pipeline_prod.py
    from components import (
        bigquery_query_job_op,
        data_quality_check_op,
        train_model_op,
        evaluate_model_op,
        model_approved_op,
        model_rejected_op,
        register_model_op
    )
"""

# =============================================================================
# CUSTOM COMPONENTS - IMPLEMENTED ✅
# =============================================================================

# Custom: Data Quality Components ✅ IMPLEMENTED (Phase 1A)
from .custom_data_quality_components import (
    data_quality_check_op,
    # Future Phase 2: schema_validation_op
    # Future Phase 3: drift_detection_op
)

# Custom: Training Components ✅ IMPLEMENTED (Phase 1B)
from .custom_training_components import (
    train_model_op,
    # Future: hyperparameter_tuning_op
)

# Custom: Evaluation Components ✅ IMPLEMENTED (Phase 1B)
from .custom_evaluation_components import (
    evaluate_model_op,
    model_approved_op,
    model_rejected_op,
)

# Custom: Registry Components ✅ IMPLEMENTED (Phase 1B)
from .custom_registry_components import (
    register_model_op,
    # Future: version_model_op
)

# =============================================================================
# CUSTOM COMPONENTS - FUTURE ⏳
# =============================================================================

# TODO: Uncomment when custom_monitoring_components.py is implemented
# from .custom_monitoring_components import (
#     model_monitoring_setup_op,
#     prediction_logging_op,
#     alert_configuration_op,
# )

# TODO: Uncomment when custom_audit_components.py is implemented
# from .custom_audit_components import (
#     audit_lineage_op,
#     compliance_check_op,
#     metadata_logging_op,
# )

# TODO: Uncomment when custom_experiment_components.py is implemented
# from .custom_experiment_components import (
#     experiment_tracking_op,
# )

# =============================================================================
# PRE-BUILT COMPONENTS - IMPLEMENTED ✅
# =============================================================================

# Pre-built: BigQuery Components ✅ IMPLEMENTED (Phase 1B)
from .prebuilt_bigquery_components import (
    bigquery_query_job_op,
    # Future: bigquery_create_model_op
    # Future: bigquery_predict_model_op
)

# =============================================================================
# EXPORTS - ALL IMPLEMENTED COMPONENTS ✅
# =============================================================================

__all__ = [
    # =========================================================================
    # CUSTOM COMPONENTS ✅ IMPLEMENTED
    # =========================================================================

    # Data Quality (Phase 1A: Completeness only)
    'data_quality_check_op',

    # Training (Phase 1B)
    'train_model_op',

    # Evaluation (Phase 1B)
    'evaluate_model_op',
    'model_approved_op',
    'model_rejected_op',

    # Registry (Phase 1B)
    'register_model_op',

    # =========================================================================
    # PRE-BUILT COMPONENTS ✅ IMPLEMENTED
    # =========================================================================

    # BigQuery (Phase 1B)
    'bigquery_query_job_op',

    # =========================================================================
    # FUTURE COMPONENTS ⏳ (Uncomment as implemented)
    # =========================================================================

    # TODO: Phase 2 - Monitoring
    # 'model_monitoring_setup_op',
    # 'prediction_logging_op',
    # 'alert_configuration_op',

    # TODO: Phase 2 - Audit
    # 'audit_lineage_op',
    # 'compliance_check_op',
    # 'metadata_logging_op',

    # TODO: Phase 2 - Experiment
    # 'experiment_tracking_op',
]


# =============================================================================
# VERSION INFORMATION
# =============================================================================

__version__ = '0.2.0'  # Phase 1B: Core pipeline components implemented

# Version History:
# - 0.1.0: Initial release with data quality (completeness only)
# - 0.2.0: Added training, evaluation, registry, BigQuery components
# - 0.3.0: (Future) Add monitoring, audit, experiment components
