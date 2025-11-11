"""
MLOps Template Pipeline Package
==========================================
Vertex AI pipeline components and orchestration for Sysco models.

This template package contains:
- Pipeline definitions (dev and prod)
- Pipeline compiler
- Pipeline execution scripts
- Template components (custom and pre-built)

Template Package Structure:
    src/
    ├── __init__.py (this file)
    ├── compiler.py
    ├── vertex_pipeline_dev.py
    ├── vertex_pipeline_prod.py
    ├── run_pipeline.py
    ├── deploy_model.py
    └── components/
        ├── __init__.py
        ├── custom_*_components.py
        └── prebuilt_*_components.py

Usage (Module Execution):
    # Compile pipelines
    python -m src.compiler --py src/vertex_pipeline_dev.py --output pipeline.yaml

    # Run pipeline
    python -m src.run_pipeline --project-id PROJECT --region REGION ...

    # Deploy model
    python -m src.deploy_model --project-id PROJECT --region REGION ...

Execution Context:
    This package is designed to be executed as a module using the -m flag.
    This allows relative imports throughout the package to work correctly.

    Example:
        ✅ python -m src.compiler (CORRECT - treats src/ as package)
        ❌ python src/compiler.py (WRONG - breaks relative imports)
"""

__version__ = '0.2.0'
__author__ = 'MLOps Team'

# Package metadata
__all__ = [
    'compiler',
    'vertex_pipeline_dev',
    'vertex_pipeline_prod',
    'run_pipeline',
    'deploy_model',
    'components',
]

# Version history
# 0.1.0: Initial release with data quality (completeness only)
# 0.2.0: Added training, evaluation, registry, BigQuery components
# 0.3.0: (Future) Add monitoring, audit, experiment components
