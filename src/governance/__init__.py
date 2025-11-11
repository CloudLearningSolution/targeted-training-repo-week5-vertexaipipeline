"""
Governance & FinOps Tools
========================
Tools for resource management, cost control, and policy enforcement.

These tools run in CI/CD pipelines (GitHub Actions) to validate
compiled pipeline YAML before deployment to Vertex AI.

Tools:
- resource_auditor: Audit CPU/memory limits in compiled pipelines
- cost_estimator: Estimate pipeline execution costs (future)
- policy_validator: Validate governance policies (future)

Usage:
    # In GitHub Actions workflow:
    python -m src.governance.resource_auditor compiled_pipelines/dev_pipeline.yaml
"""

__version__ = '0.1.0'
__all__ = ['resource_auditor']
