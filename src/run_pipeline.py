"""
Submits a Vertex AI Pipeline Job using the Vertex AI SDK for Python.

This script is the recommended, officially supported method for running
Vertex AI Pipelines from a CI/CD environment. It replaces the need for
the gcloud CLI for pipeline submission.

===============================================================================
Lab 5.4: Vertex AI Pipeline Component Architecture Exploration
===============================================================================
This script demonstrates how to submit and execute Vertex AI Pipeline jobs
programmatically. Understanding this script helps identify:
- How pipeline jobs are created and configured
- What components are required for pipeline execution
- How pipeline parameters and metadata are passed to Vertex AI

TODO: Lab 5.4.1 - Component Identification: Pipeline job submission component
TODO: Lab 5.4.2 - Purpose Recognition: Programmatic pipeline execution interface
TODO: Lab 5.4.3 - Architecture Understanding: Pipeline execution architecture

===============================================================================
Lab 5.5: Vertex AI Custom Components and Pre-built Components and Accelerator Templates
===============================================================================
This script demonstrates how pipeline execution integrates with Terraform-provisioned
infrastructure. Understanding this integration is critical for the Accelerator
Template architecture.

INFRASTRUCTURE DEPENDENCY (Terraform):
======================================
This script REQUIRES infrastructure to already exist before it can run:

Required Terraform Resources (from vertex_ai_infrastructure.tf):
-----------------------------------------------------------------
1. GCS Bucket (google_storage_bucket.mlops_bucket):
   - Used by: --pipeline-root parameter
   - Purpose: Stores pipeline execution artifacts
   - Created by: Terraform BEFORE this script runs

2. Service Account (google_service_account.vertex_pipeline_sa):
   - Used by: --service-account parameter
   - Purpose: Provides execution identity and permissions
   - Created by: Terraform BEFORE this script runs

3. Vertex AI API Enabled:
   - Used by: aiplatform.init() and PipelineJob()
   - Purpose: Enables Vertex AI operations
   - Created by: Terraform or manual setup

WORKFLOW INTEGRATION:
====================
1. Platform Team runs: terraform apply
   └─> Creates: GCS buckets, service accounts, IAM permissions

2. GitHub Actions calls this script: python run_pipeline.py
   └─> Uses: Terraform-created bucket (--pipeline-root)
   └─> Uses: Terraform-created service account (--service-account)
   └─> Submits: Compiled pipeline to Vertex AI

If Terraform infrastructure doesn't exist, this script will fail with errors like:
- "Bucket does not exist"
- "Service account not found"
- "Permission denied"

CUSTOM vs PRE-BUILT COMPONENT SUPPORT:
======================================
This script works with BOTH custom and pre-built components:

Custom Components (current pipelines):
- Reads artifacts from GCS bucket (Terraform-created)
- Writes artifacts to GCS bucket (Terraform-created)
- Uses service account permissions (Terraform-created)

Pre-built Components (if used):
- Same GCS bucket access requirements
- Same service account permissions
- Additional IAM permissions for specific operations
  (e.g., BigQuery access for BigqueryQueryJobOp)

The infrastructure provisioned by Terraform supports both component types!

TODO: Lab 5.5.3 - Accelerator Templates: This script uses Terraform-provisioned infrastructure
TODO: Lab 5.5.3 - Infrastructure Integration: GCS buckets and service accounts from Terraform
TODO: Lab 5.5.3 - Component Support: Infrastructure supports both custom and pre-built components
===============================================================================

Usage:
    python src/run_pipeline.py --project-id <PROJECT_ID> --region <REGION>
        --pipeline-spec-uri <PIPELINE_SPEC_PATH> --service-account <SERVICE_ACCOUNT>
        --pipeline-root <PIPELINE_ROOT> --display-name <DISPLAY_NAME>
        --parameter-values-json <PARAMS_JSON> --labels-json <LABELS_JSON>
        [--enable-caching]

Arguments:
    --project-id: Google Cloud project ID.
    --region: Google Cloud region for Vertex AI resources.
    --pipeline-spec-uri: Path or GCS URI to the compiled pipeline spec file.
    --service-account: Service account email for running the pipeline job.
    --pipeline-root: GCS path for pipeline output artifacts.
    --display-name: Display name for the pipeline job.
    --parameter-values-json: JSON string of pipeline parameter values.
    --labels-json: JSON string of labels for the pipeline job.
    --enable-caching: Optional flag to enable pipeline step caching.

This script is intended to be used in CI/CD workflows and supports all
required arguments for robust pipeline job submission. Logging is enabled
for all major steps and errors.
"""

import argparse
import json
import logging
from datetime import datetime
from google.cloud import aiplatform


def main():
    """
    Main function to submit a Vertex AI Pipeline job.
    
    This function demonstrates the core components required for pipeline execution:
    - Pipeline specification (compiled YAML)
    - Runtime parameters and configuration
    - Execution environment (project, region, service account)
    - Job metadata (labels, display name, caching)
    
    TODO: Lab 5.4.1 - Component Identification: Pipeline job submission requires multiple components
    TODO: Lab 5.4.2 - Purpose Recognition: Orchestrates pipeline execution in Vertex AI
    TODO: Lab 5.4.3 - Architecture Understanding: Bridge between compiled pipeline and Vertex AI runtime
    """
    # ==========================================================================
    # ARGUMENT PARSING - Pipeline Job Configuration Components
    # ==========================================================================
    # TODO: Lab 5.4.1 - Component Identification: Command-line arguments define pipeline job configuration
    # TODO: Lab 5.4.2 - Purpose Recognition: Flexible configuration enables different pipeline executions
    
    # Set up argument parser for all required pipeline job parameters.
    parser = argparse.ArgumentParser(
        description="Submit a Vertex AI Pipeline Job."
    )
    
    # TODO: Lab 5.4.1 - Component Identification: Project and region define Vertex AI resource context
    # TODO: Lab 5.4.2 - Purpose Recognition: GCP project and region determine where pipeline executes
    parser.add_argument(
        "--project-id", type=str, required=True, help="Google Cloud project ID."
    )
    parser.add_argument(
        "--region", type=str, required=True, help="Google Cloud region."
    )
    
    # TODO: Lab 5.4.1 - Component Identification: Pipeline spec is the compiled pipeline definition
    # TODO: Lab 5.4.2 - Purpose Recognition: Compiled YAML specification contains all component definitions
    # TODO: Lab 5.4.3 - Architecture Understanding: Pipeline spec is the executable pipeline blueprint
    parser.add_argument(
        "--pipeline-spec-uri", type=str, required=True,
        help="GCS URI or local path of the compiled pipeline spec."
    )
    
    # TODO: Lab 5.4.1 - Component Identification: Service account provides pipeline execution identity
    # TODO: Lab 5.4.2 - Purpose Recognition: Service account controls access to GCP resources during execution
    # TODO: Lab 5.5.3 - Terraform Dependency: Service account created by Terraform (google_service_account.vertex_pipeline_sa)
    parser.add_argument(
        "--service-account", type=str, required=True,
        help="Service account for the pipeline run."
    )
    
    # TODO: Lab 5.4.1 - Component Identification: Pipeline root is GCS storage for pipeline artifacts
    # TODO: Lab 5.4.2 - Purpose Recognition: Centralized storage location for all pipeline outputs
    # TODO: Lab 5.4.3 - Architecture Understanding: GCS serves as artifact storage layer in pipeline architecture
    # TODO: Lab 5.5.3 - Terraform Dependency: GCS bucket created by Terraform (google_storage_bucket.mlops_bucket)
    parser.add_argument(
        "--pipeline-root", type=str, required=True,
        help="GCS root path for pipeline outputs."
    )
    
    # TODO: Lab 5.4.1 - Component Identification: Display name enables pipeline job identification
    # TODO: Lab 5.4.2 - Purpose Recognition: Human-readable identifier for tracking pipeline runs
    parser.add_argument(
        "--display-name", type=str, required=True,
        help="Display name for the pipeline run."
    )
    
    # TODO: Lab 5.4.1 - Component Identification: Parameter values configure pipeline component behavior
    # TODO: Lab 5.4.2 - Purpose Recognition: Runtime parameters enable dynamic pipeline configuration
    # TODO: Lab 5.4.3 - Architecture Understanding: Parameters flow from job submission to individual components
    parser.add_argument(
        "--parameter-values-json", type=str, required=True,
        help="JSON string of pipeline parameter values."
    )
    
    # TODO: Lab 5.4.1 - Component Identification: Labels provide metadata for pipeline job organization
    # TODO: Lab 5.4.2 - Purpose Recognition: Labels enable filtering, tracking, and cost attribution
    parser.add_argument(
        "--labels-json", type=str, required=True,
        help="JSON string of labels for the pipeline run."
    )
    
    # TODO: Lab 5.4.1 - Component Identification: Caching flag enables component output reuse
    # TODO: Lab 5.4.2 - Purpose Recognition: Caching optimizes pipeline execution by reusing previous results
    # TODO: Lab 5.4.3 - Architecture Understanding: Caching is component-level feature in pipeline architecture
    
    # TODO: Lab 5.6.3 — Caching and retry guidance
    # INSPECT: This --enable-caching flag controls component-level caching.
    # TASK: Note that caching is OPTIONAL (action="store_true" means default is False).
    # TASK: Document where caching is enabled/disabled (here in CLI args, passed to PipelineJob below).
    # GUIDANCE: Enable caching for idempotent components (same inputs → same outputs, no side effects).
    # GUIDANCE: Disable caching for components with side effects (external API calls, database writes).
    
    parser.add_argument(
        "--enable-caching", action="store_true",
        help="Flag to enable caching for the pipeline run."
    )

    args = parser.parse_args()

    # ==========================================================================
    # VERTEX AI INITIALIZATION - Platform Connection Component
    # ==========================================================================
    # TODO: Lab 5.4.1 - Component Identification: Vertex AI SDK initialization creates platform connection
    # TODO: Lab 5.4.2 - Purpose Recognition: SDK initialization establishes connection to Vertex AI services
    # TODO: Lab 5.4.3 - Architecture Understanding: aiplatform.init() configures SDK for pipeline submission
    
    # TODO: Lab 5.6.5 — Observability: metrics, logging, and dashboard links
    # INSPECT: logging.basicConfig() sets up logging at INFO level.
    # TASK: Locate all logging.info() and logging.error() calls in this file (there are 4 total).
    # TASK: Note the logging pattern used for major steps (initialization, submission, success).
    
    # Configure logging for info and error messages.
    logging.basicConfig(level=logging.INFO)
    logging.info("Initializing Vertex AI Platform...")

    # Initialize Vertex AI SDK with project and region.
    # TODO: Lab 5.4.1 - Component Identification: Project and location define execution environment
    aiplatform.init(project=args.project_id, location=args.region)

    # ==========================================================================
    # PARAMETER PARSING - Pipeline Configuration Component
    # ==========================================================================
    # TODO: Lab 5.4.1 - Component Identification: JSON parsing converts string arguments to structured data
    # TODO: Lab 5.4.2 - Purpose Recognition: Structured parameters enable type-safe pipeline configuration
    
    # TODO: Lab 5.6.2 — PipelineJob construction and parameter shapes
    # INSPECT: JSON parsing converts CLI string arguments to Python dicts.
    # TASK: Note how parameter values are serialized (JSON string → Python dict via json.loads()).
    # TASK: Identify which pipeline parameters are required (all parameters in --parameter-values-json).
    # TASK: Open vertex_pipeline_dev.py or vertex_pipeline_prod.py and compare parameter names.
    
    # Parse JSON strings for parameter values and labels.
    try:
        # TODO: Lab 5.4.1 - Component Identification: Parameter values configure individual pipeline components
        # TODO: Lab 5.4.3 - Architecture Understanding: Parameters define runtime behavior across pipeline components
        parameter_values = json.loads(args.parameter_values_json)
        
        # TODO: Lab 5.4.1 - Component Identification: Labels provide pipeline job metadata
        labels = json.loads(args.labels_json)
    except json.JSONDecodeError as e:
        # TODO: Lab 5.6.7 — Failure modes and debugging tips
        # INSPECT: JSON parsing can fail if strings are malformed.
        # DEBUGGING TIP: Check for missing quotes, trailing commas, or invalid JSON syntax.
        
        logging.error(f"Failed to parse JSON arguments: {e}")
        raise

    # ==========================================================================
    # JOB CREATION - Pipeline Job Component Instantiation
    # ==========================================================================
    # TODO: Lab 5.4.1 - Component Identification: Unique job ID enables pipeline run tracking
    # TODO: Lab 5.4.2 - Purpose Recognition: Timestamp-based ID ensures uniqueness across pipeline runs
    
    # Generate a unique job ID using display name and timestamp.
    job_id = f"{args.display_name}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    logging.info(f"Submitting pipeline job: {job_id}")

    # TODO: Lab 5.4.1 - Component Identification: PipelineJob object represents a pipeline execution instance
    # TODO: Lab 5.4.2 - Purpose Recognition: PipelineJob encapsulates all configuration for pipeline execution
    # TODO: Lab 5.4.3 - Architecture Understanding: PipelineJob is the runtime representation of compiled pipeline
    
    # TODO: Lab 5.6.2 — PipelineJob construction and parameter shapes
    # INSPECT: This PipelineJob constructor is the core of pipeline submission.
    # TASK: Note ALL arguments passed: display_name, template_path, pipeline_root, parameter_values, enable_caching, labels.
    # TASK: Identify the pipeline spec URI (template_path) - this is the compiled YAML from compiler.py.
    # TASK: Identify the service account (passed to submit() below, not constructor).
    # TASK: Note the caching flag (enable_caching) - passed from CLI args above.
    
    # Create the PipelineJob object with all required parameters.
    pipeline_job = aiplatform.PipelineJob(
        # TODO: Lab 5.4.1 - Component Identification: Display name identifies the pipeline job in Vertex AI
        display_name=job_id,
        
        # TODO: Lab 5.4.1 - Component Identification: Template path points to compiled pipeline specification
        # TODO: Lab 5.4.3 - Architecture Understanding: Compiled YAML spec defines complete pipeline architecture
        template_path=args.pipeline_spec_uri,
        
        # TODO: Lab 5.4.1 - Component Identification: Pipeline root defines artifact storage location
        # TODO: Lab 5.4.3 - Architecture Understanding: GCS storage integrates with pipeline execution
        pipeline_root=args.pipeline_root,
        
        # TODO: Lab 5.4.1 - Component Identification: Parameter values configure component behavior at runtime
        # TODO: Lab 5.4.3 - Architecture Understanding: Parameters flow from job to individual components
        parameter_values=parameter_values,
        
        # TODO: Lab 5.4.1 - Component Identification: Caching flag enables component output reuse
        # TODO: Lab 5.4.2 - Purpose Recognition: Caching reduces execution time and costs
        enable_caching=args.enable_caching,
        
        # TODO: Lab 5.4.1 - Component Identification: Labels provide job metadata for organization
        labels=labels
    )

    # ==========================================================================
    # JOB SUBMISSION - Pipeline Execution Initiation
    # ==========================================================================
    # TODO: Lab 5.4.1 - Component Identification: submit() initiates pipeline execution in Vertex AI
    # TODO: Lab 5.4.2 - Purpose Recognition: Job submission transfers control to Vertex AI orchestration
    # TODO: Lab 5.4.3 - Architecture Understanding: Submission triggers Kubeflow pipeline execution in managed environment
    
    # TODO: Lab 5.6.2 — PipelineJob construction and parameter shapes
    # INSPECT: The submit() call actually submits the job to Vertex AI.
    # TASK: Note that submit() is ASYNCHRONOUS by default (does not wait for completion).
    # TASK: Note the service_account parameter passed here (not in PipelineJob constructor above).
    
    # TODO: Lab 5.6.3 — Caching and retry guidance
    # INSPECT: Retry configuration is NOT set at the job level (it's set at component level in pipeline files).
    # TASK: Note that retries must be configured in individual components using .set_retry() or num_retries parameter.
    # GUIDANCE: Review pipeline files to find retry configuration on components.
    
    # TODO: Lab 5.6.7 — Failure modes and debugging tips
    # INSPECT: No try/except around submit() - exceptions will propagate.
    # COMMON FAILURES: Invalid service account, missing pipeline spec, invalid parameters, bucket doesn't exist.
    # DEBUGGING TIP: Check exception message; verify Terraform infrastructure exists; validate parameter names.
    
    # Submit the pipeline job using the specified service account.
    pipeline_job.submit(service_account=args.service_account)

    # ==========================================================================
    # EXECUTION MONITORING - Pipeline Job Tracking Component
    # ==========================================================================
    # TODO: Lab 5.4.1 - Component Identification: Dashboard URI enables pipeline execution monitoring
    # TODO: Lab 5.4.2 - Purpose Recognition: Console access for real-time pipeline monitoring and debugging
    # TODO: Lab 5.4.3 - Architecture Understanding: Vertex AI provides managed monitoring layer for pipelines
    
    # TODO: Lab 5.6.5 — Observability: metrics, logging, and dashboard links
    # INSPECT: These logging calls provide observability for pipeline submission.
    # TASK: Note the dashboard URI construction using pipeline_job._dashboard_uri().
    # TASK: Document where to find logs: this script's logs (stdout), pipeline logs (Vertex AI console).
    # TASK: Note that this is submission success, NOT pipeline completion (pipeline runs asynchronously).
    
    # Log success and provide dashboard URI for monitoring.
    logging.info(f"✅ Successfully submitted pipeline job: {job_id}")
    logging.info(f"View in console: {pipeline_job._dashboard_uri()}")


if __name__ == "__main__":
    """
    Entry point for script execution.
    
    This script demonstrates the complete pipeline job submission architecture:
    
    PIPELINE JOB SUBMISSION ARCHITECTURE:
    =====================================
    1. CONFIGURATION COMPONENTS:
       - Project and region (execution context)
       - Pipeline specification (compiled YAML)
       - Service account (execution identity) [FROM TERRAFORM]
       - Pipeline root (artifact storage) [FROM TERRAFORM]
    
    2. RUNTIME COMPONENTS:
       - Parameter values (component configuration)
       - Labels (job metadata)
       - Caching settings (optimization)
    
    3. EXECUTION COMPONENTS:
       - PipelineJob object (job representation)
       - Vertex AI SDK (platform interface)
       - Kubeflow backend (orchestration engine)
    
    4. TERRAFORM INFRASTRUCTURE INTEGRATION:
       - GCS bucket (--pipeline-root) created by Terraform
       - Service account (--service-account) created by Terraform
       - IAM permissions configured by Terraform
       - Vertex AI APIs enabled by Terraform or manual setup
    
    DEPLOYMENT SEQUENCE:
    ===================
    1. Platform team: terraform apply
       └─> Creates infrastructure (buckets, service accounts)
    
    2. GitHub Actions: python run_pipeline.py \
           --pipeline-root gs://TERRAFORM-CREATED-BUCKET/... \
           --service-account TERRAFORM-CREATED-SA@...
       └─> Submits pipeline using Terraform infrastructure
    
    TODO: Lab 5.4.3 - Architecture Understanding: This script demonstrates the
          complete pipeline job submission architecture in Vertex AI
    TODO: Lab 5.5.3 - Accelerator Templates: This script integrates with Terraform
          infrastructure layer to submit pipelines
    
    TODO: Lab 5.6.3 — Caching and retry guidance
    CACHING GUIDANCE:
    - Caching is enabled/disabled via --enable-caching flag (passed to PipelineJob constructor).
    - Enable caching for IDEMPOTENT components (same inputs always produce same outputs, no side effects).
    - Disable caching for components with side effects (external API calls, database writes, real-time data).
    - Cache scope: project + location + pipeline_root. Changing any component input invalidates cache.
    
    RETRY GUIDANCE:
    - Retries are NOT configured in this script - they must be set at the component level.
    - In pipeline files, use .set_retry(num_retries=3) on component tasks.
    - Only retry IDEMPOTENT components that might fail transiently (network errors, API timeouts).
    - Do NOT retry non-idempotent operations (database writes, API calls with side effects).
    
    TASK: Open vertex_pipeline_dev.py and vertex_pipeline_prod.py to find retry configurations.
    
    TODO: Lab 5.6.7 — Failure modes and debugging tips
    COMMON FAILURE MODES:
    1. JSON parsing errors - Check --parameter-values-json and --labels-json format.
    2. Invalid service account - Verify Terraform created the service account and IAM permissions.
    3. Missing pipeline spec - Verify compiler.py ran successfully and YAML file exists.
    4. Bucket doesn't exist - Verify Terraform created the GCS bucket.
    5. Parameter name mismatch - Compare JSON keys to pipeline function parameters.
    
    DEBUGGING TIPS:
    - Check logs: stdout/stderr for this script, Vertex AI console for pipeline execution.
    - Verify Terraform infrastructure exists before running this script.
    - Validate parameter_values JSON keys match pipeline function signature exactly.
    - Check IAM permissions: service account needs Vertex AI User + Storage Object Admin roles.
    - Use dashboard URI to monitor pipeline execution and view component logs.
    
    BEST PRACTICES:
    - Always include environment and team labels for cost attribution.
    - Use descriptive display names that include environment (dev/prod).
    - Enable caching for dev/test pipelines to speed up iteration.
    - Disable caching for production pipelines to ensure fresh execution.
    - Log all major steps for troubleshooting.
    """
    # Entry point for script execution.
    main()
