"""
Compiles Kubeflow Pipelines (KFP v2) into YAML pipeline specs for Vertex AI.
Supports dev and prod variants of the diabetes prediction pipeline with dynamic
selection based on the provided Python file name.

This script is intended for use in CI/CD workflows to automate pipeline
compilation for Vertex AI. It ensures that the correct pipeline function is
selected and compiled to a YAML spec for submission.

===============================================================================
Lab 5.4: Vertex AI Pipeline Component Architecture Exploration
===============================================================================
This script demonstrates the pipeline compilation process, a critical component
in the Vertex AI Pipeline architecture. Understanding compilation helps identify:
- How Python pipeline definitions become executable specifications
- What the compilation process validates and transforms
- How compiled pipelines enable Vertex AI execution

TODO: Lab 5.4.1 - Component Identification: Pipeline compiler component
TODO: Lab 5.4.2 - Purpose Recognition: Transforms Python definitions to executable YAML
TODO: Lab 5.4.3 - Architecture Understanding: Compilation bridges development and execution

===============================================================================
Lab 5.5: Vertex AI Custom Components and Pre-built Components and Accelerator Templates
===============================================================================
This script compiles pipelines containing custom components (and potentially
pre-built components) into executable specifications. Understanding compilation
in the Accelerator Template context shows how different component types are
processed.

COMPILATION IN THE ACCELERATOR TEMPLATE ARCHITECTURE:
=====================================================

INFRASTRUCTURE LAYER (Terraform - Runs First):
-----------------------------------------------
terraform apply → Creates GCS buckets, service accounts, IAM permissions
Status: Infrastructure READY

COMPILATION LAYER (This Script - Runs Second):
----------------------------------------------
python compiler.py → Compiles pipeline Python code to YAML specification
Input: vertex_pipeline_dev.py or vertex_pipeline_prod.py
Output: vertex_pipeline_dev.yaml or vertex_pipeline_prod.yaml
Status: Pipeline specification READY for execution

EXECUTION LAYER (run_pipeline.py - Runs Third):
-----------------------------------------------
python run_pipeline.py → Submits compiled YAML to Vertex AI
Uses: Terraform-created infrastructure (buckets, service accounts)
Status: Pipeline RUNNING in Vertex AI

CUSTOM vs PRE-BUILT COMPONENT COMPILATION:
==========================================

Custom Components (Current Pipelines):
--------------------------------------
- Defined with @component decorator
- Compiled into container specifications
- Base image and packages embedded in YAML
- No external dependencies required during compilation

Pre-built Components (If Used):
-------------------------------
- Imported from google_cloud_pipeline_components library
- Already pre-compiled by Google
- References to pre-built container images
- Same compilation process, different component source

COMPILATION VALIDATES BOTH TYPES:
- Component input/output type compatibility
- Parameter type consistency
- Resource specifications (CPU, memory)
- Container image availability

Example: If we used BigqueryQueryJobOp (pre-built component):
- Compiler validates component exists in library
- Compiler includes reference to Google's pre-built container
- Compiled YAML references Google-maintained image
- No custom container building required

TERRAFORM INTEGRATION:
=====================
Compilation is INDEPENDENT of Terraform but compiled pipelines will:
- Store artifacts in Terraform-created GCS buckets (at runtime)
- Execute using Terraform-created service accounts (at runtime)
- Use Terraform-provisioned infrastructure (at runtime)

Compilation happens BEFORE runtime, so Terraform resources aren't needed yet.
However, the compiled YAML will eventually USE those Terraform resources.

TODO: Lab 5.5.1 - Custom Components: This script compiles custom component definitions
TODO: Lab 5.5.2 - Pre-built Components: Could compile pre-built components with same process
TODO: Lab 5.5.3 - Accelerator Templates: Compilation is step 2 of 3 in deployment workflow
===============================================================================
"""

import argparse
import sys
from typing import Callable
from kfp import compiler

# TODO: Lab 5.4.1 - Component Identification: Pipeline function imports define available pipelines
# TODO: Lab 5.4.2 - Purpose Recognition: Separate dev/prod pipeline definitions enable environment-specific configurations
# TODO: Lab 5.4.3 - Architecture Understanding: Pipeline functions are the high-level component definitions

# TODO: Lab 5.6.1 — Compilation and YAML structure
# INSPECT: These imports define which pipeline functions can be compiled.
# TASK: Note the function names (dev_diabetes_pipeline, prod_diabetes_pipeline).
# TASK: Open vertex_pipeline_dev.py and vertex_pipeline_prod.py to find these function definitions.
# TASK: In each pipeline file, identify the pipeline function name and its primary parameters.
# TASK: Record which artifact types appear in the pipeline signature (e.g., Input[Dataset], Output[Model]).

# Import pipeline functions for dev and prod environments.
from vertex_pipeline_dev import dev_diabetes_pipeline
from vertex_pipeline_prod import prod_diabetes_pipeline


def main() -> None:
    """
    Parses command-line arguments and compiles the selected pipeline into a YAML
    file suitable for Vertex AI Pipelines.
    
    This function demonstrates the core compilation architecture:
    - Pipeline function selection (dev vs prod)
    - KFP v2 compilation process
    - YAML specification generation
    
    The compilation process validates pipeline structure, resolves component
    dependencies, and generates a Kubeflow-compatible YAML specification that
    Vertex AI can execute.

    Arguments:
        --py: Path to the Python file defining the pipeline.
        --output: Path to the output compiled YAML file.

    The script selects the pipeline function based on the filename provided in
    --py. If the filename contains 'vertex_pipeline_dev.py', it uses the dev
    pipeline. If it contains 'vertex_pipeline_prod.py', it uses the prod
    pipeline. If neither, it raises a ValueError.

    Compilation errors are caught and printed to stderr. The script exits with
    status code 1 on failure.
    
    TODO: Lab 5.4.1 - Component Identification: Compilation is a transformation component
    TODO: Lab 5.4.2 - Purpose Recognition: Converts high-level Python to executable YAML
    TODO: Lab 5.4.3 - Architecture Understanding: Compilation enables pipeline portability
    """
    # ==========================================================================
    # ARGUMENT PARSING - Compilation Configuration Component
    # ==========================================================================
    # TODO: Lab 5.4.1 - Component Identification: Command-line arguments specify compilation inputs/outputs
    # TODO: Lab 5.4.2 - Purpose Recognition: Flexible CLI enables CI/CD integration
    
    parser = argparse.ArgumentParser(
        description="Compile a KFP v2 pipeline for Vertex AI."
    )
    
    # TODO: Lab 5.4.1 - Component Identification: --py argument identifies source pipeline definition
    # TODO: Lab 5.4.2 - Purpose Recognition: Python file contains pipeline component definitions
    # TODO: Lab 5.4.3 - Architecture Understanding: Python definitions use KFP v2 SDK for component specification
    parser.add_argument(
        "--py",
        type=str,
        required=True,
        help="Path to the Python file defining the pipeline."
    )
    
    # TODO: Lab 5.4.1 - Component Identification: --output argument specifies compiled YAML destination
    # TODO: Lab 5.4.2 - Purpose Recognition: YAML format is Kubeflow-compatible pipeline specification
    # TODO: Lab 5.4.3 - Architecture Understanding: YAML spec is executable blueprint for Vertex AI
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the output compiled YAML file."
    )
    args = parser.parse_args()

    # ==========================================================================
    # PIPELINE SELECTION - Environment-Specific Component Selection
    # ==========================================================================
    # TODO: Lab 5.4.1 - Component Identification: Pipeline function selection based on environment
    # TODO: Lab 5.4.2 - Purpose Recognition: Different environments require different pipeline configurations
    # TODO: Lab 5.4.3 - Architecture Understanding: Pipeline variants share architecture but differ in parameters
    # TODO: Lab 5.5.1 - Custom Components: Both dev and prod pipelines contain custom components
    # TODO: Lab 5.5.4 - Component Reusability: Same custom components compiled for different environments
    
    # Select pipeline function based on filename.
    if "vertex_pipeline_dev.py" in args.py:
        # TODO: Lab 5.4.1 - Component Identification: Development pipeline function
        # TODO: Lab 5.4.2 - Purpose Recognition: Dev pipeline optimized for experimentation and testing
        # TODO: Lab 5.5.1 - Custom Components: Dev pipeline uses custom components
        pipeline_func = dev_diabetes_pipeline
    elif "vertex_pipeline_prod.py" in args.py:
        # TODO: Lab 5.4.1 - Component Identification: Production pipeline function
        # TODO: Lab 5.4.2 - Purpose Recognition: Prod pipeline optimized for reliability and monitoring
        # TODO: Lab 5.5.1 - Custom Components: Prod pipeline uses same custom components as dev
        # TODO: Lab 5.5.4 - Component Reusability: Demonstrates component portability across environments
        pipeline_func = prod_diabetes_pipeline
    else:
        # TODO: Lab 5.4.2 - Purpose Recognition: Validation ensures only recognized pipelines are compiled
        # Raise error if filename does not match expected dev/prod pipeline files.
        raise ValueError(
            f"Unknown pipeline file specified: {args.py}. "
            "Must contain 'vertex_pipeline_dev.py' or "
            "'vertex_pipeline_prod.py'."
        )

    # ==========================================================================
    # PIPELINE COMPILATION - Transform Python to YAML Component
    # ==========================================================================
    # TODO: Lab 5.4.1 - Component Identification: KFP Compiler is the transformation engine
    # TODO: Lab 5.4.2 - Purpose Recognition: Compiler validates structure and generates executable spec
    # TODO: Lab 5.4.3 - Architecture Understanding: Compilation process overview
    #
    # COMPILATION ARCHITECTURE:
    # ========================
    # 1. VALIDATION PHASE:
    #    - Validates component definitions and signatures
    #    - Checks parameter types and dependencies
    #    - Verifies pipeline structure (no cycles, valid inputs/outputs)
    #
    # 2. TRANSFORMATION PHASE:
    #    - Converts Python component definitions to Kubeflow components
    #    - Resolves component dependencies and execution order
    #    - Generates container specifications for each component
    #
    # 3. SERIALIZATION PHASE:
    #    - Serializes pipeline structure to YAML format
    #    - Embeds component specifications and metadata
    #    - Creates Kubeflow-compatible pipeline specification
    #
    # 4. OUTPUT PHASE:
    #    - Writes compiled YAML to specified output path
    #    - YAML spec is ready for Vertex AI submission
    
    # TODO: Lab 5.6.1 — Compilation and YAML structure
    # INSPECT: This is the core compilation call. Locate the compile() method and note its parameters.
    # TASK: Note which pipeline file it references (pipeline_func - selected above based on --py argument).
    # TASK: Note the output path (package_path - from --output argument).
    # TASK: After compilation, open the generated YAML file to inspect its structure.
    # TASK: In the YAML, identify sections: components, deploymentSpec, pipelineInfo, root, schemaVersion.
    
    try:
        # TODO: Lab 5.4.1 - Component Identification: compiler.Compiler() is the KFP v2 compilation component
        # TODO: Lab 5.4.2 - Purpose Recognition: .compile() method performs complete pipeline transformation
        # TODO: Lab 5.4.3 - Architecture Understanding: Compilation creates portable pipeline specification
        
        # Compile the selected pipeline function to the specified YAML output.
        compiler.Compiler().compile(
            # TODO: Lab 5.4.1 - Component Identification: pipeline_func is the Python pipeline definition
            # TODO: Lab 5.4.3 - Architecture Understanding: Pipeline function contains all component definitions
            pipeline_func=pipeline_func,
            
            # TODO: Lab 5.4.1 - Component Identification: package_path is the output YAML specification
            # TODO: Lab 5.4.2 - Purpose Recognition: YAML format enables Vertex AI execution
            # TODO: Lab 5.4.3 - Architecture Understanding: Compiled spec is platform-independent
            package_path=args.output
        )
        print(f"✅ Successfully compiled '{args.py}' → '{args.output}'")
    except Exception as e:
        # TODO: Lab 5.4.2 - Purpose Recognition: Compilation errors indicate structural or dependency issues
        
        # TODO: Lab 5.6.7 — Failure modes and debugging tips
        # INSPECT: Exception handling for compilation failures.
        # TASK: Note common compilation errors: invalid component definitions, type mismatches, circular dependencies.
        # DEBUGGING TIP: Read exception message for specific error; check component decorators and type hints.
        
        # Print compilation errors to stderr and exit with status code 1.
        print(f"❌ Failed to compile pipeline: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    """
    Entry point for script execution.
    
    This script demonstrates the complete pipeline compilation architecture:
    
    PIPELINE COMPILATION ARCHITECTURE:
    ==================================
    1. INPUT COMPONENTS:
       - Python pipeline definitions (dev/prod variants)
       - KFP v2 component specifications
       - Pipeline parameters and dependencies
    
    2. COMPILATION COMPONENTS:
       - KFP v2 Compiler (transformation engine)
       - Validation logic (structure and dependency checking)
       - Serialization engine (Python to YAML conversion)
    
    3. OUTPUT COMPONENTS:
       - YAML pipeline specification (Kubeflow-compatible)
       - Component container specifications
       - Pipeline metadata and parameter definitions
    
    4. ARCHITECTURE BENEFITS:
       - Python-to-YAML compilation enables version control
       - Compiled specs are portable across Vertex AI projects
       - Validation at compile-time prevents runtime failures
       - Clear separation between definition and execution
    
    WHAT COMPILATION VALIDATES:
    ===========================
    - Component input/output type compatibility
    - Parameter type consistency
    - Pipeline structure (no circular dependencies)
    - Resource specifications (memory, CPU)
    - Container image availability
    - Component interface contracts
    
    COMPILED YAML SPECIFICATION CONTAINS:
    =====================================
    - Complete component definitions
    - Dependency graph (execution order)
    - Parameter schemas and defaults
    - Container specifications per component
    - Pipeline metadata (name, description)
    - Resource requirements per component
    
    CUSTOM vs PRE-BUILT COMPONENT COMPILATION:
    ==========================================
    Custom Components (Current Pipelines):
    - @component decorator definitions compiled to containers
    - Base image and packages embedded in YAML
    - Component code included in specification
    
    Pre-built Components (If Used):
    - Library imports compiled to component references
    - Google-maintained container images referenced
    - No custom container building required
    - Same validation and YAML generation process
    
    ACCELERATOR TEMPLATE WORKFLOW:
    =============================
    Step 1: terraform apply (create infrastructure)
    Step 2: python compiler.py (THIS SCRIPT - compile pipeline)
    Step 3: python run_pipeline.py (submit to Vertex AI using Terraform infrastructure)
    
    Compilation is independent of Terraform but compiled pipelines will
    eventually use Terraform-created infrastructure at runtime.
    
    TODO: Lab 5.4.3 - Architecture Understanding: Compilation is the bridge
          between pipeline development (Python) and pipeline execution (YAML)
    TODO: Lab 5.5.1 - Custom Components: Compiles custom component definitions
    TODO: Lab 5.5.2 - Pre-built Components: Would compile pre-built components identically
    TODO: Lab 5.5.3 - Accelerator Templates: Compilation is step 2 in the 3-step deployment
    """
    # Entry point for script execution.
    main()
