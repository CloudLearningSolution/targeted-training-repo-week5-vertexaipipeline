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
Lab 5.6: Code-Only Exploration - Orchestration, CI/CD, and Pipeline Compilation
===============================================================================
This script is a key artifact for understanding pipeline orchestration through
code inspection. Lab 5.6 focuses on READ-ONLY exploration without execution.

WHAT TO INSPECT IN THIS FILE:
=============================
1. Pipeline function imports and selection logic
2. Compilation call structure and parameters
3. Output YAML path and naming conventions
4. Error handling for compilation failures
5. How this script integrates into CI/CD workflows

LAB 5.6 INSPECTION GOALS:
=========================
- Understand how Python pipeline definitions become YAML specifications
- Identify which pipeline parameters are available at compilation time
- Recognize the relationship between source .py files and compiled .yaml files
- Document the compilation process flow for your deliverables

DELIVERABLE HINTS FOR THIS FILE:
================================
- components-observations.md: Note the compile() call and its parameters
- pipeline-params.md: Identify which parameters are in pipeline function signatures
- ci-cd-inspection.md: Document how this script is called in CI/CD workflows
- orchestration-notes.md: Record compilation as a pre-execution orchestration step

===============================================================================
"""

import argparse
import sys
from typing import Callable
from kfp import compiler

# TODO: Lab 5.4.1 - Component Identification: Pipeline function imports define available pipelines
# TODO: Lab 5.4.2 - Purpose Recognition: Separate dev/prod pipeline definitions enable environment-specific configurations
# TODO: Lab 5.4.3 - Architecture Understanding: Pipeline functions are the high-level component definitions

# TODO: Lab 5.6.1 — COMPILATION AND YAML STRUCTURE
# INSPECT: These imports define which pipeline functions are available for compilation.
# QUESTION: What are the names of the two pipeline functions imported here?
# QUESTION: How does the compiler know which function to compile (dev vs prod)?
# DELIVERABLE: Record these function names in pipeline-params.md under "Available Pipeline Functions"
# HINT: The function names imported here must match the function definitions in the pipeline files.

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
    
    # TODO: Lab 5.6.1 — COMPILATION AND YAML STRUCTURE
    # INSPECT: The argparse setup defines how this script receives input.
    # QUESTION: What are the two required arguments and what do they represent?
    # QUESTION: How would a CI/CD workflow call this script? (Example command format)
    # DELIVERABLE: Document this in ci-cd-inspection.md under "Compilation Step Command"
    # HINT: The --py argument points to the source, --output points to the destination.
    
    parser = argparse.ArgumentParser(
        description="Compile a KFP v2 pipeline for Vertex AI."
    )
    
    # TODO: Lab 5.4.1 - Component Identification: --py argument identifies source pipeline definition
    # TODO: Lab 5.4.2 - Purpose Recognition: Python file contains pipeline component definitions
    # TODO: Lab 5.4.3 - Architecture Understanding: Python definitions use KFP v2 SDK for component specification
    
    # TODO: Lab 5.6.1 — COMPILATION AND YAML STRUCTURE
    # INSPECT: The --py argument specifies the SOURCE pipeline Python file.
    # OBSERVATION: This path is used in the pipeline selection logic below.
    # DELIVERABLE: Note in components-observations.md that compilation requires source file path.
    
    parser.add_argument(
        "--py",
        type=str,
        required=True,
        help="Path to the Python file defining the pipeline."
    )
    
    # TODO: Lab 5.4.1 - Component Identification: --output argument specifies compiled YAML destination
    # TODO: Lab 5.4.2 - Purpose Recognition: YAML format is Kubeflow-compatible pipeline specification
    # TODO: Lab 5.4.3 - Architecture Understanding: YAML spec is executable blueprint for Vertex AI
    
    # TODO: Lab 5.6.1 — COMPILATION AND YAML STRUCTURE
    # INSPECT: The --output argument specifies the DESTINATION for compiled YAML.
    # OBSERVATION: This YAML file becomes the input to run_pipeline.py.
    # QUESTION: What naming convention should be used for output files? (dev vs prod)
    # DELIVERABLE: Document the expected output file names in pipeline-params.md.
    # HINT: Output file names typically mirror the input file names (e.g., vertex_pipeline_dev.yaml)
    
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
    
    # TODO: Lab 5.6.1 — COMPILATION AND YAML STRUCTURE
    # INSPECT: This selection logic determines WHICH pipeline function to compile.
    # OBSERVATION: The logic uses string matching on the --py argument value.
    # QUESTION: What happens if neither "vertex_pipeline_dev.py" nor "vertex_pipeline_prod.py" is found?
    # QUESTION: Could this logic be improved? (Consider edge cases and error messages)
    # DELIVERABLE: Document this selection mechanism in components-observations.md.
    # BEST PRACTICE: String-based selection is simple but fragile; consider explicit flags in production.
    
    # Select pipeline function based on filename.
    if "vertex_pipeline_dev.py" in args.py:
        # TODO: Lab 5.4.1 - Component Identification: Development pipeline function
        # TODO: Lab 5.4.2 - Purpose Recognition: Dev pipeline optimized for experimentation and testing
        # TODO: Lab 5.5.1 - Custom Components: Dev pipeline uses custom components
        
        # TODO: Lab 5.6.1 — COMPILATION AND YAML STRUCTURE
        # INSPECT: This branch selects the DEVELOPMENT pipeline function.
        # OBSERVATION: dev_diabetes_pipeline is the function name to compile.
        # ACTION: Open vertex_pipeline_dev.py and locate the dev_diabetes_pipeline function definition.
        # ACTION: Note the function signature and parameters in pipeline-params.md.
        
        pipeline_func = dev_diabetes_pipeline
    elif "vertex_pipeline_prod.py" in args.py:
        # TODO: Lab 5.4.1 - Component Identification: Production pipeline function
        # TODO: Lab 5.4.2 - Purpose Recognition: Prod pipeline optimized for reliability and monitoring
        # TODO: Lab 5.5.1 - Custom Components: Prod pipeline uses same custom components as dev
        # TODO: Lab 5.5.4 - Component Reusability: Demonstrates component portability across environments
        
        # TODO: Lab 5.6.1 — COMPILATION AND YAML STRUCTURE
        # INSPECT: This branch selects the PRODUCTION pipeline function.
        # OBSERVATION: prod_diabetes_pipeline is the function name to compile.
        # ACTION: Open vertex_pipeline_prod.py and locate the prod_diabetes_pipeline function definition.
        # ACTION: Compare dev vs prod function signatures in pipeline-params.md.
        # HINT: Note any parameter differences between dev and prod environments.
        
        pipeline_func = prod_diabetes_pipeline
    else:
        # TODO: Lab 5.4.2 - Purpose Recognition: Validation ensures only recognized pipelines are compiled
        
        # TODO: Lab 5.6.7 — FAILURE MODES AND DEBUGGING
        # INSPECT: This error handling shows a validation failure mode.
        # OBSERVATION: Script exits with clear error message if pipeline file is unrecognized.
        # DEBUGGING TIP: If you see this error, check the --py argument value for typos.
        # BEST PRACTICE: Early validation prevents wasted compilation time on invalid inputs.
        # DELIVERABLE: Document this failure mode in orchestration-notes.md under "Compilation Errors"
        
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
    
    # TODO: Lab 5.6.1 — COMPILATION AND YAML STRUCTURE
    # INSPECT: This is the CORE compilation call that transforms Python to YAML.
    # CRITICAL OBSERVATION: compiler.Compiler().compile() is the KFP v2 compilation engine.
    # QUESTION: What are the two parameters passed to the compile() method?
    # QUESTION: What format is the output file (YAML)? What does it contain?
    # ACTION: After compilation, open the generated .yaml file and inspect its structure.
    # ACTION: Note the sections: components, deploymentSpec, pipelineInfo, root, schemaVersion.
    # DELIVERABLE: In components-observations.md, list the key sections of compiled YAML.
    # DELIVERABLE: In pipeline-params.md, document which parameters appear in the YAML.
    # HINT: The YAML file is a complete executable specification - it contains ALL pipeline information.
    
    try:
        # TODO: Lab 5.4.1 - Component Identification: compiler.Compiler() is the KFP v2 compilation component
        # TODO: Lab 5.4.2 - Purpose Recognition: .compile() method performs complete pipeline transformation
        # TODO: Lab 5.4.3 - Architecture Understanding: Compilation creates portable pipeline specification
        
        # TODO: Lab 5.6.1 — COMPILATION AND YAML STRUCTURE
        # INSPECT: The compile() method call with its two critical parameters.
        # OBSERVATION 1: pipeline_func is the Python function to compile (dev or prod selected above).
        # OBSERVATION 2: package_path is the output YAML file path.
        # COMPILATION PROCESS: Python function → KFP validation → YAML serialization → File write.
        # DELIVERABLE: Document this compilation command format in ci-cd-inspection.md.
        # QUESTION: Does compilation require GCP credentials? (Answer: No - it's a local transformation)
        # QUESTION: Does compilation execute any pipeline code? (Answer: No - it only analyzes structure)
        
        # Compile the selected pipeline function to the specified YAML output.
        compiler.Compiler().compile(
            # TODO: Lab 5.4.1 - Component Identification: pipeline_func is the Python pipeline definition
            # TODO: Lab 5.4.3 - Architecture Understanding: Pipeline function contains all component definitions
            
            # TODO: Lab 5.6.1 — COMPILATION AND YAML STRUCTURE
            # INSPECT: pipeline_func parameter - the Python function object to compile.
            # OBSERVATION: This is the actual function object (not a string), imported at the top of this file.
            # ACTION: In vertex_pipeline_dev.py or vertex_pipeline_prod.py, find this function definition.
            # ACTION: Note the @dsl.pipeline decorator and function signature.
            # HINT: The function signature parameters become pipeline parameters in the YAML.
            
            pipeline_func=pipeline_func,
            
            # TODO: Lab 5.4.1 - Component Identification: package_path is the output YAML specification
            # TODO: Lab 5.4.2 - Purpose Recognition: YAML format enables Vertex AI execution
            # TODO: Lab 5.4.3 - Architecture Understanding: Compiled spec is platform-independent
            
            # TODO: Lab 5.6.1 — COMPILATION AND YAML STRUCTURE
            # INSPECT: package_path parameter - where the compiled YAML will be written.
            # OBSERVATION: This path comes from the --output CLI argument.
            # FILE FLOW: Source .py file → compiler.py (this script) → Output .yaml file
            # NEXT STEP: The .yaml file is then used by run_pipeline.py to submit to Vertex AI.
            # DELIVERABLE: Document this file flow in orchestration-notes.md under "Compilation Flow"
            
            package_path=args.output
        )
        
        # TODO: Lab 5.6.7 — FAILURE MODES AND DEBUGGING
        # INSPECT: Success message after successful compilation.
        # DEBUGGING TIP: If you see this message, the YAML file was created successfully.
        # NEXT DEBUGGING STEP: Inspect the YAML file to verify it contains expected components.
        # CI/CD SIGNAL: This success message can be parsed in CI/CD logs to confirm compilation worked.
        
        print(f"✅ Successfully compiled '{args.py}' → '{args.output}'")
    except Exception as e:
        # TODO: Lab 5.4.2 - Purpose Recognition: Compilation errors indicate structural or dependency issues
        
        # TODO: Lab 5.6.7 — FAILURE MODES AND DEBUGGING
        # INSPECT: Exception handling for compilation failures.
        # COMMON FAILURE MODES:
        # 1. Invalid component definitions (missing inputs/outputs, type mismatches)
        # 2. Circular dependencies between components
        # 3. Invalid parameter types or missing decorators
        # 4. Import errors (missing pipeline function or component definitions)
        # 5. Syntax errors in pipeline Python files
        # DEBUGGING TIPS:
        # - Read the exception message carefully - it usually indicates the exact issue
        # - Check that all @component and @dsl.pipeline decorators are correct
        # - Verify component input/output types match between components
        # - Ensure all required imports are present
        # - Test pipeline files independently before compiling
        # CI/CD IMPACT: Compilation failures will stop the CI/CD pipeline early (before execution costs)
        # DELIVERABLE: Document these common failure modes in orchestration-notes.md.
        
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
    
    TODO: Lab 5.6.1 — COMPILATION AND YAML STRUCTURE
    INSPECT: This docstring provides comprehensive context for the compilation process.
    KEY LEARNING POINTS:
    1. Compilation is a LOCAL operation - no GCP credentials needed
    2. Compilation happens BEFORE execution - it's a build step, not a runtime step
    3. YAML output is PORTABLE - can be version-controlled and shared
    4. Compilation VALIDATES structure early - prevents costly runtime failures
    5. Compiled YAML contains EVERYTHING needed to run the pipeline
    
    DELIVERABLE CHECKLIST FOR THIS FILE:
    ====================================
    □ components-observations.md: Document compile() call and parameters
    □ pipeline-params.md: List pipeline function names and parameters
    □ ci-cd-inspection.md: Document how compilation fits in CI/CD workflow
    □ orchestration-notes.md: Note compilation as pre-execution step
    
    NEXT INSPECTION STEPS:
    ======================
    1. Open vertex_pipeline_dev.py and find dev_diabetes_pipeline function
    2. Open vertex_pipeline_prod.py and find prod_diabetes_pipeline function
    3. Compare the two pipeline function signatures
    4. Open a compiled .yaml file and inspect its structure
    5. Open run_pipeline.py to see how compiled YAML is used for submission
    """
    # Entry point for script execution.
    main()
