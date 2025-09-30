"""
Compiles Kubeflow Pipelines (KFP v2) into YAML pipeline specs for Vertex AI.
YAML pipeline specs are templates used in accelerator templates.
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
"""

import argparse
import sys
from typing import Callable
from kfp import compiler

# TODO: Lab 5.4.1 - Component Identification: Pipeline function imports define available pipelines
# TODO: Lab 5.4.2 - Purpose Recognition: Separate dev/prod pipeline definitions enable environment-specific configurations
# TODO: Lab 5.4.3 - Architecture Understanding: Pipeline functions are the high-level component definitions

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
    
    # Select pipeline function based on filename.
    if "vertex_pipeline_dev.py" in args.py:
        # TODO: Lab 5.4.1 - Component Identification: Development pipeline function
        # TODO: Lab 5.4.2 - Purpose Recognition: Dev pipeline optimized for experimentation and testing
        pipeline_func = dev_diabetes_pipeline
    elif "vertex_pipeline_prod.py" in args.py:
        # TODO: Lab 5.4.1 - Component Identification: Production pipeline function
        # TODO: Lab 5.4.2 - Purpose Recognition: Prod pipeline optimized for reliability and monitoring
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
    
    TODO: Lab 5.4.3 - Architecture Understanding: Compilation is the bridge
          between pipeline development (Python) and pipeline execution (YAML)
    """
    # Entry point for script execution.
    main()
