"""
Deploy a Registered Model from Vertex AI Model Registry to an Endpoint

This script provides a complete workflow for deploying a trained and registered
diabetes prediction model to a Vertex AI endpoint for online inference serving.

Compatible with:
- google-cloud-aiplatform==1.56.0
- pandas==1.5.3
- numpy==1.23.5

Main Objectives:
1. Retrieve a registered model from Vertex AI Model Registry
2. Create or use an existing Vertex AI endpoint
3. Deploy the model to the endpoint with appropriate resource allocation
4. Test the deployed endpoint with sample diabetes prediction data
5. Provide monitoring and management capabilities

The script handles both new deployments and updates to existing endpoints,
with proper error handling and logging throughout the process.

===============================================================================
Lab 5.4: Vertex AI Pipeline Component Architecture Exploration
===============================================================================
This deployment script demonstrates how pipeline outputs integrate with Vertex AI
serving infrastructure. Understanding this script helps identify:
- How pipeline components (Model Registry) connect to serving components (Endpoints)
- The role of Model Registry as a bridge between pipeline and deployment
- Complete ML architecture from pipeline execution to model serving

TODO: Lab 5.4.1 - Component Identification: Model deployment and serving components
TODO: Lab 5.4.2 - Purpose Recognition: Pipeline-to-production architecture integration
TODO: Lab 5.4.3 - Architecture Understanding: Complete ML system architecture
===============================================================================
"""

import argparse
import json
import logging
import time
from typing import Dict, List, Optional, Tuple

from google.cloud import aiplatform
import pandas as pd


class ModelDeploymentManager:
    """
    Manages the deployment of registered models to Vertex AI endpoints.

    This class encapsulates all functionality needed to:
    - Find and validate registered models
    - Create or reuse endpoints
    - Deploy models with proper configuration
    - Test deployments with sample data
    
    TODO: Lab 5.4.1 - Component Identification: Deployment manager orchestrates serving components
    TODO: Lab 5.4.2 - Purpose Recognition: Bridges pipeline outputs to serving infrastructure
    TODO: Lab 5.4.3 - Architecture Understanding: Deployment is final stage of ML pipeline architecture
    """

    def __init__(self, project_id: str, region: str):
        """
        Initialize the deployment manager with GCP credentials.
        
        TODO: Lab 5.4.1 - Component Identification: Deployment manager initialization
        TODO: Lab 5.4.2 - Purpose Recognition: Establishes connection to Vertex AI services
        """
        self.project_id = project_id
        self.region = region
        # TODO: Lab 5.4.1 - Component Identification: aiplatform.init() configures Vertex AI SDK
        # TODO: Lab 5.4.3 - Architecture Understanding: SDK initialization enables service integration
        aiplatform.init(project=project_id, location=region)
        
        # TODO: Lab 5.6.5 — Observability: metrics, logging, and dashboard links
        # INSPECT: logging.basicConfig() sets up deployment script logging.
        # NOTE: This logging configuration applies to the entire deployment process.
        # OBSERVATION: Uses timestamp, level, and message format for structured logging.
        # DELIVERABLE: Document this logging pattern in observability.md.
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def find_latest_model(self, model_display_name: str) -> aiplatform.Model:
        """
        Retrieve the latest registered model from Vertex AI Model Registry.
        
        TODO: Lab 5.4.1 - Component Identification: Model Registry query component
        TODO: Lab 5.4.2 - Purpose Recognition: Locates pipeline-registered models for deployment
        TODO: Lab 5.4.3 - Architecture Understanding: Model Registry bridges pipeline and serving
        
        Args:
            model_display_name: Display name of the model to find
            
        Returns:
            aiplatform.Model: Latest registered model instance
        """
        # TODO: Lab 5.6.5 — Observability: metrics, logging, and dashboard links
        # INSPECT: Logging pattern for model search operations.
        # OBSERVATION: Logs model search start for debugging and audit trail.
        # TASK: Count all self.logger calls in this method (there are 2).
        
        self.logger.info(f"Searching for model: {model_display_name}")
        # TODO: Lab 5.4.1 - Component Identification: Model.list() queries Model Registry
        # TODO: Lab 5.4.2 - Purpose Recognition: Retrieves models registered by pipeline components
        # TODO: Lab 5.4.3 - Architecture Understanding: Model Registry is central component repository
        models = aiplatform.Model.list(filter=f'display_name="{model_display_name}"')
        if not models:
            raise ValueError(
                f"No models found with display name: {model_display_name}. "
                f"Ensure the model is registered in the Model Registry."
            )
        # TODO: Lab 5.4.2 - Purpose Recognition: Latest model selection for production deployment
        latest_model = max(models, key=lambda m: m.create_time)
        
        # TODO: Lab 5.6.5 — Observability: metrics, logging, and dashboard links
        # INSPECT: Logging pattern for model discovery success.
        # OBSERVATION: Logs model details (display_name, resource_name, create_time).
        # PURPOSE: Provides audit trail of which model version was selected for deployment.
        # DELIVERABLE: Document this observability pattern in observability.md.
        
        self.logger.info(
            f"Found latest model: {latest_model.display_name} "
            f"(Resource: {latest_model.resource_name}, "
            f"Created: {latest_model.create_time})"
        )
        return latest_model

    def create_or_get_endpoint(self, endpoint_display_name: str) -> aiplatform.Endpoint:
        """
        Create a new Vertex AI endpoint or retrieve an existing one.
        
        TODO: Lab 5.4.1 - Component Identification: Endpoint creation/retrieval component
        TODO: Lab 5.4.2 - Purpose Recognition: Endpoints serve deployed models for online inference
        TODO: Lab 5.4.3 - Architecture Understanding: Endpoints are serving layer in ML architecture
        
        Args:
            endpoint_display_name: Name for the endpoint
            
        Returns:
            aiplatform.Endpoint: Endpoint instance for model deployment
        """
        # TODO: Lab 5.6.5 — Observability: metrics, logging, and dashboard links
        # INSPECT: Logging for endpoint lookup operations.
        # OBSERVATION: Logs endpoint search start.
        # TASK: Count all self.logger calls in this method (there are 3).
        
        self.logger.info(f"Looking for existing endpoint: {endpoint_display_name}")
        # TODO: Lab 5.4.1 - Component Identification: Endpoint.list() queries existing endpoints
        # TODO: Lab 5.4.2 - Purpose Recognition: Reuse existing endpoints when available
        existing_endpoints = aiplatform.Endpoint.list(
            filter=f'display_name="{endpoint_display_name}"'
        )
        if existing_endpoints:
            endpoint = existing_endpoints[0]
            
            # TODO: Lab 5.6.5 — Observability: metrics, logging, and dashboard links
            # INSPECT: Logging for existing endpoint reuse.
            # OBSERVATION: Logs endpoint resource_name for tracking.
            # PURPOSE: Audit trail showing endpoint was reused (not newly created).
            
            self.logger.info(f"Using existing endpoint: {endpoint.resource_name}")
            return endpoint
        # TODO: Lab 5.4.1 - Component Identification: Endpoint.create() provisions new serving endpoint
        # TODO: Lab 5.4.2 - Purpose Recognition: New endpoints for first-time deployments
        # TODO: Lab 5.4.3 - Architecture Understanding: Endpoints are managed serving infrastructure
        
        # TODO: Lab 5.6.5 — Observability: metrics, logging, and dashboard links
        # INSPECT: Logging for new endpoint creation.
        # OBSERVATION: Logs when creating NEW endpoint (vs reusing existing).
        
        self.logger.info(f"Creating new endpoint: {endpoint_display_name}")
        endpoint = aiplatform.Endpoint.create(
            display_name=endpoint_display_name,
            description=f"Endpoint for {endpoint_display_name} diabetes prediction model",
            # TODO: Lab 5.4.1 - Component Identification: Labels enable endpoint organization and tracking
            
            # TODO: Lab 5.6.5 — Observability: metrics, logging, and dashboard links
            # INSPECT: Labels for endpoint categorization and filtering.
            # OBSERVATION: Labels enable monitoring dashboard filtering and cost attribution.
            # LABELS: model_type, environment, managed_by.
            # PURPOSE: Enables filtering in Vertex AI console and cost tracking.
            # DELIVERABLE: Document label usage in observability.md.
            
            labels={
                "model_type": "diabetes_classifier",
                "environment": "production",
                "managed_by": "vertex_ai_pipeline"
            }
        )
        
        # TODO: Lab 5.6.5 — Observability: metrics, logging, and dashboard links
        # INSPECT: Logging for successful endpoint creation.
        # OBSERVATION: Logs endpoint resource_name for tracking and monitoring.
        
        self.logger.info(f"Created endpoint: {endpoint.resource_name}")
        return endpoint

    def deploy_model_to_endpoint(
        self,
        model: aiplatform.Model,
        endpoint: aiplatform.Endpoint,
        deployment_name: str,
        machine_type: str = "n1-standard-2",
        min_replica_count: int = 1,
        max_replica_count: int = 3,
        traffic_percentage: int = 100
    ) -> Dict:
        """
        Deploy a registered model to an endpoint with resource configuration.
        
        TODO: Lab 5.4.1 - Component Identification: Model deployment component
        TODO: Lab 5.4.2 - Purpose Recognition: Connects pipeline outputs to serving infrastructure
        TODO: Lab 5.4.3 - Architecture Understanding: Deployment completes pipeline-to-production flow
        
        Args:
            model: Registered model from Model Registry
            endpoint: Target endpoint for deployment
            deployment_name: Name for this deployment
            machine_type: Compute resources for serving
            min_replica_count: Minimum serving replicas
            max_replica_count: Maximum serving replicas
            traffic_percentage: Percentage of traffic to this deployment
            
        Returns:
            Dict: Deployment configuration details
        """
        # TODO: Lab 5.6.5 — Observability: metrics, logging, and dashboard links
        # INSPECT: Logging for deployment initiation.
        # OBSERVATION: Logs both model and endpoint names for audit trail.
        # TASK: Count all self.logger calls in this method (there are 2 or 3 depending on warnings).
        
        self.logger.info(f"Deploying model {model.display_name} to endpoint {endpoint.display_name}")
        # TODO: Lab 5.4.1 - Component Identification: endpoint.deploy() creates model deployment
        # TODO: Lab 5.4.2 - Purpose Recognition: Deployment makes pipeline models available for inference
        # TODO: Lab 5.4.3 - Architecture Understanding: Deployment links pipeline output to serving layer
        endpoint.deploy(
            model=model,
            deployed_model_display_name=deployment_name,
            # TODO: Lab 5.4.1 - Component Identification: Resource configuration for serving
            # TODO: Lab 5.4.2 - Purpose Recognition: Machine type defines serving capacity
            machine_type=machine_type,
            # TODO: Lab 5.4.1 - Component Identification: Auto-scaling configuration
            # TODO: Lab 5.4.2 - Purpose Recognition: Replica counts enable scalable serving
            min_replica_count=min_replica_count,
            max_replica_count=max_replica_count,
            # TODO: Lab 5.4.1 - Component Identification: Traffic allocation for A/B testing
            # TODO: Lab 5.4.2 - Purpose Recognition: Enables gradual rollout and testing
            traffic_percentage=traffic_percentage,
            sync=True,
        )
        
        # TODO: Lab 5.6.5 — Observability: metrics, logging, and dashboard links
        # INSPECT: Logging for successful deployment completion.
        # OBSERVATION: Confirms deployment succeeded before returning details.
        # PURPOSE: Clear success signal in logs for monitoring and alerting.
        
        self.logger.info("Model deployment completed successfully")
        deployed_model_id = None
        try:
            endpoint._sync_gca_resource()
            for deployed_model in endpoint._gca_resource.deployed_models:
                if deployed_model.display_name == deployment_name:
                    deployed_model_id = deployed_model.id
                    break
        except Exception as e:
            # TODO: Lab 5.6.5 — Observability: metrics, logging, and dashboard links
            # INSPECT: Warning logging for non-critical errors.
            # OBSERVATION: Uses self.logger.warning() for recoverable issues.
            # PURPOSE: Logs issue but doesn't fail deployment if model ID can't be retrieved.
            
            self.logger.warning(f"Could not retrieve deployed model ID: {e}")
            deployed_model_id = "unknown"
        # TODO: Lab 5.4.1 - Component Identification: Deployment metadata for tracking
        
        # TODO: Lab 5.6.5 — Observability: metrics, logging, and dashboard links
        # INSPECT: Deployment metadata returned for monitoring and tracking.
        # OBSERVATION: Returns structured dict with all deployment configuration.
        # PURPOSE: Enables monitoring dashboards and deployment tracking.
        # DELIVERABLE: Document this metadata structure in observability.md.
        
        deployment_details = {
            "endpoint_id": endpoint.name,
            "deployed_model_id": deployed_model_id,
            "machine_type": machine_type,
            "replica_count": f"{min_replica_count}-{max_replica_count}",
            "traffic_percentage": traffic_percentage,
            "status": "deployed"
        }
        return deployment_details

    def test_endpoint_prediction(
        self,
        endpoint: aiplatform.Endpoint,
        test_instances: List[List]
    ) -> Tuple[List, bool]:
        """
        Test endpoint with sample data to verify deployment.
        
        TODO: Lab 5.4.1 - Component Identification: Endpoint testing component
        TODO: Lab 5.4.2 - Purpose Recognition: Validates deployment functionality
        TODO: Lab 5.4.3 - Architecture Understanding: Testing verifies complete pipeline-to-serving flow
        
        Args:
            endpoint: Deployed endpoint to test
            test_instances: Sample data for testing
            
        Returns:
            Tuple[List, bool]: Predictions and success status
        """
        # TODO: Lab 5.6.5 — Observability: metrics, logging, and dashboard links
        # INSPECT: Logging for endpoint testing operations.
        # OBSERVATION: Logs test start for deployment validation tracking.
        # TASK: Count all self.logger calls in this method (there are 4-5).
        
        self.logger.info("Testing endpoint with sample diabetes data...")
        try:
            # TODO: Lab 5.4.1 - Component Identification: endpoint.predict() invokes deployed model
            # TODO: Lab 5.4.2 - Purpose Recognition: Validates end-to-end inference capability
            # TODO: Lab 5.4.3 - Architecture Understanding: Prediction verifies complete ML architecture
            predictions = endpoint.predict(instances=test_instances)
            if predictions and predictions.predictions:
                # TODO: Lab 5.6.5 — Observability: metrics, logging, and dashboard links
                # INSPECT: Logging for successful endpoint test.
                # OBSERVATION: Logs success message and sample predictions.
                # PURPOSE: Validates deployment is functional and returns expected results.
                
                self.logger.info("Endpoint responding correctly")
                for i, prediction in enumerate(predictions.predictions[:3]):
                    # TODO: Lab 5.6.5 — Observability: metrics, logging, and dashboard links
                    # INSPECT: Logging pattern for sample predictions.
                    # OBSERVATION: Logs first 3 predictions for spot-checking results.
                    # PURPOSE: Quick validation that predictions look reasonable.
                    
                    self.logger.info(f"Sample prediction {i+1}: {prediction}")
                return predictions.predictions, True
            else:
                # TODO: Lab 5.6.5 — Observability: metrics, logging, and dashboard links
                # INSPECT: Error logging for empty prediction response.
                # OBSERVATION: Uses self.logger.error() for test failures.
                
                self.logger.error("Endpoint returned empty predictions")
                return [], False
        except Exception as e:
            # TODO: Lab 5.6.5 — Observability: metrics, logging, and dashboard links
            # INSPECT: Exception logging for endpoint test failures.
            # OBSERVATION: Logs exception details for debugging.
            # PURPOSE: Critical for diagnosing deployment issues.
            # DELIVERABLE: Document error logging pattern in observability.md.
            
            self.logger.error(f"Endpoint test failed: {str(e)}")
            return [], False

    def get_endpoint_info(self, endpoint: aiplatform.Endpoint) -> Dict:
        """
        Retrieve detailed information about an endpoint and its deployments.
        
        TODO: Lab 5.4.1 - Component Identification: Endpoint metadata retrieval component
        TODO: Lab 5.4.2 - Purpose Recognition: Provides deployment visibility and monitoring data
        
        Args:
            endpoint: Endpoint to inspect
            
        Returns:
            Dict: Comprehensive endpoint information
        """
        # TODO: Lab 5.6.5 — Observability: metrics, logging, and dashboard links
        # INSPECT: Logging for endpoint information retrieval.
        # OBSERVATION: Logs endpoint inspection operation.
        # TASK: Count all self.logger calls in this method (there is 1).
        
        self.logger.info(f"Retrieving endpoint information: {endpoint.display_name}")
        # TODO: Lab 5.4.1 - Component Identification: Endpoint metadata structure
        # TODO: Lab 5.4.2 - Purpose Recognition: Metadata enables monitoring and management
        
        # TODO: Lab 5.6.5 — Observability: metrics, logging, and dashboard links
        # INSPECT: Endpoint metadata structure for monitoring.
        # OBSERVATION: Returns structured dict with endpoint and deployment details.
        # PURPOSE: Enables monitoring dashboards and operational visibility.
        # FIELDS: endpoint_name, endpoint_id, create_time, region, project_id, deployed_models.
        # DELIVERABLE: Document this metadata structure in observability.md for monitoring setup.
        
        endpoint_info = {
            "endpoint_name": endpoint.display_name,
            "endpoint_id": endpoint.name,
            "create_time": str(endpoint.create_time),
            "region": self.region,
            "project_id": self.project_id,
            "deployed_models": []
        }
        # TODO: Lab 5.4.1 - Component Identification: Deployed model details for tracking
        
        # TODO: Lab 5.6.5 — Observability: metrics, logging, and dashboard links
        # INSPECT: Deployed model metadata collection.
        # OBSERVATION: Iterates through all deployed models on endpoint.
        # PURPOSE: Captures configuration for monitoring and capacity planning.
        # METADATA COLLECTED: display_name, model_id, machine_type, min/max_replicas, deployed_model_id.
        
        for deployed_model in endpoint._gca_resource.deployed_models:
            model_info = {
                "model_display_name": deployed_model.display_name,
                "model_id": deployed_model.model,
                "machine_type": deployed_model.dedicated_resources.machine_spec.machine_type,
                "min_replicas": deployed_model.dedicated_resources.min_replica_count,
                "max_replicas": deployed_model.dedicated_resources.max_replica_count,
                "deployed_model_id": deployed_model.id
            }
            endpoint_info["deployed_models"].append(model_info)
        return endpoint_info


def create_sample_test_data() -> List[List]:
    """
    Generate sample diabetes prediction test data for endpoint validation.
    
    TODO: Lab 5.4.1 - Component Identification: Test data generation component
    TODO: Lab 5.4.2 - Purpose Recognition: Enables deployment validation without production data
    
    Returns:
        List[List]: Sample test instances as arrays of feature values
    """
    # TODO: Lab 5.4.2 - Purpose Recognition: Sample data matches pipeline training data schema
    sample_data = [
        [6, 148, 72, 35, 0, 33.6, 0.627, 50],
        [1, 85, 66, 29, 0, 26.6, 0.351, 31],
        [8, 183, 64, 0, 0, 23.3, 0.672, 32]
    ]
    return sample_data


def main():
    """
    Main deployment workflow orchestration.
    
    This function demonstrates the complete deployment architecture:
    - Model retrieval from Model Registry (pipeline output)
    - Endpoint creation or reuse (serving infrastructure)
    - Model deployment (connecting pipeline to serving)
    - Testing and validation (verifying complete architecture)
    
    TODO: Lab 5.4.1 - Component Identification: Deployment orchestration component
    TODO: Lab 5.4.2 - Purpose Recognition: Completes pipeline-to-production architecture
    TODO: Lab 5.4.3 - Architecture Understanding: Demonstrates complete ML system architecture
    """
    # TODO: Lab 5.4.1 - Component Identification: CLI argument parsing for deployment configuration
    parser = argparse.ArgumentParser(
        description="Deploy a registered Vertex AI model to an endpoint"
    )
    parser.add_argument(
        "--project-id",
        required=True,
        help="Google Cloud project ID"
    )
    parser.add_argument(
        "--region",
        required=True,
        help="Google Cloud region (e.g., us-central1)"
    )
    parser.add_argument(
        "--model-display-name",
        required=True,
        help="Display name of the registered model to deploy"
    )
    parser.add_argument(
        "--endpoint-display-name",
        required=True,
        help="Display name for the endpoint (created if doesn't exist)"
    )
    parser.add_argument(
        "--deployment-name",
        default="diabetes-model-deployment",
        help="Name for this model deployment"
    )
    parser.add_argument(
        "--machine-type",
        default="n1-standard-2",
        help="Machine type for deployment"
    )
    parser.add_argument(
        "--min-replicas",
        type=int,
        default=1,
        help="Minimum number of replicas"
    )
    parser.add_argument(
        "--max-replicas",
        type=int,
        default=3,
        help="Maximum number of replicas"
    )
    parser.add_argument(
        "--test-endpoint",
        action="store_true",
        help="Test the endpoint after deployment"
    )

    args = parser.parse_args()

    try:
        # =======================================================================
        # DEPLOYMENT WORKFLOW - Complete Pipeline-to-Serving Architecture
        # =======================================================================
        # TODO: Lab 5.4.1 - Component Identification: Deployment manager instantiation
        # TODO: Lab 5.4.3 - Architecture Understanding: Manager orchestrates deployment components
        deployment_manager = ModelDeploymentManager(
            project_id=args.project_id,
            region=args.region
        )
        
        # TODO: Lab 5.4.1 - Component Identification: Model Registry retrieval
        # TODO: Lab 5.4.2 - Purpose Recognition: Retrieves pipeline-registered model
        # TODO: Lab 5.4.3 - Architecture Understanding: Model Registry bridges pipeline and serving
        model = deployment_manager.find_latest_model(args.model_display_name)
        
        # TODO: Lab 5.4.1 - Component Identification: Endpoint creation/retrieval
        # TODO: Lab 5.4.2 - Purpose Recognition: Establishes serving infrastructure
        # TODO: Lab 5.4.3 - Architecture Understanding: Endpoints are serving layer
        endpoint = deployment_manager.create_or_get_endpoint(args.endpoint_display_name)
        
        # TODO: Lab 5.4.1 - Component Identification: Model deployment operation
        # TODO: Lab 5.4.2 - Purpose Recognition: Connects pipeline output to serving layer
        # TODO: Lab 5.4.3 - Architecture Understanding: Deployment completes ML architecture
        deployment_details = deployment_manager.deploy_model_to_endpoint(
            model=model,
            endpoint=endpoint,
            deployment_name=args.deployment_name,
            machine_type=args.machine_type,
            min_replica_count=args.min_replicas,
            max_replica_count=args.max_replicas
        )

        # TODO: Lab 5.6.5 — Observability: metrics, logging, and dashboard links
        # INSPECT: Console output for deployment summary.
        # OBSERVATION: Prints structured deployment information to stdout.
        # PURPOSE: Provides immediate feedback on deployment success and configuration.
        # DELIVERABLE: Note this output pattern in observability.md.
        
        print("\n" + "=" * 60)
        print("MODEL DEPLOYMENT COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"Model: {model.display_name}")
        print(f"Endpoint: {endpoint.display_name}")
        print(f"Machine Type: {deployment_details['machine_type']}")
        print(f"Replica Range: {deployment_details['replica_count']}")
        print(f"Traffic Allocation: {deployment_details['traffic_percentage']}%")

        # TODO: Lab 5.4.1 - Component Identification: Optional endpoint testing
        # TODO: Lab 5.4.2 - Purpose Recognition: Validates complete pipeline-to-serving flow
        
        # TODO: Lab 5.6.5 — Observability: metrics, logging, and dashboard links
        # INSPECT: Optional endpoint testing for post-deployment validation.
        # OBSERVATION: --test-endpoint flag enables automatic testing after deployment.
        # PURPOSE: Validates deployment is functional before declaring success.
        # DELIVERABLE: Document this validation pattern in observability.md.
        
        if args.test_endpoint:
            print("\n" + "-" * 40)
            print("TESTING ENDPOINT")
            print("-" * 40)
            test_data = create_sample_test_data()
            predictions, test_passed = deployment_manager.test_endpoint_prediction(
                endpoint, test_data
            )
            if test_passed:
                print("Endpoint test passed successfully")
                print("Sample predictions:")
                for i, pred in enumerate(predictions[:2]):
                    print(f"  Test {i+1}: {pred}")
            else:
                print("Endpoint test failed")

        # TODO: Lab 5.4.1 - Component Identification: Endpoint information retrieval
        
        # TODO: Lab 5.6.5 — Observability: metrics, logging, and dashboard links
        # INSPECT: Endpoint information display for operational visibility.
        # OBSERVATION: Prints endpoint resource name and deployed model count.
        # PURPOSE: Provides operators with essential endpoint details.
        
        endpoint_info = deployment_manager.get_endpoint_info(endpoint)
        print("\n" + "-" * 40)
        print("ENDPOINT INFORMATION")
        print("-" * 40)
        print(f"Endpoint URL: {endpoint.resource_name}")
        print(f"Total Deployed Models: {len(endpoint_info['deployed_models'])}")

        # TODO: Lab 5.6.5 — Observability: metrics, logging, and dashboard links
        # INSPECT: Post-deployment guidance for monitoring and operations.
        # OBSERVATION: Prints actionable next steps for operators.
        # PURPOSE: Guides users to monitoring tools and best practices.
        # KEY OBSERVABILITY RECOMMENDATIONS:
        # 1. Monitor endpoint performance in Vertex AI console
        # 2. Set up alerting for endpoint health and latency
        # 3. Configure auto-scaling based on traffic patterns
        # 4. Implement A/B testing for model versions
        # 5. Use endpoint for predictions
        # DELIVERABLE: Document these monitoring recommendations in observability.md.
        
        print("\n" + "-" * 40)
        print("NEXT STEPS")
        print("-" * 40)
        print("1. Monitor endpoint performance in the Vertex AI console")
        print("2. Set up alerting for endpoint health and latency")
        print("3. Configure auto-scaling based on traffic patterns")
        print("4. Implement A/B testing for model versions")
        print(f"5. Use this endpoint for predictions at: {endpoint.resource_name}")

    except Exception as e:
        print(f"\nDeployment failed: {str(e)}")
        raise


if __name__ == "__main__":
    # ==========================================================================
    # Lab 5.4 Architecture Summary: Complete ML System Architecture
    # ==========================================================================
    # This deployment script demonstrates the final stage of the Vertex AI
    # Pipeline architecture, completing the pipeline-to-production flow:
    #
    # 1. PIPELINE OUTPUT CONSUMPTION:
    #    - Model Registry stores pipeline-trained models
    #    - Deployment script retrieves registered models
    #    - Model metadata enables version tracking
    #
    # 2. SERVING INFRASTRUCTURE COMPONENTS:
    #    - Endpoints provide managed serving layer
    #    - Resource configuration (machine type, replicas)
    #    - Traffic management for A/B testing
    #    - Auto-scaling for production workloads
    #
    # 3. COMPLETE ARCHITECTURE FLOW:
    #    Pipeline Components → Model Registry → Endpoint Deployment → Inference Serving
    #    - Data preprocessing (pipeline component)
    #    - Model training (pipeline component)
    #    - Model evaluation (pipeline component)
    #    - Model registration (pipeline component)
    #    - Model deployment (this script)
    #    - Online inference (endpoint)
    #
    # 4. INTEGRATION POINTS:
    #    - Model Registry: Bridge between pipeline and serving
    #    - Endpoints: Serving infrastructure for deployed models
    #    - Predictions: Final output of complete ML architecture
    #
    # TODO: Lab 5.4.3 - Architecture Understanding: This script demonstrates
    #       how pipeline components integrate with serving infrastructure to
    #       create a complete, production-ready ML system
    
    # TODO: Lab 5.6.5 — Observability: metrics, logging, and dashboard links
    # OBSERVABILITY SUMMARY FOR DEPLOY_MODEL.PY:
    # ==========================================
    #
    # LOGGING CALLS (ALL self.logger calls):
    # -------------------------------------
    # find_latest_model():
    #   - Searching for model: {model_display_name}
    #   - Found latest model: {display_name} (Resource: {resource_name}, Created: {create_time})
    #
    # create_or_get_endpoint():
    #   - Looking for existing endpoint: {endpoint_display_name}
    #   - Using existing endpoint: {resource_name} (if exists)
    #   - Creating new endpoint: {endpoint_display_name} (if new)
    #   - Created endpoint: {resource_name} (if new)
    #
    # deploy_model_to_endpoint():
    #   - Deploying model {model.display_name} to endpoint {endpoint.display_name}
    #   - Model deployment completed successfully
    #   - Could not retrieve deployed model ID: {e} (warning, if error)
    #
    # test_endpoint_prediction():
    #   - Testing endpoint with sample diabetes data...
    #   - Endpoint responding correctly (if success)
    #   - Sample prediction {i+1}: {prediction} (for each sample)
    #   - Endpoint returned empty predictions (if empty)
    #   - Endpoint test failed: {str(e)} (if exception)
    #
    # get_endpoint_info():
    #   - Retrieving endpoint information: {endpoint.display_name}
    #
    # TOTAL LOGGING CALLS: Approximately 15-17 across all methods
    #
    # LABELS FOR MONITORING AND FILTERING:
    # ------------------------------------
    # Endpoint labels (create_or_get_endpoint):
    #   - model_type: "diabetes_classifier"
    #   - environment: "production"
    #   - managed_by: "vertex_ai_pipeline"
    #
    # PURPOSE: Enable filtering in Vertex AI console and cost attribution
    #
    # METADATA STRUCTURES FOR MONITORING:
    # ----------------------------------
    # deployment_details (deploy_model_to_endpoint return value):
    #   - endpoint_id: Endpoint resource name
    #   - deployed_model_id: Deployed model ID
    #   - machine_type: Machine type used
    #   - replica_count: Min-max replica range
    #   - traffic_percentage: Traffic allocation
    #   - status: "deployed"
    #
    # endpoint_info (get_endpoint_info return value):
    #   - endpoint_name: Display name
    #   - endpoint_id: Resource name
    #   - create_time: Creation timestamp
    #   - region: GCP region
    #   - project_id: GCP project
    #   - deployed_models: List of deployed model details
    #     - model_display_name
    #     - model_id
    #     - machine_type
    #     - min_replicas
    #     - max_replicas
    #     - deployed_model_id
    #
    # CONSOLE OUTPUT (stdout):
    # -----------------------
    # - MODEL DEPLOYMENT COMPLETED SUCCESSFULLY banner
    # - Model display name
    # - Endpoint display name
    # - Machine type
    # - Replica range
    # - Traffic allocation percentage
    # - TESTING ENDPOINT section (if --test-endpoint flag)
    # - Test pass/fail status
    # - Sample predictions (first 2)
    # - ENDPOINT INFORMATION section
    # - Endpoint URL (resource name)
    # - Total deployed models count
    # - NEXT STEPS section with 5 monitoring recommendations
    #
    # WHERE TO FIND DEPLOYMENT LOGS:
    # =============================
    # - Script execution logs: stdout/stderr (captured by CI/CD or terminal)
    # - Vertex AI Model Registry: Model metadata and lineage
    # - Vertex AI Endpoints console: Endpoint monitoring dashboard
    # - Cloud Logging: Filter by project_id and deploy_model.py
    #
    # POST-DEPLOYMENT OBSERVABILITY HOOKS:
    # ===================================
    # 1. Endpoint resource_name printed for direct console access
    # 2. Deployment details returned as structured dict
    # 3. Optional endpoint testing validates functionality
    # 4. Endpoint info retrieval for operational visibility
    # 5. "Next Steps" guidance points to monitoring tools
    #
    # MONITORING RECOMMENDATIONS (from script output):
    # ===============================================
    # 1. Monitor endpoint performance in Vertex AI console
    #    - Latency metrics (p50, p95, p99)
    #    - Request rate and throughput
    #    - Error rate and status codes
    #    - Resource utilization (CPU, memory)
    #
    # 2. Set up alerting for endpoint health and latency
    #    - Use Cloud Monitoring for alerts
    #    - Alert on error rate > threshold
    #    - Alert on latency > SLA
    #    - Alert on replica count changes
    #
    # 3. Configure auto-scaling based on traffic patterns
    #    - Adjust min/max replica counts
    #    - Set appropriate machine types
    #    - Monitor scaling events
    #
    # 4. Implement A/B testing for model versions
    #    - Deploy multiple models to same endpoint
    #    - Split traffic percentage
    #    - Monitor performance differences
    #    - Gradually shift traffic to better model
    #
    # 5. Use endpoint for predictions
    #    - Endpoint resource name provided
    #    - Use Vertex AI Prediction API
    #    - Monitor prediction latency and quality
    #
    # OBSERVABILITY GAPS (NOT CURRENTLY IMPLEMENTED):
    # ==============================================
    # - No custom metrics logging (e.g., deployment duration)
    # - No integration with external monitoring systems (Prometheus, Datadog)
    # - No structured logging (JSON format for machine parsing)
    # - No deployment event tracking (success/failure metrics)
    # - No automated health checks after deployment
    # - No rollback capabilities on deployment failure
    #
    # OBSERVABILITY BEST PRACTICES DEMONSTRATED:
    # =========================================
    # ✓ Comprehensive logging at each step
    # ✓ Structured metadata returned for monitoring
    # ✓ Labels for filtering and cost attribution
    # ✓ Optional testing for validation
    # ✓ Clear success/failure indicators
    # ✓ Endpoint resource names for console access
    # ✓ Guidance on monitoring setup
    #
    # VERTEX AI CONSOLE ACCESS FOR MONITORING:
    # ========================================
    # 1. Model Registry:
    #    - Navigate to: Vertex AI → Model Registry
    #    - View: Registered models and versions
    #    - Monitor: Model lineage and metadata
    #
    # 2. Endpoints:
    #    - Navigate to: Vertex AI → Endpoints
    #    - View: All deployed endpoints
    #    - Monitor: Request metrics, latency, errors
    #    - Filter: Use labels (environment: production)
    #
    # 3. Endpoint Details (specific endpoint):
    #    - Click endpoint name from list
    #    - View: Deployed models on endpoint
    #    - Monitor: Traffic split, replica count
    #    - Access: Prediction API details
    #    - Logs: Click "View logs" for Cloud Logging
    #
    # 4. Deployed Model Details:
    #    - Click deployed model within endpoint
    #    - View: Model version, create time
    #    - Monitor: Individual model metrics
    #    - Test: Send test predictions
    #
    # 5. Cloud Logging (detailed logs):
    #    - Navigate to: Cloud Logging
    #    - Filter by: resource.type="aiplatform.googleapis.com/Endpoint"
    #    - Filter by: resource.labels.endpoint_id="<endpoint_id>"
    #    - View: Request logs, error logs, system logs
    #
    # 6. Cloud Monitoring (metrics and alerts):
    #    - Navigate to: Cloud Monitoring → Metrics Explorer
    #    - Metric: aiplatform.googleapis.com/prediction/online/*
    #    - View: Latency, error rate, request count
    #    - Create: Alerts based on metric thresholds
    #
    # INTEGRATION WITH PIPELINE OBSERVABILITY:
    # ========================================
    # Pipeline Phase          | Observability Tool
    # -----------------------|------------------------------------------
    # Training                | Pipeline component logs (Vertex AI)
    # Evaluation              | Metrics logged (accuracy, thresholds)
    # Registration            | Model Registry (version tracking)
    # Deployment (this script)| Deployment logs (stdout, Cloud Logging)
    # Serving                 | Endpoint monitoring (Vertex AI console)
    # Inference               | Prediction logs (Cloud Logging)
    #
    # COMPLETE OBSERVABILITY FLOW:
    # ===========================
    # 1. Pipeline Execution:
    #    - View in: Vertex AI Pipelines console
    #    - Monitor: Component status, execution time
    #    - Logs: Click component for logs
    #
    # 2. Model Registration:
    #    - View in: Model Registry
    #    - Monitor: Model versions, lineage
    #    - Metadata: Training accuracy, parameters
    #
    # 3. Model Deployment (THIS SCRIPT):
    #    - View in: Script stdout/logs
    #    - Monitor: Deployment success/failure
    #    - Validate: Optional endpoint testing
    #
    # 4. Endpoint Serving:
    #    - View in: Endpoints console
    #    - Monitor: Latency, throughput, errors
    #    - Alert: Cloud Monitoring alerts
    #
    # 5. Prediction Monitoring:
    #    - View in: Cloud Logging (prediction logs)
    #    - Monitor: Prediction quality, drift
    #    - Analyze: Request/response patterns
    #
    # DELIVERABLE CHECKLIST FOR observability.md:
    # ===========================================
    # □ List all logging calls in deploy_model.py (15-17 total)
    # □ Document endpoint labels and their purposes
    # □ Document deployment_details metadata structure
    # □ Document endpoint_info metadata structure
    # □ List where to find deployment logs (4 locations)
    # □ Document post-deployment observability hooks (5 items)
    # □ List monitoring recommendations from script output (5 items)
    # □ Document Vertex AI console access paths (6 sections)
    # □ Document integration with pipeline observability
    # □ Note observability gaps and potential improvements
    #
    # COMPARISON: PIPELINE vs DEPLOYMENT OBSERVABILITY:
    # =================================================
    # Pipeline Components (train_model_op, evaluate_model_op):
    #   - Logging: Component-level logs with [DEV]/[PROD] prefix
    #   - Metrics: metrics.log_metric() for accuracy tracking
    #   - Location: Vertex AI Pipelines console, Cloud Logging
    #
    # Deployment Script (deploy_model.py):
    #   - Logging: Deployment operation logs with timestamps
    #   - Metadata: Deployment details dict, endpoint info dict
    #   - Location: Script stdout, Cloud Logging, Vertex AI Endpoints console
    #
    # Both Provide:
    #   - Comprehensive logging at each step
    #   - Clear success/failure indicators
    #   - Resource names for console access
    #   - Structured metadata for monitoring
    #
    # Key Difference:
    #   - Pipeline: Logs appear in pipeline execution context
    #   - Deployment: Logs appear in deployment script context
    #   - Pipeline: Metrics logged to Vertex AI Experiments
    #   - Deployment: Endpoint metrics in Vertex AI Monitoring
    #
    # ==========================================================================
    main()
