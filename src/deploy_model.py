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
        self.logger.info(f"Looking for existing endpoint: {endpoint_display_name}")
        # TODO: Lab 5.4.1 - Component Identification: Endpoint.list() queries existing endpoints
        # TODO: Lab 5.4.2 - Purpose Recognition: Reuse existing endpoints when available
        existing_endpoints = aiplatform.Endpoint.list(
            filter=f'display_name="{endpoint_display_name}"'
        )
        if existing_endpoints:
            endpoint = existing_endpoints[0]
            self.logger.info(f"Using existing endpoint: {endpoint.resource_name}")
            return endpoint
        # TODO: Lab 5.4.1 - Component Identification: Endpoint.create() provisions new serving endpoint
        # TODO: Lab 5.4.2 - Purpose Recognition: New endpoints for first-time deployments
        # TODO: Lab 5.4.3 - Architecture Understanding: Endpoints are managed serving infrastructure
        self.logger.info(f"Creating new endpoint: {endpoint_display_name}")
        endpoint = aiplatform.Endpoint.create(
            display_name=endpoint_display_name,
            description=f"Endpoint for {endpoint_display_name} diabetes prediction model",
            # TODO: Lab 5.4.1 - Component Identification: Labels enable endpoint organization and tracking
            labels={
                "model_type": "diabetes_classifier",
                "environment": "production",
                "managed_by": "vertex_ai_pipeline"
            }
        )
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
        self.logger.info("Model deployment completed successfully")
        deployed_model_id = None
        try:
            endpoint._sync_gca_resource()
            for deployed_model in endpoint._gca_resource.deployed_models:
                if deployed_model.display_name == deployment_name:
                    deployed_model_id = deployed_model.id
                    break
        except Exception as e:
            self.logger.warning(f"Could not retrieve deployed model ID: {e}")
            deployed_model_id = "unknown"
        # TODO: Lab 5.4.1 - Component Identification: Deployment metadata for tracking
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
        self.logger.info("Testing endpoint with sample diabetes data...")
        try:
            # TODO: Lab 5.4.1 - Component Identification: endpoint.predict() invokes deployed model
            # TODO: Lab 5.4.2 - Purpose Recognition: Validates end-to-end inference capability
            # TODO: Lab 5.4.3 - Architecture Understanding: Prediction verifies complete ML architecture
            predictions = endpoint.predict(instances=test_instances)
            if predictions and predictions.predictions:
                self.logger.info("Endpoint responding correctly")
                for i, prediction in enumerate(predictions.predictions[:3]):
                    self.logger.info(f"Sample prediction {i+1}: {prediction}")
                return predictions.predictions, True
            else:
                self.logger.error("Endpoint returned empty predictions")
                return [], False
        except Exception as e:
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
        self.logger.info(f"Retrieving endpoint information: {endpoint.display_name}")
        # TODO: Lab 5.4.1 - Component Identification: Endpoint metadata structure
        # TODO: Lab 5.4.2 - Purpose Recognition: Metadata enables monitoring and management
        endpoint_info = {
            "endpoint_name": endpoint.display_name,
            "endpoint_id": endpoint.name,
            "create_time": str(endpoint.create_time),
            "region": self.region,
            "project_id": self.project_id,
            "deployed_models": []
        }
        # TODO: Lab 5.4.1 - Component Identification: Deployed model details for tracking
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
        endpoint_info = deployment_manager.get_endpoint_info(endpoint)
        print("\n" + "-" * 40)
        print("ENDPOINT INFORMATION")
        print("-" * 40)
        print(f"Endpoint URL: {endpoint.resource_name}")
        print(f"Total Deployed Models: {len(endpoint_info['deployed_models'])}")

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
    # ==========================================================================
    main()
