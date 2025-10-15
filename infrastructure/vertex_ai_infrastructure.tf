# ==============================================================================
# Vertex AI Infrastructure - Terraform Configuration
# ==============================================================================
# This Terraform configuration provisions the infrastructure layer of the
# Accelerator Template architecture for Vertex AI Pipelines.
#
# Lab 5.5: Vertex AI Custom Components and Pre-built Components and Accelerator Templates
# ==============================================================================
# CRITICAL UNDERSTANDING: Terraform vs GitHub Actions Orchestration
# ==============================================================================
# 
# TERRAFORM ROLE (This File):
# - Provisions infrastructure ONCE or when infrastructure changes
# - Creates: GCS buckets, service accounts, IAM roles, endpoints, and optionally BigQuery Shared View
# - Run BEFORE pipelines execute
# - Typically run by platform/infrastructure team
# - Changes infrequently (weeks/months)
#
# GITHUB ACTIONS ROLE (vertex-ai-cicd.yml):
# - Orchestrates ML workflow on EVERY code commit
# - Compiles pipelines → Runs pipelines → Deploys models
# - Run by ML engineers on each code change
# - Assumes infrastructure already exists
# - Runs frequently (daily/hourly)
#
# DEPLOYMENT SEQUENCE:
# ====================
# Step 1 (ONE TIME): terraform apply
#        └─> Creates infrastructure (buckets, service accounts, endpoints)
#
# Step 2 (EVERY COMMIT): git push triggers GitHub Actions
#        └─> Compiles pipelines
#        └─> Runs pipelines (uses infrastructure from Step 1)
#        └─> Deploys models (uses infrastructure from Step 1)
#
# THEY ARE COMPLEMENTARY, NOT COMPETING:
# - Terraform = Infrastructure provisioning (foundation)
# - GitHub Actions = ML workflow orchestration (uses the foundation)
#
# ==============================================================================
# This file demonstrates the INFRASTRUCTURE LAYER of the 3-layer enterprise architecture:
#
# 1. INFRASTRUCTURE LAYER (This File - Terraform):
#    - Vertex AI pipeline resources
#    - Service accounts and IAM permissions
#    - GCS buckets for data and artifacts
#    - Multi-environment workspace management
#    - RUN ONCE OR WHEN INFRASTRUCTURE CHANGES
#
# 2. PIPELINE LAYER (vertex_pipeline_dev.py, vertex_pipeline_prod.py):
#    - Custom components for business logic
#    - Pre-built components for standard operations
#    - ML workflow orchestration
#    - RUNS ON EVERY CODE COMMIT (via GitHub Actions)
#
# 3. ENTERPRISE LAYER (Best Practices):
#    - Governance and compliance patterns
#    - Reusable solution frameworks
#
# TODO: Lab 5.5.3 - Accelerator Templates: Infrastructure-as-Code for pipeline deployment
# TODO: Lab 5.5.3 - Enterprise Architecture: Complete infrastructure provisioning
# TODO: Lab 5.5.3 - Terraform vs CI/CD: Understand separation of concerns
# ==============================================================================

# ==============================================================================
# Terraform Configuration
# ==============================================================================
# TODO: Lab 5.5.3 - Infrastructure Layer: Terraform backend and provider configuration

terraform {
  required_version = ">= 1.0"
  
  # TODO: Lab 5.5.3 - State Management: Remote state storage for team collaboration
  # backend "gcs" {
  #   bucket = "your-terraform-state-bucket"
  #   prefix = "vertex-ai-pipelines"
  # }
  
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

# ==============================================================================
# Variables - Environment Configuration
# ==============================================================================
# TODO: Lab 5.5.3 - Configuration Management: Variables enable multi-environment deployment

variable "project_id" {
  description = "Google Cloud Project ID"
  type        = string
}

variable "region" {
  description = "Google Cloud region for Vertex AI resources"
  type        = string
  default     = "us-east1"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be dev, staging, or prod"
  }
}

variable "shared_mlops_bucket_name" {
  description = "Name of the shared MLOps GCS bucket"
  type        = string
}

variable "model_display_name" {
  description = "Display name for the ML model"
  type        = string
  default     = "diabetes-prediction-model"
}

# ==============================================================================
# Provider Configuration
# ==============================================================================
# TODO: Lab 5.5.3 - Provider Setup: Google Cloud provider configuration

provider "google" {
  project = var.project_id
  region  = var.region
}

# ==============================================================================
# GCS Buckets - Data and Artifact Storage
# ==============================================================================
# TODO: Lab 5.5.3 - Storage Infrastructure: GCS buckets for pipeline artifacts
# TODO: Lab 5.5.3 - Component Integration: Storage layer supports custom and pre-built components

# Main MLOps bucket for pipeline artifacts
resource "google_storage_bucket" "mlops_bucket" {
  name          = var.shared_mlops_bucket_name
  location      = var.region
  force_destroy = false
  
  # TODO: Lab 5.5.3 - Production Best Practices: Versioning for artifact management
  versioning {
    enabled = var.environment == "prod" ? true : false
  }
  
  # TODO: Lab 5.5.3 - Lifecycle Management: Automatic cleanup of old artifacts
  lifecycle_rule {
    condition {
      age = var.environment == "prod" ? 90 : 30
    }
    action {
      type = "Delete"
    }
  }
  
  # TODO: Lab 5.5.3 - Security: Uniform bucket-level access
  uniform_bucket_level_access = true
  
  labels = {
    environment = var.environment
    managed_by  = "terraform"
    purpose     = "vertex-ai-pipelines"
  }
}

# Environment-specific folders (created via objects)
resource "google_storage_bucket_object" "env_folders" {
  for_each = toset([
    "${var.environment}/raw/",
    "${var.environment}/processed/",
    "${var.environment}/models/",
    "pipeline-runs/${var.environment}/"
  ])
  
  name    = each.key
  content = "# Folder for ${each.key}"
  bucket  = google_storage_bucket.mlops_bucket.name
}

# ==============================================================================
# Service Accounts - Pipeline Execution Identity
# ==============================================================================
# TODO: Lab 5.5.3 - IAM Infrastructure: Service accounts for pipeline execution
# TODO: Lab 5.5.3 - Security: Least-privilege access for pipeline components

# Service account for Vertex AI Pipeline execution
resource "google_service_account" "vertex_pipeline_sa" {
  account_id   = "vertex-pipeline-${var.environment}"
  display_name = "Vertex AI Pipeline Service Account (${var.environment})"
  description  = "Service account for running Vertex AI Pipelines in ${var.environment}"
}

# Service account for GitHub Actions CI/CD
resource "google_service_account" "github_actions_sa" {
  account_id   = "github-actions-${var.environment}"
  display_name = "GitHub Actions Service Account (${var.environment})"
  description  = "Service account for GitHub Actions CI/CD workflows in ${var.environment}"
}

# ==============================================================================
# IAM Permissions - Resource Access Control
# ==============================================================================
# TODO: Lab 5.5.3 - Access Control: IAM roles for pipeline components
# TODO: Lab 5.5.3 - Custom Components: Permissions needed for custom component operations

# Grant Vertex AI Pipeline SA permissions
resource "google_project_iam_member" "vertex_pipeline_permissions" {
  for_each = toset([
    "roles/aiplatform.user",           # Vertex AI operations
    "roles/storage.objectAdmin",        # GCS access
    "roles/bigquery.dataEditor",        # BigQuery access (for pre-built components)
    "roles/logging.logWriter",          # Logging
  ])
  
  project = var.project_id
  role    = each.key
  member  = "serviceAccount:${google_service_account.vertex_pipeline_sa.email}"
}

# Grant GitHub Actions SA permissions
resource "google_project_iam_member" "github_actions_permissions" {
  for_each = toset([
    "roles/aiplatform.user",
    "roles/storage.objectViewer",
    "roles/iam.serviceAccountUser",
  ])
  
  project = var.project_id
  role    = each.key
  member  = "serviceAccount:${google_service_account.github_actions_sa.email}"
}

# ==============================================================================
# Vertex AI Resources - Pipeline Infrastructure
# ==============================================================================
# TODO: Lab 5.5.3 - Vertex AI Infrastructure: Pipeline and endpoint resources

# Vertex AI endpoint for model deployment
resource "google_vertex_ai_endpoint" "model_endpoint" {
  count        = var.environment == "prod" ? 1 : 0
  name         = "diabetes-prediction-endpoint-${var.environment}"
  display_name = "Diabetes Prediction Endpoint (${var.environment})"
  description  = "Production endpoint for diabetes prediction model"
  region       = var.region
  
  labels = {
    environment = var.environment
    model_type  = "diabetes-classifier"
    managed_by  = "terraform"
  }
}

# Vertex AI Dataset (optional - for managed datasets)
resource "google_vertex_ai_dataset" "diabetes_dataset" {
  count        = var.environment == "prod" ? 1 : 0
  display_name = "Diabetes Dataset (${var.environment})"
  metadata_schema_uri = "gs://google-cloud-aiplatform/schema/dataset/metadata/tabular_1.0.0.yaml"
  region       = var.region
  
  labels = {
    environment = var.environment
    managed_by  = "terraform"
  }
}

# ==============================================================================
# Outputs - Infrastructure Information
# ==============================================================================
# TODO: Lab 5.5.3 - Infrastructure Outputs: Values for pipeline configuration

output "mlops_bucket_name" {
  description = "Name of the MLOps GCS bucket"
  value       = google_storage_bucket.mlops_bucket.name
}

output "vertex_pipeline_sa_email" {
  description = "Email of the Vertex AI Pipeline service account"
  value       = google_service_account.vertex_pipeline_sa.email
}

output "github_actions_sa_email" {
  description = "Email of the GitHub Actions service account"
  value       = google_service_account.github_actions_sa.email
}

output "model_endpoint_id" {
  description = "ID of the Vertex AI endpoint (production only)"
  value       = var.environment == "prod" ? google_vertex_ai_endpoint.model_endpoint[0].id : null
}

output "environment" {
  description = "Current environment"
  value       = var.environment
}

# ==============================================================================
# Lab 5.5 Infrastructure Summary: Accelerator Template Infrastructure Layer
# ==============================================================================
# This Terraform configuration demonstrates the INFRASTRUCTURE LAYER of the
# 3-layer Accelerator Template architecture:
#
# 1. INFRASTRUCTURE PROVISIONING (This File):
#    Resource Type           | Purpose
#    ----------------------- | ------------------------------------------
#    GCS Buckets            | Artifact storage for custom components
#    Service Accounts       | Pipeline execution identity
#    IAM Permissions        | Access control for components
#    Vertex AI Endpoints    | Model serving infrastructure
#    Vertex AI Datasets     | Managed dataset storage
#
# 2. ENVIRONMENT MANAGEMENT:
#    - Workspace-based multi-environment deployment
#    - Variables for environment-specific configuration
#    - Conditional resource creation (prod-only resources)
#    - Environment-specific lifecycle policies
#
# 3. INTEGRATION WITH PIPELINE LAYER:
#    Infrastructure Resource     | Pipeline Usage
#    --------------------------- | ------------------------------------------
#    mlops_bucket               | Artifact storage for all components
#    vertex_pipeline_sa         | Pipeline execution identity
#    IAM permissions            | Enable component GCS/BigQuery access
#    Endpoint (prod)            | Model deployment target
#
# 4. CUSTOM vs PRE-BUILT COMPONENT SUPPORT:
#    Custom Components Need:
#    - GCS access for artifact storage
#    - Service account for execution
#    - Logging permissions
#    
#    Pre-built Components Need:
#    - Same GCS and service account access
#    - BigQuery permissions (for BigqueryQueryJobOp)
#    - Vertex AI permissions (for training/deployment ops)
#    
#    This infrastructure supports BOTH component types!
#
# 5. DEPLOYMENT WORKFLOW:
#    Step 1: terraform init
#    Step 2: terraform workspace select dev (or prod)
#    Step 3: terraform plan -var-file="dev.tfvars"
#    Step 4: terraform apply -var-file="dev.tfvars"
#    Step 5: Run pipelines using provisioned infrastructure
#
# 6. TERRAFORM WORKSPACE STRATEGY:
#    Workspace    | Environment | var.environment
#    ------------ | ----------- | ---------------
#    default      | dev         | "dev"
#    prod         | production  | "prod"
#
# TODO: Lab 5.5.3 - Accelerator Templates: This infrastructure enables complete
#       enterprise ML pipeline deployment with support for both custom and
#       pre-built components
# ==============================================================================

# ==============================================================================
# USAGE INSTRUCTIONS FOR LAB 5.5
# ==============================================================================
# LEARNERS: This Terraform file demonstrates Infrastructure-as-Code for Vertex AI
#
# UNDERSTANDING THE SEPARATION:
# =============================
# 
# TERRAFORM (This File):
# ----------------------
# WHEN: Run ONCE during initial setup or when infrastructure needs change
# WHO: Platform/Infrastructure team or DevOps engineer
# WHAT: Provisions foundational infrastructure
# FREQUENCY: Infrequently (weekly/monthly or on-demand)
# 
# GITHUB ACTIONS (vertex-ai-cicd.yml):
# ------------------------------------
# WHEN: Runs AUTOMATICALLY on every code commit to main branch
# WHO: Triggered by ML engineers committing code
# WHAT: Compiles pipelines, runs training, deploys models
# FREQUENCY: Frequently (multiple times per day)
# 
# REAL-WORLD WORKFLOW:
# ===================
# 
# [INFRASTRUCTURE SETUP - Done Once by Platform Team]
# 1. Platform engineer: terraform apply
#    └─> Creates: GCS buckets, service accounts, IAM permissions, endpoints
#    └─> Result: Infrastructure ready for ML workflows
# 
# [ML DEVELOPMENT - Done Repeatedly by ML Engineers]
# 2. ML engineer: Updates pipeline code (e.g., vertex_pipeline_dev.py)
# 3. ML engineer: git commit && git push
# 4. GitHub Actions (automatic):
#    ├─> Step 1: Compile pipelines (compiler.py)
#    ├─> Step 2: Run dev pipeline (run_pipeline.py) [uses GCS bucket from Terraform]
#    ├─> Step 3: Await approval
#    ├─> Step 4: Run prod pipeline (run_pipeline.py) [uses GCS bucket from Terraform]
#    └─> Step 5: Deploy model (deploy_model.py) [uses endpoint from Terraform]
# 
# 5. Repeat steps 2-4 for each code change (GitHub Actions uses existing infrastructure)
# 
# WHEN TO RUN TERRAFORM AGAIN:
# ============================
# - Adding new environment (staging)
# - Changing storage bucket configuration
# - Updating IAM permissions
# - Adding new service accounts
# - Modifying endpoint configuration
# 
# WHEN GITHUB ACTIONS RUNS:
# =========================
# - Every git push to main branch
# - Every pull request to main branch
# - Manual workflow dispatch
# 
# KEY INSIGHT FOR LAB 5.5:
# ========================
# Infrastructure (Terraform) is the FOUNDATION.
# CI/CD (GitHub Actions) is the WORKFLOW that uses the foundation.
# They work together but serve different purposes.
#
# TO EXPLORE THIS FILE:
# ====================
# 1. Review how GCS buckets support component artifact storage
# 2. Understand service account permissions for pipeline execution
# 3. See how infrastructure differs between dev and prod environments
# 4. Observe integration points with pipeline Python files
# 5. Note: This infrastructure must exist BEFORE pipelines run
#
# TO DEPLOY (Optional - requires GCP access):
# ===========================================
# 1. Create terraform.tfvars:
#    project_id               = "your-project-id"
#    shared_mlops_bucket_name = "your-bucket-name"
#    environment             = "dev"
#
# 2. Initialize Terraform:
#    terraform init
#
# 3. Plan infrastructure:
#    terraform plan
#
# 4. Apply infrastructure (ONE TIME SETUP):
#    terraform apply
#
# 5. THEN run GitHub Actions workflow (vertex-ai-cicd.yml) for ML workflows
#
# NOTE: This is for learning exploration. Actual deployment requires
# proper GCP project setup and permissions.
# ==============================================================================
