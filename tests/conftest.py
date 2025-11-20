"""
Shared Test Fixtures
===================
Pytest fixtures available to all test files.

Phase 1: Fixtures for completeness tests
- Mock GCP services (BigQuery, Metrics)
- Test data fixtures (valid, incomplete, insufficient)
- Configuration fixtures

Future phases will add:
- Fixtures for consistency tests
- Fixtures for freshness tests
- More complex test scenarios

Usage:
    Fixtures are automatically discovered and injected by pytest.

    def test_something(mock_bigquery_client, valid_diabetes_data):
        # Fixtures are automatically provided
        pass
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timedelta


# =============================================================================
# MOCK GCP SERVICE FIXTURES
# =============================================================================

@pytest.fixture
def mock_bigquery_client():
    """
    Mock BigQuery client to avoid real API calls.

    Returns a mocked BigQuery client that returns test data instead of
    making real API calls to GCP.

    Usage:
        def test_something(mock_bigquery_client):
            # BigQuery calls are automatically mocked
            # Returns fake data from valid_diabetes_data fixture
    """
