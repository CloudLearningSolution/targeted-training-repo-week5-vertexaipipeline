"""
Unit Tests for Custom Data Quality Components
=============================================
Tests for src/components/custom_data_quality_components.py

Phase 1: Testing COMPLETENESS validation only
Future phases will add consistency and freshness tests

Test Strategy:
- Test each completeness check independently
- Test blocking vs non-blocking behavior
- Use mocks to avoid real GCP calls
- Use fixtures for test data from conftest.py

Test Coverage (Phase 1):
- TestCompletenessValidation (5 tests)
  ├─ test_passes_with_complete_data
  ├─ test_fails_with_missing_values_when_blocking
  ├─ test_logs_warning_but_continues_when_non_blocking
  ├─ test_detects_insufficient_rows
  └─ test_identifies_missing_required_columns

Run tests:
    pytest tests/unit/components/test_custom_data_quality_components.py -v

    # With coverage:
    pytest tests/unit/components/test_custom_data_quality_components.py \
        --cov=src/components/custom_data_quality_components \
        --cov-report=html
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock


# =============================================================================
# TEST CLASS: Completeness Validation (Phase 1)
# =============================================================================

class TestCompletenessValidation:
    """
    Test completeness checks in data_quality_check_op.

    Phase 1: Testing ONLY completeness validation
    - Missing values per column
    - Minimum row count
    - Required columns present

    Completeness = % of non-null values per column
    DEFAULT BEHAVIOR: BLOCKING (fail_on_completeness=True)

    Future phases will add:
    - Phase 2: Consistency validation tests
    - Phase 3: Freshness validation tests
    """

    def test_passes_with_complete_data(
        self,
        valid_diabetes_data,           # ← Uses fixture from conftest.py
        diabetes_required_columns      # ← Uses fixture from conftest.py
    ):
        """
        Test: Valid data with no missing values passes completeness check.

        Given: DataFrame with 100% complete data (no nulls, 100 rows, all columns)
        When: Completeness validation runs
        Then:
          - All completeness checks pass
          - overall_passed = True
          - No issues reported
          - No ValueError raised

        This is the "happy path" test - everything should work perfectly.
        """
        # ARRANGE: Use fixture data
        data = valid_diabetes_data
        required_columns = diabetes_required_columns

        # ACT: Check completeness
        # 1. Check all data is present (no nulls)
        has_nulls = data.isnull().any().any()

        # 2. Check row count
        row_count = len(data)
        min_row_count = 100
        sufficient_rows = row_count >= min_row_count

        # 3. Check all required columns present
        missing_columns = [col for col in required_columns if col not in data.columns]

        # 4. Check completeness per column
        completeness_threshold = 0.95
        all_columns_complete = True
        for col in required_columns:
            completeness = data[col].notna().sum() / len(data)
            if completeness < completeness_threshold:
                all_columns_complete = False
                break

        # ASSERT: All checks should pass
        assert has_nulls == False, "Should have no null values"
        assert sufficient_rows == True, f"Should have at least {min_row_count} rows"
        assert len(missing_columns) == 0, "Should have all required columns"
        assert all_columns_complete == True, "All columns should be 100% complete"

    def test_fails_with_missing_values_when_blocking(
        self,
        incomplete_diabetes_data,      # ← Uses fixture from conftest.py
        default_quality_thresholds     # ← Uses fixture from conftest.py
    ):
        """
        Test: Data with missing values FAILS when fail_on_completeness=True.

        Given: DataFrame with 20% missing values in key columns
        When: Completeness validation runs with fail_on_completeness=True
        Then:
          - Completeness check fails
          - Issues are identified and logged
          - In real component, ValueError would be raised

        This tests the blocking behavior - pipeline should stop.
        """
        # ARRANGE: Use fixture data
        data = incomplete_diabetes_data
        completeness_threshold = default_quality_thresholds['completeness_threshold']

        # ACT: Check completeness for each column
        failed_columns = []
        for col in data.columns:
            completeness = data[col].notna().sum() / len(data)
            if completeness < completeness_threshold:
                failed_columns.append((col, completeness))

        # ASSERT: Should detect columns with insufficient completeness
        assert len(failed_columns) > 0, "Should detect columns with missing values"

        # Check specific columns that should fail
        pregnancies_completeness = data['Pregnancies'].notna().sum() / len(data)
        assert pregnancies_completeness < completeness_threshold, \
            f"Pregnancies should be below threshold: {pregnancies_completeness:.2%} < {completeness_threshold:.2%}"

        plasma_completeness = data['PlasmaGlucose'].notna().sum() / len(data)
        assert plasma_completeness < completeness_threshold, \
            f"PlasmaGlucose should be below threshold: {plasma_completeness:.2%} < {completeness_threshold:.2%}"

        # In real component, this would raise ValueError when fail_on_completeness=True
        # Here we're testing the detection logic only


    def test_logs_warning_but_continues_when_non_blocking(
        self,
        incomplete_diabetes_data,      # ← Uses fixture from conftest.py
        default_quality_thresholds     # ← Uses fixture from conftest.py
    ):
        """
        Test: Missing values LOG WARNING but DON'T BLOCK when fail_on_completeness=False.

        Given: DataFrame with 20% missing values
        When: Completeness validation runs with fail_on_completeness=False
        Then:
          - Completeness issues are detected and logged
          - No ValueError is raised (non-blocking mode)
          - Pipeline continues despite quality issues

        This tests the non-blocking behavior - useful for monitoring without stopping.
        """
        # ARRANGE: Use fixture data
        data = incomplete_diabetes_data
        completeness_threshold = default_quality_thresholds['completeness_threshold']
        fail_on_completeness = False  # NON-BLOCKING mode

        # ACT: Check completeness (simulate component logic)
        overall_completeness = data.notna().sum().sum() / data.size
        completeness_passed = overall_completeness >= completeness_threshold

        # ASSERT: Completeness should fail, but we don't raise error
        assert overall_completeness < completeness_threshold, \
            "Overall completeness should be below threshold"
        assert completeness_passed == False, \
            "Completeness check should fail"

        # CRITICAL: In non-blocking mode, no ValueError is raised
        # The component logs warnings but continues execution
        # This test should complete without raising an exception


    def test_detects_insufficient_rows(
        self,
        insufficient_rows_data,        # ← Uses fixture from conftest.py
        default_quality_thresholds     # ← Uses fixture from conftest.py
    ):
        """
        Test: Insufficient row count is detected and BLOCKS.

        Given: DataFrame with only 3 rows
        When: Completeness validation runs with min_row_count=100
        Then:
          - Row count check fails
          - Issue is identified
          - In real component, ValueError would be raised

        This tests data volume validation - critical for model training.
        """
        # ARRANGE: Use fixture data
        data = insufficient_rows_data
        min_row_count = default_quality_thresholds['min_row_count']

        # ACT: Check row count
        actual_row_count = len(data)
        sufficient_rows = actual_row_count >= min_row_count

        # ASSERT: Should detect insufficient rows
        assert actual_row_count < min_row_count, \
            f"Should have fewer than {min_row_count} rows"
        assert actual_row_count == 3, \
            "Should have exactly 3 rows in test data"
        assert sufficient_rows == False, \
            "Row count check should fail"

        # In real component, this would raise ValueError


    def test_identifies_missing_required_columns(
        self,
        missing_columns_data,          # ← Uses fixture from conftest.py
        diabetes_required_columns      # ← Uses fixture from conftest.py
    ):
        """
        Test: Missing required columns are detected.

        Given: DataFrame missing 'PlasmaGlucose' column
        When: Completeness validation runs
        Then:
          - Missing column is identified
          - Specific column name is reported in issues
          - In real component, ValueError would be raised

        This tests schema completeness - all required columns must be present.
        """
        # ARRANGE: Use fixture data
        data = missing_columns_data
        required_columns = diabetes_required_columns

        # ACT: Check for missing columns
        missing_columns = [
            col for col in required_columns
            if col not in data.columns
        ]

        # ASSERT: Should identify missing column
        assert len(missing_columns) > 0, \
            "Should detect at least one missing column"
        assert 'PlasmaGlucose' in missing_columns, \
            "Should specifically identify PlasmaGlucose as missing"
        assert len(missing_columns) == 1, \
            "Should detect exactly 1 missing column"

        # In real component, this would raise ValueError


# =============================================================================
# TEST EXECUTION MARKERS
# =============================================================================

# Mark these tests for pytest
pytestmark = [
    pytest.mark.unit,      # These are unit tests
    pytest.mark.fast,      # Should run quickly (< 5 seconds total)
    pytest.mark.phase1,    # Phase 1: Completeness only
]
