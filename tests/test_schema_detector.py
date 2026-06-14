"""
Unit tests for the schema_detector module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from smarteda.core.schema_detector import SchemaDetector
from smarteda.models.analysis_result import DataType


class TestSchemaDetector:
    """Test cases for SchemaDetector."""

    def test_detect_basic_schema(self, sample_df: pd.DataFrame) -> None:
        """Test basic schema detection."""
        detector = SchemaDetector()
        profile = detector.detect(sample_df)

        assert profile.row_count == len(sample_df)
        assert profile.column_count == len(sample_df.columns)
        assert len(profile.columns) == len(sample_df.columns)
        assert profile.memory_usage > 0

    def test_numeric_type_detection(self, sample_df: pd.DataFrame) -> None:
        """Test numeric column type detection."""
        detector = SchemaDetector()
        profile = detector.detect(sample_df)

        assert profile.columns["age"].inferred_type == DataType.INTEGER
        assert profile.columns["salary"].inferred_type == DataType.FLOAT
        assert profile.columns["score"].inferred_type == DataType.FLOAT

    def test_categorical_type_detection(self, sample_df: pd.DataFrame) -> None:
        """Test categorical column type detection."""
        detector = SchemaDetector()
        profile = detector.detect(sample_df)

        assert profile.columns["department"].inferred_type == DataType.CATEGORICAL

    def test_boolean_type_detection(self, sample_df: pd.DataFrame) -> None:
        """Test boolean column type detection."""
        detector = SchemaDetector()
        profile = detector.detect(sample_df)

        assert profile.columns["is_active"].inferred_type == DataType.BOOLEAN

    def test_datetime_type_detection(self, sample_df: pd.DataFrame) -> None:
        """Test datetime column type detection."""
        detector = SchemaDetector()
        profile = detector.detect(sample_df)

        assert profile.columns["join_date"].inferred_type == DataType.DATETIME

    def test_empty_dataframe(self, empty_df: pd.DataFrame) -> None:
        """Test schema detection on empty DataFrame."""
        detector = SchemaDetector()
        profile = detector.detect(empty_df)

        assert profile.row_count == 0
        assert profile.column_count == 0

    def test_constant_column_detection(self, constant_df: pd.DataFrame) -> None:
        """Test constant column detection."""
        detector = SchemaDetector()
        profile = detector.detect(constant_df)

        assert profile.columns["constant"].is_constant is True
        assert profile.columns["constant"].inferred_type == DataType.CONSTANT
        assert profile.columns["varying"].is_constant is False

    def test_semantic_type_detection(self, sample_df: pd.DataFrame) -> None:
        """Test semantic type detection."""
        detector = SchemaDetector()
        profile = detector.detect(sample_df)

        assert profile.columns["email"].semantic_type == "email"
        assert profile.columns["age"].semantic_type is None

    def test_cardinality_detection(self, sample_df: pd.DataFrame) -> None:
        """Test cardinality classification."""
        detector = SchemaDetector()
        profile = detector.detect(sample_df)

        assert profile.columns["id"].cardinality == "unique"
        assert profile.columns["department"].cardinality == "low"

    def test_null_detection(self, df_with_missing: pd.DataFrame) -> None:
        """Test null value detection in schema."""
        detector = SchemaDetector()
        profile = detector.detect(df_with_missing)

        assert profile.columns["A"].nullable is True
        assert profile.columns["A"].null_count > 0

    def test_type_distribution(self, sample_df: pd.DataFrame) -> None:
        """Test type distribution calculation."""
        detector = SchemaDetector()
        profile = detector.detect(sample_df)

        distribution = detector.get_type_distribution(profile)

        assert isinstance(distribution, dict)
        assert sum(distribution.values()) == len(sample_df.columns)

    def test_column_names_by_type(self, sample_df: pd.DataFrame) -> None:
        """Test filtering columns by type."""
        detector = SchemaDetector()
        profile = detector.detect(sample_df)

        numeric_cols = detector.get_column_names_by_type(profile, DataType.INTEGER)
        assert "age" in numeric_cols

    def test_schema_validation(self, sample_df: pd.DataFrame) -> None:
        """Test schema validation."""
        detector = SchemaDetector()
        profile = detector.detect(sample_df)

        # Validate against itself should pass
        warnings = detector.validate_schema(sample_df, profile)
        assert isinstance(warnings, list)

    def test_single_column(self, single_column_df: pd.DataFrame) -> None:
        """Test schema detection on single-column DataFrame."""
        detector = SchemaDetector()
        profile = detector.detect(single_column_df)

        assert profile.column_count == 1
        assert "value" in profile.columns

    def test_memory_usage(self, sample_df: pd.DataFrame) -> None:
        """Test memory usage tracking."""
        detector = SchemaDetector()
        profile = detector.detect(sample_df)

        assert profile.memory_usage > 0
        assert len(profile.memory_usage_human) > 0
        assert "B" in profile.memory_usage_human or "KB" in profile.memory_usage_human or "MB" in profile.memory_usage_human
