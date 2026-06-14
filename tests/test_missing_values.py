"""
Unit tests for the missing_values module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from smarteda.analysis.missing_values import MissingValueAnalyzer


class TestMissingValueAnalyzer:
    """Test cases for MissingValueAnalyzer."""

    def test_analyze_with_missing(self, df_with_missing: pd.DataFrame) -> None:
        """Test analysis on DataFrame with missing values."""
        analyzer = MissingValueAnalyzer()
        report = analyzer.analyze(df_with_missing)

        assert report.total_missing > 0
        assert report.overall_missing_ratio > 0
        assert len(report.columns_with_missing) > 0
        assert "A" in report.columns_with_missing
        assert "B" in report.columns_with_missing

    def test_analyze_complete_data(self, sample_df: pd.DataFrame) -> None:
        """Test analysis on DataFrame with no missing values."""
        analyzer = MissingValueAnalyzer()
        report = analyzer.analyze(sample_df)

        assert report.total_missing == 0
        assert report.overall_missing_ratio == 0
        assert len(report.columns_with_missing) == 0
        assert len(report.recommendations) == 1
        assert "No missing values" in report.recommendations[0]

    def test_empty_dataframe(self, empty_df: pd.DataFrame) -> None:
        """Test analysis on empty DataFrame."""
        analyzer = MissingValueAnalyzer()
        report = analyzer.analyze(empty_df)

        assert report.total_missing == 0
        assert report.total_cells == 0

    def test_column_missing_details(self, df_with_missing: pd.DataFrame) -> None:
        """Test that column-level missing details are correct."""
        analyzer = MissingValueAnalyzer()
        report = analyzer.analyze(df_with_missing)

        col_a_info = report.column_missing["A"]
        assert col_a_info["null_count"] > 0
        assert col_a_info["null_ratio"] > 0
        assert "severity" in col_a_info
        assert "suggested_strategy" in col_a_info

    def test_severity_classification(self, df_with_missing: pd.DataFrame) -> None:
        """Test severity classification levels."""
        analyzer = MissingValueAnalyzer()
        report = analyzer.analyze(df_with_missing)

        for col, info in report.column_missing.items():
            assert info["severity"] in ["low", "moderate", "high", "critical"]

    def test_recommendations_generated(self, df_with_missing: pd.DataFrame) -> None:
        """Test that recommendations are generated."""
        analyzer = MissingValueAnalyzer()
        report = analyzer.analyze(df_with_missing)

        assert len(report.recommendations) > 0

    def test_completely_empty_columns(self) -> None:
        """Test detection of completely empty columns."""
        df = pd.DataFrame({
            "full": [1, 2, 3, 4, 5],
            "empty": [np.nan, np.nan, np.nan, np.nan, np.nan],
        })

        analyzer = MissingValueAnalyzer()
        report = analyzer.analyze(df)

        assert "empty" in report.completely_empty_columns

    def test_custom_missing_indicators(self) -> None:
        """Test custom missing value indicators."""
        df = pd.DataFrame({
            "A": [1, 2, "?", 4, "N/A"],
            "B": ["x", "", "y", "?", "z"],
        })

        analyzer = MissingValueAnalyzer(missing_indicators=["?", "N/A"])
        report = analyzer.analyze(df)

        assert report.total_missing > 0

    def test_deletion_impact_row(self, df_with_missing: pd.DataFrame) -> None:
        """Test row-wise deletion impact analysis."""
        analyzer = MissingValueAnalyzer()
        impact = analyzer.get_deletion_impact(df_with_missing, strategy="row")

        assert "original_rows" in impact
        assert "remaining_rows" in impact
        assert "retention_ratio" in impact
        assert 0 <= impact["retention_ratio"] <= 1

    def test_deletion_impact_column(self, df_with_missing: pd.DataFrame) -> None:
        """Test column-wise deletion impact analysis."""
        analyzer = MissingValueAnalyzer()
        impact = analyzer.get_deletion_impact(df_with_missing, strategy="column")

        assert "original_columns" in impact
        assert "remaining_columns" in impact
        assert "dropped_columns" in impact

    def test_missing_patterns(self, df_with_missing: pd.DataFrame) -> None:
        """Test missing pattern detection."""
        analyzer = MissingValueAnalyzer()
        report = analyzer.analyze(df_with_missing)

        # Patterns may or may not be found depending on data
        assert report.missing_patterns is not None
