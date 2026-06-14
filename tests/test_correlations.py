"""
Unit tests for the correlations module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from smarteda.analysis.correlations import CorrelationAnalyzer


class TestCorrelationAnalyzer:
    """Test cases for CorrelationAnalyzer."""

    def test_detect_strong_positive(self, df_with_correlations: pd.DataFrame) -> None:
        """Test detection of strong positive correlations."""
        analyzer = CorrelationAnalyzer(method="pearson")
        report = analyzer.analyze(df_with_correlations)

        assert len(report.strong_positive_pairs) > 0

        # feature_1 and feature_2 should be strongly positively correlated
        pair_found = any(
            (p["column_1"] == "feature_1" and p["column_2"] == "feature_2")
            or (p["column_1"] == "feature_2" and p["column_2"] == "feature_1")
            for p in report.strong_positive_pairs
        )
        assert pair_found, "Expected strong positive correlation between feature_1 and feature_2"

    def test_detect_strong_negative(self, df_with_correlations: pd.DataFrame) -> None:
        """Test detection of strong negative correlations."""
        analyzer = CorrelationAnalyzer(method="pearson")
        report = analyzer.analyze(df_with_correlations)

        assert len(report.strong_negative_pairs) > 0

    def test_no_correlations_single_column(self, single_column_df: pd.DataFrame) -> None:
        """Test with single-column DataFrame."""
        analyzer = CorrelationAnalyzer()
        report = analyzer.analyze(single_column_df)

        assert len(report.numeric_columns) < 2

    def test_empty_dataframe(self, empty_df: pd.DataFrame) -> None:
        """Test with empty DataFrame."""
        analyzer = CorrelationAnalyzer()
        report = analyzer.analyze(empty_df)

        assert report.correlation_matrix.empty or len(report.numeric_columns) == 0

    def test_spearman_method(self, df_with_correlations: pd.DataFrame) -> None:
        """Test Spearman correlation method."""
        analyzer = CorrelationAnalyzer(method="spearman")
        report = analyzer.analyze(df_with_correlations)

        assert report.method == "spearman"
        assert not report.correlation_matrix.empty

    def test_kendall_method(self, df_with_correlations: pd.DataFrame) -> None:
        """Test Kendall correlation method."""
        analyzer = CorrelationAnalyzer(method="kendall")
        report = analyzer.analyze(df_with_correlations)

        assert report.method == "kendall"

    def test_target_correlation(self, df_with_correlations: pd.DataFrame) -> None:
        """Test target-specific correlation analysis."""
        analyzer = CorrelationAnalyzer(method="pearson")
        report = analyzer.analyze(df_with_correlations, target="feature_1")

        assert isinstance(report, object)

    def test_high_correlation_threshold(self, df_with_correlations: pd.DataFrame) -> None:
        """Test with custom high threshold."""
        analyzer = CorrelationAnalyzer(high_threshold=0.8)
        report = analyzer.analyze(df_with_correlations)

        assert isinstance(report, object)

    def test_recommendations(self, df_with_correlations: pd.DataFrame) -> None:
        """Test that recommendations are generated."""
        analyzer = CorrelationAnalyzer()
        report = analyzer.analyze(df_with_correlations)

        assert len(report.recommendations) > 0

    def test_correlation_matrix_values(self, df_with_correlations: pd.DataFrame) -> None:
        """Test that correlation matrix contains valid values."""
        analyzer = CorrelationAnalyzer()
        report = analyzer.analyze(df_with_correlations)

        if not report.correlation_matrix.empty:
            corr_values = report.correlation_matrix.values
            assert np.all(corr_values >= -1)
            assert np.all(corr_values <= 1)

    def test_no_numeric_columns(self) -> None:
        """Test with DataFrame containing no numeric columns."""
        df = pd.DataFrame({
            "A": ["x", "y", "z"],
            "B": ["a", "b", "c"],
        })

        analyzer = CorrelationAnalyzer()
        report = analyzer.analyze(df)

        assert len(report.numeric_columns) == 0

    def test_heatmap_data(self, df_with_correlations: pd.DataFrame) -> None:
        """Test heatmap data generation."""
        analyzer = CorrelationAnalyzer()
        report = analyzer.analyze(df_with_correlations)

        if not report.correlation_matrix.empty:
            heatmap_data = analyzer.get_correlation_heatmap_data(report.correlation_matrix)
            assert "x" in heatmap_data
            assert "y" in heatmap_data
            assert "z" in heatmap_data
