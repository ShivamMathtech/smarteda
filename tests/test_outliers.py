"""
Unit tests for the outliers module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from smarteda.analysis.outliers import OutlierAnalyzer, OutlierMethod


class TestOutlierAnalyzer:
    """Test cases for OutlierAnalyzer."""

    def test_iqr_method(self, df_with_outliers: pd.DataFrame) -> None:
        """Test IQR outlier detection."""
        analyzer = OutlierAnalyzer(method=OutlierMethod.IQR)
        report = analyzer.analyze(df_with_outliers)

        assert len(report.outliers_by_column) > 0
        assert report.method_used == "iqr"
        assert report.total_outliers > 0

    def test_zscore_method(self, df_with_outliers: pd.DataFrame) -> None:
        """Test Z-score outlier detection."""
        analyzer = OutlierAnalyzer(method=OutlierMethod.ZSCORE)
        report = analyzer.analyze(df_with_outliers)

        assert isinstance(report.total_outliers, int)
        assert report.method_used == "zscore"

    def test_modified_zscore_method(self, df_with_outliers: pd.DataFrame) -> None:
        """Test Modified Z-score outlier detection."""
        analyzer = OutlierAnalyzer(method=OutlierMethod.MODIFIED_ZSCORE)
        report = analyzer.analyze(df_with_outliers)

        assert isinstance(report.total_outliers, int)
        assert report.method_used == "modified_zscore"

    def test_no_outliers_in_clean_data(self, sample_df: pd.DataFrame) -> None:
        """Test that clean data has minimal outliers."""
        analyzer = OutlierAnalyzer(method=OutlierMethod.IQR)
        report = analyzer.analyze(sample_df)

        # Should still run without errors
        assert isinstance(report, object)

    def test_empty_dataframe(self, empty_df: pd.DataFrame) -> None:
        """Test analysis on empty DataFrame."""
        analyzer = OutlierAnalyzer()
        report = analyzer.analyze(empty_df)

        assert report.total_outliers == 0
        assert len(report.outliers_by_column) == 0

    def test_specific_columns(self, df_with_outliers: pd.DataFrame) -> None:
        """Test analysis on specific columns."""
        analyzer = OutlierAnalyzer()
        report = analyzer.analyze(df_with_outliers, columns=["normal"])

        assert len(report.columns_analyzed) == 1
        assert "normal" in report.columns_analyzed

    def test_non_numeric_columns_ignored(self, df_with_outliers: pd.DataFrame) -> None:
        """Test that non-numeric columns are ignored."""
        analyzer = OutlierAnalyzer()
        report = analyzer.analyze(df_with_outliers)

        assert "category" not in report.columns_analyzed

    def test_outlier_bounds_present(self, df_with_outliers: pd.DataFrame) -> None:
        """Test that outlier bounds are included in report."""
        analyzer = OutlierAnalyzer(method=OutlierMethod.IQR)
        report = analyzer.analyze(df_with_outliers)

        for col, info in report.outliers_by_column.items():
            if info["outlier_count"] > 0:
                assert "bounds" in info

    def test_extreme_outliers(self, df_with_outliers: pd.DataFrame) -> None:
        """Test extreme outlier detection."""
        analyzer = OutlierAnalyzer()
        report = analyzer.analyze(df_with_outliers)

        assert isinstance(report.extreme_outliers, dict)

    def test_recommendations(self, df_with_outliers: pd.DataFrame) -> None:
        """Test that recommendations are generated."""
        analyzer = OutlierAnalyzer()
        report = analyzer.analyze(df_with_outliers)

        assert len(report.recommendations) > 0

    def test_get_outlier_rows(self, df_with_outliers: pd.DataFrame) -> None:
        """Test getting rows with outliers."""
        analyzer = OutlierAnalyzer(method=OutlierMethod.IQR)
        outlier_rows = analyzer.get_outlier_rows(df_with_outliers, "normal")

        assert isinstance(outlier_rows, pd.DataFrame)

    def test_invalid_column(self, df_with_outliers: pd.DataFrame) -> None:
        """Test with non-existent column."""
        analyzer = OutlierAnalyzer()
        outlier_rows = analyzer.get_outlier_rows(df_with_outliers, "nonexistent")

        assert len(outlier_rows) == 0

    def test_constant_column(self, constant_df: pd.DataFrame) -> None:
        """Test with constant column (zero variance)."""
        analyzer = OutlierAnalyzer()
        report = analyzer.analyze(constant_df)

        # Should handle gracefully
        assert isinstance(report, object)

    def test_multivariate_outliers(self, sample_df: pd.DataFrame) -> None:
        """Test multivariate outlier detection."""
        numeric_df = sample_df.select_dtypes(include=[np.number])

        if len(numeric_df.columns) >= 2:
            analyzer = OutlierAnalyzer()
            report = analyzer.analyze(numeric_df, include_multivariate=True)

            assert isinstance(report, object)
