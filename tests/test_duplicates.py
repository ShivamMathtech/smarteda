"""
Unit tests for the duplicates module.
"""

from __future__ import annotations

import pandas as pd
import pytest

from smarteda.analysis.duplicates import DuplicateAnalyzer


class TestDuplicateAnalyzer:
    """Test cases for DuplicateAnalyzer."""

    def test_detect_exact_duplicates(self, df_with_duplicates: pd.DataFrame) -> None:
        """Test exact duplicate detection."""
        analyzer = DuplicateAnalyzer()
        report = analyzer.analyze(df_with_duplicates)

        assert report.exact_duplicates > 0
        assert report.exact_duplicate_ratio > 0

    def test_no_duplicates(self, sample_df: pd.DataFrame) -> None:
        """Test analysis on DataFrame with no duplicates."""
        analyzer = DuplicateAnalyzer()
        report = analyzer.analyze(sample_df)

        assert report.total_duplicates == 0
        assert len(report.recommendations) == 1
        assert "No duplicates" in report.recommendations[0]

    def test_empty_dataframe(self, empty_df: pd.DataFrame) -> None:
        """Test analysis on empty DataFrame."""
        analyzer = DuplicateAnalyzer()
        report = analyzer.analyze(empty_df)

        assert report.total_duplicates == 0

    def test_subset_analysis(self, df_with_duplicates: pd.DataFrame) -> None:
        """Test duplicate analysis on column subset."""
        analyzer = DuplicateAnalyzer()
        report = analyzer.analyze(df_with_duplicates, subset=["id"])

        assert isinstance(report.total_duplicates, int)

    def test_key_column_analysis(self, df_with_duplicates: pd.DataFrame) -> None:
        """Test key column duplicate detection."""
        analyzer = DuplicateAnalyzer()
        report = analyzer.analyze(df_with_duplicates, key_columns=["id"])

        assert "id" in report.key_column_duplicates

    def test_recommendations_for_duplicates(self, df_with_duplicates: pd.DataFrame) -> None:
        """Test that recommendations are generated for duplicates."""
        analyzer = DuplicateAnalyzer()
        report = analyzer.analyze(df_with_duplicates)

        assert len(report.recommendations) > 0

    def test_fuzzy_duplicate_detection(self, sample_df: pd.DataFrame) -> None:
        """Test fuzzy duplicate detection."""
        analyzer = DuplicateAnalyzer(fuzzy_threshold=0.9)
        report = analyzer.analyze(sample_df, check_fuzzy=True)

        assert isinstance(report.fuzzy_duplicates, int)

    def test_find_duplicate_groups(self, df_with_duplicates: pd.DataFrame) -> None:
        """Test finding duplicate groups."""
        analyzer = DuplicateAnalyzer()
        groups = analyzer.find_duplicate_groups(df_with_duplicates)

        assert isinstance(groups, pd.DataFrame)

    def test_custom_fuzzy_threshold(self, sample_df: pd.DataFrame) -> None:
        """Test with custom fuzzy threshold."""
        analyzer = DuplicateAnalyzer(fuzzy_threshold=0.95)
        report = analyzer.analyze(sample_df, check_fuzzy=True)

        assert isinstance(report, object)

    def test_high_duplicate_ratio_warning(self) -> None:
        """Test warning for high duplicate ratio."""
        df = pd.DataFrame({
            "A": [1, 1, 1, 1, 1],
            "B": ["x", "x", "x", "x", "x"],
        })

        analyzer = DuplicateAnalyzer()
        report = analyzer.analyze(df)

        assert report.exact_duplicate_ratio > 0.5
        assert any("CRITICAL" in rec for rec in report.recommendations)
