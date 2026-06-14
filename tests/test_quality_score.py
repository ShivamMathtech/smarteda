"""
Unit tests for the quality_score module.
"""

from __future__ import annotations

import pandas as pd
import pytest

from smarteda.analysis.quality_score import QualityScorer


class TestQualityScorer:
    """Test cases for QualityScorer."""

    def test_perfect_quality(self, sample_df: pd.DataFrame) -> None:
        """Test scoring on high-quality data."""
        scorer = QualityScorer()
        score = scorer.score(sample_df)

        assert score.overall_score > 0
        assert 0 <= score.overall_score <= 100
        assert score.grade in ["A", "B", "C", "D", "F"]
        assert len(score.score_breakdown) == 4

    def test_low_quality_data(self, low_quality_df: pd.DataFrame) -> None:
        """Test scoring on low-quality data."""
        scorer = QualityScorer()
        score = scorer.score(low_quality_df)

        assert score.overall_score < 100
        assert len(score.improvement_areas) > 0

    def test_empty_dataframe(self, empty_df: pd.DataFrame) -> None:
        """Test scoring on empty DataFrame."""
        scorer = QualityScorer()
        score = scorer.score(empty_df)

        assert score.overall_score == 0.0

    def test_completeness_score(self, df_with_missing: pd.DataFrame) -> None:
        """Test that completeness score reflects missing values."""
        scorer = QualityScorer()
        score = scorer.score(df_with_missing)

        assert score.completeness_score < 100

    def test_uniqueness_score(self, df_with_duplicates: pd.DataFrame) -> None:
        """Test that uniqueness score reflects duplicates."""
        scorer = QualityScorer()
        score = scorer.score(df_with_duplicates)

        assert score.uniqueness_score < 100

    def test_custom_weights(self, sample_df: pd.DataFrame) -> None:
        """Test with custom dimension weights."""
        weights = {
            "completeness": 0.5,
            "uniqueness": 0.2,
            "validity": 0.2,
            "consistency": 0.1,
        }

        scorer = QualityScorer(weights=weights)
        score = scorer.score(sample_df)

        assert 0 <= score.overall_score <= 100

    def test_score_breakdown_sums(self, sample_df: pd.DataFrame) -> None:
        """Test that score breakdown values are between 0 and 1."""
        scorer = QualityScorer()
        score = scorer.score(sample_df)

        for dim, val in score.score_breakdown.items():
            assert 0 <= val <= 1, f"{dim} score {val} out of range"

    def test_strengths_and_improvements(self, sample_df: pd.DataFrame) -> None:
        """Test that strengths and improvements are identified."""
        scorer = QualityScorer()
        score = scorer.score(sample_df)

        assert len(score.strengths) > 0

    def test_improvement_areas_for_bad_data(self, low_quality_df: pd.DataFrame) -> None:
        """Test improvement areas for low-quality data."""
        scorer = QualityScorer()
        score = scorer.score(low_quality_df)

        assert len(score.improvement_areas) > 0

    def test_grade_calculation(self) -> None:
        """Test grade boundaries."""
        scorer = QualityScorer()

        # Test various score levels through synthetic data
        perfect_df = pd.DataFrame({"A": [1, 2, 3, 4, 5]})
        score = scorer.score(perfect_df)

        assert score.grade in ["A", "B"]

    def test_compare_scores(self, sample_df: pd.DataFrame) -> None:
        """Test score comparison between two DataFrames."""
        scorer = QualityScorer()
        comparison = scorer.compare_scores(sample_df, sample_df)

        assert "current_overall" in comparison
        assert "previous_overall" in comparison
        assert "change" in comparison
        assert "trend" in comparison

    def test_validity_rules(self) -> None:
        """Test custom validity rules."""
        df = pd.DataFrame({
            "score": [10, 20, 30, 150, 25],  # 150 is out of range
        })

        rules = {"score": {"min": 0, "max": 100}}
        scorer = QualityScorer(validity_rules=rules)
        score = scorer.score(df)

        assert score.validity_score < 100
