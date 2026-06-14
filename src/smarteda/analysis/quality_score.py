"""
Data quality scoring module.

Computes an overall data quality score based on completeness,
uniqueness, validity, consistency, and timeliness dimensions.
Provides actionable feedback for data improvement.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from smarteda.models.analysis_result import QualityScore
from smarteda.utils.helpers import is_numeric_series

logger = logging.getLogger(__name__)


class QualityScorer:
    """
    Comprehensive data quality scoring engine.

    Evaluates data across multiple quality dimensions and computes
    a weighted overall score with detailed breakdown and improvement
    recommendations.

    Quality Dimensions:
        - Completeness: Absence of missing values
        - Uniqueness: Absence of duplicates
        - Validity: Conformance to expected formats/ranges
        - Consistency: Uniformity in representation

    Example:
        >>> scorer = QualityScorer()
        >>> score = scorer.score(df)
        >>> print(score.overall_score, score.grade)
    """

    # Weight configuration for quality dimensions
    DEFAULT_WEIGHTS = {
        "completeness": 0.30,
        "uniqueness": 0.20,
        "validity": 0.25,
        "consistency": 0.25,
    }

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        validity_rules: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize QualityScorer.

        Args:
            weights: Custom weights for quality dimensions.
                Must sum to 1.0.
            validity_rules: Custom validity rules per column.
                Format: {"column_name": {"min": 0, "max": 100}}.
        """
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self.validity_rules = validity_rules or {}

        # Normalize weights
        total = sum(self.weights.values())
        if total != 1.0:
            self.weights = {k: v / total for k, v in self.weights.items()}

    def score(self, df: pd.DataFrame) -> QualityScore:
        """
        Compute comprehensive data quality score.

        Args:
            df: Input pandas DataFrame.

        Returns:
            QualityScore with all dimension scores and recommendations.
        """
        if df.empty:
            return QualityScore(
                overall_score=0.0,
                summary="Empty dataset - cannot compute quality score",
            )

        completeness = self._score_completeness(df)
        uniqueness = self._score_uniqueness(df)
        validity = self._score_validity(df)
        consistency = self._score_consistency(df)

        overall = (
            completeness * self.weights["completeness"]
            + uniqueness * self.weights["uniqueness"]
            + validity * self.weights["validity"]
            + consistency * self.weights["consistency"]
        )

        score_breakdown = {
            "completeness": round(completeness, 4),
            "uniqueness": round(uniqueness, 4),
            "validity": round(validity, 4),
            "consistency": round(consistency, 4),
        }

        improvement_areas = self._identify_improvements(
            score_breakdown, df
        )
        strengths = self._identify_strengths(score_breakdown)

        quality_score = QualityScore(
            overall_score=round(overall * 100, 2),
            completeness_score=round(completeness * 100, 2),
            uniqueness_score=round(uniqueness * 100, 2),
            validity_score=round(validity * 100, 2),
            consistency_score=round(consistency * 100, 2),
            score_breakdown=score_breakdown,
            improvement_areas=improvement_areas,
            strengths=strengths,
        )

        logger.info(
            "Quality score: %.1f/100 (Grade: %s)",
            quality_score.overall_score,
            quality_score.grade,
        )

        return quality_score

    def _score_completeness(self, df: pd.DataFrame) -> float:
        """Score data completeness (0-1)."""
        if df.empty:
            return 0.0

        total_cells = df.size
        missing_cells = df.isna().sum().sum()

        return 1.0 - (missing_cells / total_cells) if total_cells > 0 else 0.0

    def _score_uniqueness(self, df: pd.DataFrame) -> float:
        """Score data uniqueness (0-1)."""
        if df.empty or len(df) <= 1:
            return 1.0

        # Exact duplicate penalty
        duplicate_ratio = df.duplicated().sum() / len(df)

        # Key column duplicate penalty
        key_penalty = 0.0
        for col in df.columns:
            if "id" in col.lower() or "key" in col.lower():
                dup_ratio = df[col].duplicated().sum() / len(df)
                key_penalty = max(key_penalty, dup_ratio * 0.5)

        score = 1.0 - duplicate_ratio - key_penalty
        return max(0.0, score)

    def _score_validity(self, df: pd.DataFrame) -> float:
        """Score data validity (0-1)."""
        if df.empty:
            return 0.0

        scores = []

        for col in df.columns:
            series = df[col].dropna()

            if len(series) == 0:
                continue

            col_score = self._validate_column(series, col)
            scores.append(col_score)

        return np.mean(scores) if scores else 1.0

    def _validate_column(self, series: pd.Series, col_name: str) -> float:
        """Validate a single column against rules."""
        rules = self.validity_rules.get(col_name, {})

        valid_count = len(series)

        # Range validation
        if "min" in rules and is_numeric_series(series):
            valid_count = min(valid_count, (series >= rules["min"]).sum())

        if "max" in rules and is_numeric_series(series):
            valid_count = min(valid_count, (series <= rules["max"]).sum())

        # Allowed values validation
        if "allowed" in rules:
            allowed = set(rules["allowed"])
            valid_count = min(valid_count, series.isin(allowed).sum())

        # Pattern validation for strings
        if "pattern" in rules and series.dtype == "object":
            import re
            pattern = re.compile(rules["pattern"])
            valid_count = min(
                valid_count,
                series.astype(str).str.match(pattern).sum(),
            )

        # Auto-detect anomalies for numeric columns
        if is_numeric_series(series):
            # Check for impossible values
            if (series < 0).any():
                negative_ratio = (series < 0).sum() / len(series)
                if "min" not in rules or rules.get("min", -float("inf")) >= 0:
                    valid_count = int(valid_count * (1 - negative_ratio * 0.5))

        return valid_count / len(series) if len(series) > 0 else 1.0

    def _score_consistency(self, df: pd.DataFrame) -> float:
        """Score data consistency (0-1)."""
        if df.empty:
            return 0.0

        scores = []

        for col in df.select_dtypes(include=["object"]).columns:
            series = df[col].dropna().astype(str)

            if len(series) == 0:
                continue

            # Check case consistency
            lower_ratio = series.str.islower().sum() / len(series)
            upper_ratio = series.str.isupper().sum() / len(series)
            title_ratio = series.str.istitle().sum() / len(series)

            max_case_ratio = max(lower_ratio, upper_ratio, title_ratio)
            case_consistency = max_case_ratio if max_case_ratio > 0.7 else 0.5

            # Check whitespace consistency
            has_leading = series.str.match(r"^\s").any()
            has_trailing = series.str.match(r"\s$").any()
            whitespace_penalty = 0.2 if (has_leading or has_trailing) else 0.0

            col_score = case_consistency - whitespace_penalty
            scores.append(max(0.0, col_score))

        # Type consistency
        for col in df.columns:
            series = df[col]
            non_null = series.dropna()

            if len(non_null) == 0:
                continue

            types = non_null.apply(type).nunique()
            if types > 1:
                scores.append(0.5)  # Penalty for mixed types
            else:
                scores.append(1.0)

        return np.mean(scores) if scores else 1.0

    def _identify_improvements(
        self, scores: Dict[str, float], df: pd.DataFrame
    ) -> List[str]:
        """Identify areas for improvement."""
        improvements = []

        if scores["completeness"] < 0.9:
            missing_pct = (1 - scores["completeness"]) * 100
            improvements.append(
                f"Completeness ({scores['completeness']:.1%}): "
                f"{missing_pct:.1f}% of data is missing. "
                "Consider imputation or improved data collection."
            )

        if scores["uniqueness"] < 0.9:
            improvements.append(
                f"Uniqueness ({scores['uniqueness']:.1%}): "
                "Duplicate records detected. Consider deduplication."
            )

        if scores["validity"] < 0.9:
            improvements.append(
                f"Validity ({scores['validity']:.1%}): "
                "Some values fall outside expected ranges or formats."
            )

        if scores["consistency"] < 0.9:
            improvements.append(
                f"Consistency ({scores['consistency']:.1%}): "
                "Inconsistent formatting detected in text columns. "
                "Standardize case and whitespace handling."
            )

        # Specific column recommendations
        for col in df.columns:
            null_ratio = df[col].isna().mean()
            if null_ratio > 0.5:
                improvements.append(
                    f"Column '{col}' has {null_ratio:.1%} missing - "
                    "consider dropping or heavy imputation"
                )

        return improvements

    def _identify_strengths(self, scores: Dict[str, float]) -> List[str]:
        """Identify data quality strengths."""
        strengths = []

        if scores["completeness"] >= 0.95:
            strengths.append("Excellent data completeness")

        if scores["uniqueness"] >= 0.95:
            strengths.append("No duplicate records detected")

        if scores["validity"] >= 0.95:
            strengths.append("All values within valid ranges")

        if scores["consistency"] >= 0.95:
            strengths.append("Highly consistent data formatting")

        if not strengths:
            strengths.append("Dataset has acceptable baseline quality")

        return strengths

    def compare_scores(
        self, current: pd.DataFrame, previous: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Compare quality scores between two dataset versions.

        Args:
            current: Current DataFrame.
            previous: Previous DataFrame for comparison.

        Returns:
            Dictionary with score comparison results.
        """
        current_score = self.score(current)
        previous_score = self.score(previous)

        comparison = {
            "current_overall": current_score.overall_score,
            "previous_overall": previous_score.overall_score,
            "change": round(
                current_score.overall_score - previous_score.overall_score, 2
            ),
            "dimension_changes": {},
        }

        for dim in ["completeness", "uniqueness", "validity", "consistency"]:
            current_val = getattr(current_score, f"{dim}_score")
            previous_val = getattr(previous_score, f"{dim}_score")
            comparison["dimension_changes"][dim] = {
                "current": current_val,
                "previous": previous_val,
                "change": round(current_val - previous_val, 2),
            }

        if comparison["change"] > 0:
            comparison["trend"] = "improved"
        elif comparison["change"] < 0:
            comparison["trend"] = "degraded"
        else:
            comparison["trend"] = "stable"

        return comparison
