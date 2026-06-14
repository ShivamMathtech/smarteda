"""
Correlation analysis module.

Computes correlation matrices, identifies significant relationships,
and generates interactive heatmaps for exploratory analysis.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from smarteda.models.analysis_result import CorrelationReport
from smarteda.utils.helpers import is_numeric_series

logger = logging.getLogger(__name__)


class CorrelationAnalyzer:
    """
    Correlation analysis for numeric and categorical features.

    Supports Pearson, Spearman, and Kendall correlation methods.
    Identifies strong correlations, multicollinearity risks,
    and generates interactive heatmaps.

    Example:
        >>> analyzer = CorrelationAnalyzer(method="pearson")
        >>> report = analyzer.analyze(df)
        >>> print(report.high_correlations)
    """

    def __init__(
        self,
        method: str = "pearson",
        high_threshold: float = 0.7,
        low_threshold: float = 0.1,
        min_samples: int = 3,
    ) -> None:
        """
        Initialize CorrelationAnalyzer.

        Args:
            method: Correlation method ('pearson', 'spearman', 'kendall').
            high_threshold: Threshold for flagging high correlations.
            low_threshold: Threshold for flagging negligible correlations.
            min_samples: Minimum non-null pairs required.
        """
        self.method = method
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.min_samples = min_samples

    def analyze(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        target: Optional[str] = None,
    ) -> CorrelationReport:
        """
        Perform correlation analysis on the dataset.

        Args:
            df: Input pandas DataFrame.
            columns: Specific numeric columns to analyze.
            target: Target variable for feature-target correlations.

        Returns:
            CorrelationReport with all correlation results.
        """
        if df.empty:
            return CorrelationReport(method=self.method)

        numeric_cols = self._get_numeric_columns(df, columns)

        if len(numeric_cols) < 2:
            logger.warning("At least 2 numeric columns required for correlation analysis")
            return CorrelationReport(method=self.method, numeric_columns=numeric_cols)

        # Compute correlation matrix
        corr_matrix = self._compute_correlation(df[numeric_cols])

        # Find significant correlations
        high_pos, high_neg, low_corr = self._find_significant_correlations(
            corr_matrix, numeric_cols
        )

        # Feature-target correlations if target specified
        target_correlations = []
        if target and target in df.columns and target in numeric_cols:
            target_correlations = self._analyze_target_correlations(
                df, numeric_cols, target
            )

        recommendations = self._generate_recommendations(
            high_pos, high_neg, corr_matrix, numeric_cols, target
        )

        report = CorrelationReport(
            correlation_matrix=corr_matrix,
            method=self.method,
            numeric_columns=numeric_cols,
            high_correlations=high_pos + high_neg,
            low_correlations=low_corr,
            strong_positive_pairs=high_pos,
            strong_negative_pairs=high_neg,
            recommendations=recommendations,
        )

        logger.info(
            "Correlation analysis: %d strong positive, %d strong negative",
            len(high_pos),
            len(high_neg),
        )

        return report

    def _get_numeric_columns(
        self, df: pd.DataFrame, columns: Optional[List[str]] = None
    ) -> List[str]:
        """Get numeric columns for correlation analysis."""
        if columns:
            return [c for c in columns if c in df.columns and is_numeric_series(df[c])]

        return [c for c in df.columns if is_numeric_series(df[c])]

    def _compute_correlation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute the correlation matrix."""
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])

        if numeric_df.empty or len(numeric_df.columns) < 2:
            return pd.DataFrame()

        # Handle columns with too few non-null values
        valid_cols = [
            c for c in numeric_df.columns
            if numeric_df[c].notna().sum() >= self.min_samples
        ]

        if len(valid_cols) < 2:
            return pd.DataFrame()

        try:
            corr = numeric_df[valid_cols].corr(method=self.method)
            return corr.fillna(0)
        except Exception as e:
            logger.error("Error computing correlation: %s", e)
            return pd.DataFrame()

    def _find_significant_correlations(
        self, corr_matrix: pd.DataFrame, columns: List[str]
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Find high and low correlations from the matrix."""
        high_positive = []
        high_negative = []
        low_correlations = []

        if corr_matrix.empty:
            return high_positive, high_negative, low_correlations

        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                col_i = columns[i]
                col_j = columns[j]

                if col_i not in corr_matrix.columns or col_j not in corr_matrix.columns:
                    continue

                corr_val = corr_matrix.loc[col_i, col_j]

                if np.isnan(corr_val):
                    continue

                corr_info = {
                    "column_1": col_i,
                    "column_2": col_j,
                    "correlation": round(float(corr_val), 6),
                    "abs_correlation": round(abs(float(corr_val)), 6),
                }

                if abs(corr_val) >= self.high_threshold:
                    if corr_val > 0:
                        high_positive.append(corr_info)
                    else:
                        high_negative.append(corr_info)
                elif abs(corr_val) <= self.low_threshold:
                    low_correlations.append(corr_info)

        # Sort by absolute correlation
        high_positive.sort(key=lambda x: x["abs_correlation"], reverse=True)
        high_negative.sort(key=lambda x: x["abs_correlation"], reverse=True)

        return high_positive, high_negative, low_correlations

    def _analyze_target_correlations(
        self, df: pd.DataFrame, numeric_cols: List[str], target: str
    ) -> List[Dict[str, Any]]:
        """Analyze correlations with a target variable."""
        target_corrs = []

        for col in numeric_cols:
            if col == target:
                continue

            valid_data = df[[col, target]].dropna()

            if len(valid_data) < self.min_samples:
                continue

            try:
                corr = valid_data[col].corr(valid_data[target], method=self.method)
                if not np.isnan(corr):
                    target_corrs.append({
                        "feature": col,
                        "target": target,
                        "correlation": round(float(corr), 6),
                        "abs_correlation": round(abs(float(corr)), 6),
                    })
            except Exception:
                continue

        target_corrs.sort(key=lambda x: x["abs_correlation"], reverse=True)

        return target_corrs

    def _generate_recommendations(
        self,
        high_pos: List[Dict],
        high_neg: List[Dict],
        corr_matrix: pd.DataFrame,
        columns: List[str],
        target: Optional[str],
    ) -> List[str]:
        """Generate correlation-based recommendations."""
        recommendations = []

        if corr_matrix.empty or len(columns) < 2:
            recommendations.append("Insufficient numeric columns for correlation analysis.")
            return recommendations

        total_pairs = len(columns) * (len(columns) - 1) // 2
        significant_pairs = len(high_pos) + len(high_neg)

        recommendations.append(
            f"Analyzed {total_pairs} column pairs using {self.method} correlation."
        )

        if significant_pairs == 0:
            recommendations.append(
                "No strong correlations detected (threshold: "
                f"|r| >= {self.high_threshold})."
            )

        # Multicollinearity warnings
        if len(high_pos) > 0:
            very_high = [c for c in high_pos if c["abs_correlation"] >= 0.9]

            if very_high:
                recommendations.append(
                    f"MULTICOLLINEARITY RISK: {len(very_high)} pairs with |r| >= 0.9. "
                    "Consider removing one from each pair for regression models."
                )
                for pair in very_high[:5]:
                    recommendations.append(
                        f"  - {pair['column_1']} vs {pair['column_2']}: "
                        f"r = {pair['correlation']:.3f}"
                    )

            moderate_high = [c for c in high_pos if 0.7 <= c["abs_correlation"] < 0.9]

            if moderate_high:
                recommendations.append(
                    f"{len(moderate_high)} moderate-high correlations (0.7-0.9). "
                    "May cause multicollinearity in linear models."
                )

        # Feature selection for target
        if target:
            top_features = sorted(
                [
                    {"col": col, "corr": abs(corr_matrix.loc[col, target])}
                    for col in columns
                    if col != target and col in corr_matrix.columns
                    and target in corr_matrix.columns
                    and not np.isnan(corr_matrix.loc[col, target])
                ],
                key=lambda x: x["corr"],
                reverse=True,
            )[:5]

            if top_features:
                recommendations.append(
                    f"Top features correlated with '{target}':"
                )
                for feat in top_features:
                    recommendations.append(
                        f"  - {feat['col']}: |r| = {feat['corr']:.3f}"
                    )

        # Method-specific notes
        if self.method == "pearson":
            recommendations.append(
                "Pearson correlation measures linear relationships. "
                "Use Spearman for monotonic relationships."
            )
        elif self.method == "spearman":
            recommendations.append(
                "Spearman correlation is rank-based and robust to outliers."
            )

        return recommendations

    def get_correlation_heatmap_data(
        self, corr_matrix: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Convert correlation matrix to heatmap-friendly format.

        Args:
            corr_matrix: Correlation DataFrame.

        Returns:
            Dictionary with x, y, z arrays for heatmap plotting.
        """
        if corr_matrix.empty:
            return {"x": [], "y": [], "z": [], "annotations": []}

        columns = corr_matrix.columns.tolist()

        z = corr_matrix.values.tolist()

        annotations = []
        for i, col_i in enumerate(columns):
            for j, col_j in enumerate(columns):
                annotations.append({
                    "x": col_j,
                    "y": col_i,
                    "text": f"{corr_matrix.loc[col_i, col_j]:.2f}",
                    "font": {"color": "white" if abs(corr_matrix.loc[col_i, col_j]) > 0.5 else "black"},
                })

        return {
            "x": columns,
            "y": columns,
            "z": z,
            "annotations": annotations,
        }
