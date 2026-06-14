"""
Missing value analysis module.

Detects, analyzes, and provides actionable recommendations
for handling missing values in datasets.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from smarteda.models.analysis_result import MissingValueReport

logger = logging.getLogger(__name__)


class MissingValueAnalyzer:
    """
    Comprehensive missing value analyzer.

    Identifies missing patterns, provides deletion impact analysis,
    and suggests imputation strategies based on column types and
    missingness mechanisms.

    Example:
        >>> analyzer = MissingValueAnalyzer()
        >>> report = analyzer.analyze(df)
        >>> print(report.overall_missing_ratio)
    """

    # Missingness thresholds for recommendations
    CRITICAL_THRESHOLD = 0.50  # >50% missing
    HIGH_THRESHOLD = 0.20      # >20% missing
    MODERATE_THRESHOLD = 0.05  # >5% missing

    def __init__(self, missing_indicators: Optional[List[Any]] = None) -> None:
        """
        Initialize MissingValueAnalyzer.

        Args:
            missing_indicators: Additional values to treat as missing
                (e.g., ["N/A", "?", "NULL"]).
        """
        self.missing_indicators = missing_indicators or []
        self._standard_missing = [None, np.nan, "", " ", "NA", "N/A", "n/a", "null", "NULL", "None", "?", "-"]

    def analyze(self, df: pd.DataFrame) -> MissingValueReport:
        """
        Perform comprehensive missing value analysis.

        Args:
            df: Input pandas DataFrame.

        Returns:
            MissingValueReport with all analysis results.
        """
        if df.empty:
            return MissingValueReport()

        df_clean = self._standardize_missing_values(df.copy())

        total_cells = df_clean.size
        total_missing = int(df_clean.isna().sum().sum())
        overall_ratio = total_missing / total_cells if total_cells > 0 else 0.0

        column_missing = self._analyze_by_column(df_clean)
        missing_patterns = self._find_missing_patterns(df_clean)
        columns_with_missing = [
            str(col) for col in df_clean.columns
            if df_clean[col].isna().any()
        ]
        completely_empty = [
            str(col) for col in df_clean.columns
            if df_clean[col].isna().all()
        ]

        recommendations = self._generate_recommendations(
            df_clean, column_missing, overall_ratio
        )

        report = MissingValueReport(
            total_missing=total_missing,
            total_cells=total_cells,
            overall_missing_ratio=round(overall_ratio, 6),
            column_missing=column_missing,
            missing_patterns=missing_patterns,
            columns_with_missing=columns_with_missing,
            completely_empty_columns=completely_empty,
            recommendations=recommendations,
        )

        logger.info(
            "Missing value analysis: %d/%d cells missing (%.2f%%)",
            total_missing,
            total_cells,
            overall_ratio * 100,
        )

        return report

    def _standardize_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize custom missing indicators to NaN."""
        all_indicators = set(self._standard_missing + self.missing_indicators)
        df = df.copy()

        for col in df.select_dtypes(include=["object", "category"]).columns:
            df[col] = df[col].replace(all_indicators, np.nan)
            # Avoid FutureWarning about silent downcasting
            if hasattr(df[col], "infer_objects"):
                df[col] = df[col].infer_objects(copy=False)

        return df

    def _analyze_by_column(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Analyze missing values per column."""
        column_analysis = {}

        for col in df.columns:
            series = df[col]
            null_count = int(series.isna().sum())
            null_ratio = null_count / len(df) if len(df) > 0 else 0.0

            if null_count == 0:
                continue

            analysis = {
                "null_count": null_count,
                "null_ratio": round(null_ratio, 6),
                "non_null_count": int(series.notna().sum()),
                "severity": self._classify_severity(null_ratio),
            }

            if pd.api.types.is_numeric_dtype(series):
                non_null = series.dropna()
                if len(non_null) > 0:
                    analysis["mean"] = round(float(non_null.mean()), 4)
                    analysis["median"] = float(non_null.median())
                    analysis["std"] = round(float(non_null.std()), 4)

                    # Suggest best imputation strategy
                    cv = non_null.std() / non_null.mean() if non_null.mean() != 0 else float("inf")
                    if cv < 0.1:
                        analysis["suggested_strategy"] = "mean"
                    else:
                        analysis["suggested_strategy"] = "median"
                else:
                    analysis["suggested_strategy"] = "drop_column"
            elif pd.api.types.is_datetime64_any_dtype(series):
                analysis["suggested_strategy"] = "interpolate"
            else:
                analysis["suggested_strategy"] = "mode"

            column_analysis[str(col)] = analysis

        return column_analysis

    def _classify_severity(self, ratio: float) -> str:
        """Classify missing value severity level."""
        if ratio >= self.CRITICAL_THRESHOLD:
            return "critical"
        elif ratio >= self.HIGH_THRESHOLD:
            return "high"
        elif ratio >= self.MODERATE_THRESHOLD:
            return "moderate"
        else:
            return "low"

    def _find_missing_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Find common patterns of missing values across columns."""
        missing_df = df.isna()

        if not missing_df.any().any():
            return pd.DataFrame()

        # Find rows with missing values and which columns are missing
        pattern_counts = missing_df[missing_df.any(axis=1)].value_counts()

        if len(pattern_counts) == 0:
            return pd.DataFrame()

        # Convert to readable format
        patterns = []
        for pattern_tuple, count in pattern_counts.head(10).items():
            missing_cols = [
                str(col) for col, is_missing in zip(df.columns, pattern_tuple)
                if is_missing
            ]
            patterns.append({
                "missing_columns": ", ".join(missing_cols),
                "row_count": int(count),
                "percentage": round(count / len(df) * 100, 2),
            })

        return pd.DataFrame(patterns)

    def _generate_recommendations(
        self,
        df: pd.DataFrame,
        column_missing: Dict[str, Dict[str, Any]],
        overall_ratio: float,
    ) -> List[str]:
        """Generate actionable recommendations for missing values."""
        recommendations = []

        if overall_ratio == 0:
            recommendations.append("No missing values detected - dataset is complete.")
            return recommendations

        # Overall recommendations
        if overall_ratio > 0.3:
            recommendations.append(
                f"CRITICAL: {overall_ratio:.1%} of all data is missing. "
                "Consider reviewing data collection processes."
            )
        elif overall_ratio > 0.1:
            recommendations.append(
                f"WARNING: {overall_ratio:.1%} of data is missing. "
                "Imputation recommended before analysis."
            )

        # Column-specific recommendations
        drop_candidates = []
        impute_candidates = []

        for col, info in column_missing.items():
            severity = info.get("severity", "low")

            if severity == "critical":
                drop_candidates.append(col)
            elif severity in ("high", "moderate"):
                impute_candidates.append((col, info.get("suggested_strategy", "median")))

        if drop_candidates:
            recommendations.append(
                f"Consider dropping columns with >50% missing: {drop_candidates}"
            )

        if impute_candidates:
            strategy_groups: Dict[str, List[str]] = {}
            for col, strategy in impute_candidates:
                strategy_groups.setdefault(strategy, []).append(col)

            for strategy, cols in strategy_groups.items():
                recommendations.append(
                    f"Use {strategy} imputation for: {cols}"
                )

        # Check for rows with excessive missing data
        rows_with_missing = df.isna().sum(axis=1)
        high_missing_rows = int((rows_with_missing > df.shape[1] * 0.5).sum())

        if high_missing_rows > 0:
            recommendations.append(
                f"{high_missing_rows} rows have >50% missing values - consider dropping them"
            )

        # Check if missing is systematic
        if len(column_missing) > 1:
            correlations = df.isna().corr()
            high_corr_pairs = []

            for i in range(len(correlations.columns)):
                for j in range(i + 1, len(correlations.columns)):
                    corr_val = correlations.iloc[i, j]
                    if not np.isnan(corr_val) and abs(corr_val) > 0.7:
                        high_corr_pairs.append(
                            (correlations.columns[i], correlations.columns[j], corr_val)
                        )

            if high_corr_pairs:
                recommendations.append(
                    "Correlated missing patterns detected - missingness may not be random (MNAR)"
                )

        return recommendations

    def get_deletion_impact(
        self, df: pd.DataFrame, strategy: str = "row"
    ) -> Dict[str, Any]:
        """
        Analyze the impact of listwise deletion.

        Args:
            df: Input DataFrame.
            strategy: 'row' for row-wise or 'column' for column-wise deletion.

        Returns:
            Dictionary with deletion impact analysis.
        """
        if strategy == "row":
            complete_cases = df.dropna()
            dropped = len(df) - len(complete_cases)

            return {
                "strategy": "row",
                "original_rows": len(df),
                "remaining_rows": len(complete_cases),
                "dropped_rows": dropped,
                "retention_ratio": round(len(complete_cases) / len(df), 4) if len(df) > 0 else 0,
                "data_loss_percentage": round(dropped / len(df) * 100, 2) if len(df) > 0 else 0,
            }
        else:
            complete_cols = df.dropna(axis=1)
            dropped = len(df.columns) - len(complete_cols.columns)

            return {
                "strategy": "column",
                "original_columns": len(df.columns),
                "remaining_columns": len(complete_cols.columns),
                "dropped_columns": dropped,
                "retention_ratio": round(len(complete_cols.columns) / len(df.columns), 4),
                "dropped_column_names": list(set(df.columns) - set(complete_cols.columns)),
            }
