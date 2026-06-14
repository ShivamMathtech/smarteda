"""
Data profiling module for comprehensive dataset overview.

Computes statistical summaries, distributions, and structural
information about the dataset to provide a complete picture
before detailed analysis.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from smarteda.models.analysis_result import DataType, SchemaProfile
from smarteda.utils.helpers import compute_entropy, is_numeric_series

logger = logging.getLogger(__name__)


class DataProfiler:
    """
    Comprehensive data profiler for statistical summaries.

    Generates descriptive statistics, distribution metrics,
    and structural information for all columns in a dataset.

    Example:
        >>> profiler = DataProfiler()
        >>> profile = profiler.profile(df)
        >>> print(profile["numeric_summary"])
    """

    def __init__(self, max_categories: int = 50, skewness_threshold: float = 2.0) -> None:
        """
        Initialize DataProfiler.

        Args:
            max_categories: Maximum categories to show in frequency tables.
            skewness_threshold: Threshold for flagging skewed distributions.
        """
        self.max_categories = max_categories
        self.skewness_threshold = skewness_threshold

    def profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a comprehensive profile of the dataset.

        Args:
            df: Input pandas DataFrame.

        Returns:
            Dictionary containing all profiling results.
        """
        if df.empty:
            return {"error": "Empty DataFrame"}

        profile_result = {
            "dataset_shape": df.shape,
            "total_cells": df.size,
            "memory_usage": self._get_memory_profile(df),
            "column_summary": self._get_column_summary(df),
            "numeric_summary": self._get_numeric_summary(df),
            "categorical_summary": self._get_categorical_summary(df),
            "datetime_summary": self._get_datetime_summary(df),
            "boolean_summary": self._get_boolean_summary(df),
            "distributions": self._analyze_distributions(df),
            "structural_flags": self._detect_structural_flags(df),
        }

        logger.info(
            "Profiled dataset: %d rows x %d columns",
            df.shape[0],
            df.shape[1],
        )

        return profile_result

    def _get_memory_profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get memory usage breakdown."""
        total = df.memory_usage(deep=True).sum()
        per_column = df.memory_usage(deep=True).to_dict()

        return {
            "total_bytes": int(total),
            "total_mb": round(total / (1024 * 1024), 2),
            "per_column": {str(k): int(v) for k, v in per_column.items()},
            "average_per_row": round(total / max(len(df), 1), 2),
        }

    def _get_column_summary(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Get summary for each column."""
        summary = {}

        for col in df.columns:
            series = df[col]
            null_count = int(series.isna().sum())
            null_pct = round(null_count / len(df) * 100, 2) if len(df) > 0 else 0

            col_summary = {
                "dtype": str(series.dtype),
                "non_null_count": int(series.notna().sum()),
                "null_count": null_count,
                "null_percentage": null_pct,
                "unique_count": int(series.nunique(dropna=True)),
                "is_constant": series.nunique(dropna=True) <= 1,
                "is_highly_null": null_pct > 50,
            }

            if is_numeric_series(series):
                col_summary.update({
                    "mean": round(float(series.mean()), 4) if series.notna().any() else None,
                    "std": round(float(series.std()), 4) if series.notna().sum() > 1 else None,
                    "min": float(series.min()) if series.notna().any() else None,
                    "max": float(series.max()) if series.notna().any() else None,
                })

            summary[str(col)] = col_summary

        return summary

    def _get_numeric_summary(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Get detailed summary for numeric columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) == 0:
            return {}

        summary = {}
        for col in numeric_cols:
            series = df[col].dropna()

            if len(series) == 0:
                continue

            summary[str(col)] = {
                "count": int(len(series)),
                "mean": round(float(series.mean()), 6),
                "std": round(float(series.std()), 6),
                "min": float(series.min()),
                "q1": float(series.quantile(0.25)),
                "median": float(series.median()),
                "q3": float(series.quantile(0.75)),
                "max": float(series.max()),
                "skewness": round(float(series.skew()), 4),
                "kurtosis": round(float(series.kurtosis()), 4),
                "range": float(series.max() - series.min()),
                "iqr": float(series.quantile(0.75) - series.quantile(0.25)),
                "cv": round(float(series.std() / series.mean()), 4) if series.mean() != 0 else None,
                "zeros_count": int((series == 0).sum()),
                "negatives_count": int((series < 0).sum()),
                "entropy": round(compute_entropy(series), 4),
            }

        return summary

    def _get_categorical_summary(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Get summary for categorical/object columns."""
        cat_cols = df.select_dtypes(include=["object", "category"]).columns

        if len(cat_cols) == 0:
            return {}

        summary = {}
        for col in cat_cols:
            series = df[col].dropna()

            if len(series) == 0:
                continue

            value_counts = series.value_counts()
            top_categories = value_counts.head(self.max_categories).to_dict()

            summary[str(col)] = {
                "unique_count": int(series.nunique()),
                "mode": str(series.mode().iloc[0]) if len(series.mode()) > 0 else None,
                "mode_frequency": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                "top_categories": {str(k): int(v) for k, v in top_categories.items()},
                "category_distribution": {
                    str(k): round(v / len(series) * 100, 2)
                    for k, v in list(top_categories.items())[:10]
                },
                "avg_length": round(float(series.astype(str).str.len().mean()), 2),
                "max_length": int(series.astype(str).str.len().max()),
                "empty_strings": int((series == "").sum()),
                "whitespace_only": int(series.astype(str).str.match(r"^\s+$").sum()),
                "entropy": round(compute_entropy(series), 4),
            }

        return summary

    def _get_datetime_summary(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Get summary for datetime columns."""
        dt_cols = df.select_dtypes(include=["datetime64", "datetime"]).columns

        if len(dt_cols) == 0:
            return {}

        summary = {}
        for col in dt_cols:
            series = df[col].dropna()

            if len(series) == 0:
                continue

            summary[str(col)] = {
                "count": int(len(series)),
                "min": str(series.min()),
                "max": str(series.max()),
                "range_days": int((series.max() - series.min()).days) if len(series) > 1 else 0,
                "most_common_year": int(series.dt.year.mode().iloc[0]) if len(series.dt.year.mode()) > 0 else None,
                "most_common_month": int(series.dt.month.mode().iloc[0]) if len(series.dt.month.mode()) > 0 else None,
                "weekend_ratio": round(float((series.dt.dayofweek >= 5).mean()) * 100, 2),
            }

        return summary

    def _get_boolean_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get summary for boolean columns."""
        bool_cols = df.select_dtypes(include=["bool"]).columns

        if len(bool_cols) == 0:
            # Also check for binary numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            bool_cols = [c for c in numeric_cols if df[c].dropna().isin([0, 1]).all()]

        if len(bool_cols) == 0:
            return {}

        summary = {}
        for col in bool_cols:
            series = df[col].dropna()
            true_count = int((series == True).sum()) if series.dtype == bool else int((series == 1).sum())

            summary[str(col)] = {
                "true_count": true_count,
                "false_count": int(len(series)) - true_count,
                "true_ratio": round(true_count / len(series) * 100, 2) if len(series) > 0 else 0,
            }

        return summary

    def _analyze_distributions(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Analyze distributions of numeric columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        distributions = {}
        for col in numeric_cols:
            series = df[col].dropna()

            if len(series) < 3:
                continue

            skewness = series.skew()

            dist_type = "normal"
            if abs(skewness) > self.skewness_threshold:
                dist_type = "highly_skewed_right" if skewness > 0 else "highly_skewed_left"
            elif abs(skewness) > 1:
                dist_type = "moderately_skewed_right" if skewness > 0 else "moderately_skewed_left"

            # Simple normality test using D'Agostino
            try:
                from scipy import stats
                _, p_value = stats.normaltest(series)
                is_normal = p_value > 0.05
            except ImportError:
                is_normal = abs(skewness) < 0.5

            distributions[str(col)] = {
                "distribution_type": dist_type,
                "skewness": round(float(skewness), 4),
                "kurtosis": round(float(series.kurtosis()), 4),
                "is_normal": bool(is_normal),
                "histogram_bins": self._compute_histogram_bins(series),
            }

        return distributions

    def _compute_histogram_bins(self, series: pd.Series, n_bins: int = 20) -> Dict[str, Any]:
        """Compute histogram bins for a series."""
        counts, bin_edges = np.histogram(series.dropna(), bins=n_bins)

        return {
            "counts": counts.tolist(),
            "bin_edges": [round(float(b), 4) for b in bin_edges.tolist()],
        }

    def _detect_structural_flags(self, df: pd.DataFrame) -> List[str]:
        """Detect structural issues in the dataset."""
        flags = []

        # Check for empty dataframe
        if df.empty:
            flags.append("Dataset is empty")
            return flags

        # Check for completely empty columns
        empty_cols = [c for c in df.columns if df[c].isna().all()]
        if empty_cols:
            flags.append(f"Completely empty columns: {empty_cols}")

        # Check for constant columns
        constant_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
        if constant_cols:
            flags.append(f"Constant columns (no variance): {constant_cols}")

        # Check for duplicate column names
        if len(df.columns) != len(set(df.columns)):
            flags.append("Duplicate column names detected")

        # Check for columns with very high missing rates
        high_missing = [
            str(c) for c in df.columns
            if df[c].isna().mean() > 0.8
        ]
        if high_missing:
            flags.append(f"Columns with >80% missing values: {high_missing}")

        # Check for potential index columns
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].dropna().nunique() == len(df) and df[col].is_monotonic_increasing:
                flags.append(f"Column '{col}' appears to be an index column")

        # Check for mixed types in object columns
        for col in df.select_dtypes(include=["object"]):
            types_in_col = df[col].dropna().apply(type).nunique()
            if types_in_col > 1:
                flags.append(f"Column '{col}' contains mixed data types")

        return flags

    def get_column_stats(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """
        Get detailed statistics for a specific column.

        Args:
            df: Input DataFrame.
            column: Column name to analyze.

        Returns:
            Dictionary with column statistics.
        """
        if column not in df.columns:
            return {"error": f"Column '{column}' not found"}

        series = df[column]
        stats = {
            "column": column,
            "dtype": str(series.dtype),
            "count": int(len(series)),
            "null_count": int(series.isna().sum()),
            "null_percentage": round(series.isna().mean() * 100, 2),
            "unique_count": int(series.nunique(dropna=True)),
        }

        if is_numeric_series(series):
            stats.update({
                "mean": round(float(series.mean()), 6),
                "median": float(series.median()),
                "std": round(float(series.std()), 6),
                "var": round(float(series.var()), 6),
                "min": float(series.min()),
                "max": float(series.max()),
                "range": float(series.max() - series.min()),
                "skewness": round(float(series.skew()), 4),
                "kurtosis": round(float(series.kurtosis()), 4),
            })

        return stats
