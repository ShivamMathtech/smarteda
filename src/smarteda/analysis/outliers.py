"""
Outlier detection module.

Provides multiple outlier detection methods including IQR,
Z-score, Isolation Forest, and LOF. Supports both univariate
and multivariate outlier detection with configurable parameters.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from smarteda.models.analysis_result import OutlierReport
from smarteda.utils.helpers import is_numeric_series

logger = logging.getLogger(__name__)


class OutlierMethod(str, Enum):
    """Supported outlier detection methods."""

    IQR = "iqr"
    ZSCORE = "zscore"
    ISOLATION_FOREST = "isolation_forest"
    LOF = "lof"
    MODIFIED_ZSCORE = "modified_zscore"


class OutlierAnalyzer:
    """
    Multi-method outlier detection analyzer.

    Supports IQR, Z-score, Modified Z-score, Isolation Forest,
    and Local Outlier Factor methods with automatic method
    selection based on data characteristics.

    Example:
        >>> analyzer = OutlierAnalyzer(method="iqr")
        >>> report = analyzer.analyze(df)
        >>> print(report.total_outliers)
    """

    def __init__(
        self,
        method: OutlierMethod = OutlierMethod.IQR,
        iqr_multiplier: float = 1.5,
        zscore_threshold: float = 3.0,
        contamination: float = 0.05,
        random_state: int = 42,
        max_rows_for_multivariate: int = 50000,
    ) -> None:
        """
        Initialize OutlierAnalyzer.

        Args:
            method: Outlier detection method to use.
            iqr_multiplier: Multiplier for IQR bounds (default 1.5).
            zscore_threshold: Z-score threshold (default 3.0).
            contamination: Expected outlier proportion for isolation forest.
            random_state: Random seed for reproducibility.
            max_rows_for_multivariate: Max rows for multivariate methods.
        """
        self.method = OutlierMethod(method)
        self.iqr_multiplier = iqr_multiplier
        self.zscore_threshold = zscore_threshold
        self.contamination = contamination
        self.random_state = random_state
        self.max_rows_for_multivariate = max_rows_for_multivariate

    def analyze(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        include_multivariate: bool = False,
    ) -> OutlierReport:
        """
        Perform outlier detection analysis.

        Args:
            df: Input pandas DataFrame.
            columns: Specific columns to analyze (None = all numeric).
            include_multivariate: Whether to include multivariate detection.

        Returns:
            OutlierReport with all detection results.
        """
        if df.empty:
            return OutlierReport()

        target_cols = self._get_target_columns(df, columns)

        if not target_cols:
            logger.warning("No numeric columns found for outlier detection")
            return OutlierReport()

        outliers_by_column: Dict[str, Dict[str, Any]] = {}
        total_outliers = 0
        extreme_outliers: Dict[str, int] = {}
        outlier_indices: Dict[str, List[int]] = {}

        for col in target_cols:
            series = df[col].dropna()

            if len(series) < 3:
                continue

            col_outliers, col_extreme, indices, details = self._detect_outliers(series)

            if col_outliers > 0:
                outliers_by_column[str(col)] = {
                    "outlier_count": col_outliers,
                    "outlier_ratio": round(col_outliers / len(series), 6),
                    "extreme_count": col_extreme,
                    "bounds": details.get("bounds", {}),
                    "method": self.method.value,
                }
                total_outliers += col_outliers
                extreme_outliers[str(col)] = col_extreme
                outlier_indices[str(col)] = indices

        # Multivariate outlier detection
        mv_outliers = 0
        if include_multivariate and len(target_cols) >= 2:
            mv_outliers = self._detect_multivariate_outliers(df[target_cols])

        total_outliers += mv_outliers

        outlier_ratio = total_outliers / (len(df) * len(target_cols)) if target_cols else 0

        recommendations = self._generate_recommendations(
            outliers_by_column, include_multivariate, mv_outliers
        )

        report = OutlierReport(
            columns_analyzed=[str(c) for c in target_cols],
            outliers_by_column=outliers_by_column,
            total_outliers=total_outliers,
            outlier_ratio=round(outlier_ratio, 6),
            method_used=self.method.value,
            iqr_multiplier=self.iqr_multiplier,
            zscore_threshold=self.zscore_threshold,
            extreme_outliers=extreme_outliers,
            outlier_indices={k: v[:1000] for k, v in outlier_indices.items()},
            recommendations=recommendations,
        )

        logger.info(
            "Outlier analysis: %d total outliers across %d columns",
            total_outliers,
            len(outliers_by_column),
        )

        return report

    def _get_target_columns(
        self, df: pd.DataFrame, columns: Optional[List[str]] = None
    ) -> List[str]:
        """Get the list of numeric columns to analyze (excludes boolean)."""
        def _is_valid_numeric(col_name: str) -> bool:
            if col_name not in df.columns:
                return False
            series = df[col_name]
            # Exclude boolean columns - quantile doesn't work on bool
            if pd.api.types.is_bool_dtype(series.dtype):
                return False
            return is_numeric_series(series)

        if columns:
            return [c for c in columns if _is_valid_numeric(c)]

        return [c for c in df.columns if _is_valid_numeric(c)]

    def _detect_outliers(
        self, series: pd.Series
    ) -> Tuple[int, int, List[int], Dict[str, Any]]:
        """Detect outliers in a single column using the configured method."""
        if self.method == OutlierMethod.IQR:
            return self._iqr_method(series)
        elif self.method == OutlierMethod.ZSCORE:
            return self._zscore_method(series)
        elif self.method == OutlierMethod.MODIFIED_ZSCORE:
            return self._modified_zscore_method(series)
        elif self.method == OutlierMethod.ISOLATION_FOREST:
            return self._isolation_forest_univariate(series)
        elif self.method == OutlierMethod.LOF:
            return self._lof_univariate(series)
        else:
            return self._iqr_method(series)

    def _iqr_method(self, series: pd.Series) -> Tuple[int, int, List[int], Dict[str, Any]]:
        """Detect outliers using Interquartile Range method."""
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - self.iqr_multiplier * iqr
        upper_bound = q3 + self.iqr_multiplier * iqr

        extreme_lower = q1 - 3 * iqr
        extreme_upper = q3 + 3 * iqr

        outlier_mask = (series < lower_bound) | (series > upper_bound)
        extreme_mask = (series < extreme_lower) | (series > extreme_upper)

        outlier_indices = series[outlier_mask].index.tolist()

        return (
            int(outlier_mask.sum()),
            int(extreme_mask.sum()),
            outlier_indices,
            {
                "bounds": {
                    "lower": float(lower_bound),
                    "upper": float(upper_bound),
                    "extreme_lower": float(extreme_lower),
                    "extreme_upper": float(extreme_upper),
                    "q1": float(q1),
                    "q3": float(q3),
                    "iqr": float(iqr),
                }
            },
        )

    def _zscore_method(self, series: pd.Series) -> Tuple[int, int, List[int], Dict[str, Any]]:
        """Detect outliers using Z-score method."""
        mean = series.mean()
        std = series.std()

        if std == 0:
            return 0, 0, [], {"bounds": {}}

        z_scores = np.abs((series - mean) / std)
        outlier_mask = z_scores > self.zscore_threshold
        extreme_mask = z_scores > (self.zscore_threshold * 1.5)

        outlier_indices = series[outlier_mask].index.tolist()

        return (
            int(outlier_mask.sum()),
            int(extreme_mask.sum()),
            outlier_indices,
            {
                "bounds": {
                    "threshold": self.zscore_threshold,
                    "mean": float(mean),
                    "std": float(std),
                }
            },
        )

    def _modified_zscore_method(
        self, series: pd.Series
    ) -> Tuple[int, int, List[int], Dict[str, Any]]:
        """Detect outliers using Modified Z-score (MAD-based) method."""
        median = series.median()
        mad = np.median(np.abs(series - median))

        if mad == 0:
            return 0, 0, [], {"bounds": {}}

        modified_z_scores = 0.6745 * (series - median) / mad
        outlier_mask = np.abs(modified_z_scores) > self.zscore_threshold
        extreme_mask = np.abs(modified_z_scores) > (self.zscore_threshold * 1.5)

        outlier_indices = series[outlier_mask].index.tolist()

        return (
            int(outlier_mask.sum()),
            int(extreme_mask.sum()),
            outlier_indices,
            {
                "bounds": {
                    "threshold": self.zscore_threshold,
                    "median": float(median),
                    "mad": float(mad),
                }
            },
        )

    def _isolation_forest_univariate(
        self, series: pd.Series
    ) -> Tuple[int, int, List[int], Dict[str, Any]]:
        """Apply Isolation Forest to a single column."""
        if len(series) < 10:
            return 0, 0, [], {"bounds": {}}

        data = series.values.reshape(-1, 1)

        clf = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=100,
        )

        predictions = clf.fit_predict(data)
        scores = clf.score_samples(data)

        outlier_mask = predictions == -1
        extreme_mask = scores < np.percentile(scores[outlier_mask], 25) if outlier_mask.any() else pd.Series(False, index=series.index)

        outlier_indices = series[outlier_mask].index.tolist()

        return (
            int(outlier_mask.sum()),
            int(extreme_mask.sum()) if isinstance(extreme_mask, pd.Series) else 0,
            outlier_indices,
            {"bounds": {"method": "isolation_forest", "contamination": self.contamination}},
        )

    def _lof_univariate(
        self, series: pd.Series
    ) -> Tuple[int, int, List[int], Dict[str, Any]]:
        """Apply Local Outlier Factor to a single column."""
        if len(series) < 10:
            return 0, 0, [], {"bounds": {}}

        data = series.values.reshape(-1, 1)

        n_neighbors = min(20, len(data) - 1)
        clf = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=self.contamination)

        predictions = clf.fit_predict(data)
        outlier_mask = predictions == -1

        outlier_indices = series[outlier_mask].index.tolist()

        return (
            int(outlier_mask.sum()),
            0,
            outlier_indices,
            {"bounds": {"method": "lof", "n_neighbors": n_neighbors}},
        )

    def _detect_multivariate_outliers(self, df: pd.DataFrame) -> int:
        """Detect multivariate outliers using Isolation Forest."""
        if len(df) > self.max_rows_for_multivariate:
            logger.info(
                "Sampling %d rows for multivariate outlier detection",
                self.max_rows_for_multivariate,
            )
            df = df.sample(n=self.max_rows_for_multivariate, random_state=self.random_state)

        numeric_df = df.select_dtypes(include=[np.number]).dropna()

        if len(numeric_df) < 10 or numeric_df.shape[1] < 2:
            return 0

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)

        clf = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=100,
        )

        predictions = clf.fit_predict(scaled_data)
        outlier_count = int((predictions == -1).sum())

        logger.info("Multivariate outliers detected: %d", outlier_count)

        return outlier_count

    def _generate_recommendations(
        self,
        outliers_by_column: Dict[str, Dict[str, Any]],
        include_multivariate: bool,
        mv_outliers: int,
    ) -> List[str]:
        """Generate outlier handling recommendations."""
        recommendations = []

        if not outliers_by_column:
            recommendations.append("No outliers detected using the selected method.")
            return recommendations

        total_outlier_cols = len(outliers_by_column)
        total_outliers = sum(d["outlier_count"] for d in outliers_by_column.values())

        recommendations.append(
            f"Detected outliers in {total_outlier_cols} columns "
            f"({total_outliers} total outliers)."
        )

        # Column-specific recommendations
        high_outlier_cols = [
            col for col, info in outliers_by_column.items()
            if info["outlier_ratio"] > 0.05
        ]

        if high_outlier_cols:
            recommendations.append(
                f"Columns with >5% outliers (investigate): {high_outlier_cols}"
            )

        extreme_cols = [
            col for col, count in {
                c: d.get("extreme_count", 0)
                for c, d in outliers_by_column.items()
            }.items()
            if count > 0
        ]

        if extreme_cols:
            recommendations.append(
                f"Extreme outliers detected in: {extreme_cols} - "
                "Consider verification before removal"
            )

        # Method-specific recommendations
        if self.method == OutlierMethod.IQR:
            recommendations.append(
                f"IQR method with multiplier {self.iqr_multiplier} used. "
                "Adjust multiplier for more/less sensitivity."
            )
        elif self.method == OutlierMethod.ZSCORE:
            recommendations.append(
                "Z-score assumes normality. Use IQR or MAD for non-normal distributions."
            )
        elif self.method == OutlierMethod.ISOLATION_FOREST:
            recommendations.append(
                "Isolation Forest detects global outliers. "
                "Use LOF for local anomaly detection."
            )

        if include_multivariate and mv_outliers > 0:
            recommendations.append(
                f"{mv_outliers} multivariate outliers detected. "
                "These may not appear as univariate outliers in any single column."
            )

        recommendations.append(
            "Options: cap (winsorize), remove, or transform (log/Box-Cox) outlier values."
        )

        return recommendations

    def get_outlier_rows(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Get rows containing outliers for a specific column.

        Args:
            df: Input DataFrame.
            column: Column to check for outliers.

        Returns:
            DataFrame with outlier rows only.
        """
        if column not in df.columns:
            return pd.DataFrame()

        series = df[column].dropna()

        if self.method == OutlierMethod.IQR:
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - self.iqr_multiplier * iqr
            upper = q3 + self.iqr_multiplier * iqr
            mask = (df[column] < lower) | (df[column] > upper)
        elif self.method == OutlierMethod.ZSCORE:
            mean = series.mean()
            std = series.std()
            z_scores = np.abs((df[column] - mean) / std)
            mask = z_scores > self.zscore_threshold
        else:
            return pd.DataFrame()

        return df[mask].copy()
