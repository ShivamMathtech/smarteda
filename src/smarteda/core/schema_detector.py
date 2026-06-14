"""
Schema detection module for automatic data type inference.

Detects column types, semantic meanings, cardinality, and other
schema-level characteristics to build a comprehensive profile
of the dataset structure.
"""

from __future__ import annotations

import logging
import re
import warnings
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from smarteda.models.analysis_result import ColumnInfo, DataType, SchemaProfile
from smarteda.utils.helpers import (
    detect_cardinality,
    format_bytes,
    infer_datetime_format,
    is_numeric_series,
    sanitize_column_name,
)

logger = logging.getLogger(__name__)


class SchemaDetector:
    """
    Automatic schema detection and profiling for pandas DataFrames.

    Detects data types, semantic categories, cardinality levels,
    and structural anomalies like constant or empty columns.

    Example:
        >>> detector = SchemaDetector()
        >>> profile = detector.detect(df)
        >>> print(profile.columns["age"].inferred_type)
        DataType.NUMERIC
    """

    SEMANTIC_PATTERNS: Dict[str, Dict[str, Any]] = {
        "email": {
            "patterns": [r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"],
            "keywords": ["email", "e-mail", "mail"],
        },
        "phone": {
            "patterns": [
                r"^[\+]?[(]?[0-9]{1,4}[)]?[-\s\.]?[0-9]{1,4}[-\s\.]?[0-9]{1,9}$",
                r"^\d{3}-\d{3}-\d{4}$",
                r"^\(\d{3}\)\s?\d{3}-\d{4}$",
            ],
            "keywords": ["phone", "tel", "mobile", "fax"],
        },
        "url": {
            "patterns": [r"^https?://[^\s/$.?#].[^\s]*$", r"^www\.[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"],
            "keywords": ["url", "website", "link", "homepage"],
        },
        "ip_address": {
            "patterns": [
                r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$",
                r"^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$",
            ],
            "keywords": ["ip", "ip_address", "ipv4", "ipv6"],
        },
        "date": {
            "keywords": [
                "date",
                "datetime",
                "timestamp",
                "created",
                "updated",
                "modified",
                "birth",
                "day",
            ],
        },
        "name": {
            "keywords": [
                "name",
                "first_name",
                "last_name",
                "full_name",
                "username",
                "user",
            ],
        },
        "address": {
            "keywords": ["address", "street", "city", "state", "zip", "postal", "country"],
        },
        "id": {
            "keywords": [
                "id",
                "uuid",
                "identifier",
                "key",
                "code",
                "sku",
                "serial",
            ],
        },
        "money": {
            "keywords": [
                "price",
                "cost",
                "amount",
                "salary",
                "revenue",
                "budget",
                "fee",
                "income",
                "expense",
            ],
        },
        "percentage": {
            "keywords": [
                "rate",
                "ratio",
                "percentage",
                "pct",
                "share",
                "proportion",
            ],
        },
    }

    def __init__(self, sample_size: int = 1000, confidence_threshold: float = 0.8) -> None:
        """
        Initialize SchemaDetector.

        Args:
            sample_size: Number of rows to sample for type inference.
            confidence_threshold: Minimum ratio for semantic type detection.
        """
        self.sample_size = sample_size
        self.confidence_threshold = confidence_threshold

    def detect(self, df: pd.DataFrame) -> SchemaProfile:
        """
        Detect the complete schema of a DataFrame.

        Args:
            df: Input pandas DataFrame.

        Returns:
            SchemaProfile with detected column information.
        """
        if df.empty:
            logger.warning("Empty DataFrame provided to schema detector")
            return SchemaProfile()

        columns: Dict[str, ColumnInfo] = {}

        for col_name in df.columns:
            col_data = df[col_name]
            column_info = self._analyze_column(col_name, col_data, len(df))
            columns[str(col_name)] = column_info

        total_memory = df.memory_usage(deep=True).sum()
        duplicate_rows = df.duplicated().sum()

        profile = SchemaProfile(
            columns=columns,
            row_count=len(df),
            column_count=len(df.columns),
            memory_usage=int(total_memory),
            memory_usage_human=format_bytes(int(total_memory)),
            duplicate_rows=int(duplicate_rows),
            duplicate_ratio=duplicate_rows / len(df) if len(df) > 0 else 0.0,
        )

        logger.info(
            "Schema detected: %d columns, %d rows, %s memory",
            profile.column_count,
            profile.row_count,
            profile.memory_usage_human,
        )

        return profile

    def _analyze_column(self, name: str, series: pd.Series, total_rows: int) -> ColumnInfo:
        """Analyze a single column and return its ColumnInfo."""
        null_count = int(series.isna().sum())
        null_ratio = null_count / total_rows if total_rows > 0 else 0.0
        unique_count = int(series.nunique(dropna=True))
        unique_ratio = unique_count / (total_rows - null_count) if (total_rows - null_count) > 0 else 0.0

        inferred_type = self._infer_type(series)
        semantic_type = self._detect_semantic_type(name, series, inferred_type)

        sample = series.dropna().head(5).tolist()

        is_constant = unique_count <= 1 and null_count == 0
        is_empty = unique_count == 0 or total_rows == 0

        cardinality = detect_cardinality(unique_ratio, unique_count)

        if is_constant:
            cardinality = "constant"
        elif is_empty:
            cardinality = "empty"

        memory = int(series.memory_usage(deep=True))

        return ColumnInfo(
            name=str(name),
            inferred_type=inferred_type,
            pandas_dtype=str(series.dtype),
            nullable=null_count > 0,
            unique_count=unique_count,
            unique_ratio=unique_ratio,
            sample_values=sample,
            null_count=null_count,
            null_ratio=null_ratio,
            memory_usage=memory,
            semantic_type=semantic_type,
            is_constant=is_constant,
            is_empty=is_empty,
            cardinality=cardinality,
        )

    def _infer_type(self, series: pd.Series) -> DataType:
        """Infer the logical data type of a series."""
        if series.empty or series.isna().all():
            return DataType.EMPTY

        if series.nunique(dropna=True) <= 1:
            return DataType.CONSTANT

        dtype = series.dtype

        if pd.api.types.is_bool_dtype(dtype):
            return DataType.BOOLEAN

        if pd.api.types.is_datetime64_any_dtype(dtype):
            return DataType.DATETIME

        if is_numeric_series(series):
            non_null = series.dropna()
            if len(non_null) == 0:
                return DataType.NUMERIC

            if pd.api.types.is_integer_dtype(dtype) or (
                non_null == non_null.astype(int)
            ).all():
                return DataType.INTEGER

            return DataType.FLOAT

        if dtype == "object" or pd.api.types.is_string_dtype(dtype):
            non_null = series.dropna()
            if len(non_null) == 0:
                return DataType.TEXT

            is_datetime, _ = infer_datetime_format(series, self.sample_size)
            if is_datetime:
                return DataType.DATETIME

            unique_vals = non_null.nunique()
            if unique_vals <= 2:
                return DataType.CATEGORICAL

            avg_length = non_null.astype(str).str.len().mean()
            if avg_length is not None and avg_length < 50 and unique_vals / len(non_null) < 0.5:
                return DataType.CATEGORICAL

            return DataType.TEXT

        if hasattr(series, "cat") or str(dtype).startswith("category"):
            return DataType.CATEGORICAL

        return DataType.TEXT

    def _detect_semantic_type(
        self, name: str, series: pd.Series, inferred_type: DataType
    ) -> Optional[str]:
        """Detect semantic meaning of a column based on name and values."""
        name_lower = sanitize_column_name(name).lower()

        for semantic, config in self.SEMANTIC_PATTERNS.items():
            keywords = config.get("keywords", [])
            if any(kw in name_lower for kw in keywords):
                patterns = config.get("patterns", [])
                if patterns:
                    sample = series.dropna().head(self.sample_size).astype(str)
                    matches = 0
                    for value in sample:
                        if any(re.match(p, str(value)) for p in patterns):
                            matches += 1
                    if len(sample) > 0 and matches / len(sample) >= self.confidence_threshold:
                        return semantic
                else:
                    return semantic

        if inferred_type == DataType.DATETIME:
            return "datetime"

        if inferred_type == DataType.BOOLEAN:
            return "flag"

        return None

    def get_type_distribution(self, profile: SchemaProfile) -> Dict[str, int]:
        """
        Get the distribution of data types in the profile.

        Args:
            profile: SchemaProfile to analyze.

        Returns:
            Dictionary mapping type names to counts.
        """
        distribution: Dict[str, int] = {}
        for col_info in profile.columns.values():
            type_name = col_info.inferred_type.value
            distribution[type_name] = distribution.get(type_name, 0) + 1
        return distribution

    def get_column_names_by_type(self, profile: SchemaProfile, data_type: DataType) -> List[str]:
        """
        Get column names filtered by data type.

        Args:
            profile: SchemaProfile to filter.
            data_type: DataType to filter by.

        Returns:
            List of matching column names.
        """
        return [
            name
            for name, info in profile.columns.items()
            if info.inferred_type == data_type
        ]

    def validate_schema(self, df: pd.DataFrame, profile: SchemaProfile) -> List[str]:
        """
        Validate a DataFrame against an existing schema profile.

        Args:
            df: DataFrame to validate.
            profile: Expected schema profile.

        Returns:
            List of validation warnings.
        """
        warnings_list: List[str] = []

        current_profile = self.detect(df)

        expected_cols = set(profile.columns.keys())
        current_cols = set(current_profile.columns.keys())

        missing = expected_cols - current_cols
        extra = current_cols - expected_cols

        if missing:
            warnings_list.append(f"Missing expected columns: {sorted(missing)}")
        if extra:
            warnings_list.append(f"Unexpected extra columns: {sorted(extra)}")

        for col_name in expected_cols & current_cols:
            expected_type = profile.columns[col_name].inferred_type
            current_type = current_profile.columns[col_name].inferred_type

            if expected_type != current_type:
                warnings_list.append(
                    f"Type mismatch for '{col_name}': "
                    f"expected {expected_type.value}, got {current_type.value}"
                )

        return warnings_list
