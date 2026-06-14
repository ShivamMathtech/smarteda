"""
Utility helper functions for SmartEDA.

Common utilities used across modules for data type inference,
column name sanitization, and statistical computations.
"""

from __future__ import annotations

import re
import warnings
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def infer_datetime_format(series: pd.Series, sample_size: int = 1000) -> Tuple[bool, Optional[str]]:
    """
    Infer if a series can be parsed as datetime.

    Args:
        series: Input pandas Series to check.
        sample_size: Number of samples to test for performance.

    Returns:
        Tuple of (is_datetime, detected_format).
    """
    if series.dtype.kind in "mM":
        return True, None

    sample = series.dropna().head(sample_size)
    if len(sample) == 0:
        return False, None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            converted = pd.to_datetime(sample, errors="coerce", infer_datetime_format=True)
            success_ratio = converted.notna().sum() / len(sample)

            if success_ratio > 0.8:
                return True, None
    except (ValueError, TypeError):
        pass

    return False, None


def sanitize_column_name(name: str) -> str:
    """
    Sanitize a column name for safe use in templates and outputs.

    Args:
        name: Original column name.

    Returns:
        Sanitized column name.
    """
    if not isinstance(name, str):
        name = str(name)

    sanitized = re.sub(r"[^\w\s-]", "_", name)
    sanitized = re.sub(r"[-\s]+", "_", sanitized)
    sanitized = re.sub(r"_+", "_", sanitized)
    sanitized = sanitized.strip("_")

    return sanitized or "unnamed_column"


def compute_entropy(series: pd.Series) -> float:
    """
    Compute Shannon entropy of a categorical series.

    Args:
        series: Input pandas Series.

    Returns:
        Entropy value (0 = constant, higher = more uniform).
    """
    if series.empty:
        return 0.0

    value_counts = series.value_counts(normalize=True, dropna=True)
    if len(value_counts) <= 1:
        return 0.0

    entropy = -(value_counts * np.log2(value_counts)).sum()
    return float(entropy)


def format_bytes(size: int) -> str:
    """
    Convert bytes to human-readable format.

    Args:
        size: Size in bytes.

    Returns:
        Human-readable string (e.g., '1.5 MB').
    """
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(size) < 1024.0:
            return f"{size:,.1f} {unit}"
        size /= 1024.0
    return f"{size:,.1f} TB"


def is_numeric_series(series: pd.Series) -> bool:
    """
    Check if a series is numeric (including nullable integer types).

    Args:
        series: Input pandas Series.

    Returns:
        True if series is numeric, False otherwise.
    """
    return pd.api.types.is_numeric_dtype(series) or series.dtype.name.startswith("Int")


def detect_cardinality(unique_ratio: float, unique_count: int) -> str:
    """
    Classify cardinality level based on unique ratio and count.

    Args:
        unique_ratio: Ratio of unique values to total rows.
        unique_count: Absolute number of unique values.

    Returns:
        Cardinality classification string.
    """
    if unique_ratio == 1.0:
        return "unique"
    elif unique_ratio > 0.9:
        return "very-high"
    elif unique_ratio > 0.5:
        return "high"
    elif unique_count <= 10:
        return "low"
    elif unique_count <= 100:
        return "medium"
    else:
        return "high"


def compute_gini_coefficient(series: pd.Series) -> float:
    """
    Compute Gini coefficient for inequality measurement.

    Args:
        series: Numeric pandas Series.

    Returns:
        Gini coefficient between 0 and 1.
    """
    values = series.dropna().values
    if len(values) == 0:
        return 0.0

    sorted_values = np.sort(values)
    n = len(sorted_values)
    cumsum = np.cumsum(sorted_values)

    return float((n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n)


def detect_separator(filepath: str) -> str:
    """
    Detect CSV separator by sampling the file.

    Args:
        filepath: Path to CSV file.

    Returns:
        Detected separator character.
    """
    separators = [",", "\t", ";", "|"]
    counts = {sep: 0 for sep in separators}

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for _ in range(5):
                line = f.readline()
                if not line:
                    break
                for sep in separators:
                    counts[sep] += line.count(sep)
    except (IOError, UnicodeDecodeError):
        pass

    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else ","
