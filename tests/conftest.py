"""
Pytest configuration and shared fixtures for SmartEDA tests.
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create a standard sample DataFrame for testing."""
    np.random.seed(42)
    n = 1000

    return pd.DataFrame({
        "id": range(1, n + 1),
        "name": [f"User_{i}" for i in range(n)],
        "age": np.random.randint(18, 80, n),
        "salary": np.random.normal(50000, 15000, n).round(2),
        "department": np.random.choice(["Engineering", "Sales", "Marketing", "HR"], n),
        "is_active": np.random.choice([True, False], n),
        "join_date": pd.date_range("2020-01-01", periods=n, freq="D"),
        "score": np.random.uniform(0, 100, n).round(2),
        "email": [f"user{i}@example.com" for i in range(n)],
    })


@pytest.fixture
def df_with_missing() -> pd.DataFrame:
    """Create a DataFrame with missing values for testing."""
    np.random.seed(42)
    n = 500

    df = pd.DataFrame({
        "A": np.random.normal(0, 1, n),
        "B": np.random.normal(5, 2, n),
        "C": np.random.choice(["x", "y", "z"], n),
        "D": pd.date_range("2020-01-01", periods=n, freq="D"),
    })

    # Introduce missing values
    missing_idx_A = np.random.choice(df.index, size=50, replace=False)
    missing_idx_B = np.random.choice(df.index, size=100, replace=False)
    missing_idx_C = np.random.choice(df.index, size=30, replace=False)

    df.loc[missing_idx_A, "A"] = np.nan
    df.loc[missing_idx_B, "B"] = np.nan
    df.loc[missing_idx_C, "C"] = np.nan

    return df


@pytest.fixture
def df_with_duplicates() -> pd.DataFrame:
    """Create a DataFrame with duplicate rows for testing."""
    np.random.seed(42)

    base = pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 1, 3, 6],
        "value": ["a", "b", "c", "d", "e", "a", "c", "f"],
        "score": [10, 20, 30, 40, 50, 10, 30, 60],
    })

    return base


@pytest.fixture
def df_with_outliers() -> pd.DataFrame:
    """Create a DataFrame with outliers for testing."""
    np.random.seed(42)
    n = 200

    df = pd.DataFrame({
        "normal": np.random.normal(100, 10, n),
        "skewed": np.random.exponential(5, n),
        "uniform": np.random.uniform(0, 100, n),
        "category": np.random.choice(["A", "B", "C"], n),
    })

    # Add clear outliers
    outlier_idx = np.random.choice(df.index, size=10, replace=False)
    df.loc[outlier_idx[:5], "normal"] = np.random.uniform(200, 300, 5)
    df.loc[outlier_idx[5:], "skewed"] = np.random.uniform(50, 100, 5)

    return df


@pytest.fixture
def df_with_correlations() -> pd.DataFrame:
    """Create a DataFrame with known correlations for testing."""
    np.random.seed(42)
    n = 300

    x1 = np.random.normal(0, 1, n)
    x2 = 0.9 * x1 + np.random.normal(0, 0.2, n)  # Strong positive
    x3 = -0.7 * x1 + np.random.normal(0, 0.5, n)  # Moderate negative
    x4 = np.random.normal(0, 1, n)  # Independent
    x5 = 0.3 * x1 + np.random.normal(0, 0.8, n)  # Weak positive

    return pd.DataFrame({
        "feature_1": x1,
        "feature_2": x2,
        "feature_3": x3,
        "feature_4": x4,
        "feature_5": x5,
    })


@pytest.fixture
def empty_df() -> pd.DataFrame:
    """Create an empty DataFrame for edge case testing."""
    return pd.DataFrame()


@pytest.fixture
def single_column_df() -> pd.DataFrame:
    """Create a single-column DataFrame for edge case testing."""
    return pd.DataFrame({"value": [1, 2, 3, 4, 5]})


@pytest.fixture
def constant_df() -> pd.DataFrame:
    """Create a DataFrame with constant columns for edge case testing."""
    return pd.DataFrame({
        "constant": [5, 5, 5, 5, 5],
        "varying": [1, 2, 3, 4, 5],
    })


@pytest.fixture
def temp_csv_file(tmp_path: Path) -> str:
    """Create a temporary CSV file for testing data loading."""
    df = pd.DataFrame({
        "A": [1, 2, 3, 4, 5],
        "B": ["a", "b", "c", "d", "e"],
        "C": [1.1, 2.2, 3.3, 4.4, 5.5],
    })

    filepath = tmp_path / "test_data.csv"
    df.to_csv(filepath, index=False)

    return str(filepath)


@pytest.fixture
def temp_excel_file(tmp_path: Path) -> str:
    """Create a temporary Excel file for testing data loading."""
    df = pd.DataFrame({
        "X": [10, 20, 30, 40, 50],
        "Y": ["p", "q", "r", "s", "t"],
        "Z": [True, False, True, False, True],
    })

    filepath = tmp_path / "test_data.xlsx"
    df.to_excel(filepath, index=False)

    return str(filepath)


@pytest.fixture
def temp_html_file(tmp_path: Path) -> str:
    """Create a temporary path for HTML report output."""
    return str(tmp_path / "report.html")


@pytest.fixture
def low_quality_df() -> pd.DataFrame:
    """Create a low-quality DataFrame for quality scoring tests."""
    np.random.seed(42)
    n = 100

    df = pd.DataFrame({
        "id": list(range(50)) + list(range(50)),  # 50% duplicates
        "value": [np.nan] * 60 + list(range(40)),  # 60% missing
        "category": [" A", "a ", " A", "a "] * 25,  # Inconsistent whitespace/case
        "score": np.random.uniform(-1000, 1000, n),  # Wide range
    })

    return df
