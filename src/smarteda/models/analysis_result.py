"""
Data models for SmartEDA analysis results.

Defines dataclasses for structured storage of all EDA analysis outputs,
enabling type-safe interfaces between components.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd


class DataType(str, Enum):
    """Enumeration of inferred data types."""

    NUMERIC = "numeric"
    INTEGER = "integer"
    FLOAT = "float"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    TEXT = "text"
    CONSTANT = "constant"
    EMPTY = "empty"


@dataclass
class ColumnInfo:
    """Schema information for a single column."""

    name: str
    inferred_type: DataType
    pandas_dtype: str
    nullable: bool
    unique_count: int
    unique_ratio: float
    sample_values: List[Any] = field(default_factory=list)
    null_count: int = 0
    null_ratio: float = 0.0
    memory_usage: int = 0
    semantic_type: Optional[str] = None
    is_constant: bool = False
    is_empty: bool = False
    cardinality: str = "unknown"  # low, medium, high, very-high


@dataclass
class SchemaProfile:
    """Complete schema profile of a dataset."""

    columns: Dict[str, ColumnInfo] = field(default_factory=dict)
    row_count: int = 0
    column_count: int = 0
    memory_usage: int = 0
    memory_usage_human: str = ""
    duplicate_rows: int = 0
    duplicate_ratio: float = 0.0
    estimated_cardinality: Dict[str, str] = field(default_factory=dict)


@dataclass
class MissingValueReport:
    """Report on missing values in the dataset."""

    total_missing: int = 0
    total_cells: int = 0
    overall_missing_ratio: float = 0.0
    column_missing: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    missing_patterns: pd.DataFrame = field(default_factory=pd.DataFrame)
    columns_with_missing: List[str] = field(default_factory=list)
    completely_empty_columns: List[str] = field(default_factory=list)
    missing_heatmap_data: Optional[pd.DataFrame] = None
    recommendations: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Ensure DataFrames are properly initialized."""
        if isinstance(self.missing_patterns, np.ndarray):
            self.missing_patterns = pd.DataFrame(self.missing_patterns)


@dataclass
class DuplicateReport:
    """Report on duplicate records in the dataset."""

    total_duplicates: int = 0
    duplicate_ratio: float = 0.0
    exact_duplicates: int = 0
    exact_duplicate_ratio: float = 0.0
    fuzzy_duplicates: int = 0
    fuzzy_duplicate_details: List[Dict[str, Any]] = field(default_factory=list)
    duplicate_groups: pd.DataFrame = field(default_factory=pd.DataFrame)
    columns_analyzed: List[str] = field(default_factory=list)
    key_column_duplicates: Dict[str, int] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Ensure DataFrames are properly initialized."""
        if isinstance(self.duplicate_groups, np.ndarray):
            self.duplicate_groups = pd.DataFrame(self.duplicate_groups)


@dataclass
class OutlierReport:
    """Report on outliers detected in the dataset."""

    columns_analyzed: List[str] = field(default_factory=list)
    outliers_by_column: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    total_outliers: int = 0
    outlier_ratio: float = 0.0
    method_used: str = "iqr"
    iqr_multiplier: float = 1.5
    zscore_threshold: float = 3.0
    extreme_outliers: Dict[str, int] = field(default_factory=dict)
    outlier_indices: Dict[str, List[int]] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class CorrelationReport:
    """Report on correlations between numeric features."""

    correlation_matrix: pd.DataFrame = field(default_factory=pd.DataFrame)
    method: str = "pearson"
    numeric_columns: List[str] = field(default_factory=list)
    high_correlations: List[Dict[str, Any]] = field(default_factory=list)
    low_correlations: List[Dict[str, Any]] = field(default_factory=list)
    strong_positive_pairs: List[Dict[str, Any]] = field(default_factory=list)
    strong_negative_pairs: List[Dict[str, Any]] = field(default_factory=list)
    correlation_heatmap: Optional[str] = None
    recommendations: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Ensure DataFrames are properly initialized."""
        if isinstance(self.correlation_matrix, np.ndarray):
            self.correlation_matrix = pd.DataFrame(self.correlation_matrix)


@dataclass
class QualityScore:
    """Overall data quality score."""

    overall_score: float = 0.0
    completeness_score: float = 0.0
    uniqueness_score: float = 0.0
    validity_score: float = 0.0
    consistency_score: float = 0.0
    score_breakdown: Dict[str, float] = field(default_factory=dict)
    grade: str = "F"
    summary: str = ""
    improvement_areas: List[str] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Calculate grade and summary based on score."""
        if self.overall_score >= 90:
            self.grade = "A"
            self.summary = "Excellent data quality"
        elif self.overall_score >= 80:
            self.grade = "B"
            self.summary = "Good data quality with minor issues"
        elif self.overall_score >= 70:
            self.grade = "C"
            self.summary = "Acceptable data quality, some improvements needed"
        elif self.overall_score >= 60:
            self.grade = "D"
            self.summary = "Below average data quality, significant improvements needed"
        else:
            self.grade = "F"
            self.summary = "Poor data quality, major improvements required"


@dataclass
class EDAReport:
    """Complete EDA report combining all analysis results."""

    dataset_name: str = "dataset"
    schema: Optional[SchemaProfile] = None
    missing_values: Optional[MissingValueReport] = None
    duplicates: Optional[DuplicateReport] = None
    outliers: Optional[OutlierReport] = None
    correlations: Optional[CorrelationReport] = None
    quality_score: Optional[QualityScore] = None
    execution_time: float = 0.0
    generated_at: str = ""
    version: str = "0.1.0"
    warnings: List[str] = field(default_factory=list)
    info: Dict[str, Any] = field(default_factory=dict)
