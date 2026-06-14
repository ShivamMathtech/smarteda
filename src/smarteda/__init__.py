"""
SmartEDA v0.1 - Production-grade Exploratory Data Analysis toolkit.

A comprehensive Python library for automated exploratory data analysis,
featuring data ingestion, schema detection, quality analysis, and
beautiful HTML report generation.
"""

__version__ = "0.1.0"
__author__ = "SmartEDA Team"

from smarteda.core.data_loader import DataLoader
from smarteda.core.schema_detector import SchemaDetector
from smarteda.core.data_profiler import DataProfiler
from smarteda.analysis.missing_values import MissingValueAnalyzer
from smarteda.analysis.duplicates import DuplicateAnalyzer
from smarteda.analysis.outliers import OutlierAnalyzer
from smarteda.analysis.correlations import CorrelationAnalyzer
from smarteda.analysis.quality_score import QualityScorer
from smarteda.reports.html_generator import HTMLReportGenerator
from smarteda.models.analysis_result import (
    ColumnInfo,
    SchemaProfile,
    MissingValueReport,
    DuplicateReport,
    OutlierReport,
    CorrelationReport,
    QualityScore,
    EDAReport,
)

__all__ = [
    "DataLoader",
    "SchemaDetector",
    "DataProfiler",
    "MissingValueAnalyzer",
    "DuplicateAnalyzer",
    "OutlierAnalyzer",
    "CorrelationAnalyzer",
    "QualityScorer",
    "HTMLReportGenerator",
    "ColumnInfo",
    "SchemaProfile",
    "MissingValueReport",
    "DuplicateReport",
    "OutlierReport",
    "CorrelationReport",
    "QualityScore",
    "EDAReport",
]