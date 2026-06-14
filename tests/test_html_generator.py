"""
Unit tests for the html_generator module.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from smarteda.analysis.correlations import CorrelationAnalyzer
from smarteda.analysis.duplicates import DuplicateAnalyzer
from smarteda.analysis.missing_values import MissingValueAnalyzer
from smarteda.analysis.outliers import OutlierAnalyzer
from smarteda.analysis.quality_score import QualityScorer
from smarteda.core.schema_detector import SchemaDetector
from smarteda.models.analysis_result import EDAReport
from smarteda.reports.html_generator import HTMLReportGenerator


class TestHTMLReportGenerator:
    """Test cases for HTMLReportGenerator."""

    def _create_full_report(self, df: pd.DataFrame) -> EDAReport:
        """Helper to create a complete EDAReport."""
        schema_detector = SchemaDetector()
        missing_analyzer = MissingValueAnalyzer()
        duplicate_analyzer = DuplicateAnalyzer()
        outlier_analyzer = OutlierAnalyzer()
        correlation_analyzer = CorrelationAnalyzer()
        quality_scorer = QualityScorer()

        return EDAReport(
            dataset_name="test_dataset",
            schema=schema_detector.detect(df),
            missing_values=missing_analyzer.analyze(df),
            duplicates=duplicate_analyzer.analyze(df),
            outliers=outlier_analyzer.analyze(df),
            correlations=correlation_analyzer.analyze(df),
            quality_score=quality_scorer.score(df),
            execution_time=1.5,
            generated_at=datetime.now().isoformat(),
        )

    def test_generate_html(self, sample_df: pd.DataFrame) -> None:
        """Test HTML generation."""
        report = self._create_full_report(sample_df)
        generator = HTMLReportGenerator()

        html = generator.generate(report)

        assert isinstance(html, str)
        assert len(html) > 0
        assert "SmartEDA" in html
        assert "test_dataset" in html

    def test_save_html(self, sample_df: pd.DataFrame, tmp_path) -> None:
        """Test saving HTML to file."""
        report = self._create_full_report(sample_df)
        generator = HTMLReportGenerator()

        output_path = str(tmp_path / "test_report.html")
        saved_path = generator.save(report, output_path)

        assert saved_path == output_path

        # Verify file exists and has content
        with open(saved_path, "r") as f:
            content = f.read()
        assert len(content) > 0
        assert "<html" in content.lower()

    def test_empty_report(self) -> None:
        """Test HTML generation with empty report."""
        report = EDAReport(dataset_name="empty")
        generator = HTMLReportGenerator()

        html = generator.generate(report)

        assert isinstance(html, str)
        assert "SmartEDA" in html

    def test_quality_score_display(self, sample_df: pd.DataFrame) -> None:
        """Test that quality score is displayed in HTML."""
        report = self._create_full_report(sample_df)
        generator = HTMLReportGenerator()

        html = generator.generate(report)

        assert report.quality_score is not None
        assert str(report.quality_score.overall_score) in html or f"{report.quality_score.overall_score:.1f}" in html

    def test_schema_section(self, sample_df: pd.DataFrame) -> None:
        """Test that schema section is included."""
        report = self._create_full_report(sample_df)
        generator = HTMLReportGenerator()

        html = generator.generate(report)

        assert "Dataset Overview" in html

    def test_correlation_heatmap_chart(self, df_with_correlations: pd.DataFrame) -> None:
        """Test correlation heatmap chart generation."""
        report = self._create_full_report(df_with_correlations)
        generator = HTMLReportGenerator()

        html = generator.generate(report)

        # Should contain Plotly chart code
        assert "Plotly" in html

    def test_custom_template(self, sample_df: pd.DataFrame, tmp_path) -> None:
        """Test with custom template."""
        custom_template = "<html><body><h1>{{ report.dataset_name }}</h1></body></html>"
        template_path = str(tmp_path / "custom_template.html")

        with open(template_path, "w") as f:
            f.write(custom_template)

        report = self._create_full_report(sample_df)
        generator = HTMLReportGenerator(template_path=template_path)

        html = generator.generate(report)

        assert "test_dataset" in html
        assert "<h1>" in html

    def test_missing_values_section(self, df_with_missing: pd.DataFrame) -> None:
        """Test missing values section in report."""
        report = self._create_full_report(df_with_missing)
        generator = HTMLReportGenerator()

        html = generator.generate(report)

        assert isinstance(html, str)

    def test_outliers_section(self, df_with_outliers: pd.DataFrame) -> None:
        """Test outliers section in report."""
        report = self._create_full_report(df_with_outliers)
        generator = HTMLReportGenerator()

        html = generator.generate(report)

        assert isinstance(html, str)

    def test_report_contains_grade(self, sample_df: pd.DataFrame) -> None:
        """Test that report contains quality grade."""
        report = self._create_full_report(sample_df)
        generator = HTMLReportGenerator()

        html = generator.generate(report)

        assert report.quality_score is not None
        assert report.quality_score.grade in html

    def test_report_structure(self, sample_df: pd.DataFrame) -> None:
        """Test that report has proper HTML structure."""
        report = self._create_full_report(sample_df)
        generator = HTMLReportGenerator()

        html = generator.generate(report)

        assert "<!DOCTYPE html>" in html
        assert "<html" in html.lower()
        assert "</html>" in html.lower()
        assert "<body" in html.lower()
        assert "</body>" in html.lower()
