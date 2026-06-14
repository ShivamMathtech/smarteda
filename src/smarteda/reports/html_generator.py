"""
HTML report generation module.

Generates beautiful, interactive HTML reports using Plotly
visualizations and Jinja2 templates. Reports include all
EDA analysis results with charts, tables, and recommendations.
"""

from __future__ import annotations

import base64
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from jinja2 import Template

from smarteda.models.analysis_result import (
    CorrelationReport,
    DuplicateReport,
    EDAReport,
    MissingValueReport,
    OutlierReport,
    QualityScore,
    SchemaProfile,
)

logger = logging.getLogger(__name__)

# Default HTML template
DEFAULT_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SmartEDA Report - {{ report.dataset_name }}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        :root {
            --primary: #2563eb;
            --primary-dark: #1d4ed8;
            --success: #059669;
            --warning: #d97706;
            --danger: #dc2626;
            --gray-50: #f9fafb;
            --gray-100: #f3f4f6;
            --gray-200: #e5e7eb;
            --gray-300: #d1d5db;
            --gray-400: #9ca3af;
            --gray-500: #6b7280;
            --gray-600: #4b5563;
            --gray-700: #374151;
            --gray-800: #1f2937;
            --gray-900: #111827;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: var(--gray-50);
            color: var(--gray-800);
            line-height: 1.6;
        }

        .header {
            background: linear-gradient(135deg, var(--primary), var(--primary-dark));
            color: white;
            padding: 2rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }

        .header h1 {
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }

        .header-meta {
            opacity: 0.9;
            font-size: 0.9rem;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }

        .section {
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
            border: 1px solid var(--gray-200);
        }

        .section-title {
            font-size: 1.4rem;
            font-weight: 600;
            color: var(--gray-900);
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid var(--gray-200);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .grid-4 {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 1rem;
        }

        .grid-2 {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 1rem;
        }

        .metric-card {
            background: var(--gray-50);
            border-radius: 8px;
            padding: 1.25rem;
            border-left: 4px solid var(--primary);
            transition: transform 0.2s;
        }

        .metric-card:hover {
            transform: translateY(-2px);
        }

        .metric-label {
            font-size: 0.85rem;
            color: var(--gray-500);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.25rem;
        }

        .metric-value {
            font-size: 1.75rem;
            font-weight: 700;
            color: var(--gray-900);
        }

        .metric-card.success { border-left-color: var(--success); }
        .metric-card.warning { border-left-color: var(--warning); }
        .metric-card.danger { border-left-color: var(--danger); }

        .grade {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 60px;
            height: 60px;
            border-radius: 50%;
            font-size: 1.5rem;
            font-weight: 700;
            color: white;
        }

        .grade-a { background: var(--success); }
        .grade-b { background: #10b981; }
        .grade-c { background: var(--warning); }
        .grade-d { background: #f97316; }
        .grade-f { background: var(--danger); }

        .score-display {
            display: flex;
            align-items: center;
            gap: 1.5rem;
            margin-bottom: 1rem;
        }

        .score-details {
            flex: 1;
        }

        .progress-bar {
            width: 100%;
            height: 8px;
            background: var(--gray-200);
            border-radius: 4px;
            overflow: hidden;
            margin-top: 0.5rem;
        }

        .progress-fill {
            height: 100%;
            border-radius: 4px;
            transition: width 0.5s ease;
        }

        .progress-fill.high { background: var(--success); }
        .progress-fill.medium { background: var(--warning); }
        .progress-fill.low { background: var(--danger); }

        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
            font-size: 0.9rem;
        }

        th, td {
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid var(--gray-200);
        }

        th {
            background: var(--gray-100);
            font-weight: 600;
            color: var(--gray-700);
            position: sticky;
            top: 0;
        }

        tr:hover {
            background: var(--gray-50);
        }

        .tag {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
        }

        .tag-success { background: #d1fae5; color: #065f46; }
        .tag-warning { background: #fef3c7; color: #92400e; }
        .tag-danger { background: #fee2e2; color: #991b1b; }
        .tag-info { background: #dbeafe; color: #1e40af; }

        .recommendations {
            background: #eff6ff;
            border-left: 4px solid var(--primary);
            padding: 1rem 1.25rem;
            border-radius: 0 8px 8px 0;
            margin-top: 1rem;
        }

        .recommendations h4 {
            color: var(--primary);
            margin-bottom: 0.5rem;
        }

        .recommendations ul {
            margin-left: 1.25rem;
        }

        .recommendations li {
            margin-bottom: 0.35rem;
            color: var(--gray-700);
        }

        .chart-container {
            width: 100%;
            min-height: 400px;
            margin: 1rem 0;
        }

        .footer {
            text-align: center;
            padding: 2rem;
            color: var(--gray-500);
            font-size: 0.85rem;
        }

        .summary-text {
            color: var(--gray-600);
            font-style: italic;
            margin-top: 0.5rem;
        }

        .severity-critical { color: var(--danger); font-weight: 600; }
        .severity-high { color: #f97316; font-weight: 600; }
        .severity-moderate { color: var(--warning); }
        .severity-low { color: var(--success); }

        @media (max-width: 768px) {
            .grid-2 { grid-template-columns: 1fr; }
            .grid-4 { grid-template-columns: repeat(2, 1fr); }
            .container { padding: 1rem; }
            .header h1 { font-size: 1.5rem; }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>SmartEDA Report</h1>
        <div class="header-meta">
            Dataset: {{ report.dataset_name }} | Generated: {{ report.generated_at }}
            | SmartEDA v{{ report.version }}
        </div>
    </div>

    <div class="container">
        <!-- Quality Score Section -->
        {% if report.quality_score %}
        <div class="section">
            <div class="section-title">
                <span>Data Quality Score</span>
            </div>
            <div class="score-display">
                {% set grade_class = 'grade-' + report.quality_score.grade.lower() if report.quality_score.grade else 'grade-f' %}
                <div class="grade {{ grade_class }}">
                    {{ report.quality_score.grade }}
                </div>
                <div class="score-details">
                    <div class="metric-value">{{ "%.1f" | format(report.quality_score.overall_score) }}/100</div>
                    <div class="summary-text">{{ report.quality_score.summary }}</div>
                    <div style="margin-top: 1rem;">
                        {% for dim, score in report.quality_score.score_breakdown.items() %}
                        <div style="margin-bottom: 0.75rem;">
                            <div style="display: flex; justify-content: space-between; font-size: 0.85rem;">
                                <span style="text-transform: capitalize;">{{ dim }}</span>
                                <span>{{ "%.0f" | format(score * 100) }}%</span>
                            </div>
                            <div class="progress-bar">
                                <div class="progress-fill {% if score >= 0.8 %}high{% elif score >= 0.5 %}medium{% else %}low{% endif %}"
                                     style="width: {{ score * 100 }}%"></div>
                            </div>
                        </div>
                        {% endfor %}
                    </div>
                </div>
            </div>

            {% if report.quality_score.improvement_areas %}
            <div class="recommendations">
                <h4>Improvement Areas</h4>
                <ul>
                    {% for area in report.quality_score.improvement_areas %}
                    <li>{{ area }}</li>
                    {% endfor %}
                </ul>
            </div>
            {% endif %}
        </div>
        {% endif %}

        <!-- Dataset Overview -->
        {% if report.schema %}
        <div class="section">
            <div class="section-title">
                <span>Dataset Overview</span>
            </div>
            <div class="grid-4">
                <div class="metric-card">
                    <div class="metric-label">Rows</div>
                    <div class="metric-value">{{ "{:,}".format(report.schema.row_count) }}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Columns</div>
                    <div class="metric-value">{{ report.schema.column_count }}</div>
                </div>
                <div class="metric-card {% if report.schema.duplicate_ratio > 0.1 %}danger{% else %}success{% endif %}">
                    <div class="metric-label">Duplicates</div>
                    <div class="metric-value">{{ "{:,}".format(report.schema.duplicate_rows) }}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Memory</div>
                    <div class="metric-value">{{ report.schema.memory_usage_human }}</div>
                </div>
            </div>

            <h4 style="margin-top: 1.5rem; margin-bottom: 0.75rem;">Column Details</h4>
            <div style="overflow-x: auto;">
                <table>
                    <thead>
                        <tr>
                            <th>Column</th>
                            <th>Type</th>
                            <th>Data Type</th>
                            <th>Unique</th>
                            <th>Missing</th>
                            <th>Cardinality</th>
                            <th>Semantic</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for name, col in report.schema.columns.items() %}
                        <tr>
                            <td><strong>{{ name }}</strong></td>
                            <td><span class="tag tag-info">{{ col.inferred_type.value }}</span></td>
                            <td>{{ col.pandas_dtype }}</td>
                            <td>{{ "{:,}".format(col.unique_count) }}</td>
                            <td>
                                {% if col.null_ratio > 0.5 %}
                                <span class="severity-critical">{{ "{:.1%}".format(col.null_ratio) }}</span>
                                {% elif col.null_ratio > 0.2 %}
                                <span class="severity-high">{{ "{:.1%}".format(col.null_ratio) }}</span>
                                {% elif col.null_ratio > 0 %}
                                <span class="severity-moderate">{{ "{:.1%}".format(col.null_ratio) }}</span>
                                {% else %}
                                <span class="severity-low">0%</span>
                                {% endif %}
                            </td>
                            <td><span class="tag {% if col.cardinality == 'constant' %}tag-warning{% else %}tag-success{% endif %}">{{ col.cardinality }}</span></td>
                            <td>{{ col.semantic_type or "-" }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
        </div>
        {% endif %}

        <!-- Missing Values -->
        {% if report.missing_values and report.missing_values.columns_with_missing %}
        <div class="section">
            <div class="section-title">
                <span>Missing Value Analysis</span>
            </div>
            <div class="grid-4">
                <div class="metric-card danger">
                    <div class="metric-label">Total Missing</div>
                    <div class="metric-value">{{ "{:,}".format(report.missing_values.total_missing) }}</div>
                </div>
                <div class="metric-card warning">
                    <div class="metric-label">Missing %</div>
                    <div class="metric-value">{{ "{:.2%}".format(report.missing_values.overall_missing_ratio) }}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Columns Affected</div>
                    <div class="metric-value">{{ report.missing_values.columns_with_missing | length }}</div>
                </div>
                <div class="metric-card {% if report.missing_values.completely_empty_columns %}danger{% else %}success{% endif %}">
                    <div class="metric-label">Empty Columns</div>
                    <div class="metric-value">{{ report.missing_values.completely_empty_columns | length }}</div>
                </div>
            </div>

            {% if charts.missing_values_chart %}
            <div class="chart-container" id="missing-values-chart"></div>
            {% endif %}

            {% if report.missing_values.recommendations %}
            <div class="recommendations">
                <h4>Recommendations</h4>
                <ul>
                    {% for rec in report.missing_values.recommendations %}
                    <li>{{ rec }}</li>
                    {% endfor %}
                </ul>
            </div>
            {% endif %}
        </div>
        {% endif %}

        <!-- Duplicates -->
        {% if report.duplicates and report.duplicates.total_duplicates > 0 %}
        <div class="section">
            <div class="section-title">
                <span>Duplicate Analysis</span>
            </div>
            <div class="grid-4">
                <div class="metric-card danger">
                    <div class="metric-label">Total Duplicates</div>
                    <div class="metric-value">{{ "{:,}".format(report.duplicates.total_duplicates) }}</div>
                </div>
                <div class="metric-card warning">
                    <div class="metric-label">Duplicate %</div>
                    <div class="metric-value">{{ "{:.2%}".format(report.duplicates.duplicate_ratio) }}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Exact Duplicates</div>
                    <div class="metric-value">{{ "{:,}".format(report.duplicates.exact_duplicates) }}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Fuzzy Duplicates</div>
                    <div class="metric-value">{{ "{:,}".format(report.duplicates.fuzzy_duplicates) }}</div>
                </div>
            </div>

            {% if report.duplicates.recommendations %}
            <div class="recommendations">
                <h4>Recommendations</h4>
                <ul>
                    {% for rec in report.duplicates.recommendations %}
                    <li>{{ rec }}</li>
                    {% endfor %}
                </ul>
            </div>
            {% endif %}
        </div>
        {% endif %}

        <!-- Outliers -->
        {% if report.outliers and report.outliers.outliers_by_column %}
        <div class="section">
            <div class="section-title">
                <span>Outlier Detection ({{ report.outliers.method_used }})</span>
            </div>
            <div class="grid-4">
                <div class="metric-card warning">
                    <div class="metric-label">Total Outliers</div>
                    <div class="metric-value">{{ "{:,}".format(report.outliers.total_outliers) }}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Outlier Ratio</div>
                    <div class="metric-value">{{ "{:.2%}".format(report.outliers.outlier_ratio) }}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Columns Affected</div>
                    <div class="metric-value">{{ report.outliers.outliers_by_column | length }}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Method</div>
                    <div class="metric-value" style="font-size: 1.1rem;">{{ report.outliers.method_used }}</div>
                </div>
            </div>

            <h4 style="margin-top: 1.5rem; margin-bottom: 0.75rem;">Outliers by Column</h4>
            <div style="overflow-x: auto;">
                <table>
                    <thead>
                        <tr>
                            <th>Column</th>
                            <th>Outlier Count</th>
                            <th>Outlier %</th>
                            <th>Extreme</th>
                            <th>Bounds</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for col, info in report.outliers.outliers_by_column.items() %}
                        <tr>
                            <td><strong>{{ col }}</strong></td>
                            <td>{{ "{:,}".format(info.outlier_count) }}</td>
                            <td>{{ "{:.2%}".format(info.outlier_ratio) }}</td>
                            <td>{{ "{:,}".format(info.extreme_count) }}</td>
                            <td>
                                {% if info.bounds %}
                                [{{ "{:.2f}".format(info.bounds.get("lower", 0)) }},
                                 {{ "{:.2f}".format(info.bounds.get("upper", 0)) }}]
                                {% else %}-{% endif %}
                            </td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>

            {% if charts.outliers_chart %}
            <div class="chart-container" id="outliers-chart"></div>
            {% endif %}

            {% if report.outliers.recommendations %}
            <div class="recommendations">
                <h4>Recommendations</h4>
                <ul>
                    {% for rec in report.outliers.recommendations %}
                    <li>{{ rec }}</li>
                    {% endfor %}
                </ul>
            </div>
            {% endif %}
        </div>
        {% endif %}

        <!-- Correlations -->
        {% if report.correlations and report.correlations.numeric_columns %}
        <div class="section">
            <div class="section-title">
                <span>Correlation Analysis ({{ report.correlations.method }})</span>
            </div>
            <div class="grid-4">
                <div class="metric-card">
                    <div class="metric-label">Numeric Columns</div>
                    <div class="metric-value">{{ report.correlations.numeric_columns | length }}</div>
                </div>
                <div class="metric-card {% if report.correlations.strong_positive_pairs %}warning{% else %}success{% endif %}">
                    <div class="metric-label">Strong Positive</div>
                    <div class="metric-value">{{ report.correlations.strong_positive_pairs | length }}</div>
                </div>
                <div class="metric-card {% if report.correlations.strong_negative_pairs %}warning{% else %}success{% endif %}">
                    <div class="metric-label">Strong Negative</div>
                    <div class="metric-value">{{ report.correlations.strong_negative_pairs | length }}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Method</div>
                    <div class="metric-value" style="font-size: 1.1rem;">{{ report.correlations.method }}</div>
                </div>
            </div>

            {% if charts.correlation_chart %}
            <div class="chart-container" id="correlation-chart"></div>
            {% endif %}

            {% if report.correlations.high_correlations %}
            <h4 style="margin-top: 1.5rem; margin-bottom: 0.75rem;">High Correlation Pairs</h4>
            <div style="overflow-x: auto;">
                <table>
                    <thead>
                        <tr>
                            <th>Column 1</th>
                            <th>Column 2</th>
                            <th>Correlation</th>
                            <th>Type</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for pair in report.correlations.high_correlations %}
                        <tr>
                            <td>{{ pair.column_1 }}</td>
                            <td>{{ pair.column_2 }}</td>
                            <td class="{% if pair.abs_correlation > 0.9 %}severity-critical{% elif pair.abs_correlation > 0.8 %}severity-high{% else %}severity-moderate{% endif %}">
                                {{ "{:.3f}".format(pair.correlation) }}
                            </td>
                            <td>
                                {% if pair.correlation > 0 %}
                                <span class="tag tag-success">Positive</span>
                                {% else %}
                                <span class="tag tag-danger">Negative</span>
                                {% endif %}
                            </td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
            {% endif %}

            {% if report.correlations.recommendations %}
            <div class="recommendations">
                <h4>Recommendations</h4>
                <ul>
                    {% for rec in report.correlations.recommendations %}
                    <li>{{ rec }}</li>
                    {% endfor %}
                </ul>
            </div>
            {% endif %}
        </div>
        {% endif %}

        <!-- Warnings -->
        {% if report.warnings %}
        <div class="section">
            <div class="section-title">
                <span>Warnings</span>
            </div>
            <div class="recommendations" style="background: #fef3c7; border-left-color: var(--warning);">
                <ul>
                    {% for warning in report.warnings %}
                    <li>{{ warning }}</li>
                    {% endfor %}
                </ul>
            </div>
        </div>
        {% endif %}
    </div>

    <div class="footer">
        <p>Generated by SmartEDA v{{ report.version }} | Report generated in {{ "%.2f" | format(report.execution_time) }}s</p>
        <p style="margin-top: 0.5rem;">Production-grade Exploratory Data Analysis toolkit</p>
    </div>

    <script>
        {% if charts.missing_values_chart %}
        Plotly.newPlot('missing-values-chart', {{ charts.missing_values_chart | safe }}, {
            responsive: true,
            displayModeBar: true
        });
        {% endif %}

        {% if charts.outliers_chart %}
        Plotly.newPlot('outliers-chart', {{ charts.outliers_chart | safe }}, {
            responsive: true,
            displayModeBar: true
        });
        {% endif %}

        {% if charts.correlation_chart %}
        Plotly.newPlot('correlation-chart', {{ charts.correlation_chart | safe }}, {
            responsive: true,
            displayModeBar: true
        });
        {% endif %}
    </script>
</body>
</html>
'''


class HTMLReportGenerator:
    """
    Generates interactive HTML reports from EDA analysis results.

    Creates beautiful, self-contained HTML reports with Plotly
    visualizations using Jinja2 templates.

    Example:
        >>> generator = HTMLReportGenerator()
        >>> html = generator.generate(report)
        >>> generator.save(report, "eda_report.html")
    """

    def __init__(self, template_path: Optional[str] = None) -> None:
        """
        Initialize HTMLReportGenerator.

        Args:
            template_path: Path to custom Jinja2 template.
                Uses default template if None.
        """
        if template_path and Path(template_path).exists():
            with open(template_path, "r") as f:
                self.template_content = f.read()
        else:
            self.template_content = DEFAULT_TEMPLATE

        self.template = Template(self.template_content)

    def generate(self, report: EDAReport) -> str:
        """
        Generate HTML report from EDAReport.

        Args:
            report: Complete EDA analysis report.

        Returns:
            HTML string.
        """
        charts = self._generate_charts(report)

        html = self.template.render(
            report=report,
            charts=charts,
        )

        logger.info("Generated HTML report for '%s'", report.dataset_name)

        return html

    def save(self, report: EDAReport, output_path: str) -> str:
        """
        Generate and save HTML report to file.

        Args:
            report: Complete EDA analysis report.
            output_path: Path to save the HTML file.

        Returns:
            Path to the saved file.
        """
        html = self.generate(report)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        logger.info("Saved HTML report to: %s", output_path)

        return str(output_path)

    def _generate_charts(self, report: EDAReport) -> Dict[str, Optional[str]]:
        """Generate Plotly charts for the report."""
        charts = {
            "missing_values_chart": None,
            "outliers_chart": None,
            "correlation_chart": None,
        }

        if report.missing_values and report.missing_values.column_missing:
            charts["missing_values_chart"] = self._create_missing_values_chart(
                report.missing_values
            )

        if report.outliers and report.outliers.outliers_by_column:
            charts["outliers_chart"] = self._create_outliers_chart(
                report.outliers
            )

        if (
            report.correlations
            and not report.correlations.correlation_matrix.empty
        ):
            charts["correlation_chart"] = self._create_correlation_chart(
                report.correlations
            )

        return charts

    def _create_missing_values_chart(
        self, missing_report: MissingValueReport
    ) -> Optional[str]:
        """Create missing values bar chart."""
        if not missing_report.column_missing:
            return None

        columns = list(missing_report.column_missing.keys())
        missing_counts = [
            missing_report.column_missing[c]["null_count"] for c in columns
        ]
        missing_ratios = [
            missing_report.column_missing[c]["null_ratio"] * 100 for c in columns
        ]

        colors = [
            "#dc2626" if r > 50 else "#d97706" if r > 20 else "#2563eb"
            for r in missing_ratios
        ]

        fig = go.Figure(data=[
            go.Bar(
                x=columns,
                y=missing_ratios,
                text=[f"{r:.1f}%" for r in missing_ratios],
                textposition="auto",
                marker_color=colors,
                name="Missing %",
            )
        ])

        fig.update_layout(
            title="Missing Values by Column",
            xaxis_title="Column",
            yaxis_title="Missing %",
            template="plotly_white",
            height=400,
            showlegend=False,
        )

        return fig.to_json()

    def _create_outliers_chart(self, outlier_report: OutlierReport) -> Optional[str]:
        """Create outliers summary chart."""
        if not outlier_report.outliers_by_column:
            return None

        columns = list(outlier_report.outliers_by_column.keys())
        counts = [
            outlier_report.outliers_by_column[c]["outlier_count"]
            for c in columns
        ]

        fig = go.Figure(data=[
            go.Bar(
                x=columns,
                y=counts,
                text=[str(c) for c in counts],
                textposition="auto",
                marker_color="#d97706",
                name="Outlier Count",
            )
        ])

        fig.update_layout(
            title=f"Outlier Count by Column ({outlier_report.method_used})",
            xaxis_title="Column",
            yaxis_title="Outlier Count",
            template="plotly_white",
            height=400,
            showlegend=False,
        )

        return fig.to_json()

    def _create_correlation_chart(
        self, correlation_report: CorrelationReport
    ) -> Optional[str]:
        """Create correlation heatmap chart."""
        if correlation_report.correlation_matrix.empty:
            return None

        corr = correlation_report.correlation_matrix

        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale="RdBu",
            zmin=-1,
            zmax=1,
            text=[[f"{v:.2f}" for v in row] for row in corr.values],
            texttemplate="%{text}",
            textfont={"size": 10},
        ))

        fig.update_layout(
            title=f"Correlation Matrix ({correlation_report.method})",
            template="plotly_white",
            height=600,
            width=800,
            xaxis={"tickangle": -45},
        )

        return fig.to_json()
