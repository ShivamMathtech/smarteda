"""
SmartEDA Basic Usage Example
============================

This example demonstrates the complete SmartEDA workflow
from data loading to HTML report generation.
"""

from datetime import datetime

import numpy as np
import pandas as pd

from smarteda import (
    CorrelationAnalyzer,
    DataLoader,
    DuplicateAnalyzer,
    EDAReport,
    HTMLReportGenerator,
    MissingValueAnalyzer,
    OutlierAnalyzer,
    QualityScorer,
    SchemaDetector,
)


def create_sample_data(n_rows: int = 1000) -> pd.DataFrame:
    """Create realistic sample data for demonstration."""
    np.random.seed(42)

    # Create correlated features
    age = np.random.randint(18, 80, n_rows)
    experience = age - 18 + np.random.normal(0, 2, n_rows).astype(int)
    salary = 30000 + experience * 2500 + np.random.normal(0, 10000, n_rows)

    # Add some missing values
    missing_idx = np.random.choice(n_rows, size=50, replace=False)
    experience_with_missing = experience.copy().astype(float)
    experience_with_missing[missing_idx[:25]] = np.nan

    salary_with_missing = salary.copy()
    salary_with_missing[missing_idx[25:]] = np.nan

    return pd.DataFrame({
        "employee_id": range(1, n_rows + 1),
        "name": [f"Employee_{i}" for i in range(n_rows)],
        "age": age,
        "experience_years": experience_with_missing.round(0),
        "salary": salary_with_missing.round(2),
        "department": np.random.choice(
            ["Engineering", "Sales", "Marketing", "HR", "Finance"],
            n_rows,
        ),
        "is_full_time": np.random.choice([True, False], n_rows, p=[0.8, 0.2]),
        "performance_score": np.random.uniform(0, 100, n_rows).round(2),
        "email": [f"emp{i}@company.com" for i in range(n_rows)],
        "join_date": pd.date_range("2018-01-01", periods=n_rows, freq="D"),
    })


def main() -> None:
    """Run the complete SmartEDA workflow."""
    print("=" * 60)
    print("SmartEDA v0.1 - Automated EDA Report Generation")
    print("=" * 60)

    # Step 1: Create sample data (or load from file)
    print("\n[Step 1] Loading data...")
    df = create_sample_data(n_rows=1000)
    print(f"  Loaded DataFrame: {df.shape[0]:,} rows x {df.shape[1]} columns")

    # Alternative: Load from file
    # df = DataLoader().load("your_data.csv")
    # df = DataLoader().load("your_data.xlsx", sheet_name="Sheet1")

    # Step 2: Schema Detection
    print("\n[Step 2] Detecting schema...")
    schema_detector = SchemaDetector()
    schema = schema_detector.detect(df)

    print(f"  Rows: {schema.row_count:,}")
    print(f"  Columns: {schema.column_count}")
    print(f"  Memory: {schema.memory_usage_human}")
    print(f"  Detected types:")
    for name, col in schema.columns.items():
        print(f"    - {name:20s}: {col.inferred_type.value:12s} ({col.cardinality})")

    # Step 3: Missing Value Analysis
    print("\n[Step 3] Analyzing missing values...")
    missing_analyzer = MissingValueAnalyzer()
    missing_report = missing_analyzer.analyze(df)

    print(f"  Total missing: {missing_report.total_missing:,} cells")
    print(f"  Missing ratio: {missing_report.overall_missing_ratio:.2%}")
    if missing_report.columns_with_missing:
        print(f"  Affected columns: {missing_report.columns_with_missing}")

    # Step 4: Duplicate Analysis
    print("\n[Step 4] Checking for duplicates...")
    duplicate_analyzer = DuplicateAnalyzer()
    duplicate_report = duplicate_analyzer.analyze(df)

    print(f"  Exact duplicates: {duplicate_report.exact_duplicates}")
    print(f"  Duplicate ratio: {duplicate_report.exact_duplicate_ratio:.2%}")

    # Step 5: Outlier Detection
    print("\n[Step 5] Detecting outliers (IQR method)...")
    outlier_analyzer = OutlierAnalyzer()
    outlier_report = outlier_analyzer.analyze(df)

    print(f"  Total outliers: {outlier_report.total_outliers:,}")
    print(f"  Affected columns: {len(outlier_report.outliers_by_column)}")
    for col, info in outlier_report.outliers_by_column.items():
        print(f"    - {col}: {info['outlier_count']} outliers ({info['outlier_ratio']:.2%})")

    # Step 6: Correlation Analysis
    print("\n[Step 6] Analyzing correlations...")
    correlation_analyzer = CorrelationAnalyzer(method="pearson")
    correlation_report = correlation_analyzer.analyze(df)

    print(f"  Numeric columns: {len(correlation_report.numeric_columns)}")
    print(f"  Strong positive: {len(correlation_report.strong_positive_pairs)}")
    print(f"  Strong negative: {len(correlation_report.strong_negative_pairs)}")

    if correlation_report.strong_positive_pairs:
        print("  Top positive correlations:")
        for pair in correlation_report.strong_positive_pairs[:3]:
            print(f"    - {pair['column_1']} vs {pair['column_2']}: {pair['correlation']:.3f}")

    # Step 7: Quality Scoring
    print("\n[Step 7] Computing quality score...")
    quality_scorer = QualityScorer()
    quality_score = quality_scorer.score(df)

    print(f"  Overall Score: {quality_score.overall_score:.1f}/100")
    print(f"  Grade: {quality_score.grade}")
    print(f"  Summary: {quality_score.summary}")
    print(f"  Breakdown:")
    for dim, score in quality_score.score_breakdown.items():
        print(f"    - {dim}: {score * 100:.1f}%")

    # Step 8: Generate HTML Report
    print("\n[Step 8] Generating HTML report...")
    start_time = datetime.now()

    report = EDAReport(
        dataset_name="employee_data",
        schema=schema,
        missing_values=missing_report,
        duplicates=duplicate_report,
        outliers=outlier_report,
        correlations=correlation_report,
        quality_score=quality_score,
        execution_time=(datetime.now() - start_time).total_seconds(),
        generated_at=datetime.now().isoformat(),
    )

    generator = HTMLReportGenerator()
    output_path = generator.save(report, "eda_report.html")

    print(f"  Report saved to: {output_path}")
    print(f"  Execution time: {report.execution_time:.2f}s")

    # Print recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)

    if quality_score.improvement_areas:
        print("\nPriority Improvements:")
        for area in quality_score.improvement_areas[:3]:
            print(f"  - {area}")

    if missing_report.recommendations:
        print("\nMissing Value Actions:")
        for rec in missing_report.recommendations[:2]:
            print(f"  - {rec}")

    if outlier_report.recommendations:
        print("\nOutlier Handling:")
        for rec in outlier_report.recommendations[:2]:
            print(f"  - {rec}")

    print("\n" + "=" * 60)
    print("EDA Complete! Open eda_report.html in your browser.")
    print("=" * 60)


if __name__ == "__main__":
    main()
