"""
Duplicate detection module.

Identifies exact and fuzzy duplicate records, analyzes
duplicate groups, and provides deduplication recommendations.
"""

from __future__ import annotations

import logging
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from smarteda.models.analysis_result import DuplicateReport

logger = logging.getLogger(__name__)


class DuplicateAnalyzer:
    """
    Comprehensive duplicate record analyzer.

    Detects exact duplicates, near-duplicates using string similarity,
and key-column duplicates. Provides actionable deduplication strategies.

    Example:
        >>> analyzer = DuplicateAnalyzer()
        >>> report = analyzer.analyze(df)
        >>> print(report.total_duplicates)
    """

    def __init__(
        self,
        fuzzy_threshold: float = 0.85,
        max_comparisons: int = 10000,
    ) -> None:
        """
        Initialize DuplicateAnalyzer.

        Args:
            fuzzy_threshold: Similarity threshold for fuzzy matching (0-1).
            max_comparisons: Maximum pairwise comparisons for fuzzy detection.
        """
        self.fuzzy_threshold = fuzzy_threshold
        self.max_comparisons = max_comparisons

    def analyze(
        self,
        df: pd.DataFrame,
        subset: Optional[List[str]] = None,
        key_columns: Optional[List[str]] = None,
        check_fuzzy: bool = False,
    ) -> DuplicateReport:
        """
        Perform comprehensive duplicate analysis.

        Args:
            df: Input pandas DataFrame.
            subset: Column subset to consider for exact duplicates.
            key_columns: Columns to check for key-level duplicates.
            check_fuzzy: Whether to perform fuzzy duplicate detection.

        Returns:
            DuplicateReport with all analysis results.
        """
        if df.empty:
            return DuplicateReport()

        # Exact duplicate analysis
        exact_dups = self._find_exact_duplicates(df, subset)

        # Key column duplicate analysis
        key_dup_analysis = {}
        if key_columns:
            for col in key_columns:
                if col in df.columns:
                    dup_count = int(df[col].duplicated().sum())
                    key_dup_analysis[str(col)] = dup_count

        # Fuzzy duplicate analysis
        fuzzy_dups = 0
        fuzzy_details = []
        if check_fuzzy and len(df) <= self.max_comparisons:
            fuzzy_dups, fuzzy_details = self._find_fuzzy_duplicates(df, subset)

        # Build duplicate groups summary
        duplicate_groups = self._summarize_duplicate_groups(df, subset)

        total_duplicates = exact_dups + fuzzy_dups
        duplicate_ratio = total_duplicates / len(df) if len(df) > 0 else 0.0

        recommendations = self._generate_recommendations(
            df, exact_dups, fuzzy_dups, key_dup_analysis, subset
        )

        report = DuplicateReport(
            total_duplicates=total_duplicates,
            duplicate_ratio=round(duplicate_ratio, 6),
            exact_duplicates=exact_dups,
            exact_duplicate_ratio=round(exact_dups / len(df), 6) if len(df) > 0 else 0.0,
            fuzzy_duplicates=fuzzy_dups,
            fuzzy_duplicate_details=fuzzy_details,
            duplicate_groups=duplicate_groups,
            columns_analyzed=[str(c) for c in (subset or df.columns)],
            key_column_duplicates=key_dup_analysis,
            recommendations=recommendations,
        )

        logger.info(
            "Duplicate analysis: %d exact, %d fuzzy (%.2f%% total)",
            exact_dups,
            fuzzy_dups,
            duplicate_ratio * 100,
        )

        return report

    def _find_exact_duplicates(
        self, df: pd.DataFrame, subset: Optional[List[str]] = None
    ) -> int:
        """Count exact duplicate rows."""
        if subset:
            valid_cols = [c for c in subset if c in df.columns]
            if not valid_cols:
                return 0
            return int(df.duplicated(subset=valid_cols, keep=False).sum())

        return int(df.duplicated(keep=False).sum())

    def _find_fuzzy_duplicates(
        self,
        df: pd.DataFrame,
        subset: Optional[List[str]] = None,
    ) -> Tuple[int, List[Dict[str, Any]]]:
        """
        Find near-duplicate rows using string similarity.

        Uses a sampling approach for large datasets.
        """
        cols = subset or df.select_dtypes(include=["object"]).columns.tolist()
        cols = [c for c in cols if c in df.columns]

        if not cols or len(df) < 2:
            return 0, []

        # Sample for performance
        sample_size = min(len(df), 5000)
        sample_df = df[cols].sample(n=sample_size, random_state=42) if len(df) > sample_size else df[cols]

        fuzzy_count = 0
        details = []
        checked = set()

        # Create string representation for comparison
        sample_strings = sample_df.astype(str).fillna("").apply(
            lambda row: " ".join(row.values), axis=1
        )

        indices = sample_strings.index.tolist()
        max_pairs = min(len(indices) * (len(indices) - 1) // 2, self.max_comparisons)

        pair_count = 0
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                if pair_count >= max_pairs:
                    break

                idx1, idx2 = indices[i], indices[j]
                if (idx1, idx2) in checked:
                    continue

                checked.add((idx1, idx2))
                pair_count += 1

                str1 = sample_strings.loc[idx1]
                str2 = sample_strings.loc[idx2]

                if len(str1) < 3 or len(str2) < 3:
                    continue

                similarity = SequenceMatcher(None, str1, str2).ratio()

                if similarity >= self.fuzzy_threshold and similarity < 1.0:
                    fuzzy_count += 1
                    details.append({
                        "index_1": int(idx1),
                        "index_2": int(idx2),
                        "similarity": round(similarity, 4),
                        "preview_1": str1[:100],
                        "preview_2": str2[:100],
                    })

            if pair_count >= max_pairs:
                break

        # Scale up estimate if sampled
        if len(df) > sample_size:
            scale_factor = (len(df) / sample_size) ** 2
            fuzzy_count = int(fuzzy_count * min(scale_factor, 10))

        return fuzzy_count, details[:100]  # Cap details

    def _summarize_duplicate_groups(
        self, df: pd.DataFrame, subset: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Create summary of duplicate groups."""
        cols = subset or list(df.columns)
        cols = [c for c in cols if c in df.columns]

        if not cols:
            return pd.DataFrame()

        dup_mask = df.duplicated(subset=cols, keep=False)

        if not dup_mask.any():
            return pd.DataFrame()

        dup_df = df[dup_mask].copy()

        group_sizes = dup_df.groupby(cols[0] if len(cols) == 1 else cols).size()
        group_sizes = group_sizes[group_sizes > 1].sort_values(ascending=False)

        if len(group_sizes) == 0:
            return pd.DataFrame()

        summary = pd.DataFrame({
            "group_size": group_sizes.values,
            "times_seen": group_sizes.values,
        }).head(20)

        return summary

    def _generate_recommendations(
        self,
        df: pd.DataFrame,
        exact_dups: int,
        fuzzy_dups: int,
        key_dups: Dict[str, int],
        subset: Optional[List[str]],
    ) -> List[str]:
        """Generate deduplication recommendations."""
        recommendations = []

        if exact_dups == 0 and fuzzy_dups == 0:
            recommendations.append("No duplicates detected - dataset is clean.")
            return recommendations

        if exact_dups > 0:
            exact_ratio = exact_dups / len(df) if len(df) > 0 else 0

            if exact_ratio > 0.1:
                recommendations.append(
                    f"CRITICAL: {exact_dups} exact duplicates ({exact_ratio:.1%}). "
                    "Immediate deduplication required."
                )
            else:
                recommendations.append(
                    f"{exact_dups} exact duplicates found ({exact_ratio:.1%}). "
                    "Consider dropping duplicates with keep='first'."
                )

        if fuzzy_dups > 0:
            recommendations.append(
                f"{fuzzy_dups} potential near-duplicates detected. "
                f"Review using similarity threshold {self.fuzzy_threshold}."
            )

        if subset:
            recommendations.append(
                f"Duplicates analyzed on subset: {subset}. "
                "Consider analyzing on full columns for complete picture."
            )

        for col, count in key_dups.items():
            if count > 0:
                recommendations.append(
                    f"{count} duplicate values in key column '{col}'. "
                    "Verify if this is expected for composite keys."
                )

        if exact_dups > 0:
            recommendations.append(
                "Deduplication code: df = df.drop_duplicates()"
            )

        return recommendations

    def find_duplicate_groups(
        self, df: pd.DataFrame, subset: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Return all rows that are part of duplicate groups.

        Args:
            df: Input DataFrame.
            subset: Columns to consider for duplication.

        Returns:
            DataFrame with duplicate rows marked.
        """
        cols = subset or list(df.columns)
        cols = [c for c in cols if c in df.columns]

        if not cols:
            return df.copy()

        dup_mask = df.duplicated(subset=cols, keep=False)
        result = df[dup_mask].copy()
        result["__duplicate_group__"] = result.groupby(cols).ngroup()

        return result
