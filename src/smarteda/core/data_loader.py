"""
Data loading module for CSV and Excel file ingestion.

Handles automatic delimiter detection, encoding inference,
type optimization, and robust error handling for common
edge cases in real-world data files.
"""

from __future__ import annotations

import logging
import warnings
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from smarteda.utils.helpers import detect_separator

logger = logging.getLogger(__name__)


class DataLoadError(Exception):
    """Custom exception for data loading failures."""

    pass


class UnsupportedFormatError(DataLoadError):
    """Raised when file format is not supported."""

    pass


class DataLoader:
    """
    Production-grade data loader for CSV and Excel files.

    Features:
        - Automatic delimiter/encoding detection for CSV
        - Multi-sheet Excel support with sheet selection
        - Type optimization to reduce memory usage
        - Robust error handling with fallback strategies
        - Progress logging for large files

    Example:
        >>> loader = DataLoader()
        >>> df = loader.load("data.csv")
        >>> df = loader.load("data.xlsx", sheet_name="Sheet1")
    """

    SUPPORTED_CSV_EXTENSIONS = {".csv", ".txt", ".tsv", ".dat"}
    SUPPORTED_EXCEL_EXTENSIONS = {".xls", ".xlsx", ".xlsm", ".ods"}

    def __init__(
        self,
        optimize_types: bool = True,
        parse_dates: bool = True,
        encoding_fallbacks: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the DataLoader.

        Args:
            optimize_types: Whether to downcast numeric types for memory efficiency.
            parse_dates: Whether to attempt automatic date parsing.
            encoding_fallbacks: List of encodings to try if UTF-8 fails.
            **kwargs: Additional arguments passed to pandas read functions.
        """
        self.optimize_types = optimize_types
        self.parse_dates = parse_dates
        self.encoding_fallbacks = encoding_fallbacks or [
            "utf-8",
            "latin-1",
            "iso-8859-1",
            "cp1252",
        ]
        self.default_kwargs = kwargs

    def load(
        self,
        filepath: Union[str, Path],
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Load a data file (CSV or Excel) into a DataFrame.

        Args:
            filepath: Path to the data file.
            **kwargs: Additional arguments for pandas readers.
                For CSV: sep, header, skiprows, etc.
                For Excel: sheet_name, header, etc.

        Returns:
            Loaded and optimized pandas DataFrame.

        Raises:
            UnsupportedFormatError: If file format is not supported.
            DataLoadError: If all loading strategies fail.

        Example:
            >>> loader = DataLoader()
            >>> df = loader.load("customers.csv")
            >>> df = loader.load("sales.xlsx", sheet_name=0)
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise DataLoadError(f"File not found: {filepath}")

        suffix = filepath.suffix.lower()

        if suffix in self.SUPPORTED_CSV_EXTENSIONS:
            return self._load_csv(filepath, **kwargs)
        elif suffix in self.SUPPORTED_EXCEL_EXTENSIONS:
            return self._load_excel(filepath, **kwargs)
        else:
            raise UnsupportedFormatError(
                f"Unsupported file format: '{suffix}'. "
                f"Supported formats: CSV {self.SUPPORTED_CSV_EXTENSIONS}, "
                f"Excel {self.SUPPORTED_EXCEL_EXTENSIONS}"
            )

    def load_from_buffer(
        self,
        buffer: Union[BytesIO, bytes],
        file_extension: str = ".csv",
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Load data from a buffer (useful for web uploads).

        Args:
            buffer: BytesIO or bytes object containing the data.
            file_extension: File extension to determine parser (".csv" or ".xlsx").
            **kwargs: Additional arguments for pandas readers.

        Returns:
            Loaded and optimized pandas DataFrame.
        """
        if isinstance(buffer, bytes):
            buffer = BytesIO(buffer)

        ext = file_extension.lower()

        if ext in self.SUPPORTED_CSV_EXTENSIONS:
            return self._load_csv_from_buffer(buffer, **kwargs)
        elif ext in self.SUPPORTED_EXCEL_EXTENSIONS:
            return self._load_excel_from_buffer(buffer, **kwargs)
        else:
            raise UnsupportedFormatError(f"Unsupported buffer format: '{ext}'")

    def _load_csv(self, filepath: Path, **kwargs: Any) -> pd.DataFrame:
        """Load a CSV file with automatic delimiter and encoding detection."""
        merged_kwargs = {**self.default_kwargs, **kwargs}
        encoding = merged_kwargs.pop("encoding", None)

        if encoding is None:
            encoding = self._detect_encoding(filepath)

        sep = merged_kwargs.pop("sep", None)
        if sep is None:
            sep = detect_separator(str(filepath))

        try:
            df = pd.read_csv(
                filepath,
                sep=sep,
                encoding=encoding,
                low_memory=False,
                **merged_kwargs,
            )
        except UnicodeDecodeError as e:
            logger.warning("Encoding %s failed, trying fallbacks: %s", encoding, e)
            df = self._try_encoding_fallbacks(filepath, sep, merged_kwargs)

        if self.parse_dates:
            df = self._auto_parse_dates(df)

        if self.optimize_types:
            df = self._optimize_dataframe_types(df)

        logger.info(
            "Loaded CSV: %s | Shape: %s | Memory: %.2f MB",
            filepath.name,
            df.shape,
            df.memory_usage(deep=True).sum() / 1e6,
        )

        return df

    def _load_csv_from_buffer(self, buffer: BytesIO, **kwargs: Any) -> pd.DataFrame:
        """Load CSV from a BytesIO buffer."""
        merged_kwargs = {**self.default_kwargs, **kwargs}
        sep = merged_kwargs.pop("sep", ",")

        for encoding in self.encoding_fallbacks:
            try:
                buffer.seek(0)
                df = pd.read_csv(
                    buffer,
                    sep=sep,
                    encoding=encoding,
                    low_memory=False,
                    **merged_kwargs,
                )
                break
            except UnicodeDecodeError:
                continue
        else:
            raise DataLoadError("Failed to decode buffer with all encodings")

        if self.parse_dates:
            df = self._auto_parse_dates(df)

        if self.optimize_types:
            df = self._optimize_dataframe_types(df)

        return df

    def _load_excel(self, filepath: Path, **kwargs: Any) -> pd.DataFrame:
        """Load an Excel file with sheet selection support."""
        merged_kwargs = {**self.default_kwargs, **kwargs}
        sheet_name = merged_kwargs.pop("sheet_name", 0)

        try:
            df = pd.read_excel(
                filepath,
                sheet_name=sheet_name,
                engine="openpyxl",
                **merged_kwargs,
            )

            if isinstance(df, dict):
                first_sheet = list(df.keys())[0]
                logger.info("Multi-sheet Excel detected. Using sheet: %s", first_sheet)
                df = df[first_sheet]

        except ValueError as e:
            raise DataLoadError(f"Failed to load Excel file: {e}")

        if self.parse_dates:
            df = self._auto_parse_dates(df)

        if self.optimize_types:
            df = self._optimize_dataframe_types(df)

        logger.info(
            "Loaded Excel: %s | Sheet: %s | Shape: %s",
            filepath.name,
            sheet_name,
            df.shape,
        )

        return df

    def _load_excel_from_buffer(self, buffer: BytesIO, **kwargs: Any) -> pd.DataFrame:
        """Load Excel from a BytesIO buffer."""
        merged_kwargs = {**self.default_kwargs, **kwargs}
        sheet_name = merged_kwargs.pop("sheet_name", 0)

        df = pd.read_excel(
            buffer,
            sheet_name=sheet_name,
            engine="openpyxl",
            **merged_kwargs,
        )

        if isinstance(df, dict):
            df = df[list(df.keys())[0]]

        if self.parse_dates:
            df = self._auto_parse_dates(df)

        if self.optimize_types:
            df = self._optimize_dataframe_types(df)

        return df

    def _detect_encoding(self, filepath: Path) -> str:
        """Detect file encoding by trying common encodings."""
        for encoding in self.encoding_fallbacks:
            try:
                with open(filepath, "r", encoding=encoding) as f:
                    f.read(1024)
                return encoding
            except (UnicodeDecodeError, UnicodeError):
                continue

        logger.warning("Could not detect encoding, falling back to utf-8 with errors='replace'")
        return "utf-8"

    def _try_encoding_fallbacks(
        self, filepath: Path, sep: str, kwargs: Dict[str, Any]
    ) -> pd.DataFrame:
        """Try multiple encodings when the primary fails."""
        for encoding in self.encoding_fallbacks[1:]:
            try:
                df = pd.read_csv(
                    filepath,
                    sep=sep,
                    encoding=encoding,
                    low_memory=False,
                    **kwargs,
                )
                logger.info("Successfully loaded with encoding: %s", encoding)
                return df
            except UnicodeDecodeError:
                continue

        raise DataLoadError("Failed to load file with any encoding")

    def _auto_parse_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Automatically detect and parse datetime columns."""
        for col in df.select_dtypes(include=["object"]).columns:
            if df[col].dropna().empty:
                continue

            try:
                sample = df[col].dropna().head(100)
                converted = pd.to_datetime(sample, errors="coerce")
                success_rate = converted.notna().sum() / len(sample)

                if success_rate > 0.8:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                    logger.debug("Parsed datetime column: %s", col)
            except (ValueError, TypeError):
                continue

        return df

    def _optimize_dataframe_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage by downcasting types."""
        optimized = df.copy()

        for col in optimized.columns:
            col_type = optimized[col].dtype

            if col_type == "object":
                num_unique = optimized[col].nunique(dropna=True)
                num_total = len(optimized[col].dropna())

                if num_unique / max(num_total, 1) < 0.5 and num_total > 0:
                    optimized[col] = optimized[col].astype("category")

            elif np.issubdtype(col_type, np.integer):
                optimized[col] = pd.to_numeric(optimized[col], downcast="integer")

            elif np.issubdtype(col_type, np.floating):
                optimized[col] = pd.to_numeric(optimized[col], downcast="float")

        original_mem = df.memory_usage(deep=True).sum()
        optimized_mem = optimized.memory_usage(deep=True).sum()

        if original_mem > 0:
            reduction = (1 - optimized_mem / original_mem) * 100
            if reduction > 10:
                logger.info(
                    "Memory optimized: %.1f%% reduction (%.2f MB -> %.2f MB)",
                    reduction,
                    original_mem / 1e6,
                    optimized_mem / 1e6,
                )

        return optimized

    @staticmethod
    def get_file_info(filepath: Union[str, Path]) -> Dict[str, Any]:
        """
        Get information about a file without loading it.

        Args:
            filepath: Path to the data file.

        Returns:
            Dictionary with file metadata.
        """
        filepath = Path(filepath)

        info = {
            "name": filepath.name,
            "extension": filepath.suffix.lower(),
            "size_bytes": filepath.stat().st_size if filepath.exists() else 0,
            "exists": filepath.exists(),
            "is_supported": filepath.suffix.lower()
            in (
                DataLoader.SUPPORTED_CSV_EXTENSIONS
                | DataLoader.SUPPORTED_EXCEL_EXTENSIONS
            ),
        }

        if info["size_bytes"] > 0:
            for unit in ["B", "KB", "MB", "GB"]:
                if info["size_bytes"] < 1024:
                    info["size_human"] = f"{info['size_bytes']:.1f} {unit}"
                    break
                info["size_bytes"] /= 1024

        return info

    def load_multiple(
        self,
        filepaths: List[Union[str, Path]],
        concat: bool = True,
        **kwargs: Any,
    ) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        """
        Load multiple files and optionally concatenate them.

        Args:
            filepaths: List of file paths to load.
            concat: Whether to concatenate all files into one DataFrame.
            **kwargs: Additional arguments for loading.

        Returns:
            Single DataFrame if concat=True, else list of DataFrames.
        """
        dataframes = []

        for fp in filepaths:
            try:
                df = self.load(fp, **kwargs)
                df["__source_file__"] = Path(fp).name
                dataframes.append(df)
            except DataLoadError as e:
                logger.error("Failed to load %s: %s", fp, e)

        if not dataframes:
            raise DataLoadError("No files were successfully loaded")

        if concat:
            return pd.concat(dataframes, ignore_index=True)

        return dataframes
