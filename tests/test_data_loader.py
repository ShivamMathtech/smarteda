"""
Unit tests for the data_loader module.
"""

from __future__ import annotations

import pandas as pd
import pytest
from pathlib import Path

from smarteda.core.data_loader import DataLoader, DataLoadError, UnsupportedFormatError


class TestDataLoader:
    """Test cases for DataLoader."""

    def test_load_csv_success(self, temp_csv_file: str) -> None:
        """Test loading a valid CSV file."""
        loader = DataLoader()
        df = loader.load(temp_csv_file)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert list(df.columns) == ["A", "B", "C"]

    def test_load_excel_success(self, temp_excel_file: str) -> None:
        """Test loading a valid Excel file."""
        loader = DataLoader()
        df = loader.load(temp_excel_file)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert list(df.columns) == ["X", "Y", "Z"]

    def test_load_nonexistent_file(self) -> None:
        """Test loading a non-existent file raises error."""
        loader = DataLoader()

        with pytest.raises(DataLoadError, match="File not found"):
            loader.load("/path/to/nonexistent/file.csv")

    def test_load_unsupported_format(self, tmp_path: Path) -> None:
        """Test loading an unsupported file format raises error."""
        loader = DataLoader()
        filepath = tmp_path / "data.pdf"
        filepath.write_text("test")

        with pytest.raises(UnsupportedFormatError, match="Unsupported file format"):
            loader.load(str(filepath))

    def test_load_from_buffer_csv(self, temp_csv_file: str) -> None:
        """Test loading CSV from buffer."""
        loader = DataLoader()

        with open(temp_csv_file, "rb") as f:
            buffer = f.read()

        df = loader.load_from_buffer(buffer, file_extension=".csv")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5

    def test_load_from_buffer_excel(self, temp_excel_file: str) -> None:
        """Test loading Excel from buffer."""
        loader = DataLoader()

        with open(temp_excel_file, "rb") as f:
            buffer = f.read()

        df = loader.load_from_buffer(buffer, file_extension=".xlsx")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5

    def test_type_optimization(self, sample_df: pd.DataFrame, tmp_path: Path) -> None:
        """Test that type optimization reduces memory usage."""
        filepath = tmp_path / "opt_test.csv"
        sample_df.to_csv(filepath, index=False)

        loader = DataLoader(optimize_types=True)
        df_optimized = loader.load(str(filepath))

        loader_unoptimized = DataLoader(optimize_types=False)
        df_unoptimized = loader_unoptimized.load(str(filepath))

        assert isinstance(df_optimized, pd.DataFrame)
        assert len(df_optimized) == len(df_unoptimized)

    def test_auto_date_parsing(self, tmp_path: Path) -> None:
        """Test automatic date column detection."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "date": ["2023-01-01", "2023-06-15", "2023-12-31"],
            "value": [10, 20, 30],
        })

        filepath = tmp_path / "dates.csv"
        df.to_csv(filepath, index=False)

        loader = DataLoader(parse_dates=True)
        result = loader.load(str(filepath))

        assert isinstance(result, pd.DataFrame)

    def test_load_multiple_files(self, tmp_path: Path) -> None:
        """Test loading and concatenating multiple files."""
        for i in range(3):
            df = pd.DataFrame({"A": [i, i + 1], "B": ["x", "y"]})
            df.to_csv(tmp_path / f"file_{i}.csv", index=False)

        loader = DataLoader()
        filepaths = [str(tmp_path / f"file_{i}.csv") for i in range(3)]
        result = loader.load_multiple(filepaths, concat=True)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 6

    def test_get_file_info(self, temp_csv_file: str) -> None:
        """Test getting file information."""
        info = DataLoader.get_file_info(temp_csv_file)

        assert info["exists"] is True
        assert info["extension"] == ".csv"
        assert info["is_supported"] is True
        assert "size_human" in info

    def test_get_file_info_nonexistent(self) -> None:
        """Test getting info for non-existent file."""
        info = DataLoader.get_file_info("/nonexistent/file.csv")

        assert info["exists"] is False
        assert info["is_supported"] is True

    def test_csv_with_different_separator(self, tmp_path: Path) -> None:
        """Test loading CSV with semicolon separator."""
        filepath = tmp_path / "semicolon.csv"
        filepath.write_text("A;B;C\n1;a;10\n2;b;20\n")

        loader = DataLoader()
        df = loader.load(str(filepath), sep=";")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_empty_dataframe_load(self, tmp_path: Path) -> None:
        """Test loading a CSV that results in empty DataFrame."""
        filepath = tmp_path / "empty.csv"
        pd.DataFrame(columns=["A", "B"]).to_csv(filepath, index=False)

        loader = DataLoader()
        df = loader.load(str(filepath))

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
