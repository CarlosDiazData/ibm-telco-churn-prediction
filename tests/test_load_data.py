"""Unit tests for src/data/load_data.py."""
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


class TestLoadTelcoData:
    """Test suite for load_telco_data function."""

    def test_load_telco_data_returns_dataframe(self):
        """GIVEN a valid Excel file path
        WHEN load_telco_data is called
        THEN a pandas DataFrame is returned."""
        mock_df = pd.DataFrame({"Col1": [1, 2], "Churn": ["No", "Yes"]})

        with patch("pandas.read_excel", return_value=mock_df), \
             patch("pathlib.Path.exists", return_value=True):
            from src.data.load_data import load_telco_data

            result = load_telco_data("dummy_path.xlsx")

            assert isinstance(result, pd.DataFrame)

    def test_load_telco_data_shape_and_columns(self):
        """GIVEN mocked Excel data
        WHEN loaded
        THEN the DataFrame has the expected shape and columns."""
        mock_data = {
            "CustomerID": ["C001", "C002", "C003"],
            "tenure": [12, 24, 6],
            "Churn": ["No", "Yes", "No"],
        }
        mock_df = pd.DataFrame(mock_data)

        with patch("pandas.read_excel", return_value=mock_df), \
             patch("pathlib.Path.exists", return_value=True):
            from src.data.load_data import load_telco_data

            result = load_telco_data("dummy_path.xlsx")

            assert result.shape == (3, 3)
            assert list(result.columns) == ["CustomerID", "tenure", "Churn"]

    def test_load_telco_data_file_not_found(self):
        """GIVEN a non-existent file path
        WHEN load_telco_data is called
        THEN a FileNotFoundError is raised."""
        from src.data.load_data import load_telco_data

        with pytest.raises(FileNotFoundError):
            load_telco_data("/nonexistent/path/Telco_customer_churn.xlsx")
