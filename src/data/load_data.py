"""Load Telco customer churn data from Excel."""
from pathlib import Path

import pandas as pd


def load_telco_data(path: str) -> pd.DataFrame:
    """Load Telco churn Excel file, return DataFrame with all columns.

    Args:
        path: Path to the Telco_customer_churn.xlsx file.

    Returns:
        DataFrame with all columns intact.

    Raises:
        FileNotFoundError: If the Excel file does not exist at the given path.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_excel(path, engine="openpyxl")

    # Coerce Total Charges to numeric (it's stored as object/string in the Excel)
    if "Total Charges" in df.columns:
        df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")

    return df
