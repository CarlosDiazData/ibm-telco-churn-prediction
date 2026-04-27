"""Exploratory data analysis for Telco churn dataset."""
import pandas as pd


def generate_eda_report(df: pd.DataFrame) -> dict:
    """Generate data profile report including missing values and class distribution.

    Args:
        df: The Telco churn DataFrame.

    Returns:
        Dictionary with:
        - missing_summary: dict of {column: (count, percentage)}
        - class_distribution: dict of {class_label: count}
        - dtypes: Series of column dtypes
    """
    # Missing value summary
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / len(df)) * 100
    missing_summary = {
        col: {"count": int(count), "percentage": round(pct, 2)}
        for col, count, pct in zip(df.columns, missing_counts, missing_pct)
        if count > 0
    }

    # Class distribution for Churn column — check both "Churn Label" and "Churn Value"
    class_distribution = {}
    churn_col = None
    if "Churn Label" in df.columns:
        churn_col = "Churn Label"
    elif "Churn" in df.columns:
        churn_col = "Churn"

    if churn_col:
        churn_counts = df[churn_col].value_counts()
        class_distribution = {
            label: int(count) for label, count in zip(churn_counts.index, churn_counts)
        }

    # Data types
    dtypes = df.dtypes

    report = {
        "missing_summary": missing_summary,
        "class_distribution": class_distribution,
        "dtypes": dtypes,
        "total_rows": len(df),
        "total_columns": len(df.columns),
    }

    return report


def print_eda_report(report: dict) -> None:
    """Print the EDA report to stdout."""
    print("=" * 60)
    print("TELCO CHURN DATA — EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    print(f"\nDataset shape: {report['total_rows']} rows × {report['total_columns']} columns")

    # Missing values
    print("\n--- Missing Values ---")
    missing = report["missing_summary"]
    if missing:
        for col, info in missing.items():
            print(f"  {col}: {info['count']} ({info['percentage']}%)")
    else:
        print("  No missing values detected.")

    # Class distribution
    print("\n--- Target Class Distribution (Churn) ---")
    if report["class_distribution"]:
        total = sum(report["class_distribution"].values())
        for label, count in report["class_distribution"].items():
            pct = (count / total) * 100
            print(f"  {label}: {count} ({pct:.1f}%)")
    else:
        print("  'Churn' column not found in dataset.")

    # Data types
    print("\n--- Column Data Types ---")
    for col, dtype in report["dtypes"].items():
        print(f"  {col}: {dtype}")

    print("=" * 60)
