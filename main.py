#!/usr/bin/env python3
"""Orchestrator for the Telco churn prediction pipeline.

Loads raw data → runs EDA → preprocesses → trains models with MLflow tracking
→ evaluates on holdout set → serializes best model.
"""
import warnings

from sklearn.model_selection import train_test_split

from src.data.eda import generate_eda_report, print_eda_report
from src.data.load_data import load_telco_data
from src.features.preprocess import build_preprocessor, configure_preprocessor
from src.models.evaluate import evaluate
from src.models.serialize import serialize_artifacts
from src.models.train import train_and_log


def identify_column_types(df, target_col="Churn Value"):
    """Identify numerical and categorical columns automatically.

    Args:
        df: Input DataFrame.
        target_col: Name of the target column to exclude.

    Returns:
        Tuple of (numerical_cols, categorical_cols).
    """
    numerical_cols = []
    categorical_cols = []

    for col in df.columns:
        if col == target_col:
            continue
        # Exclude other churn-related metadata columns that shouldn't be features
        if col in ["Churn Label", "Churn Score", "Churn Reason", "CustomerID", "Count", "Country", "State", "City", "Zip Code", "Lat Long", "Latitude", "Longitude", "CLTV"]:
            continue
        if df[col].dtype in ["int64", "float64"]:
            numerical_cols.append(col)
        else:
            categorical_cols.append(col)

    return numerical_cols, categorical_cols


def main():
    warnings.filterwarnings("ignore")

    # Configuration
    DATA_PATH = "data/raw/Telco_customer_churn.xlsx"
    # Since the actual file is at project root, use the actual path
    import os
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(PROJECT_ROOT, "Telco_customer_churn.xlsx")

    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    TARGET_COL = "Churn Value"  # Use numeric target (0/1), not "Churn Label" (Yes/No)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("Loading data...")
    df = load_telco_data(DATA_PATH)
    print(f"Loaded {len(df)} rows × {len(df.columns)} columns")

    # ── 2. EDA ─────────────────────────────────────────────────────────────────
    print("\nRunning EDA...")
    eda_report = generate_eda_report(df)
    print_eda_report(eda_report)

    # ── 3. Preprocessing ───────────────────────────────────────────────────────
    # Separate features and target
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # Encode target if it's string labels (shouldn't happen with Churn Value but handle it)
    if y.dtype == "object":
        y = y.map({"Yes": 1, "No": 0, "Yes ": 1, "No ": 0}).fillna(y)
    # Ensure y is integer type
    y = y.astype(int)

    # Identify column types
    numerical_cols, categorical_cols = identify_column_types(X, TARGET_COL)
    print(f"\nColumn types — Numerical: {len(numerical_cols)}, Categorical: {len(categorical_cols)}")

    # Build and configure preprocessor
    preprocessor = build_preprocessor()
    preprocessor = configure_preprocessor(preprocessor, numerical_cols, categorical_cols)

    # ── 4. Train/Test split ────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

    # ── 4b. Compute CLTV sample weights ───────────────────────────────────────
    # Extract CLTV weights aligned with X_train to prevent leakage
    weights_train = df.loc[X_train.index, "CLTV"] / df["CLTV"].mean()
    print(f"CLTV weight stats — min: {weights_train.min():.3f}, max: {weights_train.max():.3f}, mean: {weights_train.mean():.3f}")

    # ── 5. Train and log to MLflow ─────────────────────────────────────────────
    print("\nTraining models (with MLflow tracking)...")
    run_ids = train_and_log(X_train, y_train, preprocessor, sample_weight=weights_train.values)
    print(f"MLflow run IDs: {run_ids}")

    # ── 6. Evaluate ────────────────────────────────────────────────────────────
    # Re-train full pipeline to get the fitted preprocessor + classifier
    from sklearn.pipeline import Pipeline as SkPipeline
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC

    results = {}
    for model_name, model in [
        ("LogisticRegression", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ("RandomForest", RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)),
        ("GradientBoosting", GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ("SVM", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)),
    ]:
        pipeline = SkPipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", model),
            ]
        )
        pipeline.fit(X_train, y_train)
        metrics = evaluate(pipeline, X_test, y_test)
        results[model_name] = metrics
        print(f"\n{model_name} — ROC-AUC: {metrics['roc_auc']}, F1: {metrics['f1']}")

    # ── 7. Select best model ───────────────────────────────────────────────────
    best_model_name = max(results, key=lambda k: results[k]["roc_auc"])
    best_metrics = results[best_model_name]
    print(f"\nBest model: {best_model_name} (ROC-AUC: {best_metrics['roc_auc']})")

    # Re-fit best model for serialization
    model_registry = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42),
    }
    best_clf = model_registry[best_model_name]

    best_pipeline = SkPipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", best_clf),
        ]
    )
    best_pipeline.fit(X_train, y_train)

    # ── 8. Serialize ────────────────────────────────────────────────────────────
    print("\nSerializing artifacts to models/ ...")
    serialize_artifacts(preprocessor, best_pipeline)

    # ── 9. Final summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Best model: {best_model_name}")
    print(f"ROC-AUC: {best_metrics['roc_auc']}")
    print(f"F1: {best_metrics['f1']}")
    print(f"Precision: {best_metrics['precision']}")
    print(f"Recall: {best_metrics['recall']}")
    print(f"\nArtifacts saved to: models/")
    print(f"MLflow tracking: mlruns/")
    print("=" * 60)


if __name__ == "__main__":
    main()
