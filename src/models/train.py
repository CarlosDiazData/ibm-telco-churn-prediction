"""Model training with MLflow experiment tracking."""
import warnings

import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline as SkPipeline


def train_and_log(
    X_train,
    y_train,
    preprocessor,
    experiment_name: str = "churn-prediction",
) -> dict:
    """Train Logistic Regression and Random Forest, log each run to MLflow.

    Args:
        X_train: Training features DataFrame or array.
        y_train: Training target array or Series.
        preprocessor: Fitted ColumnTransformer preprocessor.
        experiment_name: MLflow experiment name.

    Returns:
        Dictionary mapping model name to MLflow run ID.
    """
    warnings.filterwarnings("ignore")

    # Set tracking URI — local mlruns directory (must be before set_experiment)
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment(experiment_name)
    mlflow.autolog()

    run_ids = {}

    models_config = [
        ("LogisticRegression", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ("RandomForest", RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)),
    ]

    for model_name, model in models_config:
        with mlflow.start_run(run_name=model_name) as run:
            run_id = run.info.run_id
            run_ids[model_name] = run_id

            # Build full pipeline: preprocessor + classifier
            pipeline = SkPipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("classifier", model),
                ]
            )

            # Fit pipeline
            pipeline.fit(X_train, y_train)

            # Log explicitly for clarity (autolog handles most of this)
            mlflow.sklearn.log_model(pipeline, f"model_{model_name}")

    return run_ids
