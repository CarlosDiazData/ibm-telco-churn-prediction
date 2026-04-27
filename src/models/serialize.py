"""Model and preprocessor serialization utilities."""
import joblib
from pathlib import Path


def serialize_artifacts(preprocessor, best_model, output_dir: str = "models") -> None:
    """Save preprocessor and best model to disk using joblib.

    Args:
        preprocessor: Fitted preprocessor to save.
        best_model: Fitted model or pipeline to save.
        output_dir: Directory where artifacts will be saved.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    joblib.dump(preprocessor, output_path / "preprocessor.joblib")
    joblib.dump(best_model, output_path / "best_model.joblib")


def load_artifacts(output_dir: str = "models") -> tuple:
    """Load preprocessor and model from disk.

    Args:
        output_dir: Directory where artifacts are stored.

    Returns:
        Tuple of (preprocessor, best_model).
    """
    preprocessor = joblib.load(Path(output_dir) / "preprocessor.joblib")
    best_model = joblib.load(Path(output_dir) / "best_model.joblib")
    return preprocessor, best_model
