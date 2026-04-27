"""Integration tests for the full main.py pipeline run."""
import os
import subprocess
import sys
from pathlib import Path


class TestPipelineIntegration:
    """Integration tests for the churn prediction pipeline."""

    def test_main_pipeline_runs_without_error(self):
        """GIVEN the project is set up with data and dependencies
        WHEN main.py is executed
        THEN it completes without raising an exception."""
        project_root = Path(__file__).parent.parent
        result = subprocess.run(
            [sys.executable, "main.py"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=120,
        )

        # Pipeline should exit cleanly
        assert result.returncode == 0, f"main.py failed with stderr: {result.stderr}"

    def test_models_directory_contains_artifacts(self):
        """GIVEN main.py has been executed successfully
        WHEN the run completes
        THEN models/ contains preprocessor.joblib and best_model.joblib."""
        project_root = Path(__file__).parent.parent
        models_dir = project_root / "models"

        # Run the pipeline if artifacts don't exist yet
        if not (models_dir / "preprocessor.joblib").exists() or not (models_dir / "best_model.joblib").exists():
            subprocess.run([sys.executable, "main.py"], cwd=project_root, capture_output=True, text=True, timeout=120)

        assert (models_dir / "preprocessor.joblib").exists(), "preprocessor.joblib not found"
        assert (models_dir / "best_model.joblib").exists(), "best_model.joblib not found"

    def test_mlruns_directory_created_after_training(self):
        """GIVEN main.py has been executed
        WHEN training runs
        THEN mlruns/ directory is created with run artifacts."""
        project_root = Path(__file__).parent.parent

        # Ensure pipeline runs first
        if not (project_root / "mlruns").exists():
            subprocess.run([sys.executable, "main.py"], cwd=project_root, capture_output=True, text=True, timeout=120)

        mlruns_dir = project_root / "mlruns"
        assert mlruns_dir.exists(), "mlruns directory was not created"
        # MLflow creates a .trash folder inside mlruns; check it has content
        assert len(list(mlruns_dir.glob("**/*"))) > 0, "mlruns directory is empty"
