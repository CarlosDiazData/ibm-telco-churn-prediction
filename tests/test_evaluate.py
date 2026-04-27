"""Unit tests for src/models/evaluate.py."""
from unittest.mock import MagicMock

import numpy as np
import pytest


class TestEvaluate:
    """Test suite for evaluate function."""

    def test_evaluate_returns_all_metric_keys(self):
        """GIVEN a mock model and data
        WHEN evaluate is called
        THEN all expected keys are present in the result."""
        from src.models.evaluate import evaluate

        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1]])
        mock_model.predict.return_value = np.array([0, 1, 0])

        X_test = np.array([[1, 2], [3, 4], [5, 6]])
        y_test = np.array([0, 1, 0])

        result = evaluate(mock_model, X_test, y_test)

        expected_keys = ["roc_auc", "precision", "recall", "f1", "confusion_matrix"]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_evaluate_roc_auc_is_valid_float(self):
        """GIVEN a mock model
        WHEN evaluate is called
        THEN roc_auc is a float between 0 and 1."""
        from src.models.evaluate import evaluate

        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.8, 0.2], [0.3, 0.7]])
        mock_model.predict.return_value = np.array([0, 1])

        result = evaluate(mock_model, np.array([[1, 2], [3, 4]]), np.array([0, 1]))

        assert isinstance(result["roc_auc"], float)
        assert 0.0 <= result["roc_auc"] <= 1.0

    def test_evaluate_confusion_matrix_is_2x2_list(self):
        """GIVEN a mock model
        WHEN evaluate is called
        THEN confusion_matrix is a 2x2 nested list."""
        from src.models.evaluate import evaluate

        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.8, 0.2], [0.3, 0.7]])
        mock_model.predict.return_value = np.array([0, 1])

        result = evaluate(mock_model, np.array([[1, 2], [3, 4]]), np.array([0, 1]))

        cm = result["confusion_matrix"]
        assert isinstance(cm, list)
        assert len(cm) == 2
        assert all(len(row) == 2 for row in cm)

    def test_evaluate_precision_recall_f1_are_floats(self):
        """GIVEN a mock model
        WHEN evaluate is called
        THEN precision, recall, and f1 are floats."""
        from src.models.evaluate import evaluate

        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.8, 0.2], [0.3, 0.7]])
        mock_model.predict.return_value = np.array([0, 1])

        result = evaluate(mock_model, np.array([[1, 2], [3, 4]]), np.array([0, 1]))

        assert isinstance(result["precision"], float)
        assert isinstance(result["recall"], float)
        assert isinstance(result["f1"], float)
