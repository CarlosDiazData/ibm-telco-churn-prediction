"""Unit tests for sample_weight support in train.py."""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestTrainAndLogSampleWeight:
    """Test suite for sample_weight parameter in train_and_log."""

    def test_train_and_log_passes_sample_weight_to_fit(self):
        """GIVEN train_and_log called with sample_weight
        WHEN pipeline.fit is called
        THEN sample_weight is passed to fit via classifier__sample_weight."""
        from src.models.train import train_and_log

        X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y_train = np.array([0, 1, 0, 1])
        sample_weight = np.array([1.0, 2.0, 0.5, 1.5])

        mock_preprocessor = MagicMock()

        # Mock SkPipeline to capture fit calls
        with patch("src.models.train.SkPipeline") as mock_pipeline_class:
            mock_pipeline_instance = MagicMock()
            mock_pipeline_class.return_value = mock_pipeline_instance

            with patch("src.models.train.mlflow") as mock_mlflow:
                mock_run = MagicMock()
                mock_run.info.run_id = "test-run-id"
                mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
                mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)
                mock_mlflow.sklearn.log_model = MagicMock()

                train_and_log(X_train, y_train, mock_preprocessor, sample_weight=sample_weight)

                # Verify fit was called with sample_weight routed to classifier
                call_args = mock_pipeline_instance.fit.call_args
                assert call_args is not None, "pipeline.fit was not called"
                args, kwargs = call_args

                assert "classifier__sample_weight" in kwargs, \
                    f"Expected 'classifier__sample_weight' in kwargs, got {kwargs.keys()}"
                np.testing.assert_array_equal(kwargs["classifier__sample_weight"], sample_weight)

    def test_train_and_log_without_sample_weight_calls_fit_without_it(self):
        """GIVEN train_and_log called without sample_weight
        WHEN pipeline.fit is called
        THEN no sample_weight parameter is passed."""
        from src.models.train import train_and_log

        X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y_train = np.array([0, 1, 0, 1])

        mock_preprocessor = MagicMock()

        with patch("src.models.train.SkPipeline") as mock_pipeline_class:
            mock_pipeline_instance = MagicMock()
            mock_pipeline_class.return_value = mock_pipeline_instance

            with patch("src.models.train.mlflow") as mock_mlflow:
                mock_run = MagicMock()
                mock_run.info.run_id = "test-run-id"
                mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
                mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)
                mock_mlflow.sklearn.log_model = MagicMock()

                train_and_log(X_train, y_train, mock_preprocessor)

                call_args = mock_pipeline_instance.fit.call_args
                assert call_args is not None, "pipeline.fit was not called"
                args, kwargs = call_args

                assert "sample_weight" not in kwargs, "sample_weight should not be passed"
                assert "classifier__sample_weight" not in kwargs, \
                    "classifier__sample_weight should not be passed"

    def test_sample_weight_normalized_produces_mean_of_one(self):
        """GIVEN CLTV values
        WHEN weights = cltv / cltv.mean()
        THEN the mean weight is approximately 1.0."""
        cltv = np.array([2003, 3000, 4000, 5000, 6500])
        weights = cltv / cltv.mean()

        assert weights.shape == cltv.shape
        assert abs(weights.mean() - 1.0) < 1e-10, f"Mean weight should be ~1.0, got {weights.mean()}"

    def test_sample_weight_different_values_produce_correct_mean(self):
        """GIVEN a specific set of CLTV values
        WHEN weights are computed as cltv / cltv.mean()
        THEN mean of weights equals 1.0 regardless of input values."""
        # Values from spec: range 2003-6500, mean ~4400
        cltv = np.array([2003, 3000, 4400, 5000, 6500])
        weights = cltv / cltv.mean()

        # Sum of weights should equal number of elements
        assert abs(weights.sum() - len(weights)) < 1e-10
        assert abs(weights.mean() - 1.0) < 1e-10
