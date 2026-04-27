"""Unit tests for src/features/preprocess.py."""
import numpy as np
import pandas as pd
import pytest


class TestBuildPreprocessor:
    """Test suite for build_preprocessor function."""

    def test_build_preprocessor_returns_column_transformer(self):
        """GIVEN no arguments
        WHEN build_preprocessor is called
        THEN a ColumnTransformer is returned."""
        from sklearn.compose import ColumnTransformer
        from src.features.preprocess import build_preprocessor

        result = build_preprocessor()

        assert isinstance(result, ColumnTransformer)

    def test_preprocessor_has_numerical_and_categorical(self):
        """GIVEN a built preprocessor
        WHEN inspected
        THEN it has both 'numerical' and 'categorical' transformers."""
        from src.features.preprocess import build_preprocessor

        preprocessor = build_preprocessor()
        # Use transformers (pre-fit) — named_transformers_ is only available post-fit
        transformer_names = [name for name, _, _ in preprocessor.transformers]

        assert "numerical" in transformer_names
        assert "categorical" in transformer_names

    def test_preprocessor_fit_transform_shape(self):
        """GIVEN a simple known DataFrame
        WHEN preprocessor is fit and transforms
        THEN the output shape matches expected dimensions."""
        from src.features.preprocess import build_preprocessor, configure_preprocessor

        # Simple test data with known column types
        df = pd.DataFrame({
            "Age": [25.0, 30.0, 35.0],          # numerical
            "Gender": ["M", "F", "M"],          # categorical
            "Churn": ["No", "Yes", "No"],       # target (to be dropped)
        })

        X = df.drop(columns=["Churn"])
        numerical_cols = ["Age"]
        categorical_cols = ["Gender"]

        preprocessor = build_preprocessor()
        preprocessor = configure_preprocessor(preprocessor, numerical_cols, categorical_cols)

        X_transformed = preprocessor.fit_transform(X)

        # Numerical: 1 column → 1 scaled value
        # Categorical: 1 column → 2 one-hot columns (M, F)
        # Total expected: 3 columns
        assert X_transformed.shape[1] == 3

    def test_configure_preprocessor_sets_columns(self):
        """GIVEN a preprocessor and column lists
        WHEN configure_preprocessor is called
        THEN the preprocessor is updated with correct column assignments."""
        from src.features.preprocess import build_preprocessor, configure_preprocessor

        preprocessor = build_preprocessor()
        numerical_cols = ["tenure", "MonthlyCharges"]
        categorical_cols = ["PaymentMethod"]

        result = configure_preprocessor(preprocessor, numerical_cols, categorical_cols)

        # Verify the result is the same object (modified in place)
        assert result is preprocessor
        # Get the transformer column assignments
        transformers = preprocessor.transformers
        num_transformer_cols = transformers[0][2]
        cat_transformer_cols = transformers[1][2]

        assert num_transformer_cols == numerical_cols
        assert cat_transformer_cols == categorical_cols
