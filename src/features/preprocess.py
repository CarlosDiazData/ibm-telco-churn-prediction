"""Preprocessing pipeline for Telco churn data."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor() -> ColumnTransformer:
    """Build a ColumnTransformer preprocessor for Telco churn data.

    The preprocessor handles heterogeneous column types:
    - Numerical columns: median imputation + StandardScaler
    - Categorical columns: most frequent imputation + OneHotEncoder

    Returns:
        ColumnTransformer ready to be fit on training data.
    """
    # Numerical pipeline: median impute + scale
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Categorical pipeline: mode impute + one-hot encode
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    # Placeholder column lists — will be overridden when fit is called
    # The actual column assignment is done at fit time via set_config or by passing
    # column names to configure the transformer.
    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numerical_pipeline, []),  # Empty, to be determined at fit time
            ("categorical", categorical_pipeline, []),
        ],
        remainder="drop",
    )

    return preprocessor


def configure_preprocessor(preprocessor: ColumnTransformer, numerical_cols: list, categorical_cols: list) -> ColumnTransformer:
    """Configure the preprocessor with actual column names.

    Args:
        preprocessor: The ColumnTransformer built by build_preprocessor().
        numerical_cols: List of numerical column names.
        categorical_cols: List of categorical column names.

    Returns:
        The same preprocessor, reconfigured with column assignments.
    """
    # Access transformers directly (pre-fit) — named_transformers_ is only available post-fit
    # transformers is a list of (name, transformer, columns) tuples
    num_transformer = preprocessor.transformers[0][1]
    cat_transformer = preprocessor.transformers[1][1]
    preprocessor.set_params(
        transformers=[
            ("numerical", num_transformer, numerical_cols),
            ("categorical", cat_transformer, categorical_cols),
        ]
    )
    return preprocessor
