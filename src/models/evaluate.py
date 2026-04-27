"""Model evaluation utilities."""
from typing import Any

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate(model: Any, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate a fitted model on test data.

    Args:
        model: Fitted sklearn compatible model or pipeline.
        X_test: Test features.
        y_test: Test labels.

    Returns:
        Dictionary with:
        - roc_auc: ROC-AUC score
        - precision: Precision score
        - recall: Recall score
        - f1: F1 score
        - confusion_matrix: 2x2 confusion matrix as nested list
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_proba)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    return {
        "roc_auc": round(roc_auc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "confusion_matrix": cm.tolist(),
    }
