"""Evaluation metrics (manuscript Section III-H). Dysarthric = positive class."""

from __future__ import annotations

import numpy as np
from scipy.stats import beta
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact binomial confidence interval for k successes out of n."""
    lo = beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict:
    """Compute the metrics reported in Tables VII and VIII.

    ``y_score`` is the predicted probability of the dysarthric class.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    n = len(y_true)
    correct = int(tp + tn)
    ci_lo, ci_hi = clopper_pearson(correct, n)

    def safe_div(a, b):
        return float(a / b) if b else 0.0

    return {
        "n_test": n,
        "accuracy": correct / n,
        "accuracy_ci95": [ci_lo, ci_hi],
        "precision": safe_div(tp, tp + fp),
        "sensitivity": safe_div(tp, tp + fn),
        "specificity": safe_div(tn, tn + fp),
        "f1_dysarthric": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_score)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
