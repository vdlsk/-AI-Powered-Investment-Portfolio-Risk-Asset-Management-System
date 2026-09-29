"""
Model evaluation.

README §24: accuracy alone is insufficient for risk prediction because
risk events (large losses) are rare -- a model that always predicts "no
risk" can have high accuracy while catching zero real risk events. We
therefore always report ROC-AUC/PR-AUC (threshold-independent) alongside
recall specifically on the positive (risk) class, and false-negative rate,
since missing a real risk event is the costly failure mode for this
application.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_report(y_true: pd.Series, y_pred: pd.Series, y_proba: pd.Series) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else float("nan")

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),   # a.k.a. risk-event recall
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "false_negative_rate": false_negative_rate,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }

    # ROC-AUC / PR-AUC need both classes present in y_true to be defined.
    if y_true.nunique() > 1:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        metrics["pr_auc"] = average_precision_score(y_true, y_proba)
    else:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")

    return metrics


def calibration_table(y_true: pd.Series, y_proba: pd.Series, n_bins: int = 10) -> pd.DataFrame:
    """Bucket predictions into probability bins and compare predicted vs
    actual positive rate per bin -- a well-calibrated model should have
    predicted ~= actual within each bin.
    """
    df = pd.DataFrame({"y_true": y_true.values, "y_proba": y_proba.values})
    df["bin"] = pd.qcut(df["y_proba"], q=n_bins, duplicates="drop")
    grouped = df.groupby("bin", observed=True).agg(
        predicted_mean=("y_proba", "mean"),
        actual_rate=("y_true", "mean"),
        count=("y_true", "size"),
    )
    return grouped.reset_index()


def regression_report(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    nonzero = y_true != 0
    mape = (
        float((np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])).mean() * 100)
        if nonzero.any()
        else float("nan")
    )
    return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape}
