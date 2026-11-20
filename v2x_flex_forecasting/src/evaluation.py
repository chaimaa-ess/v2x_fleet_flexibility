"""
Evaluation utilities for V2X flexibility forecasting.

Expected array shape:
    y_true: (n_samples, horizon, n_targets)
    y_pred: (n_samples, horizon, n_targets)

For this project:
    horizon   = 96
    n_targets = 3
    targets   = [Pmax, Cmax, Cmin]

IMPORTANT:
Metrics should be computed after inverse-transforming the
standardized model outputs back to their original physical units.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


TARGET_NAMES = ["Pmax", "Cmax", "Cmin"]


# ============================================================
# Validation
# ============================================================

def _validate_arrays(y_true, y_pred):
    """
    Validate target and prediction arrays.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: "
            f"y_true={y_true.shape}, "
            f"y_pred={y_pred.shape}"
        )

    if y_true.ndim != 3:
        raise ValueError(
            "Expected arrays with shape "
            "(n_samples, horizon, n_targets), "
            f"but received {y_true.shape}."
        )

    if not np.all(np.isfinite(y_true)):
        raise ValueError(
            "y_true contains NaN or infinite values."
        )

    if not np.all(np.isfinite(y_pred)):
        raise ValueError(
            "y_pred contains NaN or infinite values."
        )

    return y_true, y_pred


# ============================================================
# Main metrics
# ============================================================

def compute_metrics(
    y_true,
    y_pred,
    target_names=TARGET_NAMES,
):
    """
    Compute forecasting metrics for each flexibility variable
    and for all variables combined.

    Returns
    -------
    pandas.DataFrame
        Columns:
            target
            MAE
            RMSE
            R2
    """

    y_true, y_pred = _validate_arrays(
        y_true,
        y_pred,
    )

    n_targets = y_true.shape[-1]

    if len(target_names) != n_targets:
        raise ValueError(
            f"{len(target_names)} target names supplied "
            f"for {n_targets} targets."
        )

    results = []

    # --------------------------------------------------------
    # Individual flexibility variables
    # --------------------------------------------------------

    for j, target in enumerate(target_names):

        true_j = y_true[:, :, j].reshape(-1)
        pred_j = y_pred[:, :, j].reshape(-1)

        mae = mean_absolute_error(
            true_j,
            pred_j,
        )

        rmse = np.sqrt(
            mean_squared_error(
                true_j,
                pred_j,
            )
        )

        r2 = r2_score(
            true_j,
            pred_j,
        )

        results.append({
            "target": target,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
        })

    # --------------------------------------------------------
    # Overall metrics
    # --------------------------------------------------------

    true_all = y_true.reshape(-1)
    pred_all = y_pred.reshape(-1)

    results.append({
        "target": "Overall",
        "MAE": mean_absolute_error(
            true_all,
            pred_all,
        ),
        "RMSE": np.sqrt(
            mean_squared_error(
                true_all,
                pred_all,
            )
        ),
        "R2": r2_score(
            true_all,
            pred_all,
        ),
    })

    return pd.DataFrame(results)


# ============================================================
# Metrics across the 96-step forecasting horizon
# ============================================================

def compute_horizon_metrics(
    y_true,
    y_pred,
    target_names=TARGET_NAMES,
):
    """
    Compute MAE and RMSE at every forecast step.

    For 15-minute data:

        step 1  = +15 min relative to target start
        step 4  = +1 hour
        step 96 = +24 hours

    Note that the actual target start already incorporates
    the forecasting look-ahead k.

    Returns
    -------
    pandas.DataFrame

        forecast_step
        horizon_hours
        target
        MAE
        RMSE
    """

    y_true, y_pred = _validate_arrays(
        y_true,
        y_pred,
    )

    horizon = y_true.shape[1]
    n_targets = y_true.shape[2]

    if len(target_names) != n_targets:
        raise ValueError(
            f"{len(target_names)} target names supplied "
            f"for {n_targets} targets."
        )

    results = []

    for h in range(horizon):

        for j, target in enumerate(target_names):

            true_h = y_true[:, h, j]
            pred_h = y_pred[:, h, j]

            mae = mean_absolute_error(
                true_h,
                pred_h,
            )

            rmse = np.sqrt(
                mean_squared_error(
                    true_h,
                    pred_h,
                )
            )

            results.append({
                "forecast_step": h + 1,
                "horizon_hours": (h + 1) * 0.25,
                "target": target,
                "MAE": mae,
                "RMSE": rmse,
            })

    return pd.DataFrame(results)


# ============================================================
# Error arrays
# ============================================================

def compute_errors(y_true, y_pred):
    """
    Return signed and absolute forecasting errors.

    signed error:
        prediction - ground truth
    """

    y_true, y_pred = _validate_arrays(
        y_true,
        y_pred,
    )

    error = y_pred - y_true

    return {
        "error": error,
        "absolute_error": np.abs(error),
        "squared_error": error ** 2,
    }


# ============================================================
# Save results
# ============================================================

def save_evaluation_results(
    metrics,
    horizon_metrics,
    output_dir,
    model_name,
    k,
):
    """
    Save evaluation results to CSV files.

    Example
    -------
    results/commercial/
        transformer_k1_metrics.csv
        transformer_k1_horizon_metrics.csv
    """

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_path = (
        output_dir
        / f"{model_name}_k{k}_metrics.csv"
    )

    horizon_path = (
        output_dir
        / f"{model_name}_k{k}_horizon_metrics.csv"
    )

    metrics.to_csv(
        metrics_path,
        index=False,
    )

    horizon_metrics.to_csv(
        horizon_path,
        index=False,
    )

    print(f"Saved metrics to: {metrics_path}")
    print(
        f"Saved horizon metrics to: "
        f"{horizon_path}"
    )


# ============================================================
# Complete evaluation
# ============================================================

def evaluate_forecast(
    y_true,
    y_pred,
    model_name=None,
    k=None,
    output_dir=None,
    target_names=TARGET_NAMES,
):
    """
    Complete forecasting evaluation.

    Computes:
        1. Overall/per-target metrics
        2. Per-horizon metrics
        3. Error arrays

    Optionally saves metric tables.

    Returns
    -------
    dict
    """

    metrics = compute_metrics(
        y_true,
        y_pred,
        target_names=target_names,
    )

    horizon_metrics = compute_horizon_metrics(
        y_true,
        y_pred,
        target_names=target_names,
    )

    errors = compute_errors(
        y_true,
        y_pred,
    )

    if output_dir is not None:

        if model_name is None or k is None:
            raise ValueError(
                "model_name and k must be supplied "
                "when output_dir is specified."
            )

        save_evaluation_results(
            metrics=metrics,
            horizon_metrics=horizon_metrics,
            output_dir=output_dir,
            model_name=model_name,
            k=k,
        )

    return {
        "metrics": metrics,
        "horizon_metrics": horizon_metrics,
        "errors": errors,
    }