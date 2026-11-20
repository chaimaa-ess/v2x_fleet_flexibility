"""
Model-agnostic analysis utilities for V2X flexibility forecasting.

Works with:
    - Random Forest
    - Transformer
    - CNN-LSTM

Expected prediction shape:
    (n_samples, horizon, n_targets)

For this project:
    horizon = 96
    targets = [Pmax, Cmax, Cmin]
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


TARGET_NAMES = ["Pmax", "Cmax", "Cmin"]


# ============================================================
# Load saved results
# ============================================================

def load_model_results(
    results_dir,
    model_name,
    k,
):
    """
    Load metrics, horizon metrics and predictions
    for one trained model.
    """

    results_dir = Path(results_dir)

    metrics_path = (
        results_dir
        / f"{model_name}_k{k}_metrics.csv"
    )

    horizon_path = (
        results_dir
        / f"{model_name}_k{k}_horizon_metrics.csv"
    )

    prediction_path = (
        results_dir
        / f"{model_name}_k{k}_predictions.npz"
    )

    metrics = pd.read_csv(metrics_path)

    horizon_metrics = pd.read_csv(
        horizon_path
    )

    predictions = np.load(
        prediction_path,
        allow_pickle=True,
    )

    return {
        "metrics": metrics,
        "horizon_metrics": horizon_metrics,
        "y_true": predictions["y_true"],
        "y_pred": predictions["y_pred"],
        "target_start": predictions["target_start"],
    }


# ============================================================
# Metric summary table
# ============================================================

def create_metric_summary(
    results_dir,
    models,
    k_values,
):
    """
    Combine model metrics into one table.
    """

    rows = []

    for model in models:

        for k in k_values:

            path = (
                Path(results_dir)
                / f"{model}_k{k}_metrics.csv"
            )

            if not path.exists():
                continue

            df = pd.read_csv(path)

            df.insert(0, "k", k)
            df.insert(0, "model", model)

            rows.append(df)

    if not rows:
        raise ValueError(
            "No metric files were found."
        )

    return pd.concat(
        rows,
        ignore_index=True,
    )


# ============================================================
# Horizon summary
# ============================================================

def create_horizon_summary(
    results_dir,
    models,
    k_values,
):
    """
    Combine horizon-level results from all models.
    """

    rows = []

    for model in models:

        for k in k_values:

            path = (
                Path(results_dir)
                / f"{model}_k{k}_horizon_metrics.csv"
            )

            if not path.exists():
                continue

            df = pd.read_csv(path)

            df.insert(0, "k", k)
            df.insert(0, "model", model)

            rows.append(df)

    if not rows:
        raise ValueError(
            "No horizon metric files were found."
        )

    return pd.concat(
        rows,
        ignore_index=True,
    )


# ============================================================
# Actual vs predicted plot
# ============================================================

def plot_actual_vs_predicted(
    y_true,
    y_pred,
    target_start,
    model_name,
    k,
    output_dir,
    target_names=TARGET_NAMES,
    sample_index=0,
):
    """
    Plot one complete 24-hour forecast window.

    One figure is produced for each target.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    horizon = y_true.shape[1]

    forecast_hours = (
        np.arange(horizon) * 0.25
    )

    for j, target in enumerate(target_names):

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        ax.plot(
            forecast_hours,
            y_true[sample_index, :, j],
            label="Actual",
            linewidth=2,
        )

        ax.plot(
            forecast_hours,
            y_pred[sample_index, :, j],
            label="Predicted",
            linewidth=2,
        )

        ax.set_xlabel(
            "Forecast horizon (hours)"
        )

        ax.set_ylabel(target)

        ax.set_title(
            f"{model_name.upper()} | "
            f"{target} | k={k}\n"
            f"Forecast start: "
            f"{target_start[sample_index]}"
        )

        ax.legend()
        ax.grid(alpha=0.3)

        fig.tight_layout()

        path = (
            output_dir
            / f"{model_name}_k{k}_"
              f"{target}_forecast.png"
        )

        fig.savefig(
            path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)


# ============================================================
# Horizon error plot
# ============================================================

def plot_horizon_metric(
    horizon_summary,
    k,
    target,
    metric,
    output_dir,
):
    """
    Compare models across the 24-hour forecast horizon.

    metric:
        "MAE" or "RMSE"
    """

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    subset = horizon_summary[
        (horizon_summary["k"] == k)
        & (horizon_summary["target"] == target)
    ]

    if subset.empty:
        return

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    for model in subset["model"].unique():

        model_data = subset[
            subset["model"] == model
        ]

        ax.plot(
            model_data["horizon_hours"],
            model_data[metric],
            label=model,
            linewidth=2,
        )

    ax.set_xlabel(
        "Forecast horizon (hours)"
    )

    ax.set_ylabel(metric)

    ax.set_title(
        f"{target} {metric} across forecast horizon | "
        f"k={k}"
    )

    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()

    path = (
        output_dir
        / f"k{k}_{target}_{metric}_horizon.png"
    )

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# Error distribution
# ============================================================

def plot_error_distribution(
    y_true,
    y_pred,
    model_name,
    k,
    output_dir,
    target_names=TARGET_NAMES,
):
    """
    Plot signed prediction-error distribution.

    Error = prediction - actual
    """

    output_dir = Path(output_dir)

    for j, target in enumerate(target_names):

        error = (
            y_pred[:, :, j]
            - y_true[:, :, j]
        ).reshape(-1)

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        ax.hist(
            error,
            bins=80,
            alpha=0.8,
        )

        ax.axvline(
            0,
            linestyle="--",
            linewidth=2,
        )

        ax.set_xlabel(
            "Prediction error"
        )

        ax.set_ylabel(
            "Frequency"
        )

        ax.set_title(
            f"{model_name.upper()} | "
            f"{target} error distribution | "
            f"k={k}"
        )

        ax.grid(
            alpha=0.3
        )

        fig.tight_layout()

        path = (
            output_dir
            / f"{model_name}_k{k}_"
              f"{target}_error_distribution.png"
        )

        fig.savefig(
            path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)


# ============================================================
# Error statistics
# ============================================================

def compute_error_statistics(
    y_true,
    y_pred,
    model_name,
    k,
    target_names=TARGET_NAMES,
):
    """
    Additional diagnostic statistics.

    Bias:
        mean(prediction - actual)

    Error SD:
        standard deviation of prediction errors
    """

    rows = []

    for j, target in enumerate(target_names):

        actual = (
            y_true[:, :, j]
            .reshape(-1)
        )

        predicted = (
            y_pred[:, :, j]
            .reshape(-1)
        )

        error = predicted - actual

        rows.append({
            "model": model_name,
            "k": k,
            "target": target,
            "bias": np.mean(error),
            "error_std": np.std(error),
            "mean_actual": np.mean(actual),
            "mean_predicted": np.mean(predicted),
            "actual_std": np.std(actual),
            "predicted_std": np.std(predicted),
        })

    return pd.DataFrame(rows)


# ============================================================
# R2 comparison plot
# ============================================================

def plot_r2_comparison(
    metric_summary,
    output_dir,
):
    """
    Generate one R2 comparison plot for each k.
    """

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    k_values = sorted(
        metric_summary["k"].unique()
    )

    for k in k_values:

        subset = metric_summary[
            metric_summary["k"] == k
        ]

        pivot = subset.pivot(
            index="target",
            columns="model",
            values="R2",
        )

        fig, ax = plt.subplots(
            figsize=(9, 5)
        )

        pivot.plot(
            kind="bar",
            ax=ax,
        )

        ax.set_ylabel("R²")
        ax.set_xlabel("Target")

        ax.set_title(
            f"Model comparison | k={k}"
        )

        ax.axhline(
            0,
            linewidth=1,
        )

        ax.legend(
            title="Model"
        )

        ax.grid(
            axis="y",
            alpha=0.3,
        )

        plt.xticks(
            rotation=0
        )

        fig.tight_layout()

        path = (
            output_dir
            / f"k{k}_r2_comparison.png"
        )

        fig.savefig(
            path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)


# ============================================================
# MAE / RMSE comparison
# ============================================================

def plot_metric_comparison(
    metric_summary,
    metric,
    output_dir,
):
    """
    Compare MAE or RMSE across models.
    """

    output_dir = Path(output_dir)

    for k in sorted(
        metric_summary["k"].unique()
    ):

        subset = metric_summary[
            metric_summary["k"] == k
        ]

        pivot = subset.pivot(
            index="target",
            columns="model",
            values=metric,
        )

        fig, ax = plt.subplots(
            figsize=(9, 5)
        )

        pivot.plot(
            kind="bar",
            ax=ax,
        )

        ax.set_ylabel(metric)
        ax.set_xlabel("Target")

        ax.set_title(
            f"{metric} comparison | k={k}"
        )

        ax.legend(
            title="Model"
        )

        ax.grid(
            axis="y",
            alpha=0.3,
        )

        plt.xticks(
            rotation=0
        )

        fig.tight_layout()

        path = (
            output_dir
            / f"k{k}_{metric.lower()}_comparison.png"
        )

        fig.savefig(
            path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)