from pathlib import Path

import pandas as pd

from config import K_VALUES, TARGET_COLS

from src.analysis import (
    load_model_results,
    create_metric_summary,
    create_horizon_summary,
    plot_actual_vs_predicted,
    plot_horizon_metric,
    plot_error_distribution,
    compute_error_statistics,
    plot_r2_comparison,
    plot_metric_comparison,
)


# ============================================================
# Configuration
# ============================================================

DATASETS = [
    "residential",
    "commercial",
]

MODELS = [
    "naive",
    "rf",
    "transformer",
    "cnn_lstm",
]


# ============================================================
# Analyse one dataset
# ============================================================

def analyse_dataset(dataset):

    print("\n")
    print("=" * 70)
    print(f"ANALYSIS: {dataset.upper()}")
    print("=" * 70)

    results_dir = (
        Path("results")
        / dataset
    )

    analysis_dir = (
        results_dir
        / "analysis"
    )

    analysis_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    # ========================================================
    # 1. Summary metrics
    # ========================================================

    metric_summary = create_metric_summary(
        results_dir=results_dir,
        models=MODELS,
        k_values=K_VALUES,
    )

    metric_path = (
        analysis_dir
        / "model_metrics_summary.csv"
    )

    metric_summary.to_csv(
        metric_path,
        index=False,
    )

    print("\nModel metric summary:")
    print(metric_summary.to_string(index=False))


    # ========================================================
    # 2. Horizon metrics
    # ========================================================

    horizon_summary = create_horizon_summary(
        results_dir=results_dir,
        models=MODELS,
        k_values=K_VALUES,
    )

    horizon_summary.to_csv(
        analysis_dir
        / "model_horizon_summary.csv",
        index=False,
    )


    # ========================================================
    # 3. Model comparison plots
    # ========================================================

    plot_r2_comparison(
        metric_summary=metric_summary,
        output_dir=analysis_dir,
    )

    plot_metric_comparison(
        metric_summary=metric_summary,
        metric="MAE",
        output_dir=analysis_dir,
    )

    plot_metric_comparison(
        metric_summary=metric_summary,
        metric="RMSE",
        output_dir=analysis_dir,
    )


    # ========================================================
    # 4. Horizon plots
    # ========================================================

    for k in K_VALUES:

        for target in TARGET_COLS:

            plot_horizon_metric(
                horizon_summary=horizon_summary,
                k=k,
                target=target,
                metric="MAE",
                output_dir=analysis_dir,
            )

            plot_horizon_metric(
                horizon_summary=horizon_summary,
                k=k,
                target=target,
                metric="RMSE",
                output_dir=analysis_dir,
            )


    # ========================================================
    # 5. Prediction diagnostics
    # ========================================================

    error_statistics = []

    for model in MODELS:

        for k in K_VALUES:

            prediction_file = (
                results_dir
                / f"{model}_k{k}_predictions.npz"
            )

            # Model may not have been trained yet.
            if not prediction_file.exists():

                print(
                    f"Skipping {model}, k={k}: "
                    "predictions not found."
                )

                continue


            print(
                f"\nAnalysing predictions: "
                f"{model}, k={k}"
            )

            results = load_model_results(
                results_dir=results_dir,
                model_name=model,
                k=k,
            )


            # -----------------------------------------------
            # Actual vs predicted
            # -----------------------------------------------

            plot_actual_vs_predicted(
                y_true=results["y_true"],
                y_pred=results["y_pred"],
                target_start=results["target_start"],
                model_name=model,
                k=k,
                output_dir=analysis_dir,
                target_names=TARGET_COLS,
                sample_index=0,
            )


            # -----------------------------------------------
            # Error distributions
            # -----------------------------------------------

            plot_error_distribution(
                y_true=results["y_true"],
                y_pred=results["y_pred"],
                model_name=model,
                k=k,
                output_dir=analysis_dir,
                target_names=TARGET_COLS,
            )


            # -----------------------------------------------
            # Error statistics
            # -----------------------------------------------

            stats = compute_error_statistics(
                y_true=results["y_true"],
                y_pred=results["y_pred"],
                model_name=model,
                k=k,
                target_names=TARGET_COLS,
            )

            error_statistics.append(
                stats
            )


    # ========================================================
    # 6. Save diagnostic statistics
    # ========================================================

    if error_statistics:

        error_statistics = pd.concat(
            error_statistics,
            ignore_index=True,
        )

        error_statistics.to_csv(
            analysis_dir
            / "error_statistics.csv",
            index=False,
        )

        print("\nError statistics:")
        print(
            error_statistics.to_string(
                index=False
            )
        )


    print(
        f"\nAnalysis saved to:\n"
        f"{analysis_dir}"
    )


# ============================================================
# Main
# ============================================================

def main():

    for dataset in DATASETS:

        analyse_dataset(
            dataset
        )


if __name__ == "__main__":
    main()