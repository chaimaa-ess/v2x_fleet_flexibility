"""
Run origin-aware seasonal naive baseline.

The baseline respects the information available at each
forecast origin, preventing future-information leakage.

Outputs use the same structure as:
    Random Forest
    CNN-LSTM
    Transformer
"""

from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    K_VALUES,
    TARGET_COLS,
    HORIZON,
    RECENT_STEPS,
    DAILY_LAGS,
    WEEKLY_LAGS,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
)

from src.data_preparation import (
    prepare_forecasting_samples,
)

from src.preprocessing import (
    chronological_split,
    chronological_split_by_date,
)

from src.models.naive import (
    predict_origin_aware_naive,
)

from src.evaluation import (
    evaluate_forecast,
)


# ============================================================
# Configuration
# ============================================================

DATASETS = [
    "residential",
    "commercial",
]


DATA_FILES = {
    "residential":
        Path("data")
        / "residential"
        / "polytope_agg_res_df.pkl",

    "commercial":
        Path("data")
        / "commercial"
        / "polytope_agg_com_df.pkl",
}


COMMERCIAL_TRAIN_END = "2023-03-31 23:59:59"
COMMERCIAL_VAL_END = "2023-04-30 23:59:59"


# ============================================================
# Load data
# ============================================================

def load_data(filepath):

    print("\nLoading data:")
    print(filepath)

    df = pd.read_pickle(filepath)

    df.index = pd.to_datetime(
        df.index
    )

    df = df.sort_index()

    print(
        "Dataset shape:",
        df.shape,
    )

    print(
        "Start:",
        df.index.min(),
    )

    print(
        "End:",
        df.index.max(),
    )

    return df


# ============================================================
# Create identical split
# ============================================================

def create_split(
    samples,
    dataset,
):

    if dataset == "residential":

        print(
            "\nUsing ratio-based chronological split."
        )

        return chronological_split(
            samples=samples,
            train_ratio=TRAIN_RATIO,
            val_ratio=VAL_RATIO,
            test_ratio=TEST_RATIO,
        )

    elif dataset == "commercial":

        print(
            "\nUsing date-based chronological split."
        )

        return chronological_split_by_date(
            samples=samples,
            train_end=COMMERCIAL_TRAIN_END,
            val_end=COMMERCIAL_VAL_END,
        )

    else:

        raise ValueError(
            f"Unknown dataset: {dataset}"
        )


# ============================================================
# Run one experiment
# ============================================================

def run_naive(
    df,
    dataset,
    k,
):

    print("\n")
    print("=" * 70)

    print(
        f"ORIGIN-AWARE NAIVE | "
        f"DATASET={dataset} | k={k}"
    )

    print("=" * 70)


    # ========================================================
    # 1. Reconstruct forecasting samples
    # ========================================================

    samples = prepare_forecasting_samples(
        df=df,
        k=k,
        target_cols=TARGET_COLS,
        recent_steps=RECENT_STEPS,
        daily_lags=DAILY_LAGS,
        weekly_lags=WEEKLY_LAGS,
        horizon=HORIZON,
        freq="15min",
    )


    # ========================================================
    # 2. Same split as RF / neural models
    # ========================================================

    split = create_split(
        samples=samples,
        dataset=dataset,
    )

    test_data = split["test"]


    # ========================================================
    # 3. Get test information
    # ========================================================

    y_true = np.asarray(
        test_data["Y"],
        dtype=float,
    )

    origins = pd.to_datetime(
        test_data["origin"]
    )

    target_start = pd.to_datetime(
        test_data["target_start"]
    )


    print(
        "\nTest samples:",
        len(y_true),
    )

    print(
        "First origin:",
        origins[0],
    )

    print(
        "First target:",
        target_start[0],
    )


    # ========================================================
    # 4. Generate origin-aware predictions
    # ========================================================

    print(
        "\nGenerating origin-aware "
        "seasonal-naive predictions..."
    )


    y_pred, lag_days = (
        predict_origin_aware_naive(
            df=df,
            origins=origins,
            target_start=target_start,
            target_cols=TARGET_COLS,
            horizon=HORIZON,
            freq="15min",
            seasonal_period="1D",
            max_lookback_days=14,
        )
    )


    print(
        "Prediction shape:",
        y_pred.shape,
    )

    print(
        "Target shape    :",
        y_true.shape,
    )


    # ========================================================
    # 5. Check forecast validity
    # ========================================================

    valid_mask = np.all(
        np.isfinite(y_pred),
        axis=(1, 2),
    )


    n_valid = int(
        valid_mask.sum()
    )

    n_invalid = int(
        (~valid_mask).sum()
    )


    print(
        "\nValid complete forecasts  :",
        n_valid,
    )

    print(
        "Invalid complete forecasts:",
        n_invalid,
    )


    # ========================================================
    # 6. Lag diagnostics
    # ========================================================

    valid_lags = lag_days[
        np.isfinite(lag_days)
    ]


    if len(valid_lags) > 0:

        unique_lags, counts = np.unique(
            valid_lags.astype(int),
            return_counts=True,
        )

        print(
            "\nSeasonal lag usage:"
        )

        for lag, count in zip(
            unique_lags,
            counts,
        ):

            percentage = (
                100
                * count
                / len(valid_lags)
            )

            print(
                f"  {lag} day(s) back: "
                f"{count:,} predictions "
                f"({percentage:.2f}%)"
            )


    # ========================================================
    # 7. Remove incomplete forecasts if necessary
    # ========================================================

    if n_invalid > 0:

        print(
            "\nWARNING:"
            "\nSome complete 24-hour forecasts could not "
            "be generated."
        )

        print(
            "Those samples will be excluded from "
            "naive evaluation."
        )


        y_true = y_true[
            valid_mask
        ]

        y_pred = y_pred[
            valid_mask
        ]

        origins = origins[
            valid_mask
        ]

        target_start = target_start[
            valid_mask
        ]

        lag_days = lag_days[
            valid_mask
        ]


    # ========================================================
    # 8. Safety checks
    # ========================================================

    if len(y_true) == 0:

        raise ValueError(
            "No valid naive forecasts remain."
        )


    if y_true.shape != y_pred.shape:

        raise ValueError(
            f"Shape mismatch: "
            f"y_true={y_true.shape}, "
            f"y_pred={y_pred.shape}"
        )


    if not np.all(
        np.isfinite(y_true)
    ):

        raise ValueError(
            "y_true contains NaN or Inf."
        )


    if not np.all(
        np.isfinite(y_pred)
    ):

        raise ValueError(
            "y_pred contains NaN or Inf."
        )


    # ========================================================
    # 9. Results directory
    # ========================================================

    results_dir = (
        Path("results")
        / dataset
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    # ========================================================
    # 10. Evaluate
    #
    # No scaling / inverse scaling is required.
    # These values are already in physical units.
    # ========================================================

    print(
        "\nEvaluating origin-aware naive baseline..."
    )


    evaluation = evaluate_forecast(
        y_true=y_true,
        y_pred=y_pred,
        model_name="naive",
        k=k,
        output_dir=results_dir,
        target_names=TARGET_COLS,
    )


    # ========================================================
    # 11. Save predictions
    #
    # Same format as RF / CNN-LSTM / Transformer.
    # ========================================================

    prediction_path = (
        results_dir
        / f"naive_k{k}_predictions.npz"
    )


    np.savez_compressed(
        prediction_path,

        y_true=y_true,
        y_pred=y_pred,

        origin=np.asarray(
            origins
        ),

        target_start=np.asarray(
            target_start
        ),

        lag_days=lag_days,
    )


    print(
        "\nSaved predictions to:"
    )

    print(
        prediction_path
    )


    # ========================================================
    # 12. Print metrics
    # ========================================================

    print(
        "\nOrigin-aware naive metrics"
    )

    print("-" * 70)

    print(
        evaluation["metrics"].to_string(
            index=False
        )
    )


    return evaluation


# ============================================================
# Main
# ============================================================

def main():

    print("\n")
    print("=" * 70)

    print(
        "ORIGIN-AWARE SEASONAL NAIVE BASELINE"
    )

    print("=" * 70)


    for dataset in DATASETS:

        print("\n")
        print("#" * 70)

        print(
            f"DATASET: {dataset.upper()}"
        )

        print("#" * 70)


        df = load_data(
            DATA_FILES[dataset]
        )


        for k in K_VALUES:

            run_naive(
                df=df,
                dataset=dataset,
                k=k,
            )


    print("\n")
    print("=" * 70)

    print(
        "NAIVE BASELINE COMPLETE"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()