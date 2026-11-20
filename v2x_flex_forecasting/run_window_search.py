"""
History-window selection experiment for EV fleet-flexibility forecasting.

Purpose
-------
Select an appropriate continuous history length L from:

    L in {96, 192, 384, 672}

using validation performance only.

The script:
1. Loads residential and commercial aggregate-flexibility DataFrames.
2. Builds leakage-safe continuous-history samples for k in {1, 4, 48}.
3. Records the number of valid/rejected samples for every L.
4. Splits samples chronologically.
5. Fits scalers on training data only.
6. Trains the same small TCN configuration for every L.
7. Evaluates validation MAE/RMSE/R2 in ORIGINAL physical units.
8. Saves all results to CSV.

The test partition is created and preserved but is NOT evaluated here. It must
not influence history-window selection.

Expected project structure
--------------------------
v2x_forecasting/
├── data/
│   ├── residential/polytope_agg_res_df.pkl
│   └── commercial/polytope_agg_com_df.pkl
├── src/
│   ├── data_preparation.py
│   ├── preprocessing.py
│   ├── datasets.py
│   └── training.py
└── run_window_search.py
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.data_preparation import prepare_forecasting_samples
from src.preprocessing import (
    chronological_split,
    inverse_transform_targets,
    preprocess_splits,
)
from src.datasets import create_dataloaders
from src.training import (
    TrainingConfig,
    count_trainable_parameters,
    predict,
    train_model,
)


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

DATASETS = {
    "residential": Path("data/residential/polytope_agg_res_df.pkl"),
    "commercial": Path("data/commercial/polytope_agg_com_df.pkl"),
}

HISTORY_WINDOWS = (960, 1344) #(96, 192, 384, 672)
LEADS = (1, 4, 48)
HORIZON = 96
FREQ = "15min"

# Change these names here if the stored DataFrames use different labels.
TARGET_COLS = ("Pmax", "Cmax", "Cmin")

RESULTS_DIR = Path("results/window_search")
CHECKPOINT_DIR = Path("checkpoints/window_search")

# This is intentionally a modest fixed architecture/training budget.
# Only L changes during the history-window experiment.
BATCH_SIZE = 128
MAX_EPOCHS = 50
PATIENCE = 8
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
SEED = 42

# Kept consistent with the current preprocessing defaults. The commercial
# split should be revisited if its validation partition proves too small.
TRAIN_RATIO = 0.80
VAL_RATIO = 0.05
TEST_RATIO = 0.15


# ---------------------------------------------------------------------------
# Fixed window-selection TCN
# ---------------------------------------------------------------------------

class Chomp1d(nn.Module):
    """Remove right padding so a Conv1d is strictly causal."""

    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class ResidualCausalBlock(nn.Module):
    """
    Lightweight residual causal block with one dilated convolution.

    One convolution per dilation is deliberate: with kernel_size=3 and
    dilations 1,2,...,256, the receptive field is

        R = 1 + (3-1) * (1+2+4+...+256)
          = 1023 steps,

    which exceeds the largest candidate history L=672.

    Therefore the final temporal state can, in principle, depend on every
    input position for every candidate L. This avoids the receptive-field
    problem in the old TCN.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()

        padding = (kernel_size - 1) * dilation

        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.chomp = Chomp1d(padding)
        self.norm = nn.BatchNorm1d(channels)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        z = self.conv(x)
        z = self.chomp(z)
        z = self.norm(z)
        z = self.activation(z)
        z = self.dropout(z)

        return self.activation(z + residual)


class WindowSearchTCN(nn.Module):
    """
    Fixed TCN used ONLY for selecting history length.

    Input:
        (B, L, n_features)

    Output:
        (B, horizon, n_targets)

    The same model hyperparameters are used for every candidate L.
    """

    def __init__(
        self,
        n_features: int = 3,
        n_targets: int = 3,
        horizon: int = 96,
        channels: int = 32,
        kernel_size: int = 3,
        dropout: float = 0.10,
        dilations: Tuple[int, ...] = (
            1, 2, 4, 8, 16, 32, 64, 128, 256
        ),
        head_hidden: int = 128,
    ) -> None:
        super().__init__()

        self.horizon = horizon
        self.n_targets = n_targets

        self.input_projection = nn.Conv1d(
            n_features,
            channels,
            kernel_size=1,
        )

        self.blocks = nn.Sequential(
            *[
                ResidualCausalBlock(
                    channels=channels,
                    kernel_size=kernel_size,
                    dilation=d,
                    dropout=dropout,
                )
                for d in dilations
            ]
        )

        # The final causal state has a receptive field of 1023 steps and thus
        # can access the complete 96/192/384/672-step input window.
        self.head = nn.Sequential(
            nn.Linear(channels, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, horizon * n_targets),
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        # (B, L, F) -> (B, F, L)
        x = history.transpose(1, 2)

        x = self.input_projection(x)
        x = self.blocks(x)

        # Last causal state summarizes all accessible history.
        x = x[:, :, -1]

        y = self.head(x)
        return y.view(-1, self.horizon, self.n_targets)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Overall metrics across samples, horizon steps, and target variables."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Metric shapes differ: true={y_true.shape}, pred={y_pred.shape}"
        )

    error = y_pred - y_true

    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error ** 2))

    flat_true = y_true.reshape(-1)
    flat_pred = y_pred.reshape(-1)

    ss_res = np.sum((flat_true - flat_pred) ** 2)
    ss_tot = np.sum((flat_true - flat_true.mean()) ** 2)
    r2 = np.nan if ss_tot == 0 else 1.0 - ss_res / ss_tot

    return {
        "val_mae": float(mae),
        "val_rmse": float(rmse),
        "val_r2": float(r2),
    }


def per_target_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_cols: Iterable[str],
) -> Dict[str, float]:
    """Validation metrics for each target variable separately."""
    result: Dict[str, float] = {}

    for j, name in enumerate(target_cols):
        true_j = y_true[..., j]
        pred_j = y_pred[..., j]
        err = pred_j - true_j

        result[f"{name}_mae"] = float(np.mean(np.abs(err)))
        result[f"{name}_rmse"] = float(np.sqrt(np.mean(err ** 2)))

        true_flat = true_j.reshape(-1)
        pred_flat = pred_j.reshape(-1)
        ss_res = np.sum((true_flat - pred_flat) ** 2)
        ss_tot = np.sum((true_flat - true_flat.mean()) ** 2)

        result[f"{name}_r2"] = (
            float("nan")
            if ss_tot == 0
            else float(1.0 - ss_res / ss_tot)
        )

    return result


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

def load_dataframe(path: Path) -> pd.DataFrame:
    """Load and minimally validate one aggregate-flexibility DataFrame."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path.resolve()}")

    df = pd.read_pickle(path)

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{path} did not contain a pandas DataFrame.")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(
            f"{path}: DataFrame index must be a DatetimeIndex."
        )

    missing_cols = [c for c in TARGET_COLS if c not in df.columns]
    if missing_cols:
        raise KeyError(
            f"{path}: missing target columns {missing_cols}. "
            f"Available columns: {list(df.columns)}"
        )

    return df.sort_index()


def run_one_experiment(
    dataset_name: str,
    df: pd.DataFrame,
    k: int,
    history_steps: int,
    device: str | None,
) -> Dict[str, object]:
    """Prepare, train, and validate one (dataset, k, L) configuration."""

    print()
    print("=" * 78)
    print(
        f"Dataset={dataset_name} | k={k} | "
        f"L={history_steps} ({history_steps / 96:.1f} days)"
    )
    print("=" * 78)

    samples = prepare_forecasting_samples(
        df=df,
        k=k,
        history_steps=history_steps,
        horizon=HORIZON,
        freq=FREQ,
        target_cols=TARGET_COLS,
        feature_cols=TARGET_COLS,
        verbose=True,
    )

    diagnostics = samples["diagnostics"]

    # If gaps/history length leave too little data, fail this configuration
    # explicitly rather than silently training on an invalid split.
    if diagnostics["valid_samples"] < 20:
        raise ValueError(
            f"Only {diagnostics['valid_samples']} valid samples remain."
        )

    splits = chronological_split(
        samples=samples,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        verbose=True,
    )

    scaled_splits, scalers = preprocess_splits(splits)

    loaders = create_dataloaders(
        scaled_splits,
        batch_size=BATCH_SIZE,
        num_workers=0,
        seed=SEED,
    )

    model = WindowSearchTCN(
        n_features=len(TARGET_COLS),
        n_targets=len(TARGET_COLS),
        horizon=HORIZON,
    )

    n_params = count_trainable_parameters(model)

    checkpoint_path = (
        CHECKPOINT_DIR
        / dataset_name
        / f"k{k}_L{history_steps}.pt"
    )

    training_config = TrainingConfig(
        max_epochs=MAX_EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        patience=PATIENCE,
        min_delta=0.0,
        gradient_clip_norm=1.0,
        device=device,
        seed=SEED,
        verbose=True,
    )

    training_result = train_model(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        config=training_config,
        criterion=nn.MSELoss(),
        checkpoint_path=checkpoint_path,
    )

    # IMPORTANT: validation only. Do not call predict() on the test loader here.
    val_output = predict(
        model=model,
        loader=loaders["val"],
        device=device,
        return_targets=True,
    )

    val_pred = inverse_transform_targets(
        val_output["predictions"],
        scalers.target_scaler,
    )
    val_true = inverse_transform_targets(
        val_output["targets"],
        scalers.target_scaler,
    )

    metrics = regression_metrics(val_true, val_pred)
    target_metrics = per_target_metrics(
        val_true,
        val_pred,
        TARGET_COLS,
    )

    row: Dict[str, object] = {
        "dataset": dataset_name,
        "k": k,
        "history_steps": history_steps,
        "history_days": history_steps / 96.0,
        "candidate_origins": diagnostics["candidate_origins"],
        "valid_samples": diagnostics["valid_samples"],
        "rejected_samples": diagnostics["rejected_samples"],
        "rejected_missing_history":
            diagnostics["rejected_missing_history"],
        "rejected_missing_target":
            diagnostics["rejected_missing_target"],
        "rejected_missing_both":
            diagnostics["rejected_missing_both"],
        "rejected_nonfinite_history":
            diagnostics["rejected_nonfinite_history"],
        "rejected_nonfinite_target":
            diagnostics["rejected_nonfinite_target"],
        "rejected_nonfinite_both":
            diagnostics["rejected_nonfinite_both"],
        "train_samples": len(scaled_splits["train"]["Y"]),
        "val_samples": len(scaled_splits["val"]["Y"]),
        "test_samples_reserved": len(scaled_splits["test"]["Y"]),
        "n_parameters": n_params,
        "best_epoch": training_result.best_epoch,
        "best_val_loss_scaled": training_result.best_val_loss,
        "epochs_trained": training_result.epochs_trained,
        "training_seconds": training_result.training_seconds,
        **metrics,
        **target_metrics,
    }

    print()
    print(
        f"Validation | MAE={row['val_mae']:.4f} | "
        f"RMSE={row['val_rmse']:.4f} | "
        f"R2={row['val_r2']:.6f}"
    )

    return row


def save_results(rows: list[Dict[str, object]]) -> pd.DataFrame:
    """Save raw and ranked window-search results."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = pd.DataFrame(rows)
    raw_path = RESULTS_DIR / "window_search_results.csv"
    results.to_csv(raw_path, index=False)

    # Rank L within each dataset/k by validation MAE.
    ranked = results.copy()
    ranked["mae_rank"] = ranked.groupby(
        ["dataset", "k"]
    )["val_mae"].rank(method="min", ascending=True)

    ranked = ranked.sort_values(
        ["dataset", "k", "mae_rank", "history_steps"]
    )

    ranked_path = RESULTS_DIR / "window_search_ranked.csv"
    ranked.to_csv(ranked_path, index=False)

    # A compact summary is convenient for inspection.
    summary_cols = [
        "dataset",
        "k",
        "history_steps",
        "history_days",
        "valid_samples",
        "train_samples",
        "val_samples",
        "test_samples_reserved",
        "val_mae",
        "val_rmse",
        "val_r2",
        "mae_rank",
    ]
    summary = ranked[summary_cols]
    summary.to_csv(
        RESULTS_DIR / "window_search_summary.csv",
        index=False,
    )

    return ranked


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select continuous history length using validation only."
    )
    parser.add_argument(
        "--dataset",
        choices=("residential", "commercial", "all"),
        default="all",
        help="Dataset(s) to run.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help='PyTorch device, e.g. "cuda", "cuda:0", or "cpu".',
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="*",
        default=list(LEADS),
        help="Lead steps to evaluate. Default: 1 4 48.",
    )
    parser.add_argument(
        "--windows",
        type=int,
        nargs="*",
        default=list(HISTORY_WINDOWS),
        help="History lengths to evaluate. Default: 96 192 384 672.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    invalid_k = [k for k in args.k if k < 1]
    if invalid_k:
        raise ValueError(f"All k values must be >=1: {invalid_k}")

    invalid_windows = [L for L in args.windows if L < 1]
    if invalid_windows:
        raise ValueError(
            f"All history windows must be >=1: {invalid_windows}"
        )

    selected_datasets = (
        DATASETS.keys()
        if args.dataset == "all"
        else (args.dataset,)
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Save the experiment definition alongside the results.
    config_record = {
        "datasets": [str(x) for x in selected_datasets],
        "history_windows": list(args.windows),
        "leads": list(args.k),
        "horizon": HORIZON,
        "freq": FREQ,
        "target_cols": list(TARGET_COLS),
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
        "test_ratio": TEST_RATIO,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "seed": SEED,
        "device": args.device,
        "selection_metric": "validation MAE in original units",
        "test_evaluated": False,
    }

    with open(
        RESULTS_DIR / "window_search_config.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(config_record, f, indent=2)

    rows: list[Dict[str, object]] = []

    for dataset_name in selected_datasets:
        df = load_dataframe(DATASETS[dataset_name])

        print()
        print("#" * 78)
        print(f"{dataset_name.upper()} DATASET")
        print(f"Shape: {df.shape}")
        print(f"Range: {df.index.min()} -> {df.index.max()}")
        print("#" * 78)

        for k in args.k:
            for history_steps in args.windows:
                try:
                    row = run_one_experiment(
                        dataset_name=dataset_name,
                        df=df,
                        k=k,
                        history_steps=history_steps,
                        device=args.device,
                    )
                    rows.append(row)

                    # Save incrementally so completed runs survive interruption.
                    save_results(rows)

                except Exception as exc:
                    print(
                        f"\nFAILED | dataset={dataset_name} | "
                        f"k={k} | L={history_steps}\n"
                        f"{type(exc).__name__}: {exc}"
                    )

                    rows.append(
                        {
                            "dataset": dataset_name,
                            "k": k,
                            "history_steps": history_steps,
                            "history_days": history_steps / 96.0,
                            "status": "failed",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    pd.DataFrame(rows).to_csv(
                        RESULTS_DIR / "window_search_progress.csv",
                        index=False,
                    )

    successful = [
        row for row in rows
        if "val_mae" in row and np.isfinite(row["val_mae"])
    ]

    if not successful:
        raise RuntimeError("No window-search experiment completed successfully.")

    ranked = save_results(successful)

    print()
    print("=" * 78)
    print("WINDOW SEARCH COMPLETE")
    print("=" * 78)

    display_cols = [
        "dataset",
        "k",
        "history_steps",
        "valid_samples",
        "val_samples",
        "val_mae",
        "val_rmse",
        "val_r2",
        "mae_rank",
    ]

    print(ranked[display_cols].to_string(index=False))

    winners = (
        ranked.sort_values(["dataset", "k", "mae_rank"])
        .groupby(["dataset", "k"], as_index=False)
        .first()
    )

    print()
    print("Best validation window within each dataset/k:")
    print(
        winners[
            ["dataset", "k", "history_steps", "val_mae", "val_rmse", "val_r2"]
        ].to_string(index=False)
    )

    print()
    print(
        "Do not automatically adopt these winners yet. Inspect whether one "
        "common L is consistently competitive across datasets/k values and "
        "whether longer windows materially reduce commercial sample counts."
    )
    print(
        f"Results saved under: {RESULTS_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()
