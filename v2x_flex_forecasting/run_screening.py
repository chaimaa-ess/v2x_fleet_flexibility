"""
Equal-budget model screening for EV fleet-flexibility forecasting.

Screening scenario (fixed before model comparison)
--------------------------------------------------
Dataset : residential
Lead    : k = 48 (12-hour lead at 15-min resolution)
History : L = 672 (7 days)
Horizon : H = 96 (24 hours)
Input   : 3 flexibility variables + 4 HISTORICAL calendar covariates
          [Pmax, Cmax, Cmin, tod_sin, tod_cos, dow_sin, dow_cos]

Candidates
----------
    Random Forest
    CNN-LSTM
    TCN
    PatchTST

Each model family receives the same number of randomly sampled hyperparameter
configurations. All candidates use exactly the same train/validation/test
samples. Selection uses validation MAE in original physical units. The test
partition is created and reserved but NEVER evaluated in this script.

Default budget: 20 configurations per model (80 learned-model runs total).

Notes
-----
* Calendar covariates are generated dynamically from each sample's origin.
* No future-calendar covariates are used.
* The TCN uses the same one-convolution residual causal-block family used in
  the history-window/calendar-selection experiments.
* Random Forest receives the same 672 x 7 information, flattened to 4704 features.
* Random search is deterministic under --seed.
"""

from __future__ import annotations

import argparse
import copy
import gc
import itertools
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data_preparation import prepare_forecasting_samples
from src.preprocessing import (
    chronological_split,
    inverse_transform_targets,
    preprocess_splits,
)
from src.datasets import create_dataloaders

from sklearn.ensemble import RandomForestRegressor


# =============================================================================
# Fixed experimental definition
# =============================================================================

DATA_PATH = Path("data/residential/polytope_agg_res_df.pkl")
DATASET_NAME = "residential"
TARGET_COLS = ("Pmax", "Cmax", "Cmin")

K = 48
HISTORY_STEPS = 672
HORIZON = 96
FREQ = "15min"
CALENDAR_DIM = 4

TRAIN_RATIO = 0.80
VAL_RATIO = 0.05
TEST_RATIO = 0.15

DEFAULT_BUDGET = 20
DEFAULT_SEED = 42
MAX_EPOCHS = 50
PATIENCE = 8
GRADIENT_CLIP_NORM = 1.0
BATCH_SIZE = 128

RESULTS_DIR = Path("results/model_screening")
CHECKPOINT_DIR = Path("checkpoints/model_screening")

MODEL_NAMES = ("random_forest",)  #  #, "cnn_lstm", "tcn", "patchtst")


# =============================================================================
# Reproducibility / device
# =============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return dev


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# Metrics
# =============================================================================

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: true={y_true.shape}, pred={y_pred.shape}")

    err = y_pred - y_true
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err ** 2))
    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - yt.mean()) ** 2)
    r2 = np.nan if ss_tot == 0 else 1.0 - ss_res / ss_tot
    return {"val_mae": float(mae), "val_rmse": float(rmse), "val_r2": float(r2)}


def per_target_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    names: Iterable[str],
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for j, name in enumerate(names):
        yt = y_true[..., j]
        yp = y_pred[..., j]
        err = yp - yt
        out[f"{name}_mae"] = float(np.mean(np.abs(err)))
        out[f"{name}_rmse"] = float(np.sqrt(np.mean(err ** 2)))
        a = yt.reshape(-1)
        b = yp.reshape(-1)
        ss_res = np.sum((a - b) ** 2)
        ss_tot = np.sum((a - a.mean()) ** 2)
        out[f"{name}_r2"] = np.nan if ss_tot == 0 else float(1 - ss_res / ss_tot)
    return out


# =============================================================================
# Calendar helpers
# =============================================================================

def build_calendar_lookup() -> np.ndarray:
    slots = np.arange(672)
    dow = slots // 96
    quarter = slots % 96
    tod_angle = 2.0 * np.pi * quarter / 96.0
    dow_angle = 2.0 * np.pi * dow / 7.0
    return np.column_stack([
        np.sin(tod_angle), np.cos(tod_angle),
        np.sin(dow_angle), np.cos(dow_angle),
    ]).astype(np.float32)


def timestamps_to_week_slots(timestamps: Sequence[Any]) -> np.ndarray:
    ts = pd.DatetimeIndex(pd.to_datetime(timestamps))
    quarter = ts.hour.to_numpy() * 4 + ts.minute.to_numpy() // 15
    return (ts.dayofweek.to_numpy() * 96 + quarter).astype(np.int16)


def history_calendar_from_origins(origins: Sequence[Any], history_steps: int) -> np.ndarray:
    """Return (N,L,4) calendar tensor. Use only when materialization is needed."""
    lookup = build_calendar_lookup()
    origin_slots = timestamps_to_week_slots(origins).astype(np.int32)
    offsets = np.arange(-(history_steps - 1), 1, dtype=np.int32)
    slots = (origin_slots[:, None] + offsets[None, :]) % 672
    return lookup[slots]


def make_tree_matrix(split: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Flatten the same 672x7 information used by neural models for tree ensembles."""
    history = np.asarray(split["X_history"], dtype=np.float32)
    calendar = history_calendar_from_origins(split["origin"], history.shape[1])
    x = np.concatenate([history, calendar], axis=-1)
    x = np.ascontiguousarray(x.reshape(x.shape[0], -1), dtype=np.float32)
    y = np.asarray(split["Y"], dtype=np.float32)
    y = np.ascontiguousarray(y.reshape(y.shape[0], -1), dtype=np.float32)
    del calendar
    return x, y


# =============================================================================
# Neural models
# =============================================================================

class Chomp1d(nn.Module):
    def __init__(self, n: int) -> None:
        super().__init__()
        self.n = n

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x if self.n == 0 else x[:, :, :-self.n].contiguous()


class ResidualCausalBlock(nn.Module):
    """One-convolution residual causal block used in the selection TCN."""
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size,
                              padding=pad, dilation=dilation)
        self.chomp = Chomp1d(pad)
        self.norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.drop(self.act(self.norm(self.chomp(self.conv(x)))))
        return self.act(z + x)


class ScreeningTCN(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dropout: float,
                 head_hidden: int, horizon: int = HORIZON, n_targets: int = 3):
        super().__init__()
        self.horizon = horizon
        self.n_targets = n_targets
        self.input_projection = nn.Conv1d(7, channels, 1)
        dilations = (1, 2, 4, 8, 16, 32, 64, 128, 256)
        self.blocks = nn.Sequential(*[
            ResidualCausalBlock(channels, kernel_size, d, dropout)
            for d in dilations
        ])
        self.head = nn.Sequential(
            nn.Linear(channels, head_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(head_hidden, horizon * n_targets),
        )

    def forward(self, history: torch.Tensor, history_time: torch.Tensor) -> torch.Tensor:
        x = torch.cat([history, history_time], dim=-1).transpose(1, 2)
        x = self.blocks(self.input_projection(x))[:, :, -1]
        return self.head(x).view(-1, self.horizon, self.n_targets)


class ScreeningCNNLSTM(nn.Module):
    def __init__(self, conv_channels: int, kernel_size: int, lstm_hidden: int,
                 lstm_layers: int, dropout: float, head_hidden: int,
                 horizon: int = HORIZON, n_targets: int = 3):
        super().__init__()
        self.horizon = horizon
        self.n_targets = n_targets
        pad = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv1d(7, conv_channels, kernel_size, padding=pad),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, head_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(head_hidden, horizon * n_targets),
        )

    def forward(self, history: torch.Tensor, history_time: torch.Tensor) -> torch.Tensor:
        x = torch.cat([history, history_time], dim=-1)
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        return self.head(x).view(-1, self.horizon, self.n_targets)


class ScreeningPatchTST(nn.Module):
    """Compact patch-based Transformer encoder for equal-budget screening."""
    def __init__(self, patch_len: int, stride: int, d_model: int, n_heads: int,
                 n_layers: int, dim_feedforward: int, dropout: float,
                 horizon: int = HORIZON, n_targets: int = 3):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if patch_len > HISTORY_STEPS:
            raise ValueError("patch_len cannot exceed history length")

        self.patch_len = patch_len
        self.stride = stride
        self.horizon = horizon
        self.n_targets = n_targets
        self.n_patches = 1 + (HISTORY_STEPS - patch_len) // stride

        self.patch_projection = nn.Linear(patch_len * 7, d_model)
        self.position = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        nn.init.normal_(self.position, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, horizon * n_targets)

    def forward(self, history: torch.Tensor, history_time: torch.Tensor) -> torch.Tensor:
        x = torch.cat([history, history_time], dim=-1)  # (B,L,7)
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        # unfold -> (B,n_patches,7,patch_len)
        patches = patches.contiguous().reshape(x.shape[0], self.n_patches, -1)
        z = self.patch_projection(patches) + self.position
        z = self.norm(self.encoder(z)).mean(dim=1)
        return self.head(z).view(-1, self.horizon, self.n_targets)


# =============================================================================
# Shared neural training
# =============================================================================

@dataclass
class NeuralTrainResult:
    best_epoch: int
    best_val_loss: float
    epochs_trained: int
    training_seconds: float


def move_history_batch(batch: Mapping[str, torch.Tensor], device: torch.device):
    history = batch["history"].to(device=device, dtype=torch.float32, non_blocking=True)
    target = batch["target"].to(device=device, dtype=torch.float32, non_blocking=True)
    if "history_time" not in batch:
        raise KeyError(
            "Screening requires history_time. Create loaders with calendar_mode='history'."
        )
    history_time = batch["history_time"].to(
        device=device, dtype=torch.float32, non_blocking=True
    )
    return history, history_time, target


def train_neural_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
    seed: int,
    checkpoint_path: Path,
) -> NeuralTrainResult:
    set_seed(seed)
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    best_val = float("inf")
    best_epoch = 0
    best_state = None
    no_improve = 0
    start = time.perf_counter()

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_sum = 0.0
        train_n = 0

        for batch in train_loader:
            history, history_time, target = move_history_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(history, history_time)
            loss = criterion(pred, target)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss: {loss.item()}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            optimizer.step()
            train_sum += loss.item() * history.shape[0]
            train_n += history.shape[0]

        model.eval()
        val_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for batch in val_loader:
                history, history_time, target = move_history_batch(batch, device)
                pred = model(history, history_time)
                loss = criterion(pred, target)
                val_sum += loss.item() * history.shape[0]
                val_n += history.shape[0]

        train_loss = train_sum / train_n
        val_loss = val_sum / val_n
        improved = val_loss < best_val
        mark = " *" if improved else ""
        print(
            f"Epoch {epoch:03d}/{MAX_EPOCHS:03d} | train={train_loss:.6f} | "
            f"val={val_loss:.6f}{mark}"
        )

        if improved:
            best_val = float(val_loss)
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, checkpoint_path)
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"Early stopping; best epoch={best_epoch}, val={best_val:.6f}")
                break

    if best_state is None:
        raise RuntimeError("Training produced no valid model state.")
    model.load_state_dict(best_state)
    return NeuralTrainResult(best_epoch, best_val, epoch, time.perf_counter() - start)


@torch.no_grad()
def predict_neural(model: nn.Module, loader: DataLoader, device: torch.device):
    model.to(device).eval()
    preds, targets = [], []
    for batch in loader:
        history, history_time, target = move_history_batch(batch, device)
        pred = model(history, history_time)
        preds.append(pred.cpu().numpy())
        targets.append(target.cpu().numpy())
    return np.concatenate(preds), np.concatenate(targets)


# =============================================================================
# Hyperparameter spaces / deterministic random search
# =============================================================================

SEARCH_SPACES: Dict[str, Dict[str, Sequence[Any]]] = {
    "tcn": {
        "channels": (32, 64),
        "kernel_size": (3, 5),
        "dropout": (0.0, 0.1, 0.2),
        "head_hidden": (128, 256),
        "learning_rate": (3e-4, 1e-3),
        "weight_decay": (0.0, 1e-4),
    },
    "cnn_lstm": {
        "conv_channels": (32, 64),
        "kernel_size": (3, 5, 7),
        "lstm_hidden": (64, 128),
        "lstm_layers": (1, 2),
        "dropout": (0.0, 0.1, 0.2),
        "head_hidden": (128, 256),
        "learning_rate": (3e-4, 1e-3),
        "weight_decay": (0.0, 1e-4),
    },
    "patchtst": {
        "patch_len": (16, 24, 32),
        "stride": (8, 16),
        "d_model": (64, 128),
        "n_heads": (4, 8),
        "n_layers": (2, 3),
        "dim_feedforward": (128, 256),
        "dropout": (0.0, 0.1, 0.2),
        "learning_rate": (3e-4, 1e-3),
        "weight_decay": (0.0, 1e-4),
    },
    "random_forest": {
        "n_estimators": (200, 400, 600),
        "max_depth": (None, 20, 40),
        "max_features": ("sqrt", 0.5, 1.0),
        "min_samples_leaf": (1, 2, 4),
        "min_samples_split": (2, 5, 10),
        "bootstrap": (True,),
    },
}


def sample_configs(model_name: str, budget: int, seed: int) -> list[Dict[str, Any]]:
    space = SEARCH_SPACES[model_name]
    keys = list(space)
    combinations = [dict(zip(keys, vals)) for vals in itertools.product(*(space[k] for k in keys))]

    # PatchTST validity constraint.
    if model_name == "patchtst":
        combinations = [
            c for c in combinations if c["d_model"] % c["n_heads"] == 0
        ]

    if budget > len(combinations):
        raise ValueError(
            f"Budget {budget} exceeds {len(combinations)} unique {model_name} configs."
        )

    rng = random.Random(seed)
    indices = rng.sample(range(len(combinations)), budget)
    return [combinations[i] for i in indices]


def build_neural_model(model_name: str, config: Mapping[str, Any]) -> nn.Module:
    arch = dict(config)
    arch.pop("learning_rate")
    arch.pop("weight_decay")
    if model_name == "tcn":
        return ScreeningTCN(**arch)
    if model_name == "cnn_lstm":
        return ScreeningCNNLSTM(**arch)
    if model_name == "patchtst":
        return ScreeningPatchTST(**arch)
    raise KeyError(model_name)


# =============================================================================
# Data preparation
# =============================================================================

def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH.resolve()}")
    df = pd.read_pickle(DATA_PATH)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Residential DataFrame index must be a DatetimeIndex.")
    missing = [c for c in TARGET_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")
    return df.sort_index()


def prepare_fixed_screening_data(df: pd.DataFrame):
    print("\nPreparing the FIXED screening dataset once...")
    samples = prepare_forecasting_samples(
        df=df,
        k=K,
        history_steps=HISTORY_STEPS,
        horizon=HORIZON,
        freq=FREQ,
        target_cols=TARGET_COLS,
        feature_cols=TARGET_COLS,
        verbose=True,
    )
    splits = chronological_split(
        samples=samples,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        verbose=True,
    )
    scaled_splits, scalers = preprocess_splits(splits)
    return samples, splits, scaled_splits, scalers


# =============================================================================
# Individual screening runs
# =============================================================================

def run_neural_config(
    model_name: str,
    config_id: int,
    config: Dict[str, Any],
    loaders: Mapping[str, DataLoader],
    scalers: Any,
    scaled_splits: Mapping[str, Mapping[str, Any]],
    device: torch.device,
    seed: int,
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print(f"{model_name.upper()} | config {config_id:02d}")
    print(json.dumps(config, indent=2))
    print("=" * 80)

    # Reset before construction so initialization is comparable/reproducible.
    set_seed(seed)
    model = build_neural_model(model_name, config)
    n_params = count_parameters(model)
    checkpoint = CHECKPOINT_DIR / model_name / f"config_{config_id:02d}.pt"

    result = train_neural_model(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        learning_rate=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
        device=device,
        seed=seed,
        checkpoint_path=checkpoint,
    )

    pred_s, true_s = predict_neural(model, loaders["val"], device)
    pred = inverse_transform_targets(pred_s, scalers.target_scaler)
    true = inverse_transform_targets(true_s, scalers.target_scaler)

    row: Dict[str, Any] = {
        "model": model_name,
        "config_id": config_id,
        "k": K,
        "history_steps": HISTORY_STEPS,
        "seed": seed,
        "n_parameters": n_params,
        "train_samples": len(scaled_splits["train"]["Y"]),
        "val_samples": len(scaled_splits["val"]["Y"]),
        "test_samples_reserved": len(scaled_splits["test"]["Y"]),
        "best_epoch": result.best_epoch,
        "best_val_loss_scaled": result.best_val_loss,
        "epochs_trained": result.epochs_trained,
        "training_seconds": result.training_seconds,
        "hyperparameters": json.dumps(config, sort_keys=True),
        **regression_metrics(true, pred),
        **per_target_metrics(true, pred, TARGET_COLS),
    }

    print(
        f"Validation | MAE={row['val_mae']:.4f} | RMSE={row['val_rmse']:.4f} | "
        f"R2={row['val_r2']:.6f} | params={n_params:,}"
    )
    del model, pred_s, true_s, pred, true
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return row


def run_random_forest_config(
    config_id: int,
    config: Dict[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    scalers: Any,
    scaled_splits: Mapping[str, Mapping[str, Any]],
    seed: int,
    n_jobs: int,
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print(f"RANDOM_FOREST | config {config_id:02d}")
    print(json.dumps(config, indent=2))
    print("=" * 80)

    start = time.perf_counter()
    model = RandomForestRegressor(
        random_state=seed,
        n_jobs=n_jobs,
        **config,
    )
    model.fit(x_train, y_train)
    pred_s_flat = model.predict(x_val)
    elapsed = time.perf_counter() - start

    pred_s = np.asarray(pred_s_flat, dtype=np.float32).reshape(-1, HORIZON, len(TARGET_COLS))
    true_s = y_val.reshape(-1, HORIZON, len(TARGET_COLS))
    pred = inverse_transform_targets(pred_s, scalers.target_scaler)
    true = inverse_transform_targets(true_s, scalers.target_scaler)

    row: Dict[str, Any] = {
        "model": "random_forest",
        "config_id": config_id,
        "k": K,
        "history_steps": HISTORY_STEPS,
        "seed": seed,
        "n_parameters": np.nan,
        "train_samples": len(scaled_splits["train"]["Y"]),
        "val_samples": len(scaled_splits["val"]["Y"]),
        "test_samples_reserved": len(scaled_splits["test"]["Y"]),
        "best_epoch": np.nan,
        "best_val_loss_scaled": np.nan,
        "epochs_trained": np.nan,
        "training_seconds": elapsed,
        "hyperparameters": json.dumps(config, sort_keys=True),
        **regression_metrics(true, pred),
        **per_target_metrics(true, pred, TARGET_COLS),
    }
    print(
        f"Validation | MAE={row['val_mae']:.4f} | RMSE={row['val_rmse']:.4f} | "
        f"R2={row['val_r2']:.6f}"
    )
    del model, pred_s_flat, pred_s, true_s, pred, true
    gc.collect()
    return row


# =============================================================================
# Results
# =============================================================================

def save_results(rows: list[Dict[str, Any]]) -> pd.DataFrame:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS_DIR / "screening_results.csv", index=False)

    if "val_mae" not in frame.columns:
        return pd.DataFrame()
    ok = frame[frame["val_mae"].notna()].copy()
    if ok.empty:
        return ok

    ok["within_model_rank"] = ok.groupby("model")["val_mae"].rank(
        method="min", ascending=True
    )
    ok.sort_values(["model", "within_model_rank", "config_id"]).to_csv(
        RESULTS_DIR / "screening_ranked_configs.csv", index=False
    )

    winners = (
        ok.sort_values(["model", "val_mae", "config_id"])
        .groupby("model", as_index=False)
        .first()
        .sort_values("val_mae")
        .reset_index(drop=True)
    )
    winners["model_rank"] = np.arange(1, len(winners) + 1)
    winners.to_csv(RESULTS_DIR / "screening_model_winners.csv", index=False)
    return winners


# =============================================================================
# Main
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Equal-budget model screening at residential k=48.")
    p.add_argument("--budget", type=int, default=DEFAULT_BUDGET,
                   help="Random configurations per model. Default: 20.")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--device", default=None,
                   help='Neural device, e.g. "cuda", "cuda:0", or "cpu".')
    p.add_argument("--models", nargs="+", choices=MODEL_NAMES,
                   default=list(MODEL_NAMES))
    p.add_argument("--rf-n-jobs", type=int, default=-1,
                   help="Random Forest CPU workers. Default: all available cores.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.budget < 1:
        raise ValueError("--budget must be >= 1")

    set_seed(args.seed)
    device = resolve_device(args.device)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("EQUAL-BUDGET MODEL SCREENING")
    print("=" * 80)
    print(f"Dataset          : {DATASET_NAME}")
    print(f"Lead k           : {K} ({K * 15 / 60:.1f} hours)")
    print(f"History          : {HISTORY_STEPS} steps ({HISTORY_STEPS / 96:.1f} days)")
    print(f"Horizon          : {HORIZON} steps ({HORIZON / 4:.1f} hours)")
    print("Input            : flexibility + historical TOD/DOW (7 channels)")
    print(f"Models           : {args.models}")
    print(f"Budget/model     : {args.budget}")
    print(f"Neural device    : {device}")
    print("Test evaluated   : NO")
    print("=" * 80)

    df = load_data()
    print(f"Data shape: {df.shape}")
    print(f"Range: {df.index.min()} -> {df.index.max()}")

    samples, raw_splits, scaled_splits, scalers = prepare_fixed_screening_data(df)
    diagnostics = samples.get("diagnostics", {})

    # Neural models all use the same dynamic historical-calendar loaders.
    loaders = create_dataloaders(
        scaled_splits,
        batch_size=BATCH_SIZE,
        num_workers=0,
        seed=args.seed,
        calendar_mode="history",
    )

    # Fail early if the expected 7-channel information is not available.
    debug_batch = next(iter(loaders["train"]))
    if "history_time" not in debug_batch:
        raise RuntimeError(
            "calendar_mode='history' did not produce history_time. "
            "Update src/datasets.py before screening."
        )
    print(
        "Neural batch check: history=", tuple(debug_batch["history"].shape),
        "history_time=", tuple(debug_batch["history_time"].shape),
        "target=", tuple(debug_batch["target"].shape),
    )
    del debug_batch

    sampled: Dict[str, list[Dict[str, Any]]] = {}
    for i, name in enumerate(args.models):
        sampled[name] = sample_configs(name, args.budget, args.seed + 1000 * i)

    experiment_record = {
        "dataset": DATASET_NAME,
        "k": K,
        "history_steps": HISTORY_STEPS,
        "horizon": HORIZON,
        "input_channels": 7,
        "calendar": "historical TOD/DOW cyclic only",
        "models": args.models,
        "budget_per_model": args.budget,
        "seed": args.seed,
        "max_epochs_neural": MAX_EPOCHS,
        "patience_neural": PATIENCE,
        "batch_size_neural": BATCH_SIZE,
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
        "test_ratio": TEST_RATIO,
        "selection_metric": "validation MAE in original physical units",
        "test_evaluated": False,
        "diagnostics": diagnostics,
        "sampled_configs": sampled,
    }
    with open(RESULTS_DIR / "screening_config.json", "w", encoding="utf-8") as f:
        json.dump(experiment_record, f, indent=2, default=str)

    rows: list[Dict[str, Any]] = []

    # Materialize flattened arrays only when the Random Forest candidate is requested.
    x_train = y_train = x_val = y_val = None
    if "random_forest" in args.models:
        print("\nMaterializing common 672x7 flattened matrices for Random Forest...")
        x_train, y_train = make_tree_matrix(scaled_splits["train"])
        x_val, y_val = make_tree_matrix(scaled_splits["val"])
        print(f"Random Forest train: X={x_train.shape}, Y={y_train.shape}")
        print(f"Random Forest val  : X={x_val.shape}, Y={y_val.shape}")

    for model_name in args.models:
        for config_id, config in enumerate(sampled[model_name], start=1):
            try:
                if model_name == "random_forest":
                    row = run_random_forest_config(
                        config_id=config_id,
                        config=config,
                        x_train=x_train,
                        y_train=y_train,
                        x_val=x_val,
                        y_val=y_val,
                        scalers=scalers,
                        scaled_splits=scaled_splits,
                        seed=args.seed,
                        n_jobs=args.rf_n_jobs,
                    )
                else:
                    row = run_neural_config(
                        model_name=model_name,
                        config_id=config_id,
                        config=config,
                        loaders=loaders,
                        scalers=scalers,
                        scaled_splits=scaled_splits,
                        device=device,
                        seed=args.seed,
                    )
                rows.append(row)
                save_results(rows)

            except Exception as exc:
                print(
                    f"\nFAILED | model={model_name} | config={config_id} | "
                    f"{type(exc).__name__}: {exc}"
                )
                rows.append({
                    "model": model_name,
                    "config_id": config_id,
                    "k": K,
                    "history_steps": HISTORY_STEPS,
                    "seed": args.seed,
                    "hyperparameters": json.dumps(config, sort_keys=True),
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                })
                pd.DataFrame(rows).to_csv(
                    RESULTS_DIR / "screening_progress.csv", index=False
                )

    winners = save_results(rows)
    if winners.empty:
        raise RuntimeError("No screening configuration completed successfully.")

    print("\n" + "=" * 80)
    print("MODEL SCREENING COMPLETE")
    print("=" * 80)
    cols = [
        "model_rank", "model", "config_id", "val_mae", "val_rmse", "val_r2",
        "Pmax_mae", "Cmax_mae", "Cmin_mae", "training_seconds",
    ]
    print(winners[cols].to_string(index=False))
    print("\nTop two model families by validation MAE:")
    print(winners.head(2)[["model_rank", "model", "val_mae"]].to_string(index=False))
    print("\nThe test partition was reserved and NOT evaluated.")
    print(f"Results saved under: {RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
