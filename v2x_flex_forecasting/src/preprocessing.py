"""
Preprocessing utilities for continuous-history EV flexibility forecasting.

Expected sample structure from src.data_preparation.prepare_forecasting_samples:

    X_history   : (N, L, n_features)
    Y           : (N, H, n_targets)
    origin      : (N,)
    target_start: (N,)

Core principles
---------------
1. Split chronologically; never randomly shuffle before splitting.
2. Fit scalers on the training partition only.
3. Apply the fitted training scalers unchanged to validation and test.
4. Keep input and target scalers separate.
5. Preserve timestamps and sample metadata.
6. Support date-based splitting, which is useful for discontinuous datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


ArrayDict = Dict[str, Any]
SplitDict = Dict[str, ArrayDict]


@dataclass
class ForecastScalers:
    """Container for fitted input and target StandardScalers."""

    input_scaler: StandardScaler
    target_scaler: StandardScaler


def _validate_samples(samples: Mapping[str, Any]) -> int:
    """Validate the minimum structure required by this preprocessing module."""
    required = ("X_history", "Y", "origin", "target_start")
    missing = [key for key in required if key not in samples]
    if missing:
        raise KeyError(f"Samples are missing required keys: {missing}")

    X = np.asarray(samples["X_history"])
    Y = np.asarray(samples["Y"])
    origin = np.asarray(samples["origin"])
    target_start = np.asarray(samples["target_start"])

    if X.ndim != 3:
        raise ValueError(
            f"X_history must have shape (N, L, F); received {X.shape}."
        )
    if Y.ndim != 3:
        raise ValueError(
            f"Y must have shape (N, H, T); received {Y.shape}."
        )

    n = X.shape[0]
    if Y.shape[0] != n or len(origin) != n or len(target_start) != n:
        raise ValueError(
            "X_history, Y, origin, and target_start must contain the same "
            "number of samples."
        )

    if n == 0:
        raise ValueError("No valid forecasting samples were provided.")

    if not np.isfinite(X).all():
        raise ValueError("X_history contains NaN or infinite values.")
    if not np.isfinite(Y).all():
        raise ValueError("Y contains NaN or infinite values.")

    # Optional calendar covariates
    if "X_history_time" in samples:
        X_history_time = np.asarray(samples["X_history_time"])

        if X_history_time.ndim != 3:
            raise ValueError("X_history_time must have shape (N, L, C); "  f"received {X_history_time.shape}." )

        if X_history_time.shape[0] != n:
            raise ValueError("X_history_time must contain the same number of samples as X_history.")

        if X_history_time.shape[1] != X.shape[1]:
            raise ValueError( "X_history_time must have the same history length as X_history.")

        if X_history_time.shape[2] != 4:
            raise ValueError("X_history_time must contain 4 calendar features [tod_sin, tod_cos, dow_sin, dow_cos]." )

        if not np.isfinite(X_history_time).all():
            raise ValueError( "X_history_time contains NaN or infinite values.")

    if "X_future_time" in samples:
        X_future_time = np.asarray(samples["X_future_time"])

        if X_future_time.ndim != 3:
            raise ValueError( "X_future_time must have shape (N, H, C); "  f"received {X_future_time.shape}.")

        if X_future_time.shape[0] != n:
            raise ValueError( "X_future_time must contain the same number of samples as Y.")

        if X_future_time.shape[1] != Y.shape[1]:
            raise ValueError( "X_future_time must have the same forecast horizon as Y.")

        if X_future_time.shape[2] != 4:
            raise ValueError( "X_future_time must contain 4 calendar features [tod_sin, tod_cos, dow_sin, dow_cos]." )

        if not np.isfinite(X_future_time).all():
            raise ValueError( "X_future_time contains NaN or infinite values." )

    has_history_time = "X_history_time" in samples
    has_future_time = "X_future_time" in samples

    if has_history_time != has_future_time:
        raise ValueError( "X_history_time and X_future_time must either both be present or both be absent.")

    return n


def _subset_samples(samples: Mapping[str, Any], indices: np.ndarray) -> ArrayDict:
    """
    Subset sample-level arrays while preserving useful scalar metadata.

    Only objects whose first dimension/length equals the number of forecasting
    samples are sliced. Other metadata are copied unchanged.
    """
    n = len(samples["target_start"])
    out: ArrayDict = {}

    for key, value in samples.items():
        if key == "diagnostics":
            # Diagnostics describe the complete prepared dataset, not a split.
            out[key] = value
            continue

        if isinstance(value, np.ndarray) and value.ndim >= 1 and len(value) == n:
            out[key] = value[indices]
        elif isinstance(value, pd.Series) and len(value) == n:
            out[key] = value.iloc[indices].to_numpy()
        elif isinstance(value, pd.DatetimeIndex) and len(value) == n:
            out[key] = value[indices].to_numpy()
        else:
            out[key] = value

    return out


def _sort_samples_chronologically(samples: Mapping[str, Any]) -> ArrayDict:
    """Sort samples by target_start, with origin as a deterministic tie-breaker."""
    _validate_samples(samples)

    target_start = pd.to_datetime(np.asarray(samples["target_start"]))
    origin = pd.to_datetime(np.asarray(samples["origin"]))

    order = np.lexsort(
        (
            origin.to_numpy(dtype="datetime64[ns]"),
            target_start.to_numpy(dtype="datetime64[ns]"),
        )
    )
    return _subset_samples(samples, order)


def _validate_ratios(
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> None:
    ratios = np.asarray([train_ratio, val_ratio, test_ratio], dtype=float)

    if np.any(ratios <= 0):
        raise ValueError("train_ratio, val_ratio, and test_ratio must be > 0.")

    if not np.isclose(ratios.sum(), 1.0, atol=1e-8):
        raise ValueError(
            "train_ratio + val_ratio + test_ratio must equal 1.0; "
            f"received {ratios.sum():.8f}."
        )


def chronological_split(
    samples: Mapping[str, Any],
    train_ratio: float = 0.80,
    val_ratio: float = 0.05,
    test_ratio: float = 0.15,
    verbose: bool = True,
) -> SplitDict:
    """
    Split forecasting samples chronologically by sample count.

    This function is appropriate when percentage-based chronological splitting
    is desired. For highly discontinuous datasets, consider
    chronological_split_by_date so that calendar boundaries are explicit.

    The split is performed using target_start after chronological sorting.
    """
    _validate_ratios(train_ratio, val_ratio, test_ratio)
    ordered = _sort_samples_chronologically(samples)
    n = _validate_samples(ordered)

    n_train = int(np.floor(n * train_ratio))
    n_val = int(np.floor(n * val_ratio))
    n_test = n - n_train - n_val

    if min(n_train, n_val, n_test) < 1:
        raise ValueError(
            "The requested split creates an empty partition. "
            f"N={n}, sizes=({n_train}, {n_val}, {n_test})."
        )

    train_idx = np.arange(0, n_train)
    val_idx = np.arange(n_train, n_train + n_val)
    test_idx = np.arange(n_train + n_val, n)

    splits = {
        "train": _subset_samples(ordered, train_idx),
        "val": _subset_samples(ordered, val_idx),
        "test": _subset_samples(ordered, test_idx),
    }

    if verbose:
        print_split_summary(splits)

    return splits


def chronological_split_by_date(
    samples: Mapping[str, Any],
    train_end: Any,
    val_end: Any,
    verbose: bool = True,
) -> SplitDict:
    """
    Split samples using explicit chronological target-start boundaries.

    Definitions:
        train: target_start <= train_end
        val  : train_end < target_start <= val_end
        test : target_start > val_end

    This is useful for datasets containing long missing intervals because the
    split boundaries are tied to calendar time rather than sample counts.

    Parameters
    ----------
    train_end
        Last target_start included in training.
    val_end
        Last target_start included in validation. Must be later than train_end.
    """
    ordered = _sort_samples_chronologically(samples)
    target_start = pd.to_datetime(np.asarray(ordered["target_start"]))

    train_end = pd.Timestamp(train_end)
    val_end = pd.Timestamp(val_end)

    if val_end <= train_end:
        raise ValueError("val_end must be later than train_end.")

    train_mask = target_start <= train_end
    val_mask = (target_start > train_end) & (target_start <= val_end)
    test_mask = target_start > val_end

    masks = {
        "train": np.flatnonzero(train_mask),
        "val": np.flatnonzero(val_mask),
        "test": np.flatnonzero(test_mask),
    }

    empty = [name for name, idx in masks.items() if len(idx) == 0]
    if empty:
        raise ValueError(
            "Date boundaries create empty partition(s): "
            f"{empty}. Choose boundaries that contain valid samples."
        )

    splits = {
        name: _subset_samples(ordered, idx)
        for name, idx in masks.items()
    }

    if verbose:
        print_split_summary(splits)

    return splits


def fit_scalers(train_split: Mapping[str, Any]) -> ForecastScalers:
    """
    Fit independent StandardScalers using training data only.

    Input scaler:
        fitted over all historical timestamps from X_history.

    Target scaler:
        fitted over all forecast timestamps from Y.

    Each variable is standardized independently; time positions are not
    treated as separate features.
    """
    _validate_samples(train_split)

    X = np.asarray(train_split["X_history"], dtype=np.float64)
    Y = np.asarray(train_split["Y"], dtype=np.float64)

    n_features = X.shape[-1]
    n_targets = Y.shape[-1]

    input_scaler = StandardScaler()
    target_scaler = StandardScaler()

    input_scaler.fit(X.reshape(-1, n_features))
    target_scaler.fit(Y.reshape(-1, n_targets))

    return ForecastScalers(
        input_scaler=input_scaler,
        target_scaler=target_scaler,
    )


def transform_split(
    split: Mapping[str, Any],
    scalers: ForecastScalers,
) -> ArrayDict:
    """
    Apply already-fitted training scalers to one dataset partition.

    X_history and Y are standardized.

    Optional calendar features (X_history_time and X_future_time) are
    retained without standardization because their cyclical sine/cosine
    values are already bounded in [-1, 1].
    """
    _validate_samples(split)

    X = np.asarray(split["X_history"], dtype=np.float64)
    Y = np.asarray(split["Y"], dtype=np.float64)

    n, history_steps, n_features = X.shape
    n_y, horizon, n_targets = Y.shape

    X_scaled = scalers.input_scaler.transform( X.reshape(-1, n_features)).reshape(n, history_steps, n_features)

    Y_scaled = scalers.target_scaler.transform(  Y.reshape(-1, n_targets) ).reshape(n_y, horizon, n_targets)

    out = dict(split)

    out["X_history"] = X_scaled.astype(  np.float32, copy=False, )

    out["Y"] = Y_scaled.astype( np.float32, copy=False,)

    # Calendar features are already bounded in [-1, 1],
    # so retain them without standardization.
    if "X_history_time" in split:
        out["X_history_time"] = np.asarray(split["X_history_time"], dtype=np.float32,  )

    if "X_future_time" in split:
        out["X_future_time"] = np.asarray( split["X_future_time"],  dtype=np.float32,  )

    return out


def preprocess_splits(
    splits: Mapping[str, Mapping[str, Any]],
) -> Tuple[SplitDict, ForecastScalers]:
    """
    Fit scalers on training only and transform train/validation/test.

    Returns
    -------
    scaled_splits, scalers
    """
    required = ("train", "val", "test")
    missing = [name for name in required if name not in splits]
    if missing:
        raise KeyError(f"Missing split(s): {missing}")

    scalers = fit_scalers(splits["train"])

    scaled = {
        name: transform_split(splits[name], scalers)
        for name in required
    }

    return scaled, scalers


def inverse_transform_targets(
    Y_scaled: np.ndarray,
    target_scaler: StandardScaler,
) -> np.ndarray:
    """
    Convert scaled forecasts/targets back to their original physical units.

    Accepts any array whose final dimension equals the number of target
    variables, including (N, H, T) and (H, T).
    """
    Y_scaled = np.asarray(Y_scaled)

    if Y_scaled.ndim < 2:
        raise ValueError(
            "Y_scaled must have at least two dimensions with target variables "
            "on the final axis."
        )

    n_targets = Y_scaled.shape[-1]
    expected = target_scaler.n_features_in_

    if n_targets != expected:
        raise ValueError(
            f"Expected {expected} target variables; received {n_targets}."
        )

    original_shape = Y_scaled.shape
    restored = target_scaler.inverse_transform(
        Y_scaled.reshape(-1, n_targets)
    )
    return restored.reshape(original_shape)


def inverse_transform_history(
    X_scaled: np.ndarray,
    input_scaler: StandardScaler,
) -> np.ndarray:
    """Convert scaled historical inputs back to their original units."""
    X_scaled = np.asarray(X_scaled)

    if X_scaled.ndim < 2:
        raise ValueError(
            "X_scaled must have at least two dimensions with features on "
            "the final axis."
        )

    n_features = X_scaled.shape[-1]
    expected = input_scaler.n_features_in_

    if n_features != expected:
        raise ValueError(
            f"Expected {expected} input variables; received {n_features}."
        )

    original_shape = X_scaled.shape
    restored = input_scaler.inverse_transform(
        X_scaled.reshape(-1, n_features)
    )
    return restored.reshape(original_shape)


def split_summary_frame(splits: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    """Return sample counts and temporal ranges for each split."""
    rows = []

    for name in ("train", "val", "test"):
        if name not in splits:
            continue

        split = splits[name]
        target_start = pd.to_datetime(np.asarray(split["target_start"]))
        origin = pd.to_datetime(np.asarray(split["origin"]))

        rows.append(
            {
                "split": name,
                "n_samples": len(target_start),
                "first_origin": origin.min(),
                "last_origin": origin.max(),
                "first_target": target_start.min(),
                "last_target": target_start.max(),
            }
        )

    return pd.DataFrame(rows)


def print_split_summary(splits: Mapping[str, Mapping[str, Any]]) -> None:
    """Print chronological split sizes and target-start ranges."""
    summary = split_summary_frame(splits)

    print("Chronological split")
    print("-------------------")

    for row in summary.itertuples(index=False):
        print(
            f"{row.split:5s}: {row.n_samples:6d} samples | "
            f"{row.first_target} -> {row.last_target}"
        )
