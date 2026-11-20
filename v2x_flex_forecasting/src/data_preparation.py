"""Leakage-safe continuous-history sample construction for V2X forecasting."""

from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import pandas as pd

DEFAULT_TARGET_COLS: Tuple[str, ...] = ("Pmax", "Cmax", "Cmin")


@dataclass(frozen=True)
class SampleDiagnostics:
    k: int
    history_steps: int
    horizon: int
    freq: str
    candidate_origins: int
    valid_samples: int
    rejected_samples: int
    rejected_missing_history: int
    rejected_missing_target: int
    rejected_missing_both: int
    rejected_nonfinite_history: int
    rejected_nonfinite_target: int
    rejected_nonfinite_both: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _validate_arguments(df, k, history_steps, horizon, freq,
                        feature_cols, target_cols):
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df.index must be a pandas DatetimeIndex.")
    if df.index.has_duplicates:
        raise ValueError("df.index contains duplicate timestamps.")
    if k < 1:
        raise ValueError("k must be >= 1.")
    if history_steps < 1:
        raise ValueError("history_steps must be >= 1.")
    if horizon < 1:
        raise ValueError("horizon must be >= 1.")

    required = list(dict.fromkeys([*feature_cols, *target_cols]))
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    dt = pd.to_timedelta(freq)
    if dt <= pd.Timedelta(0):
        raise ValueError("freq must be positive.")
    return dt


def _candidate_origins(index, history_steps, horizon, k, dt):
    start = index.min() + (history_steps - 1) * dt
    end = index.max() - (k + horizon - 1) * dt
    if start > end:
        return pd.DatetimeIndex([], dtype="datetime64[ns]")
    return pd.date_range(start=start, end=end, freq=dt)


def _expected_times(origin, history_steps, horizon, k, dt):
    history_times = pd.date_range(end=origin, periods=history_steps, freq=dt)
    target_times = pd.date_range(start=origin + k * dt,
                                 periods=horizon, freq=dt)
    return history_times, target_times

def _calendar_features(timestamps: pd.DatetimeIndex) -> np.ndarray:
    """
    Generate cyclical calendar features for a sequence of timestamps.

    Returns
    -------
    np.ndarray
        Shape (n_timestamps, 4), with columns:
        [tod_sin, tod_cos, dow_sin, dow_cos]
    """
    timestamps = pd.DatetimeIndex(timestamps)

    # Time of day: minutes since midnight
    minutes = (
        timestamps.hour.to_numpy(dtype=np.float64) * 60.0
        + timestamps.minute.to_numpy(dtype=np.float64)
        + timestamps.second.to_numpy(dtype=np.float64) / 60.0
    )

    tod_angle = 2.0 * np.pi * minutes / 1440.0

    tod_sin = np.sin(tod_angle)
    tod_cos = np.cos(tod_angle)

    # Day of week: Monday=0, ..., Sunday=6
    dow = timestamps.dayofweek.to_numpy(dtype=np.float64)
    dow_angle = 2.0 * np.pi * dow / 7.0

    dow_sin = np.sin(dow_angle)
    dow_cos = np.cos(dow_angle)

    return np.column_stack(
        [tod_sin, tod_cos, dow_sin, dow_cos]
    ).astype(np.float32)

def prepare_forecasting_samples(
    df: pd.DataFrame,
    k: int,
    history_steps: int,
    horizon: int = 96,
    freq: str = "15min",
    target_cols: Sequence[str] = DEFAULT_TARGET_COLS,
    feature_cols: Sequence[str] | None = None,
    verbose: bool = True,
    #include_calendar: bool = False,
) -> Dict[str, Any]:
    """Build continuous-history forecasting samples.

    For forecast origin t::

        X_history = [y_(t-(L-1)dt), ..., y_t]
        Y = [y_(t+k*dt), ..., y_(t+(k+H-1)dt)]

    If include_calendar=True, the function also returns:

        X_history_time : (N, history_steps, 4)
        X_future_time  : (N, horizon, 4)

    where the four calendar channels are:

        [tod_sin, tod_cos, dow_sin, dow_cos]

    No interpolation or forward filling is performed. A sample is retained
    only if every required input and target timestamp exists and every
    required value is finite.
    """

    target_cols = tuple(target_cols)
    feature_cols = tuple(
        target_cols if feature_cols is None else feature_cols
    )

    dt = _validate_arguments(
        df,
        k,
        history_steps,
        horizon,
        freq,
        feature_cols,
        target_cols,
    )

    data = df.sort_index()

    required = list(
        dict.fromkeys([*feature_cols, *target_cols])
    )

    try:
        numeric = data.loc[:, required].astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "feature_cols and target_cols must contain numeric values."
        ) from exc

    origins = _candidate_origins(
        numeric.index,
        history_steps,
        horizon,
        k,
        dt,
    )

    # ---------------------------------------------------------
    # Storage
    # ---------------------------------------------------------
    X_list = []
    Y_list = []

    valid_origins = []
    target_starts = []

    missing_history = 0
    missing_target = 0
    missing_both = 0

    nonfinite_history = 0
    nonfinite_target = 0
    nonfinite_both = 0

    index = numeric.index
    
    
    # Only allocate calendar storage if requested.
    #if include_calendar:
    #    X_history_time_list = []
    #    X_future_time_list = []

    # ---------------------------------------------------------
    # Construct samples
    # ---------------------------------------------------------
    for origin in origins:

        history_times, target_times = _expected_times(
            origin,
            history_steps,
            horizon,
            k,
            dt,
        )

        # -----------------------------------------------------
        # Check required timestamps
        # -----------------------------------------------------
        history_exists = history_times.isin(index).all()
        target_exists = target_times.isin(index).all()

        if not history_exists or not target_exists:

            if not history_exists and not target_exists:
                missing_both += 1

            elif not history_exists:
                missing_history += 1

            else:
                missing_target += 1

            continue

        # -----------------------------------------------------
        # Extract numerical data
        # -----------------------------------------------------
        X = numeric.loc[ history_times,list(feature_cols), ].to_numpy(  dtype=np.float32,  copy=True,)
        Y = numeric.loc[ target_times,list(target_cols), ].to_numpy( dtype=np.float32, copy=True, )

        # -----------------------------------------------------
        # Check finite values
        # -----------------------------------------------------
        history_finite = np.isfinite(X).all()
        target_finite = np.isfinite(Y).all()

        if not history_finite or not target_finite:

            if not history_finite and not target_finite:
                nonfinite_both += 1

            elif not history_finite:
                nonfinite_history += 1

            else:
                nonfinite_target += 1

            continue

        # -----------------------------------------------------
        # Valid sample
        # -----------------------------------------------------
        X_list.append(X)
        Y_list.append(Y)

        valid_origins.append(origin)
        target_starts.append(target_times[0])
        """
        # Generate calendar features only for valid samples.
        if include_calendar:
            history_calendar = _calendar_features( history_times)
            future_calendar = _calendar_features(  target_times  )
            X_history_time_list.append( history_calendar )
            X_future_time_list.append( future_calendar )
        """
    # ---------------------------------------------------------
    # Convert lists to arrays
    # ---------------------------------------------------------
    n_features = len(feature_cols)
    n_targets = len(target_cols)

    if X_list:

        X_history = np.stack(   X_list  ).astype( np.float32,copy=False,  )
        Y = np.stack(    Y_list).astype( np.float32, copy=False, )
    else:

        X_history = np.empty( (0, history_steps, n_features),dtype=np.float32, )
        Y = np.empty( (0, horizon, n_targets),  dtype=np.float32, )

    # ---------------------------------------------------------
    # Calendar arrays
    # ---------------------------------------------------------
    """
    if include_calendar:

        if X_history_time_list:
            X_history_time = np.stack(  X_history_time_list ).astype( np.float32, copy=False, )
            X_future_time = np.stack(  X_future_time_list).astype(  np.float32,  copy=False,)

        else:
            X_history_time = np.empty(   (0, history_steps, 4),  dtype=np.float32,  )
            X_future_time = np.empty((0, horizon, 4),dtype=np.float32, )
    """
    # ---------------------------------------------------------
    # Diagnostics
    # ---------------------------------------------------------
    diagnostics = SampleDiagnostics(
        k=k,
        history_steps=history_steps,
        horizon=horizon,
        freq=freq,
        candidate_origins=len(origins),
        valid_samples=len(valid_origins),
        rejected_samples=(
            len(origins) - len(valid_origins)
        ),
        rejected_missing_history=missing_history,
        rejected_missing_target=missing_target,
        rejected_missing_both=missing_both,
        rejected_nonfinite_history=nonfinite_history,
        rejected_nonfinite_target=nonfinite_target,
        rejected_nonfinite_both=nonfinite_both,
    )

    # ---------------------------------------------------------
    # Base result
    # ---------------------------------------------------------
    result = {
        "X_history": X_history,
        "Y": Y,
        "origin": np.asarray(
            valid_origins,
            dtype="datetime64[ns]",
        ),
        "target_start": np.asarray(
            target_starts,
            dtype="datetime64[ns]",
        ),
        "diagnostics": diagnostics.to_dict(),
        "feature_cols": feature_cols,
        "target_cols": target_cols,
        "k": k,
        "history_steps": history_steps,
        "horizon": horizon,
        "freq": freq,
       # "include_calendar": include_calendar,
    }

    """

    # Add calendar arrays only when requested.
    if include_calendar:
        result["X_history_time"] = X_history_time
        result["X_future_time"] = X_future_time
    """

    if verbose:
        print_sample_summary(result)

    return result


def print_sample_summary(samples: Dict[str, Any]) -> None:
    """Print sample shapes and mutually exclusive rejection diagnostics."""
    d = samples["diagnostics"]
    print(f"k = {d['k']}")
    print(f"History steps = {d['history_steps']}")
    print(f"Horizon = {d['horizon']}")
    print(f"Frequency = {d['freq']}")
    print()
    print(f"Candidate origins: {d['candidate_origins']}")
    print(f"Valid samples:     {d['valid_samples']}")
    print(f"Rejected samples:  {d['rejected_samples']}")
    print()
    print("Rejected: missing required timestamps")
    print(f"  history only : {d['rejected_missing_history']}")
    print(f"  target only  : {d['rejected_missing_target']}")
    print(f"  both         : {d['rejected_missing_both']}")
    print()
    print("Rejected: NaN/Inf values")
    print(f"  history only : {d['rejected_nonfinite_history']}")
    print(f"  target only  : {d['rejected_nonfinite_target']}")
    print(f"  both         : {d['rejected_nonfinite_both']}")
    print()
    print("Shapes:")
    print(f"X_history : {samples['X_history'].shape}")
    print(f"Y         : {samples['Y'].shape}")

    if d["valid_samples"]:
        print(f"First origin: {pd.Timestamp(samples['origin'][0])}")
        print(f"Last origin : {pd.Timestamp(samples['origin'][-1])}")
        print(f"First target: {pd.Timestamp(samples['target_start'][0])}")
        print(f"Last target : {pd.Timestamp(samples['target_start'][-1])}")


def diagnostics_to_frame(samples: Dict[str, Any]) -> pd.DataFrame:
    """Return preparation diagnostics as a one-row DataFrame."""
    return pd.DataFrame([samples["diagnostics"]])
