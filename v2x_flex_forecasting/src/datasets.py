"""PyTorch datasets/loaders for continuous-history EV flexibility forecasting."""
from __future__ import annotations
from typing import Any, Dict, Mapping, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd

CALENDAR_MODES = {
    "none",
    "history",
    "history_future",
}

def _build_calendar_lookup() -> np.ndarray:
    """
    Build cyclical calendar features for all 672 quarter-hour
    positions in one week.

    Columns:
        [tod_sin, tod_cos, dow_sin, dow_cos]

    Shape:
        (672, 4)
    """
    slots = np.arange(672)

    dow = slots // 96
    quarter = slots % 96

    tod_angle = 2.0 * np.pi * quarter / 96.0
    dow_angle = 2.0 * np.pi * dow / 7.0

    lookup = np.column_stack([
        np.sin(tod_angle),
        np.cos(tod_angle),
        np.sin(dow_angle),
        np.cos(dow_angle),
    ])

    return lookup.astype(np.float32)

class ForecastDataset(Dataset):
    """Dataset yielding history (L,F) and target (H,T) tensors."""
    def __init__(self, split: Mapping[str, Any], calendar_mode: str = "none",) -> None:
        missing = [k for k in ("X_history", "Y") if k not in split]
        if missing:
            raise KeyError(f"Split is missing required keys: {missing}")
        X = np.asarray(split["X_history"])
        Y = np.asarray(split["Y"])
        if X.ndim != 3:
            raise ValueError(f"X_history must have shape (N,L,F); got {X.shape}.")
        if Y.ndim != 3:
            raise ValueError(f"Y must have shape (N,H,T); got {Y.shape}.")
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X_history and Y must have the same sample count.")
        if X.shape[0] == 0:
            raise ValueError("Cannot create ForecastDataset from an empty split.")
        if not np.isfinite(X).all() or not np.isfinite(Y).all():
            raise ValueError("X_history/Y contains NaN or infinite values.")

        if calendar_mode not in CALENDAR_MODES:
            raise ValueError( f"calendar_mode must be one of "f"{sorted(CALENDAR_MODES)}; "  f"got {calendar_mode!r}." )
        self.calendar_mode = calendar_mode

        self.X_history = np.ascontiguousarray(X, dtype=np.float32)
        self.Y = np.ascontiguousarray(Y, dtype=np.float32)

        self.origin = (np.asarray(split["origin"]) if "origin" in split  else None )
        self.target_start = (  np.asarray(split["target_start"]) if "target_start" in split else None )

        if calendar_mode != "none":
            if self.origin is None:
                raise KeyError( "origin is required when calendar_mode is not 'none'." )

        if calendar_mode == "history_future":
            if self.target_start is None:
                raise KeyError("target_start is required when calendar_mode='history_future'.") 

        """
        # Optional calendar covariates
        self.X_history_time = None
        self.X_future_time = None

        has_history_time = "X_history_time" in split
        has_future_time = "X_future_time" in split

        if has_history_time != has_future_time:
            raise ValueError( "X_history_time and X_future_time must either both be present or both be absent." )

        if has_history_time:
            X_history_time = np.asarray(split["X_history_time"])
            X_future_time = np.asarray(split["X_future_time"])

            if X_history_time.shape != ( X.shape[0],X.shape[1], 4, ):
                raise ValueError("X_history_time must have shape " f"({X.shape[0]}, {X.shape[1]}, 4); " f"got {X_history_time.shape}.")

            if X_future_time.shape != ( Y.shape[0], Y.shape[1],4, ):
                raise ValueError( "X_future_time must have shape " f"({Y.shape[0]}, {Y.shape[1]}, 4); " f"got {X_future_time.shape}." )

            if not np.isfinite(X_history_time).all():
                raise ValueError("X_history_time contains NaN or infinite values." )

            if not np.isfinite(X_future_time).all():
                raise ValueError( "X_future_time contains NaN or infinite values." )

            self.X_history_time = np.ascontiguousarray( X_history_time,dtype=np.float32,)

            self.X_future_time = np.ascontiguousarray( X_future_time,dtype=np.float32,)

        self.has_calendar = self.X_history_time is not None
        """

        n = len(self.X_history)
        self.origin = np.asarray(split["origin"]) if "origin" in split else None
        self.target_start = np.asarray(split["target_start"]) if "target_start" in split else None
        if self.origin is not None and len(self.origin) != n:
            raise ValueError("origin must contain one timestamp per sample.")
        if self.target_start is not None and len(self.target_start) != n:
            raise ValueError("target_start must contain one timestamp per sample.")

        self.history_steps = X.shape[1]
        self.n_features = X.shape[2]
        self.horizon = Y.shape[1]
        self.n_targets = Y.shape[2]


        self.calendar_lookup = None
        self.origin_slots = None
        self.target_start_slots = None

        if calendar_mode != "none":
            self.calendar_lookup = _build_calendar_lookup()
            self.origin_slots = _timestamps_to_week_slots(self.origin )
            self.history_offsets = np.arange( -(self.history_steps - 1),  1, dtype=np.int32,)

        if self.calendar_mode == "history_future":
            self.target_start_slots = _timestamps_to_week_slots(self.target_start)
            self.future_offsets = np.arange(self.horizon, dtype=np.int32, )

    def __len__(self) -> int:
        return len(self.X_history)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:

        item = {
            "history": torch.from_numpy( self.X_history[idx]),
            "target": torch.from_numpy( self.Y[idx] ),}

        # -----------------------------------------
        # Historical calendar
        # -----------------------------------------
        if self.calendar_mode in {
            "history",
            "history_future",
        }:
            history_slots = (  self.origin_slots[idx]  + self.history_offsets  ) % 672
            history_time = self.calendar_lookup[ history_slots  ]
            item["history_time"] = torch.from_numpy(  history_time  )

        # -----------------------------------------
        # Future calendar
        # -----------------------------------------
        if self.calendar_mode == "history_future":

            future_slots = ( self.target_start_slots[idx]  + self.future_offsets   ) % 672
            future_time = self.calendar_lookup[ future_slots ]
            item["future_time"] = torch.from_numpy( future_time )

        return item

def create_datasets(splits: Mapping[str, Mapping[str, Any]], calendar_mode: str = "none",) -> Dict[str, ForecastDataset]:
    missing = [n for n in ("train", "val", "test") if n not in splits]
    if missing:
        raise KeyError(f"Missing split(s): {missing}")
    return  { name: ForecastDataset( splits[name], calendar_mode=calendar_mode, ) for name in ("train", "val", "test") }

def create_dataloaders(
    splits: Mapping[str, Mapping[str, Any]],
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: Optional[bool] = None,
    drop_last_train: bool = False,
    persistent_workers: bool = False,
    seed: int = 42,
    calendar_mode: str = "none",
) -> Dict[str, DataLoader]:
    """Create loaders; shuffle training only, preserving val/test chronology."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1.")
    if num_workers < 0:
        raise ValueError("num_workers must be >= 0.")
    datasets = create_datasets( splits, calendar_mode=calendar_mode,)
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    if num_workers == 0:
        persistent_workers = False
    generator = torch.Generator().manual_seed(seed)
    common = dict(batch_size=batch_size, num_workers=num_workers,
                  pin_memory=pin_memory, persistent_workers=persistent_workers)
    return {
        "train": DataLoader(datasets["train"], shuffle=True, drop_last=drop_last_train,
                            generator=generator, **common),
        "val": DataLoader(datasets["val"], shuffle=False, drop_last=False, **common),
        "test": DataLoader(datasets["test"], shuffle=False, drop_last=False, **common),
    }

def dataset_shapes(datasets: Mapping[str, ForecastDataset],) -> Dict[str, Dict[str, tuple]]:

    result = {}
    for name, ds in datasets.items():
        shapes = {
            "X_history": tuple(ds.X_history.shape),
            "Y": tuple(ds.Y.shape),
        }

        if ds.X_history_time is not None:
            shapes["X_history_time"] = tuple(ds.X_history_time.shape)
            shapes["X_future_time"] = tuple( ds.X_future_time.shape )

        result[name] = shapes
    return result

def _timestamps_to_week_slots( timestamps: np.ndarray,) -> np.ndarray:
    """
    Convert timestamps to integer quarter-hour positions
    within the week:

        Monday 00:00 -> 0
        ...
        Sunday 23:45 -> 671
    """
    timestamps = pd.DatetimeIndex(  pd.to_datetime(timestamps) )
    quarter = ( timestamps.hour.to_numpy() * 4+ timestamps.minute.to_numpy() // 15  )
    slots = (  timestamps.dayofweek.to_numpy() * 96  + quarter)
    return slots.astype(np.int16)


def print_dataloader_summary(loaders: Mapping[str, DataLoader],) -> None:

    print("DataLoaders\n-----------")

    for name in ("train", "val", "test"):

        if name not in loaders:
            continue

        loader = loaders[name]
        ds = loader.dataset

        msg = (
            f"{name:5s}: {len(ds):6d} samples | "
            f"history={tuple(ds.X_history.shape)} | "
            f"target={tuple(ds.Y.shape)}"
        )

        if ds.X_history_time is not None:
            msg += (
                f" | history_time={tuple(ds.X_history_time.shape)}"
                f" | future_time={tuple(ds.X_future_time.shape)}"
            )

        msg += f" | batches={len(loader)}"

        print(msg)
