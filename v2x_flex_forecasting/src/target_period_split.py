"""Target-period chronological split for already-constructed forecasting samples.

Place in src/target_period_split.py. No sample's target horizon crosses a
partition boundary. Historical context is allowed to cross earlier boundaries
because it is available at the forecast origin.
"""
from __future__ import annotations

from typing import Any, Mapping
import numpy as np
import pandas as pd


def _subset(samples: Mapping[str, Any], positions: np.ndarray) -> dict:
    n = len(samples['target_start'])
    result = {}
    for key, value in samples.items():
        if isinstance(value, np.ndarray) and value.ndim and len(value) == n:
            result[key] = value[positions]
        elif isinstance(value, (pd.Series, pd.DatetimeIndex)) and len(value) == n:
            result[key] = value.to_numpy()[positions]
        else:
            result[key] = value
    return result


def target_period_split(
    samples: Mapping[str, Any],
    *,
    train_end: Any,
    val_end: Any,
    verbose: bool = True,
) -> dict:
    """Split by fixed exclusive target-period boundaries.

    train: entire target horizon < train_end
    val:   entire target horizon >= train_end and < val_end
    test:  entire target horizon >= val_end

    Boundaries must be timestamps on the data's time axis. A sample that
    straddles a boundary is discarded. Split boundaries should be chosen
    independently of model performance (ideally from the raw date index).
    """
    required = ('X_history', 'Y', 'origin', 'target_start', 'freq')
    missing = [k for k in required if k not in samples]
    if missing:
        raise KeyError(f'Missing sample keys: {missing}')
    n = len(samples['target_start'])
    if n == 0 or samples['Y'].shape[0] != n or len(samples['origin']) != n:
        raise ValueError('Empty or inconsistent sample arrays')
    if samples['Y'].ndim != 3:
        raise ValueError('Y must have shape (N,H,T)')
    dt = pd.to_timedelta(samples['freq'])
    if dt <= pd.Timedelta(0):
        raise ValueError('freq must be positive')
    start = pd.DatetimeIndex(pd.to_datetime(samples['target_start']))
    origin = pd.DatetimeIndex(pd.to_datetime(samples['origin']))
    end = start + (samples['Y'].shape[1] - 1) * dt
    a, b = pd.Timestamp(train_end), pd.Timestamp(val_end)
    if not a < b:
        raise ValueError('Require train_end < val_end')
    if start.tz != a.tz or start.tz != b.tz:
        raise ValueError('Boundary timezone must match sample timestamps')
    if not np.all(origin < start):
        raise AssertionError('Every forecast origin must precede target_start')
    masks = {
        'train': end < a,
        'val': (start >= a) & (end < b),
        'test': start >= b,
    }
    splits = {}
    for name, mask in masks.items():
        positions = np.flatnonzero(mask)
        if len(positions) == 0:
            raise ValueError(f'{name} partition has no complete target windows')
        positions = positions[np.argsort(start[positions].asi8, kind='stable')]
        splits[name] = _subset(samples, positions)
    # Strong audit: no target timestamp from one partition occurs in another.
    spans = {}
    for name, part in splits.items():
        s = pd.DatetimeIndex(part['target_start'])
        e = s + (part['Y'].shape[1] - 1) * dt
        spans[name] = (s.min(), e.max())
    assert spans['train'][1] < spans['val'][0]
    assert spans['val'][1] < spans['test'][0]
    assert spans['train'][1] < a <= spans['val'][0]
    assert spans['val'][1] < b <= spans['test'][0]
    if verbose:
        print('Target-period split (exclusive boundaries):')
        print(f'  train_end={a}, val_end={b}')
        for name, (lo, hi) in spans.items():
            print(f'  {name:5s}: {len(splits[name]["Y"]):6d} samples; target span {lo} .. {hi}')
        print(f'  Discarded boundary-crossing samples: {n - sum(len(s["Y"]) for s in splits.values())}')
        print('  Verified: target timestamp spans do not overlap across partitions.')
    return splits


def boundaries_from_raw_index(df: pd.DataFrame, train_ratio=0.80, val_ratio=0.05):
    """Choose fixed raw-calendar boundaries independent of forecasting samples.

    Uses time duration, not the number of usable windows. For other leads,
    histories or models, reuse the SAME returned timestamps.
    """
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex) or len(idx) < 2:
        raise ValueError('df must have a nonempty DatetimeIndex')
    if not (0 < train_ratio < 1 and 0 < val_ratio < 1 and train_ratio + val_ratio < 1):
        raise ValueError('Invalid ratios')
    lo, hi = idx.min(), idx.max()
    duration = hi - lo
    return lo + train_ratio * duration, lo + (train_ratio + val_ratio) * duration
