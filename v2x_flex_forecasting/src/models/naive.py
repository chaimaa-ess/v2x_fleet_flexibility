"""
Origin-aware seasonal naive baseline.

For each target timestamp, predict using the most recent
observation at the same time of day that was available
at or before the forecast origin.

This prevents future-information leakage for different
forecast lead times k.
"""

import numpy as np
import pandas as pd


def predict_origin_aware_naive(
    df,
    origins,
    target_start,
    target_cols,
    horizon=96,
    freq="15min",
    seasonal_period="1D",
    max_lookback_days=14,
):
    """
    Generate origin-aware seasonal-naive predictions.

    For each target timestamp T and forecast origin O:

        candidate = T - seasonal_period

    If candidate > O, the candidate would not have been
    known at forecast time, so move back another seasonal
    period until:

        candidate <= O

    If that timestamp is missing from the dataframe, continue
    moving backwards until a valid historical observation is
    found, up to max_lookback_days.

    Parameters
    ----------
    df : pd.DataFrame
        Original dataframe with DatetimeIndex.

    origins : array-like
        Forecast origin for each sample.

    target_start : array-like
        Start timestamp of each forecast horizon.

    target_cols : list[str]
        Target columns, e.g.
        ["Pmax", "Cmax", "Cmin"].

    horizon : int
        Number of forecast steps.
        Default = 96.

    freq : str
        Sampling interval.
        Default = "15min".

    seasonal_period : str
        Seasonal lag.
        Default = "1D".

    max_lookback_days : int
        Maximum number of daily periods to search backwards
        when timestamps are missing.

    Returns
    -------
    y_pred : np.ndarray
        Shape:
        (n_samples, horizon, n_targets)

    lag_days : np.ndarray
        Number of seasonal periods used for every prediction.
        Shape:
        (n_samples, horizon)

        For example:
            1 = previous day
            2 = two days earlier

        NaN means no valid historical value was found.
    """

    df = df.copy()

    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    origins = pd.to_datetime(origins)
    target_start = pd.to_datetime(target_start)

    if len(origins) != len(target_start):
        raise ValueError(
            "origins and target_start must have the same length."
        )

    dt = pd.Timedelta(freq)
    seasonal_delta = pd.Timedelta(seasonal_period)

    n_samples = len(target_start)
    n_targets = len(target_cols)

    y_pred = np.full(
        (n_samples, horizon, n_targets),
        np.nan,
        dtype=float,
    )

    lag_days = np.full(
        (n_samples, horizon),
        np.nan,
        dtype=float,
    )

    # --------------------------------------------------------
    # Generate predictions
    # --------------------------------------------------------

    for i in range(n_samples):

        origin = pd.Timestamp(origins[i])
        start = pd.Timestamp(target_start[i])

        for h in range(horizon):

            target_time = (
                start
                + h * dt
            )

            # Start with the same time one seasonal period ago
            candidate_time = (
                target_time
                - seasonal_delta
            )

            periods_back = 1

            # ------------------------------------------------
            # Leakage protection
            #
            # Candidate MUST be available at forecast origin.
            # ------------------------------------------------

            while (
                candidate_time > origin
                and periods_back < max_lookback_days
            ):

                candidate_time -= seasonal_delta
                periods_back += 1

            # ------------------------------------------------
            # Search backwards if timestamp is missing
            # ------------------------------------------------

            while periods_back <= max_lookback_days:

                if (
                    candidate_time <= origin
                    and candidate_time in df.index
                ):

                    values = df.loc[
                        candidate_time,
                        target_cols,
                    ]

                    values = np.asarray(
                        values,
                        dtype=float,
                    )

                    # Ensure valid numerical values
                    if np.all(
                        np.isfinite(values)
                    ):

                        y_pred[i, h, :] = values

                        lag_days[i, h] = (
                            periods_back
                        )

                        break

                candidate_time -= seasonal_delta
                periods_back += 1

    return y_pred, lag_days