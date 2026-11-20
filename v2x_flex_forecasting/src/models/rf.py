"""
Random Forest baseline for V2X flexibility forecasting.

Inputs
------
X_recent : (N, 96, 3)
X_daily  : (N, 6, 3)
X_weekly : (N, 5, 3)

These are flattened and concatenated:

    96*3 + 6*3 + 5*3 = 321 features

Targets
-------
Y : (N, 96, 3)

Targets are flattened to:

    96*3 = 288 outputs

The RandomForestRegressor therefore performs multi-output
regression from 321 input features to 288 forecast outputs.
"""

from pathlib import Path

import joblib
import numpy as np

from sklearn.ensemble import RandomForestRegressor


# ============================================================
# Data preparation
# ============================================================

def prepare_rf_features(
    X_recent,
    X_daily,
    X_weekly,
):
    """
    Flatten and concatenate the three historical inputs.

    Parameters
    ----------
    X_recent : np.ndarray
        Shape (N, 96, 3)

    X_daily : np.ndarray
        Shape (N, 6, 3)

    X_weekly : np.ndarray
        Shape (N, 5, 3)

    Returns
    -------
    X : np.ndarray
        Shape (N, 321)
    """

    X_recent = np.asarray(X_recent)
    X_daily = np.asarray(X_daily)
    X_weekly = np.asarray(X_weekly)

    n_samples = X_recent.shape[0]

    if (
        X_daily.shape[0] != n_samples
        or X_weekly.shape[0] != n_samples
    ):
        raise ValueError(
            "X_recent, X_daily, and X_weekly "
            "must contain the same number of samples."
        )

    X_recent_flat = X_recent.reshape(
        n_samples,
        -1,
    )

    X_daily_flat = X_daily.reshape(
        n_samples,
        -1,
    )

    X_weekly_flat = X_weekly.reshape(
        n_samples,
        -1,
    )

    X = np.concatenate(
        [
            X_recent_flat,
            X_daily_flat,
            X_weekly_flat,
        ],
        axis=1,
    )

    return X


def prepare_rf_targets(Y):
    """
    Flatten forecasting targets for multi-output regression.

    Parameters
    ----------
    Y : np.ndarray
        Shape (N, 96, 3)

    Returns
    -------
    Y_flat : np.ndarray
        Shape (N, 288)
    """

    Y = np.asarray(Y)

    if Y.ndim != 3:
        raise ValueError(
            "Y must have shape "
            "(N, horizon, n_targets)."
        )

    return Y.reshape(
        Y.shape[0],
        -1,
    )


def restore_rf_targets(
    Y_flat,
    horizon=96,
    n_targets=3,
):
    """
    Restore flattened RF predictions to forecasting shape.

    Parameters
    ----------
    Y_flat : np.ndarray
        Shape (N, horizon * n_targets)

    Returns
    -------
    Y : np.ndarray
        Shape (N, horizon, n_targets)
    """

    Y_flat = np.asarray(Y_flat)

    expected_outputs = horizon * n_targets

    if Y_flat.shape[1] != expected_outputs:
        raise ValueError(
            f"Expected {expected_outputs} outputs, "
            f"received {Y_flat.shape[1]}."
        )

    return Y_flat.reshape(
        Y_flat.shape[0],
        horizon,
        n_targets,
    )


# ============================================================
# Model
# ============================================================

def build_random_forest(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=1.0,
    n_jobs=-1,
    random_state=42,
):
    """
    Build the multi-output Random Forest forecasting model.

    Returns
    -------
    sklearn.ensemble.RandomForestRegressor
    """

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        n_jobs=n_jobs,
        random_state=random_state,
    )

    return model


# ============================================================
# Training
# ============================================================

def train_random_forest(
    model,
    X_recent,
    X_daily,
    X_weekly,
    Y,
):
    """
    Train Random Forest.

    Inputs and targets are assumed to have already been
    preprocessed/scaled using the training-set scalers.
    """

    X = prepare_rf_features(
        X_recent,
        X_daily,
        X_weekly,
    )

    Y_flat = prepare_rf_targets(Y)

    if not np.all(np.isfinite(X)):
        raise ValueError(
            "Random Forest input contains NaN "
            "or infinite values."
        )

    if not np.all(np.isfinite(Y_flat)):
        raise ValueError(
            "Random Forest target contains NaN "
            "or infinite values."
        )

    model.fit(
        X,
        Y_flat,
    )

    return model


# ============================================================
# Prediction
# ============================================================

def predict_random_forest(
    model,
    X_recent,
    X_daily,
    X_weekly,
    horizon=96,
    n_targets=3,
):
    """
    Generate RF forecasts.

    Returns
    -------
    Y_pred : np.ndarray
        Shape (N, 96, 3)
    """

    X = prepare_rf_features(
        X_recent,
        X_daily,
        X_weekly,
    )

    Y_pred_flat = model.predict(X)

    Y_pred = restore_rf_targets(
        Y_pred_flat,
        horizon=horizon,
        n_targets=n_targets,
    )

    return Y_pred


# ============================================================
# Save / load
# ============================================================

def save_random_forest(
    model,
    filepath,
):
    """
    Save trained Random Forest model.
    """

    filepath = Path(filepath)

    filepath.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        filepath,
    )

    print(
        f"Random Forest saved to: {filepath}"
    )


def load_random_forest(filepath):
    """
    Load trained Random Forest model.
    """

    return joblib.load(filepath)