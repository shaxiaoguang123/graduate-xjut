"""Regression metrics on normalised RUL."""

from __future__ import annotations

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    return float(np.sqrt(np.mean(diff**2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    return float(np.mean(np.abs(diff)))


def mean_std(values: list[float]) -> tuple[float, float]:
    """Mean and sample standard deviation (0 for a single value)."""
    arr = np.asarray(values, dtype=np.float64)
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return float(arr.mean()), std
