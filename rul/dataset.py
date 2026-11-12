"""Degradation-phase samples: FPT filtering, RUL labels, train-only scaling, windows.

A sample for snapshot t (t >= FPT) is the feature sequence of degradation
snapshots [t-L+1, t]. Only degradation-phase snapshots are used (the paper
filters out the healthy phase); positions before the FPT are left-padded with
zeros and marked False in the mask. Windows end at t, so they never contain
future data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rul import features, fpt


@dataclass(frozen=True)
class FptConfig:
    method: str = "paper_table"  # "paper_table" or "rms_3sigma"
    n_healthy: int = 30
    n_sigma: float = 3.0
    n_consecutive: int = 5


@dataclass
class BearingData:
    bearing: str
    fpt: int  # 1-based snapshot index
    fpt_triggered: bool
    snapshot_index: np.ndarray  # (n,) 1-based indices fpt..last
    features: np.ndarray  # (n, D) raw (unscaled) features
    labels: np.ndarray  # (n,) RUL in [0, 1]


def resolve_fpt(bearing: str, all_features: np.ndarray, config: FptConfig) -> tuple[int, bool]:
    if config.method == "paper_table":
        return fpt.paper_fpt(bearing), True
    if config.method == "rms_3sigma":
        result = fpt.detect_fpt(
            all_features[:, features.RMS_INDEX],
            n_healthy=config.n_healthy,
            n_sigma=config.n_sigma,
            n_consecutive=config.n_consecutive,
        )
        return result.index, result.triggered
    raise ValueError(f"unknown FPT method {config.method!r}")


def degradation_data(bearing: str, all_features: np.ndarray, fpt_config: FptConfig) -> BearingData:
    """Keep snapshots fpt..last and attach linear RUL labels."""
    n = all_features.shape[0]
    first, triggered = resolve_fpt(bearing, all_features, fpt_config)
    labels = fpt.rul_labels(n, first)
    return BearingData(
        bearing=bearing,
        fpt=first,
        fpt_triggered=triggered,
        snapshot_index=np.arange(first, n + 1),
        features=all_features[first - 1 :].copy(),
        labels=labels,
    )


@dataclass
class MinMaxScaler:
    """Per-feature min-max scaling; fitted only on the arrays passed to ``fit``."""

    data_min: np.ndarray | None = field(default=None)
    data_max: np.ndarray | None = field(default=None)

    def fit(self, arrays: list[np.ndarray]) -> "MinMaxScaler":
        stacked = np.concatenate(arrays, axis=0)
        self.data_min = stacked.min(axis=0)
        self.data_max = stacked.max(axis=0)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.data_min is None or self.data_max is None:
            raise RuntimeError("scaler is not fitted")
        span = self.data_max - self.data_min
        span = np.where(span > 0, span, 1.0)
        return (x - self.data_min) / span

    def state(self) -> dict[str, list[float]]:
        assert self.data_min is not None and self.data_max is not None
        return {"min": self.data_min.tolist(), "max": self.data_max.tolist()}


def make_windows(x: np.ndarray, length: int) -> tuple[np.ndarray, np.ndarray]:
    """Causal windows ending at every row of ``x``.

    Returns ``windows`` of shape (n, length, D) (float32, left zero padding) and a
    boolean ``mask`` of shape (n, length) that is True on real rows.
    """
    if length < 1:
        raise ValueError("window length must be >= 1")
    n, d = x.shape
    padded = np.concatenate([np.zeros((length - 1, d), dtype=x.dtype), x], axis=0)
    valid = np.concatenate([np.zeros(length - 1, dtype=bool), np.ones(n, dtype=bool)])
    idx = np.arange(n)[:, None] + np.arange(length)[None, :]
    return padded[idx].astype(np.float32), valid[idx]


@dataclass
class WindowedSet:
    bearing: str
    windows: np.ndarray  # (n, L, D) float32
    mask: np.ndarray  # (n, L) bool
    labels: np.ndarray  # (n,) float32
    snapshot_index: np.ndarray  # (n,)


def windowed(data: BearingData, scaler: MinMaxScaler, length: int) -> WindowedSet:
    scaled = scaler.transform(data.features)
    windows, mask = make_windows(scaled, length)
    return WindowedSet(
        bearing=data.bearing,
        windows=windows,
        mask=mask,
        labels=data.labels.astype(np.float32),
        snapshot_index=data.snapshot_index,
    )
