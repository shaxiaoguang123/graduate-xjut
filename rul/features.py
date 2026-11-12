"""Per-snapshot statistical features (TFG-Mamba, Tables 3 and 4).

Two formula sets are provided:

``printed``
    Exactly the formulas printed in the paper (default). Several differ from the
    usual textbook definitions, e.g. skewness divides by STD**4, kurtosis is the
    raw 4th central moment, the margin factor has no square root, and "Peak" is
    identical to "Peak to peak". See papers/tfg-mamba/notes.md.
``standard``
    The conventional definitions (Lei et al. frequency features, normalised
    moments). Used only as a documented variant, never silently.

Every feature depends only on one snapshot, so the transform is causal.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from rul.data import DATA_ROOT, SAMPLING_RATE, read_snapshot, snapshot_files

FEATURE_VERSION = "v1"
FORMULAS = ("printed", "standard")
DEFAULT_CACHE_DIR = Path("artifacts/features")

TIME_FEATURE_NAMES: tuple[str, ...] = (
    "mean",
    "std",
    "rms",
    "skewness",
    "kurtosis",
    "shape_factor",
    "crest_factor",
    "impulse_factor",
    "margin_factor",
    "peak",
    "peak_to_peak",
    "shannon_entropy",
    "log_energy_entropy",
)
FREQ_FEATURE_NAMES: tuple[str, ...] = tuple(f"FF{i}" for i in range(1, 14))
FEATURE_NAMES: tuple[str, ...] = TIME_FEATURE_NAMES + FREQ_FEATURE_NAMES
N_TIME_FEATURES = len(TIME_FEATURE_NAMES)
N_FREQ_FEATURES = len(FREQ_FEATURE_NAMES)
RMS_INDEX = TIME_FEATURE_NAMES.index("rms")


def _check_formula(formula: str) -> None:
    if formula not in FORMULAS:
        raise ValueError(f"formula must be one of {FORMULAS}, got {formula!r}")


def time_features(x: np.ndarray, formula: str = "printed") -> np.ndarray:
    """13 time-domain features of one snapshot (Table 3)."""
    _check_formula(formula)
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    mean = x.mean()
    centered = x - mean
    std = np.sqrt(np.sum(centered**2) / (n - 1))
    rms = np.sqrt(np.mean(x**2))
    abs_x = np.abs(x)
    abs_mean = abs_x.mean()
    max_abs = abs_x.max()
    m3 = np.mean(centered**3)
    m4 = np.mean(centered**4)
    peak_to_peak = x.max() - x.min()

    if formula == "printed":
        skewness = m3 / std**4
        kurtosis = m4
        margin = max_abs / abs_mean**2
        peak = peak_to_peak
    else:
        skewness = m3 / std**3
        kurtosis = m4 / std**4
        margin = max_abs / np.mean(np.sqrt(abs_x)) ** 2
        peak = max_abs

    # s_i is not defined in the paper; we use the raw samples (MATLAB wentropy
    # convention) with 0*log(0) = 0 and log(0) terms skipped.
    s2 = x**2
    nonzero = s2 > 0
    shannon = -np.sum(s2[nonzero] * np.log(s2[nonzero]))
    log_energy = np.sum(np.log(s2[nonzero]))

    return np.array(
        [
            mean,
            std,
            rms,
            skewness,
            kurtosis,
            rms / abs_mean,
            max_abs / rms,
            max_abs / abs_mean,
            margin,
            peak,
            peak_to_peak,
            shannon,
            log_energy,
        ],
        dtype=np.float64,
    )


def amplitude_spectrum(x: np.ndarray, fs: float = SAMPLING_RATE) -> tuple[np.ndarray, np.ndarray]:
    """Single-sided amplitude spectrum: K = N/2 lines, f_k = k * fs / N, y(k) = 2|X_k|/N."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    lines = n // 2
    y = np.abs(np.fft.rfft(x))[:lines] * (2.0 / n)
    f = np.arange(lines, dtype=np.float64) * (fs / n)
    return f, y


def freq_features(x: np.ndarray, fs: float = SAMPLING_RATE, formula: str = "printed") -> np.ndarray:
    """13 frequency-domain features of one snapshot (Table 4)."""
    _check_formula(formula)
    f, y = amplitude_spectrum(x, fs)
    k = y.size
    ff1 = np.sum(y) / k
    d = y - ff1
    ff2 = np.sum(d**2) / (k - 1)
    if formula == "printed":
        ff3 = np.sum(d**3) / (k / ff2)
        ff4 = np.sum(d**4) / (k * ff2**3)
    else:
        ff3 = np.sum(d**3) / (k * ff2**1.5)
        ff4 = np.sum(d**4) / (k * ff2**2)

    sum_y = np.sum(y)
    ff5 = np.sum(f * y) / sum_y
    df = f - ff5
    ff6 = np.sqrt(np.sum(df**2 * y) / k)
    ff7 = np.sqrt(np.sum(f**2 * y) / sum_y)
    s2 = np.sum(f**2 * y)
    s4 = np.sum(f**4 * y)
    ff8 = s4 / s2 if formula == "printed" else np.sqrt(s4 / s2)
    ff9 = s2 / np.sqrt(s4 * sum_y)
    ff10 = ff6 / ff5
    ff11 = np.sum(df**3 * y) / (k * ff6**3)
    ff12 = np.sum(df**4 * y) / (k * ff6**4)
    root_term = np.sum(np.sqrt(np.abs(df)) * y)
    if formula == "printed":
        ff13 = root_term / (k / np.sqrt(ff6))
    else:
        ff13 = root_term / (k * np.sqrt(ff6))

    return np.array(
        [ff1, ff2, ff3, ff4, ff5, ff6, ff7, ff8, ff9, ff10, ff11, ff12, ff13],
        dtype=np.float64,
    )


def snapshot_features(x: np.ndarray, fs: float = SAMPLING_RATE, formula: str = "printed") -> np.ndarray:
    """26 features: 13 time-domain followed by 13 frequency-domain."""
    return np.concatenate([time_features(x, formula), freq_features(x, fs, formula)])


def _features_of_file(args: tuple[str, str]) -> np.ndarray:
    path, channel = args
    x = read_snapshot(path, channel)
    return np.stack([snapshot_features(x, formula=formula) for formula in FORMULAS])


def cache_path(bearing: str, cache_dir: Path | str = DEFAULT_CACHE_DIR, channel: str = "horizontal") -> Path:
    return Path(cache_dir) / f"{FEATURE_VERSION}_{channel}" / f"{bearing}.npz"


def build_bearing_features(
    bearing: str,
    root: Path | str = DATA_ROOT,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    channel: str = "horizontal",
    workers: int = 1,
    overwrite: bool = False,
) -> Path:
    """Compute both formula sets for every snapshot of a bearing and cache them."""
    out = cache_path(bearing, cache_dir, channel)
    if out.exists() and not overwrite:
        return out
    files = snapshot_files(bearing, root)
    tasks = [(str(p), channel) for p in files]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(_features_of_file, tasks, chunksize=16))
    else:
        rows = [_features_of_file(t) for t in tasks]
    stacked = np.stack(rows)  # (n_snapshots, n_formulas, 26)
    if not np.all(np.isfinite(stacked)):
        raise ValueError(f"non-finite features for bearing {bearing}")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez(
        tmp,
        printed=stacked[:, 0, :],
        standard=stacked[:, 1, :],
        feature_names=np.array(FEATURE_NAMES),
        version=np.array(FEATURE_VERSION),
        channel=np.array(channel),
        n_snapshots=np.array(len(files)),
    )
    os.replace(tmp, out)
    return out


def load_features(
    bearing: str,
    formula: str = "printed",
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    channel: str = "horizontal",
) -> np.ndarray:
    """Cached (n_snapshots, 26) feature matrix; row i is snapshot i + 1."""
    _check_formula(formula)
    path = cache_path(bearing, cache_dir, channel)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `python -m rul.cli features` first")
    with np.load(path) as data:
        if str(data["version"]) != FEATURE_VERSION:
            raise ValueError(f"{path} has feature version {data['version']}, expected {FEATURE_VERSION}")
        return np.array(data[formula], dtype=np.float64)
