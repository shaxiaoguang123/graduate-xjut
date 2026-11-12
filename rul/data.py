"""XJTU-SY raw data access.

Snapshots are stored as ``<n>.csv`` (n = minute index starting at 1) with two
columns: horizontal and vertical vibration. Files must be ordered numerically.
The dataset directory is read-only; nothing in this module writes to it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_ROOT = Path("date/Data/XJTU-SY_Bearing_Datasets")
SAMPLING_RATE = 25_600
SNAPSHOT_LENGTH = 32_768

CONDITION_DIRS = {"1": "35Hz12kN", "2": "37.5Hz11kN", "3": "40Hz10kN"}
CHANNEL_COLUMNS = {
    "horizontal": "Horizontal_vibration_signals",
    "vertical": "Vertical_vibration_signals",
}

ALL_BEARINGS: tuple[str, ...] = tuple(f"{c}_{i}" for c in "123" for i in range(1, 6))
EXCLUDED_BEARINGS: tuple[str, ...] = ("1_4", "3_2")
EVAL_BEARINGS: tuple[str, ...] = tuple(b for b in ALL_BEARINGS if b not in EXCLUDED_BEARINGS)


def condition_of(bearing: str) -> str:
    """Operating-condition id ("1", "2" or "3") of a bearing such as "2_3"."""
    _check_bearing(bearing)
    return bearing.split("_")[0]


def bearing_dir(bearing: str, root: Path | str = DATA_ROOT) -> Path:
    _check_bearing(bearing)
    return Path(root) / CONDITION_DIRS[condition_of(bearing)] / f"Bearing{bearing}"


def snapshot_files(bearing: str, root: Path | str = DATA_ROOT) -> list[Path]:
    """All snapshot files of a bearing, sorted by their numeric minute index.

    Raises if the indices are not exactly 1..n (a missing file would silently
    shift every RUL label otherwise).
    """
    directory = bearing_dir(bearing, root)
    files = [p for p in directory.iterdir() if p.suffix == ".csv"]
    if not files:
        raise FileNotFoundError(f"no snapshot files in {directory}")
    files.sort(key=lambda p: int(p.stem))
    indices = [int(p.stem) for p in files]
    if indices != list(range(1, len(files) + 1)):
        raise ValueError(f"snapshot indices of {bearing} are not contiguous 1..{len(files)}")
    return files


def read_snapshot(path: Path | str, channel: str = "horizontal") -> np.ndarray:
    """One snapshot as a float64 vector."""
    column = CHANNEL_COLUMNS[channel]
    frame = pd.read_csv(path, usecols=[column], dtype=np.float64)
    return frame[column].to_numpy(dtype=np.float64, copy=True)


def _check_bearing(bearing: str) -> None:
    if bearing not in ALL_BEARINGS:
        raise ValueError(f"unknown bearing {bearing!r}; expected one of {ALL_BEARINGS}")
