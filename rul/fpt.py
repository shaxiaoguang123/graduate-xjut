"""First prediction time (FPT) and RUL labels.

Indices are 1-based snapshot indices: index k is file ``k.csv``. The paper's
"minute k" is mapped to ``k.csv`` (see papers/tfg-mamba/notes.md A12).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# TFG-Mamba Table 1: bearing -> (total lifetime [min], FPT [min]).
PAPER_TABLE1: dict[str, tuple[int, int]] = {
    "1_1": (123, 1),
    "1_2": (160, 22),
    "1_3": (157, 49),
    "1_4": (121, 63),
    "1_5": (51, 1),
    "2_1": (490, 429),
    "2_2": (160, 37),
    "2_3": (532, 280),
    "2_4": (41, 21),
    "2_5": (338, 106),
    "3_1": (2537, 2235),
    "3_2": (2495, 924),
    "3_3": (370, 324),
    "3_4": (1514, 930),
    "3_5": (113, 1),
}


@dataclass(frozen=True)
class FptResult:
    index: int  # 1-based snapshot index of the FPT
    triggered: bool  # False when the rule never fired and the fallback was used
    mu: float
    sigma: float
    threshold: float


def paper_fpt(bearing: str) -> int:
    """FPT from the paper's Table 1, as a 1-based snapshot index."""
    return PAPER_TABLE1[bearing][1]


def detect_fpt(
    rms: np.ndarray,
    n_healthy: int = 30,
    n_sigma: float = 3.0,
    n_consecutive: int = 5,
) -> FptResult:
    """3-sigma RMS rule (paper Eqs. 21-22), evaluated causally.

    mu and sigma come from the first ``n_healthy`` snapshots (sigma with N-1,
    matching the paper's STD definition). S_i = 1 when rms_i >= mu + n_sigma*sigma.
    The FPT is the first index i >= n_healthy at which the last ``n_consecutive``
    states are all 1; that decision uses only snapshots 1..i. If the rule never
    fires, the FPT falls back to n_healthy + 1 and ``triggered`` is False.
    """
    rms = np.asarray(rms, dtype=np.float64)
    if n_healthy < 2:
        raise ValueError("n_healthy must be at least 2")
    if rms.size <= n_healthy:
        raise ValueError(f"need more than {n_healthy} snapshots, got {rms.size}")
    baseline = rms[:n_healthy]
    mu = float(baseline.mean())
    sigma = float(baseline.std(ddof=1))
    threshold = mu + n_sigma * sigma
    above = rms >= threshold
    run = 0
    for i in range(rms.size):  # i is 0-based; snapshot index is i + 1
        run = run + 1 if above[i] else 0
        if run >= n_consecutive and i + 1 >= n_healthy:
            return FptResult(i + 1, True, mu, sigma, threshold)
    return FptResult(n_healthy + 1, False, mu, sigma, threshold)


def rul_labels(n_snapshots: int, fpt: int) -> np.ndarray:
    """Linear RUL for snapshots fpt..n_snapshots: 1 at the FPT, 0 at the last snapshot."""
    if not 1 <= fpt < n_snapshots:
        raise ValueError(f"fpt must be in [1, {n_snapshots - 1}], got {fpt}")
    t = np.arange(fpt, n_snapshots + 1, dtype=np.float64)
    return (n_snapshots - t) / (n_snapshots - fpt)
