"""RUL prediction curves from a run's predictions.csv (cf. TFG-Mamba Fig. 12)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from rul.data import EVAL_BEARINGS  # noqa: E402


def plot_predictions(run_dir: Path | str, split: str = "test", out: Path | str | None = None) -> Path:
    """One panel per bearing: true RUL and the mean prediction over seeds (band = min..max)."""
    run_dir = Path(run_dir)
    df = pd.read_csv(run_dir / "predictions.csv")
    df = df[df["split"] == split]
    if df.empty:
        raise ValueError(f"no {split} predictions in {run_dir}")
    bearings = [b for b in EVAL_BEARINGS if b in set(df["bearing"].astype(str))]
    cols = 4
    rows = (len(bearings) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.8 * rows), squeeze=False)
    for ax, bearing in zip(axes.flat, bearings):
        part = df[df["bearing"].astype(str) == bearing]
        grouped = part.groupby("snapshot_index")
        t = grouped["true_rul"].first()
        p = grouped["pred_rul"]
        ax.plot(t.index, t.values, color="black", lw=1.2, label="true")
        ax.plot(t.index, p.mean().values, color="tab:blue", lw=1.0, label="pred (seed mean)")
        if part["seed"].nunique() > 1:
            ax.fill_between(t.index, p.min().values, p.max().values, color="tab:blue", alpha=0.2, lw=0)
        ax.set_title(f"Bearing{bearing}")
        ax.set_xlabel("snapshot (min)")
        ax.set_ylabel("RUL")
        ax.set_ylim(-0.2, 1.3)
    for ax in list(axes.flat)[len(bearings):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=8)
    fig.tight_layout()
    out_path = Path(out) if out else run_dir / f"curves_{split}.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
