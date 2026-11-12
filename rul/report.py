"""Markdown summaries generated from run directories (never typed by hand)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from rul.data import EVAL_BEARINGS

# TFG-Mamba Table 7 (TFG-Mamba column): bearing -> (RMSE, MAE) on normalised RUL.
TFG_MAMBA_TABLE7: dict[str, tuple[float, float]] = {
    "1_1": (0.078, 0.066),
    "1_2": (0.079, 0.065),
    "1_3": (0.089, 0.074),
    "1_5": (0.083, 0.058),
    "2_1": (0.093, 0.078),
    "2_2": (0.053, 0.041),
    "2_3": (0.081, 0.067),
    "2_4": (0.078, 0.060),
    "2_5": (0.065, 0.051),
    "3_1": (0.089, 0.064),
    "3_3": (0.059, 0.048),
    "3_4": (0.108, 0.087),
    "3_5": (0.105, 0.085),
}
TFG_MAMBA_TABLE7_PRINTED_MEAN = (0.082, 0.071)


def load_run(run_dir: Path | str) -> dict[str, Any]:
    run_dir = Path(run_dir)
    return {
        "dir": run_dir,
        "config": yaml.safe_load((run_dir / "config.yaml").read_text()),
        "meta": json.loads((run_dir / "meta.json").read_text()),
        "metrics": json.loads((run_dir / "metrics.json").read_text()),
    }


def selection_table(run_dirs: list[Path | str]) -> str:
    """Validation-only comparison of runs (e.g. window lengths)."""
    lines = ["| 实验 | 窗口 L | 协议 | 种子 | 验证 RMSE（13 折平均） | 验证 MAE | 平均最佳轮数 | 运行目录 |", "|---|---|---|---|---|---|---|---|"]
    for rd in run_dirs:
        run = load_run(rd)
        s = run["metrics"]["summary"]
        jobs = run["metrics"]["jobs"]
        best = sum(j["best_epoch"] for j in jobs) / len(jobs)
        lines.append(
            f"| {run['config']['experiment']} | {run['config']['window']['length']} | {run['config']['protocol']} "
            f"| {','.join(map(str, run['meta']['seeds']))} | {s['val_rmse_mean']:.4f} | {s['val_mae_mean']:.4f} "
            f"| {best:.1f} | `{run['dir']}` |"
        )
    return "\n".join(lines)


def test_table(run_dir: Path | str, compare_paper: bool = True) -> str:
    """Per-bearing test metrics (mean ± std over seeds) next to the paper's Table 7."""
    run = load_run(run_dir)
    s = run["metrics"]["summary"]
    per = s["per_bearing"]
    head = "| 轴承 | RMSE（均值 ± 标准差） | MAE（均值 ± 标准差） |"
    sep = "|---|---|---|"
    if compare_paper:
        head += " 论文 RMSE | 论文 MAE | RMSE 差值 |"
        sep += "---|---|---|"
    lines = [head, sep]
    for b in EVAL_BEARINGS:
        if b not in per:
            continue
        e = per[b]
        row = f"| {b} | {e['rmse_mean']:.3f} ± {e['rmse_std']:.3f} | {e['mae_mean']:.3f} ± {e['mae_std']:.3f} |"
        if compare_paper:
            pr, pm = TFG_MAMBA_TABLE7[b]
            row += f" {pr:.3f} | {pm:.3f} | {e['rmse_mean'] - pr:+.3f} |"
        lines.append(row)
    total = f"| 平均（{len(per)} 个轴承，按种子求均值 ± 标准差） | {s['test_rmse_mean']:.3f} ± {s['test_rmse_std']:.3f} | {s['test_mae_mean']:.3f} ± {s['test_mae_std']:.3f} |"
    if compare_paper:
        pr = sum(v[0] for v in TFG_MAMBA_TABLE7.values()) / len(TFG_MAMBA_TABLE7)
        pm = sum(v[1] for v in TFG_MAMBA_TABLE7.values()) / len(TFG_MAMBA_TABLE7)
        total += f" {pr:.4f}（表中写 {TFG_MAMBA_TABLE7_PRINTED_MEAN[0]}） | {pm:.4f}（表中写 {TFG_MAMBA_TABLE7_PRINTED_MEAN[1]}） | {s['test_rmse_mean'] - pr:+.3f} |"
    lines.append(total)
    val = f"\n验证集（每折一个验证轴承，13 折平均）：RMSE {s['val_rmse_mean']:.3f} ± {s['val_rmse_std']:.3f}，MAE {s['val_mae_mean']:.3f} ± {s['val_mae_std']:.3f}。"
    meta = run["meta"]
    info = (
        f"\n运行目录 `{run['dir']}`；协议 `{meta['protocol']}` {meta['protocol_version']}；设备 `{meta['device']}`；"
        f"种子 {meta['seeds']}；git `{meta['git_commit'][:7]}`{'（有未提交改动）' if meta['git_dirty'] else ''}；"
        f"参数量 {meta.get('n_params', '?')}；torch {meta['torch']}。"
    )
    return "\n".join(lines) + "\n" + val + info
