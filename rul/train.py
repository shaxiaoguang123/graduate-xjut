"""Training and evaluation of TFG-Mamba under a split manifest.

One *job* is one (fold, seed). Jobs can run in parallel worker processes; the
parent process writes every result file. Run directories follow
docs/protocol.md section 9.
"""

from __future__ import annotations

import copy
import csv
import json
import platform
import random
import socket
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn

from rul import dataset, features, metrics, splits
from rul.models.tfg_mamba import TFGMamba, TFGMambaConfig, count_parameters

# ----------------------------------------------------------------------- config


def load_config(path: Path | str, overrides: list[str] | None = None) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(path).read_text())
    for item in overrides or []:
        key, _, raw = item.partition("=")
        if not _:
            raise ValueError(f"override must look like a.b=value, got {item!r}")
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(raw)
    for section in ("paper", "experiment", "protocol", "protocol_version", "data", "fpt", "window", "model", "train"):
        if section not in cfg:
            raise ValueError(f"config is missing {section!r}")
    if cfg["protocol"] not in ("paper", "strict"):
        raise ValueError("protocol must be 'paper' or 'strict'")
    return cfg


def model_config(cfg: dict[str, Any]) -> TFGMambaConfig:
    m = dict(cfg["model"])
    if "tcn_channels" in m:
        m["tcn_channels"] = tuple(m["tcn_channels"])
    return TFGMambaConfig(**m)


def resolve_device(name: str) -> torch.device:
    """Explicitly requested accelerators must exist; never fall back silently."""
    if name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("--device mps requested but MPS is not available")
    elif name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is not available")
    elif name != "cpu":
        raise ValueError(f"unknown device {name!r}")
    return torch.device(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ------------------------------------------------------------------------- data


@dataclass
class FoldData:
    fold: splits.Fold
    train_x: np.ndarray
    train_mask: np.ndarray
    train_y: np.ndarray
    val: dataset.WindowedSet
    test: dataset.WindowedSet
    scaler_state: dict[str, list[float]]
    fpt: dict[str, dict[str, Any]]


def prepare_fold(cfg: dict[str, Any], fold: splits.Fold) -> FoldData:
    data_cfg = cfg["data"]
    fpt_cfg = dataset.FptConfig(**cfg["fpt"])
    length = int(cfg["window"]["length"])
    bearings = list(fold.train) + [fold.val, fold.test]
    series: dict[str, dataset.BearingData] = {}
    for bearing in bearings:
        feats = features.load_features(bearing, data_cfg["feature_formula"], data_cfg["cache_dir"], data_cfg["channel"])
        series[bearing] = dataset.degradation_data(bearing, feats, fpt_cfg)
    # Scaling parameters come from the training bearings only.
    scaler = dataset.MinMaxScaler().fit([series[b].features for b in fold.train])
    train_sets = [dataset.windowed(series[b], scaler, length) for b in fold.train]
    return FoldData(
        fold=fold,
        train_x=np.concatenate([s.windows for s in train_sets]),
        train_mask=np.concatenate([s.mask for s in train_sets]),
        train_y=np.concatenate([s.labels for s in train_sets]),
        val=dataset.windowed(series[fold.val], scaler, length),
        test=dataset.windowed(series[fold.test], scaler, length),
        scaler_state=scaler.state(),
        fpt={b: {"fpt": s.fpt, "triggered": s.fpt_triggered, "n_samples": int(s.labels.size)} for b, s in series.items()},
    )


# --------------------------------------------------------------------- training


def _predict(model: nn.Module, x: np.ndarray, mask: np.ndarray, device: torch.device, batch: int = 256) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            xb = torch.from_numpy(x[i : i + batch]).to(device)
            mb = torch.from_numpy(mask[i : i + batch]).to(device)
            out.append(model(xb, mb).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def train_job(cfg: dict[str, Any], fold: splits.Fold, seed: int, device_name: str, evaluate_test: bool) -> dict[str, Any]:
    """Train one (fold, seed); early stopping on validation MSE. Returns metrics and predictions."""
    torch.set_num_threads(int(cfg["train"].get("torch_threads", 2)))
    device = resolve_device(device_name)
    data = prepare_fold(cfg, fold)
    set_seed(seed)
    model = TFGMamba(model_config(cfg)).to(device)
    n_params = count_parameters(model)
    train_cfg = cfg["train"]
    if train_cfg.get("optimizer", "adam") != "adam":
        raise ValueError("only adam is implemented")
    if train_cfg.get("loss", "mse") != "mse":
        raise ValueError("only mse is implemented")
    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_cfg["lr"]))
    forward = torch.compile(model) if train_cfg.get("compile", False) else model

    x = torch.from_numpy(data.train_x)
    m = torch.from_numpy(data.train_mask)
    y = torch.from_numpy(data.train_y)
    batch_size = int(train_cfg["batch_size"])
    generator = torch.Generator().manual_seed(seed)
    max_epochs = int(train_cfg["max_epochs"])
    patience = int(train_cfg["patience"])

    best_val = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    history = []
    start = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.randperm(len(x), generator=generator)
        total = 0.0
        for i in range(0, len(order), batch_size):
            idx = order[i : i + batch_size]
            xb, mb, yb = x[idx].to(device), m[idx].to(device), y[idx].to(device)
            loss = torch.mean((forward(xb, mb) - yb) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(idx)
        val_pred = _predict(model, data.val.windows, data.val.mask, device)  # eager: no recompiles
        val_loss = float(np.mean((val_pred - data.val.labels) ** 2))
        history.append({"epoch": epoch, "train_loss": total / len(x), "val_loss": val_loss})
        if val_loss < best_val:
            best_val, best_epoch = val_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= patience:
            break
    model.load_state_dict(best_state)

    result: dict[str, Any] = {
        "fold": fold.name,
        "test_bearing": fold.test,
        "val_bearing": fold.val,
        "seed": seed,
        "n_params": n_params,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "train_seconds": round(time.time() - start, 1),
        "n_train_samples": int(len(x)),
        "fpt": data.fpt,
        "history": history,
    }
    val_pred = _predict(model, data.val.windows, data.val.mask, device)
    result["val"] = {
        "bearing": fold.val,
        "rmse": metrics.rmse(data.val.labels, val_pred),
        "mae": metrics.mae(data.val.labels, val_pred),
        "mse": float(np.mean((val_pred - data.val.labels) ** 2)),
    }
    predictions = [("val", fold.val, int(t), float(a), float(p)) for t, a, p in zip(data.val.snapshot_index, data.val.labels, val_pred)]
    if evaluate_test:
        test_pred = _predict(model, data.test.windows, data.test.mask, device)
        result["test"] = {
            "bearing": fold.test,
            "rmse": metrics.rmse(data.test.labels, test_pred),
            "mae": metrics.mae(data.test.labels, test_pred),
        }
        predictions += [("test", fold.test, int(t), float(a), float(p)) for t, a, p in zip(data.test.snapshot_index, data.test.labels, test_pred)]
    result["predictions"] = predictions
    return result


def _job_entry(args: tuple) -> dict[str, Any]:
    return train_job(*args)


# ------------------------------------------------------------------ run records


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _meta(cfg: dict[str, Any], device: str, seeds: list[int], evaluate_test: bool) -> dict[str, Any]:
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "paper": cfg["paper"],
        "experiment": cfg["experiment"],
        "protocol": cfg["protocol"],
        "protocol_version": cfg["protocol_version"],
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(status),
        "device": device,
        "seeds": seeds,
        "evaluate_test": evaluate_test,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "host": socket.gethostname(),
        "start": datetime.now().isoformat(timespec="seconds"),
    }


def summarize(results: list[dict[str, Any]], evaluate_test: bool) -> dict[str, Any]:
    """Per seed: arithmetic mean over folds (bearings). Across seeds: mean and std."""
    seeds = sorted({r["seed"] for r in results})
    summary: dict[str, Any] = {"per_seed": {}, "per_bearing": {}}
    parts = ["val", "test"] if evaluate_test else ["val"]
    for part in parts:
        for metric in ("rmse", "mae"):
            per_seed = []
            for seed in seeds:
                vals = [r[part][metric] for r in results if r["seed"] == seed]
                per_seed.append(float(np.mean(vals)))
                summary["per_seed"].setdefault(str(seed), {})[f"{part}_{metric}"] = per_seed[-1]
            mean, std = metrics.mean_std(per_seed)
            summary[f"{part}_{metric}_mean"] = mean
            summary[f"{part}_{metric}_std"] = std
    if evaluate_test:
        for r in results:
            entry = summary["per_bearing"].setdefault(r["test_bearing"], {"rmse": [], "mae": []})
            entry["rmse"].append(r["test"]["rmse"])
            entry["mae"].append(r["test"]["mae"])
        for entry in summary["per_bearing"].values():
            entry["rmse_mean"], entry["rmse_std"] = metrics.mean_std(entry.pop("rmse"))
            entry["mae_mean"], entry["mae_std"] = metrics.mean_std(entry.pop("mae"))
    summary["n_jobs"] = len(results)
    return summary


def run_experiment(
    cfg: dict[str, Any],
    device: str,
    folds: list[str] | None = None,
    seeds: list[int] | None = None,
    workers: int = 1,
    evaluate_test: bool = True,
    out_root: Path | str = "runs",
) -> Path:
    resolve_device(device)  # fail fast in the parent
    split = splits.load_split(cfg["data"]["split"])
    selected = [split.fold(t) for t in folds] if folds else list(split.folds)
    seeds = seeds if seeds is not None else [int(s) for s in cfg.get("seeds", [0])]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(out_root) / cfg["paper"] / cfg["protocol"] / cfg["experiment"] / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    meta = _meta(cfg, device, seeds, evaluate_test)
    meta["n_params"] = count_parameters(TFGMamba(model_config(cfg)))
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    if meta["git_dirty"]:
        print("WARNING: tracked files have uncommitted changes; results are not citable", flush=True)

    jobs = [(cfg, fold, seed, device, evaluate_test) for seed in seeds for fold in selected]
    results: list[dict[str, Any]] = []
    pred_path = run_dir / "predictions.csv"
    with pred_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["seed", "fold", "split", "bearing", "snapshot_index", "true_rul", "pred_rul"])

        def record(result: dict[str, Any]) -> None:
            for row in result.pop("predictions"):
                writer.writerow([result["seed"], result["fold"], *row])
            fh.flush()
            results.append(result)
            msg = f"[{len(results)}/{len(jobs)}] {result['fold']} seed={result['seed']} best_epoch={result['best_epoch']} "
            msg += f"val_rmse={result['val']['rmse']:.4f}"
            if evaluate_test:
                msg += f" test_rmse={result['test']['rmse']:.4f} test_mae={result['test']['mae']:.4f}"
            print(msg + f" ({result['train_seconds']}s)", flush=True)
            (run_dir / "metrics.json").write_text(json.dumps({"jobs": results}, indent=2))

        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
                for result in pool.map(_job_entry, jobs):
                    record(result)
        else:
            for job in jobs:
                record(train_job(*job))

    summary = summarize(results, evaluate_test)
    (run_dir / "metrics.json").write_text(json.dumps({"summary": summary, "jobs": results}, indent=2))
    meta["end"] = datetime.now().isoformat(timespec="seconds")
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_bearing"}, indent=2), flush=True)
    print(f"run directory: {run_dir}", flush=True)
    return run_dir

