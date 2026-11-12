"""Command-line entry point: ``python -m rul.cli <command>``."""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence

from rul import data, features


def _cmd_features(args: argparse.Namespace) -> None:
    bearings = args.bearings or list(data.ALL_BEARINGS)
    for bearing in bearings:
        start = time.time()
        path = features.build_bearing_features(
            bearing,
            root=args.root,
            cache_dir=args.cache_dir,
            channel=args.channel,
            workers=args.workers,
            overwrite=args.overwrite,
        )
        print(f"{bearing}: {path} ({time.time() - start:.1f}s)", flush=True)


def _cmd_fpt(args: argparse.Namespace) -> None:
    from rul import fpt

    print(f"| 轴承 | 快照数 | 论文寿命 | 论文 FPT | 3σ FPT (N_h={args.n_healthy}) | 触发 | 退化段长度 (论文 / 3σ) |")
    print("|---|---|---|---|---|---|---|")
    for bearing in data.ALL_BEARINGS:
        feats = features.load_features(bearing, "printed", args.cache_dir, args.channel)
        rms = feats[:, features.RMS_INDEX]
        result = fpt.detect_fpt(rms, n_healthy=args.n_healthy, n_sigma=args.n_sigma, n_consecutive=args.n_consecutive)
        life, paper = fpt.PAPER_TABLE1[bearing]
        n = rms.size
        mark = "" if bearing in data.EVAL_BEARINGS else "（排除）"
        print(
            f"| {bearing}{mark} | {n} | {life} | {paper} | {result.index} | {'是' if result.triggered else '否'} "
            f"| {n - paper + 1} / {n - result.index + 1} |"
        )


def _cmd_train(args: argparse.Namespace) -> None:
    from rul import train

    overrides = list(args.set or [])
    if args.experiment:
        overrides.append(f"experiment={args.experiment}")
    cfg = train.load_config(args.config, overrides)
    train.run_experiment(
        cfg,
        device=args.device,
        folds=args.folds,
        seeds=args.seeds,
        workers=args.workers,
        evaluate_test=not args.no_test,
        out_root=args.out_root,
    )


def _cmd_select(args: argparse.Namespace) -> None:
    from rul import report

    print(report.selection_table(args.run_dirs))


def _cmd_report(args: argparse.Namespace) -> None:
    from rul import report

    print(report.test_table(args.run_dir, compare_paper=not args.no_paper))


def _cmd_plot(args: argparse.Namespace) -> None:
    from rul import plots

    print(plots.plot_predictions(args.run_dir, split=args.split, out=args.out))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m rul.cli")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("features", help="compute and cache per-snapshot features")
    p.add_argument("--bearings", nargs="*", help="default: all 15 bearings")
    p.add_argument("--root", default=str(data.DATA_ROOT))
    p.add_argument("--cache-dir", default=str(features.DEFAULT_CACHE_DIR))
    p.add_argument("--channel", default="horizontal", choices=sorted(data.CHANNEL_COLUMNS))
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=_cmd_features)

    p = sub.add_parser("fpt", help="compare 3-sigma FPT with the paper's Table 1")
    p.add_argument("--cache-dir", default=str(features.DEFAULT_CACHE_DIR))
    p.add_argument("--channel", default="horizontal", choices=sorted(data.CHANNEL_COLUMNS))
    p.add_argument("--n-healthy", type=int, default=30)
    p.add_argument("--n-sigma", type=float, default=3.0)
    p.add_argument("--n-consecutive", type=int, default=5)
    p.set_defaults(func=_cmd_fpt)

    p = sub.add_parser("train", help="train and evaluate every (fold, seed) of an experiment config")
    p.add_argument("--config", required=True)
    p.add_argument("--device", required=True, choices=["mps", "cuda", "cpu"])
    p.add_argument("--folds", nargs="*", help="test bearings to run (default: all folds)")
    p.add_argument("--seeds", nargs="*", type=int, help="default: seeds from the config")
    p.add_argument("--workers", type=int, default=1, help="parallel worker processes")
    p.add_argument("--no-test", action="store_true", help="validation only; never compute test metrics")
    p.add_argument("--set", nargs="*", help="config overrides such as window.length=10")
    p.add_argument("--experiment", help="override the experiment name")
    p.add_argument("--out-root", default="runs")
    p.set_defaults(func=_cmd_train)

    p = sub.add_parser("select", help="compare validation metrics of several runs")
    p.add_argument("run_dirs", nargs="+")
    p.set_defaults(func=_cmd_select)

    p = sub.add_parser("report", help="per-bearing test metrics of a run vs. the paper")
    p.add_argument("run_dir")
    p.add_argument("--no-paper", action="store_true", help="do not add the paper's Table 7 columns")
    p.set_defaults(func=_cmd_report)

    p = sub.add_parser("plot", help="plot true vs. predicted RUL curves of a run")
    p.add_argument("run_dir")
    p.add_argument("--split", default="test", choices=["test", "val"])
    p.add_argument("--out")
    p.set_defaults(func=_cmd_plot)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
