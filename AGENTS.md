# Repository Guidelines

> This workspace was reinitialized locally after its source tree was cleared. The original dataset remains in `date/`; the layout and commands below describe the known project when code is restored.

## Project Structure & Module Organization

- `rul/` contains the Python package: data preparation, causal models, training/evaluation, metrics, research datasets, and CLI.
- `scripts/` holds experiment runners and report-generation utilities; `tests/` contains `unittest` suites named `test_*.py`.
- `configs/splits/` stores whole-bearing split manifests. `docs/` records protocols and findings; `artifacts/` and `runs/` hold generated summaries and experiment outputs.
- Raw XJTU-SY files live in `date/Data/XJTU-SY_Bearing_Datasets/`. Preserve this directory. Keep feature caches, checkpoints, and run outputs out of source control.

## Build, Test, and Development Commands

Use the existing Conda environment `pt` (Python 3.12). There is no separate build step documented.

```bash
conda run --no-capture-output -n pt python -m rul.cli --help
conda run --no-capture-output -n pt python -m unittest discover -s tests -v
conda run --no-capture-output -n pt python -m rul.cli inspect
conda run --no-capture-output -n pt python -m rul.cli prepare --workers 4
```

Use `--device mps` for native Apple Silicon experiments. An explicit MPS request must fail clearly if unavailable; do not silently fall back to CPU.

## Coding Style & Naming Conventions

Use four-space indentation, Python type hints, `snake_case` for modules/functions, and `PascalCase` for classes. Keep data transforms causal and fit learned scaling only on training bearings. Match nearby code; no formatter or linter was configured in the prior checkout.

## Testing Guidelines

Use standard-library `unittest`; add focused cases under `tests/test_<area>.py`. Test split isolation, padding/masks, causal prefix behavior, and model/device paths when relevant. Synthetic unit tests may use CPU; report the actual device for research runs. No coverage threshold was established.

## Commit & Pull Request Guidelines

Existing commits commonly use `<scope>: <summary>`, such as `research:`, `protocol:`, `report:`, and `artifacts:`. Keep summaries concise. A change proposal should explain its purpose, protocol/split, seeds and device, validation performed, and limitations. Clearly label validation versus held-out results; never tune from held-out metrics. Do not commit raw data, feature caches, checkpoints, or generated run directories.
