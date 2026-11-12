import json
import tempfile
import unittest
from pathlib import Path

import yaml

from rul import report


def _write_run(root: Path, with_test: bool = True) -> Path:
    run = root / "run"
    run.mkdir()
    (run / "config.yaml").write_text(yaml.safe_dump({"experiment": "x", "protocol": "paper", "window": {"length": 20}}))
    (run / "meta.json").write_text(json.dumps({"protocol": "paper", "protocol_version": "v1", "device": "mps", "seeds": [0, 1], "git_commit": "abcdef0123", "git_dirty": False, "torch": "2.14.1", "n_params": 115973}))
    summary = {"val_rmse_mean": 0.1, "val_rmse_std": 0.01, "val_mae_mean": 0.08, "val_mae_std": 0.01, "per_bearing": {}}
    if with_test:
        summary.update({"test_rmse_mean": 0.09, "test_rmse_std": 0.005, "test_mae_mean": 0.07, "test_mae_std": 0.004})
        summary["per_bearing"] = {"1_1": {"rmse_mean": 0.08, "rmse_std": 0.01, "mae_mean": 0.06, "mae_std": 0.01}}
    jobs = [{"best_epoch": 10}, {"best_epoch": 20}]
    (run / "metrics.json").write_text(json.dumps({"summary": summary, "jobs": jobs}))
    return run


class ReportTest(unittest.TestCase):
    def test_paper_table_mean(self) -> None:
        rmse = sum(v[0] for v in report.TFG_MAMBA_TABLE7.values()) / 13
        self.assertAlmostEqual(rmse, 0.0815, places=4)
        self.assertEqual(len(report.TFG_MAMBA_TABLE7), 13)

    def test_tables_render(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = _write_run(Path(tmp))
            text = report.test_table(run)
            self.assertIn("| 1_1 | 0.080 ± 0.010", text)
            self.assertIn("+0.002", text)  # 0.080 - 0.078
            self.assertIn("设备 `mps`", text)
            self.assertIn("| 15.0 |", report.selection_table([run]))


if __name__ == "__main__":
    unittest.main()
