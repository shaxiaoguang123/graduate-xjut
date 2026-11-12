import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from rul import features, train

ROOT = Path(__file__).resolve().parents[1]
LIFETIMES = {"1_1": 40, "1_2": 45, "1_3": 50, "1_5": 30, "2_1": 60, "2_2": 42, "2_3": 55, "2_4": 35, "2_5": 48, "3_1": 70, "3_3": 44, "3_4": 65, "3_5": 38}


def _fake_load_features(bearing, formula="printed", cache_dir=None, channel="horizontal"):
    n = LIFETIMES[bearing]
    rng = np.random.default_rng(abs(hash(bearing)) % 2**32)
    x = rng.normal(size=(n, 26))
    x[:, features.RMS_INDEX] = np.linspace(1.0, 3.0, n)
    return x


def _config(tmp: Path) -> dict:
    cfg = train.load_config(ROOT / "papers" / "tfg-mamba" / "configs" / "strict.yaml")
    cfg["data"]["split"] = str(ROOT / cfg["data"]["split"])
    cfg["fpt"] = {"method": "rms_3sigma", "n_healthy": 5, "n_sigma": 3.0, "n_consecutive": 5}
    cfg["window"]["length"] = 6
    cfg["model"]["n_layers"] = 1
    cfg["train"].update({"max_epochs": 2, "patience": 1, "compile": False, "batch_size": 16})
    return cfg


class ConfigTest(unittest.TestCase):
    def test_overrides(self) -> None:
        cfg = train.load_config(ROOT / "papers" / "tfg-mamba" / "configs" / "paper.yaml", ["window.length=10", "train.lr=0.01"])
        self.assertEqual(cfg["window"]["length"], 10)
        self.assertEqual(cfg["train"]["lr"], 0.01)
        self.assertEqual(train.model_config(cfg).tcn_channels, (32, 64))

    def test_explicit_device_never_falls_back(self) -> None:
        with mock.patch("torch.cuda.is_available", return_value=False):
            with self.assertRaises(RuntimeError):
                train.resolve_device("cuda")
        with mock.patch("torch.backends.mps.is_available", return_value=False):
            with self.assertRaises(RuntimeError):
                train.resolve_device("mps")


class PipelineTest(unittest.TestCase):
    def test_scaler_uses_training_bearings_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(features, "load_features", _fake_load_features):
            cfg = _config(Path(tmp))
            split = train.splits.load_split(cfg["data"]["split"])
            fold = split.fold("1_1")
            data = train.prepare_fold(cfg, fold)
            from rul import dataset

            fpt_cfg = dataset.FptConfig(**cfg["fpt"])
            train_feats = [dataset.degradation_data(b, _fake_load_features(b), fpt_cfg).features for b in fold.train]
            np.testing.assert_allclose(data.scaler_state["min"], np.concatenate(train_feats).min(axis=0))
            self.assertNotIn(fold.test, fold.train)

    def test_run_writes_records_and_respects_no_test(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(features, "load_features", _fake_load_features):
            cfg = _config(Path(tmp))
            run_dir = train.run_experiment(cfg, "cpu", folds=["1_1", "2_4"], seeds=[0], out_root=tmp, evaluate_test=False)
            for name in ("config.yaml", "meta.json", "metrics.json", "predictions.csv"):
                self.assertTrue((run_dir / name).exists(), name)
            payload = json.loads((run_dir / "metrics.json").read_text())
            self.assertEqual(len(payload["jobs"]), 2)
            self.assertNotIn("test", payload["jobs"][0])
            self.assertNotIn("test_rmse_mean", payload["summary"])
            self.assertNotIn(",test,", (run_dir / "predictions.csv").read_text())
            meta = json.loads((run_dir / "meta.json").read_text())
            self.assertEqual(meta["device"], "cpu")
            self.assertIn("git_commit", meta)

    def test_run_with_test_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(features, "load_features", _fake_load_features):
            cfg = _config(Path(tmp))
            run_dir = train.run_experiment(cfg, "cpu", folds=["1_1"], seeds=[0, 1], out_root=tmp)
            summary = json.loads((run_dir / "metrics.json").read_text())["summary"]
            self.assertIn("test_rmse_mean", summary)
            self.assertIn("1_1", summary["per_bearing"])


if __name__ == "__main__":
    unittest.main()
