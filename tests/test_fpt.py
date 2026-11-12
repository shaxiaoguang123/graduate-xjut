import unittest

import numpy as np

from rul import data, fpt


class DetectFptTest(unittest.TestCase):
    def _series(self) -> np.ndarray:
        rng = np.random.default_rng(0)
        healthy = 1.0 + 0.01 * rng.standard_normal(60)
        degrading = np.linspace(1.5, 3.0, 40)
        return np.concatenate([healthy, degrading])

    def test_detects_first_full_run_of_five(self) -> None:
        result = fpt.detect_fpt(self._series(), n_healthy=10)
        self.assertTrue(result.triggered)
        # Degradation starts at snapshot 61; the fifth consecutive exceedance is 65.
        self.assertEqual(result.index, 65)

    def test_short_spikes_do_not_trigger(self) -> None:
        rms = np.ones(50) + 0.01 * np.random.default_rng(1).standard_normal(50)
        rms[20:24] = 5.0  # only four in a row
        result = fpt.detect_fpt(rms, n_healthy=10)
        self.assertFalse(result.triggered)
        self.assertEqual(result.index, 11)

    def test_causal_future_changes_do_not_move_fpt(self) -> None:
        rms = self._series()
        before = fpt.detect_fpt(rms).index
        changed = rms.copy()
        changed[before:] = np.random.default_rng(2).uniform(0, 10, changed.size - before)
        self.assertEqual(fpt.detect_fpt(changed).index, before)

    def test_sigma_uses_n_minus_one(self) -> None:
        rms = np.concatenate([[1.0, 2.0, 3.0], np.zeros(5)])
        result = fpt.detect_fpt(rms, n_healthy=3)
        self.assertAlmostEqual(result.sigma, 1.0)
        self.assertAlmostEqual(result.threshold, 2.0 + 3.0)


class LabelsTest(unittest.TestCase):
    def test_linear_from_one_to_zero(self) -> None:
        y = fpt.rul_labels(n_snapshots=11, fpt=1)
        self.assertEqual(y.size, 11)
        self.assertEqual(y[0], 1.0)
        self.assertEqual(y[-1], 0.0)
        np.testing.assert_allclose(np.diff(y), -0.1)

    def test_invalid_fpt(self) -> None:
        with self.assertRaises(ValueError):
            fpt.rul_labels(10, 10)


class PaperTableTest(unittest.TestCase):
    def test_table_covers_all_bearings(self) -> None:
        self.assertEqual(set(fpt.PAPER_TABLE1), set(data.ALL_BEARINGS))
        for life, first in fpt.PAPER_TABLE1.values():
            self.assertLess(first, life)


if __name__ == "__main__":
    unittest.main()
