import unittest

import numpy as np

from rul import dataset, features


def _fake_features(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 26))
    x[:, features.RMS_INDEX] = 1.0 + 0.01 * rng.standard_normal(n)
    x[n // 2 :, features.RMS_INDEX] += np.linspace(1.0, 3.0, n - n // 2)
    return x


class DegradationDataTest(unittest.TestCase):
    def test_paper_table_fpt(self) -> None:
        data = dataset.degradation_data("1_2", _fake_features(161), dataset.FptConfig("paper_table"))
        self.assertEqual(data.fpt, 22)
        self.assertEqual(data.snapshot_index[0], 22)
        self.assertEqual(data.snapshot_index[-1], 161)
        self.assertEqual(data.features.shape, (140, 26))
        self.assertEqual(data.labels[0], 1.0)
        self.assertEqual(data.labels[-1], 0.0)

    def test_rms_3sigma_fpt(self) -> None:
        data = dataset.degradation_data("1_1", _fake_features(100), dataset.FptConfig("rms_3sigma", n_healthy=30))
        self.assertTrue(data.fpt_triggered)
        self.assertEqual(data.fpt, 55)  # degradation starts at 51, fifth exceedance at 55

    def test_unknown_method(self) -> None:
        with self.assertRaises(ValueError):
            dataset.degradation_data("1_1", _fake_features(50), dataset.FptConfig("other"))


class ScalerTest(unittest.TestCase):
    def test_fit_only_on_given_arrays(self) -> None:
        train = [np.array([[0.0, 1.0], [2.0, 1.0]])]
        scaler = dataset.MinMaxScaler().fit(train)
        test = np.array([[4.0, 5.0]])
        np.testing.assert_allclose(scaler.transform(test), [[2.0, 4.0]])  # constant column -> span 1
        # Changing test data never changes the fitted parameters.
        np.testing.assert_allclose(scaler.data_min, [0.0, 1.0])
        np.testing.assert_allclose(scaler.data_max, [2.0, 1.0])

    def test_unfitted_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            dataset.MinMaxScaler().transform(np.zeros((1, 2)))


class WindowTest(unittest.TestCase):
    def test_shapes_padding_and_mask(self) -> None:
        x = np.arange(12, dtype=np.float64).reshape(6, 2)
        windows, mask = dataset.make_windows(x, 4)
        self.assertEqual(windows.shape, (6, 4, 2))
        self.assertEqual(windows.dtype, np.float32)
        np.testing.assert_array_equal(windows[0], [[0, 0], [0, 0], [0, 0], [0, 1]])
        np.testing.assert_array_equal(mask[0], [False, False, False, True])
        np.testing.assert_array_equal(windows[5], x[2:6])
        self.assertTrue(mask[5].all())

    def test_window_ends_at_its_own_row(self) -> None:
        x = np.random.default_rng(0).normal(size=(30, 3))
        windows, _ = dataset.make_windows(x, 7)
        for i in range(30):
            np.testing.assert_allclose(windows[i, -1], x[i].astype(np.float32))

    def test_causal_future_rows_do_not_change_earlier_windows(self) -> None:
        x = np.random.default_rng(1).normal(size=(20, 3))
        before, _ = dataset.make_windows(x, 5)
        changed = x.copy()
        changed[12:] += 100.0
        after, _ = dataset.make_windows(changed, 5)
        np.testing.assert_array_equal(before[:12], after[:12])


if __name__ == "__main__":
    unittest.main()
