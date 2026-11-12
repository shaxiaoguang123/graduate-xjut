import unittest

import numpy as np

from rul import features as F

FS = 25_600.0
N = 32_768


def _sine(amplitude: float = 2.0, freq: float = 100.0) -> np.ndarray:
    t = np.arange(N) / FS
    return amplitude * np.sin(2 * np.pi * freq * t)


class TimeFeaturesTest(unittest.TestCase):
    def test_names_and_shape(self) -> None:
        self.assertEqual(len(F.FEATURE_NAMES), 26)
        self.assertEqual(F.snapshot_features(_sine()).shape, (26,))

    def test_sine_values(self) -> None:
        a = 2.0
        feats = dict(zip(F.TIME_FEATURE_NAMES, F.time_features(_sine(a))))
        self.assertAlmostEqual(feats["mean"], 0.0, places=6)
        self.assertAlmostEqual(feats["rms"], a / np.sqrt(2), places=4)
        self.assertAlmostEqual(feats["peak_to_peak"], 2 * a, places=3)
        self.assertAlmostEqual(feats["crest_factor"], np.sqrt(2), places=3)
        self.assertAlmostEqual(feats["shape_factor"], np.pi / (2 * np.sqrt(2)), places=3)

    def test_printed_formulas_match_table_3(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.normal(size=1000) * 3 + 0.5
        n = x.size
        std = np.sqrt(np.sum((x - x.mean()) ** 2) / (n - 1))
        feats = dict(zip(F.TIME_FEATURE_NAMES, F.time_features(x, "printed")))
        self.assertAlmostEqual(feats["skewness"], np.mean((x - x.mean()) ** 3) / std**4)
        self.assertAlmostEqual(feats["kurtosis"], np.mean((x - x.mean()) ** 4))
        self.assertAlmostEqual(feats["margin_factor"], np.abs(x).max() / np.mean(np.abs(x)) ** 2)
        self.assertEqual(feats["peak"], feats["peak_to_peak"])
        self.assertAlmostEqual(feats["shannon_entropy"], -np.sum(x**2 * np.log(x**2)))
        self.assertAlmostEqual(feats["log_energy_entropy"], np.sum(np.log(x**2)))

    def test_standard_formulas_on_gaussian_noise(self) -> None:
        x = np.random.default_rng(1).normal(size=200_000)
        feats = dict(zip(F.TIME_FEATURE_NAMES, F.time_features(x, "standard")))
        self.assertAlmostEqual(feats["skewness"], 0.0, delta=0.03)
        self.assertAlmostEqual(feats["kurtosis"], 3.0, delta=0.05)
        self.assertAlmostEqual(feats["peak"], np.abs(x).max())

    def test_exact_zero_samples_are_finite(self) -> None:
        x = _sine()
        x[::100] = 0.0
        self.assertTrue(np.all(np.isfinite(F.snapshot_features(x, formula="printed"))))
        self.assertTrue(np.all(np.isfinite(F.snapshot_features(x, formula="standard"))))

    def test_unknown_formula_rejected(self) -> None:
        with self.assertRaises(ValueError):
            F.time_features(_sine(), "other")


class FreqFeaturesTest(unittest.TestCase):
    def test_spectrum_lines_and_peak(self) -> None:
        f, y = F.amplitude_spectrum(_sine(2.0, 100.0), FS)
        self.assertEqual(y.size, N // 2)
        self.assertAlmostEqual(f[1] - f[0], FS / N)
        self.assertAlmostEqual(f[int(np.argmax(y))], 100.0)
        self.assertAlmostEqual(y.max(), 2.0, places=6)

    def test_centroid_of_pure_tone(self) -> None:
        feats = dict(zip(F.FREQ_FEATURE_NAMES, F.freq_features(_sine(1.0, 100.0), FS)))
        self.assertAlmostEqual(feats["FF5"], 100.0, delta=0.5)
        self.assertAlmostEqual(feats["FF7"], 100.0, delta=0.5)
        self.assertAlmostEqual(feats["FF9"], 1.0, delta=1e-3)

    def test_printed_formulas_match_table_4(self) -> None:
        x = np.random.default_rng(2).normal(size=4096)
        f, y = F.amplitude_spectrum(x, FS)
        k = y.size
        ff1 = y.sum() / k
        ff2 = np.sum((y - ff1) ** 2) / (k - 1)
        ff5 = np.sum(f * y) / y.sum()
        ff6 = np.sqrt(np.sum((f - ff5) ** 2 * y) / k)
        got = F.freq_features(x, FS, "printed")
        expected = {
            2: np.sum((y - ff1) ** 3) / (k / ff2),
            3: np.sum((y - ff1) ** 4) / (k * ff2**3),
            7: np.sum(f**4 * y) / np.sum(f**2 * y),
            12: np.sum(np.sqrt(np.abs(f - ff5)) * y) / (k / np.sqrt(ff6)),
        }
        for index, value in expected.items():
            self.assertAlmostEqual(got[index] / value, 1.0, places=10, msg=f"FF{index + 1}")

    def test_standard_variant_differs_only_where_documented(self) -> None:
        x = np.random.default_rng(3).normal(size=4096)
        printed = F.freq_features(x, FS, "printed")
        standard = F.freq_features(x, FS, "standard")
        differs = {i + 1 for i in range(13) if not np.isclose(printed[i], standard[i])}
        self.assertEqual(differs, {3, 4, 8, 13})


if __name__ == "__main__":
    unittest.main()
