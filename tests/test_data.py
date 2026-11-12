import tempfile
import unittest
from pathlib import Path

import numpy as np

from rul import data


def _write_snapshot(path: Path, horizontal: list[float], vertical: list[float]) -> None:
    lines = ["Horizontal_vibration_signals,Vertical_vibration_signals"]
    lines += [f"{h},{v}" for h, v in zip(horizontal, vertical)]
    path.write_text("\n".join(lines) + "\n")


class BearingListsTest(unittest.TestCase):
    def test_eval_bearings_exclude_1_4_and_3_2(self) -> None:
        self.assertEqual(len(data.ALL_BEARINGS), 15)
        self.assertEqual(len(data.EVAL_BEARINGS), 13)
        self.assertNotIn("1_4", data.EVAL_BEARINGS)
        self.assertNotIn("3_2", data.EVAL_BEARINGS)

    def test_condition_of(self) -> None:
        self.assertEqual(data.condition_of("2_5"), "2")
        with self.assertRaises(ValueError):
            data.condition_of("4_1")


class SnapshotFilesTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.bearing_dir = self.root / "35Hz12kN" / "Bearing1_1"
        self.bearing_dir.mkdir(parents=True)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_numeric_order_not_string_order(self) -> None:
        for i in range(1, 12):
            _write_snapshot(self.bearing_dir / f"{i}.csv", [float(i)], [0.0])
        files = data.snapshot_files("1_1", self.root)
        self.assertEqual([int(p.stem) for p in files], list(range(1, 12)))

    def test_gap_in_indices_raises(self) -> None:
        for i in (1, 2, 4):
            _write_snapshot(self.bearing_dir / f"{i}.csv", [0.0], [0.0])
        with self.assertRaises(ValueError):
            data.snapshot_files("1_1", self.root)

    def test_read_snapshot_selects_channel(self) -> None:
        path = self.bearing_dir / "1.csv"
        _write_snapshot(path, [1.0, -2.0, 3.5], [9.0, 9.0, 9.0])
        np.testing.assert_array_equal(data.read_snapshot(path, "horizontal"), [1.0, -2.0, 3.5])
        np.testing.assert_array_equal(data.read_snapshot(path, "vertical"), [9.0, 9.0, 9.0])
        self.assertEqual(data.read_snapshot(path).dtype, np.float64)


if __name__ == "__main__":
    unittest.main()
