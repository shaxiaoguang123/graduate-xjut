import tempfile
import unittest
from pathlib import Path

from rul import data, splits

MANIFEST = Path(__file__).resolve().parents[1] / "configs" / "splits" / "xjtu_sy_loo13.yaml"


class Loo13Test(unittest.TestCase):
    def setUp(self) -> None:
        self.split = splits.load_split(MANIFEST)

    def test_thirteen_folds_each_bearing_tested_once(self) -> None:
        self.assertEqual(len(self.split.folds), 13)
        self.assertEqual(sorted(f.test for f in self.split.folds), sorted(data.EVAL_BEARINGS))

    def test_sets_are_disjoint_and_exclude_1_4_3_2(self) -> None:
        for fold in self.split.folds:
            train = set(fold.train)
            self.assertNotIn(fold.test, train)
            self.assertNotIn(fold.val, train)
            self.assertNotEqual(fold.test, fold.val)
            self.assertEqual(len(train), 11)
            self.assertTrue({"1_4", "3_2"}.isdisjoint(train | {fold.test, fold.val}))

    def test_validation_rule_same_condition_next_bearing(self) -> None:
        for fold in self.split.folds:
            self.assertEqual(data.condition_of(fold.val), data.condition_of(fold.test))
            same = [b for b in data.EVAL_BEARINGS if data.condition_of(b) == data.condition_of(fold.test)]
            expected = same[(same.index(fold.test) + 1) % len(same)]
            self.assertEqual(fold.val, expected)


class ValidationTest(unittest.TestCase):
    def _load(self, text: str) -> splits.Split:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "split.yaml"
            path.write_text(text)
            return splits.load_split(path)

    def test_val_equal_test_rejected(self) -> None:
        folds = "\n".join(f'  - {{test: "{b}", val: "{b}"}}' for b in data.EVAL_BEARINGS)
        with self.assertRaises(ValueError):
            self._load(f'name: x\nprotocol_version: v1\nexcluded: ["1_4", "3_2"]\nfolds:\n{folds}\n')

    def test_excluded_bearing_as_val_rejected(self) -> None:
        lines = [f'  - {{test: "{b}", val: "1_4"}}' for b in data.EVAL_BEARINGS]
        with self.assertRaises(ValueError):
            self._load('name: x\nprotocol_version: v1\nexcluded: ["1_4", "3_2"]\nfolds:\n' + "\n".join(lines) + "\n")

    def test_missing_fold_rejected(self) -> None:
        lines = [f'  - {{test: "{b}", val: "1_1"}}' for b in data.EVAL_BEARINGS[1:]]
        with self.assertRaises(ValueError):
            self._load('name: x\nprotocol_version: v1\nexcluded: ["1_4", "3_2"]\nfolds:\n' + "\n".join(lines) + "\n")


if __name__ == "__main__":
    unittest.main()
