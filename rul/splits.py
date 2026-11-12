"""Whole-bearing split manifests (configs/splits/*.yaml)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from rul.data import ALL_BEARINGS


@dataclass(frozen=True)
class Fold:
    test: str
    val: str
    train: tuple[str, ...]

    @property
    def name(self) -> str:
        return f"test_{self.test}"


@dataclass(frozen=True)
class Split:
    name: str
    protocol_version: str
    excluded: tuple[str, ...]
    folds: tuple[Fold, ...]

    @property
    def eval_bearings(self) -> tuple[str, ...]:
        return tuple(b for b in ALL_BEARINGS if b not in self.excluded)

    def fold(self, test_bearing: str) -> Fold:
        for fold in self.folds:
            if fold.test == test_bearing:
                return fold
        raise KeyError(f"no fold with test bearing {test_bearing}")


def load_split(path: Path | str) -> Split:
    raw = yaml.safe_load(Path(path).read_text())
    excluded = tuple(str(b) for b in raw.get("excluded", []))
    eval_bearings = [b for b in ALL_BEARINGS if b not in excluded]
    folds = []
    for entry in raw["folds"]:
        test, val = str(entry["test"]), str(entry["val"])
        train = tuple(b for b in eval_bearings if b not in (test, val))
        folds.append(Fold(test=test, val=val, train=train))
    split = Split(
        name=str(raw["name"]),
        protocol_version=str(raw["protocol_version"]),
        excluded=excluded,
        folds=tuple(folds),
    )
    validate_split(split)
    return split


def validate_split(split: Split) -> None:
    """Raise if any fold leaks a bearing between sets or coverage is wrong."""
    unknown = set(split.excluded) - set(ALL_BEARINGS)
    if unknown:
        raise ValueError(f"unknown excluded bearings: {sorted(unknown)}")
    eval_set = set(split.eval_bearings)
    tested = [fold.test for fold in split.folds]
    if sorted(tested) != sorted(eval_set):
        raise ValueError("every evaluation bearing must be the test bearing of exactly one fold")
    for fold in split.folds:
        groups = {"test": {fold.test}, "val": {fold.val}, "train": set(fold.train)}
        if fold.val == fold.test:
            raise ValueError(f"{fold.name}: validation bearing equals test bearing")
        for name, members in groups.items():
            if not members <= eval_set:
                raise ValueError(f"{fold.name}: {name} contains excluded or unknown bearings")
        if fold.test in fold.train or fold.val in fold.train:
            raise ValueError(f"{fold.name}: train overlaps test/val")
        if len(fold.train) + 2 != len(eval_set):
            raise ValueError(f"{fold.name}: train set has wrong size")
