"""
Label definitions and utilities for the CWRU bearing dataset.
"""

from __future__ import annotations

from enum import IntEnum
from pathlib import Path


class BearingLabel(IntEnum):
    """Enumeration of CWRU fault classes."""

    NORMAL = 0

    BALL_007 = 1
    BALL_014 = 2
    BALL_021 = 3

    INNER_007 = 4
    INNER_014 = 5
    INNER_021 = 6

    OUTER_007 = 7
    OUTER_014 = 8
    OUTER_021 = 9


LABEL_MAP = {
    "Normal": BearingLabel.NORMAL,

    "B007": BearingLabel.BALL_007,
    "B014": BearingLabel.BALL_014,
    "B021": BearingLabel.BALL_021,

    "IR007": BearingLabel.INNER_007,
    "IR014": BearingLabel.INNER_014,
    "IR021": BearingLabel.INNER_021,

    "OR007": BearingLabel.OUTER_007,
    "OR014": BearingLabel.OUTER_014,
    "OR021": BearingLabel.OUTER_021,
}


CLASS_NAMES = {
    BearingLabel.NORMAL: "Normal",

    BearingLabel.BALL_007: "Ball Fault (0.007\")",
    BearingLabel.BALL_014: "Ball Fault (0.014\")",
    BearingLabel.BALL_021: "Ball Fault (0.021\")",

    BearingLabel.INNER_007: "Inner Race Fault (0.007\")",
    BearingLabel.INNER_014: "Inner Race Fault (0.014\")",
    BearingLabel.INNER_021: "Inner Race Fault (0.021\")",

    BearingLabel.OUTER_007: "Outer Race Fault (0.007\")",
    BearingLabel.OUTER_014: "Outer Race Fault (0.014\")",
    BearingLabel.OUTER_021: "Outer Race Fault (0.021\")",
}


ID_TO_NAME = {int(k): v for k, v in CLASS_NAMES.items()}

NAME_TO_ID = {v: int(k) for k, v in CLASS_NAMES.items()}


def filename_to_label(file: str | Path) -> int:
    """
    Convert a CWRU filename into an integer class label.

    Examples
    --------
    >>> filename_to_label("B007_0.mat")
    1

    >>> filename_to_label("IR014_2.mat")
    5

    >>> filename_to_label("Normal_3.mat")
    0
    """

    stem = Path(file).stem

    for prefix, label in LABEL_MAP.items():
        if stem.startswith(prefix):
            return int(label)

    raise ValueError(f"Unknown CWRU filename: {file}")


def label_to_name(label: int) -> str:
    """
    Convert numeric label to human-readable class name.
    """

    if label not in ID_TO_NAME:
        raise KeyError(f"Unknown label: {label}")

    return ID_TO_NAME[label]


def name_to_label(name: str) -> int:
    """
    Convert human-readable class name to label.
    """

    if name not in NAME_TO_ID:
        raise KeyError(f"Unknown class name: {name}")

    return NAME_TO_ID[name]


def get_num_classes() -> int:
    """
    Number of fault classes.
    """

    return len(BearingLabel)


def get_class_names() -> list[str]:
    """
    Return ordered list of class names.
    """

    return [
        ID_TO_NAME[i]
        for i in range(get_num_classes())
    ]