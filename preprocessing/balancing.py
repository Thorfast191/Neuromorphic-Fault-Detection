"""
Data balancing utilities.

Supports:
- Random oversampling
- Random undersampling
"""

from __future__ import annotations

import numpy as np


class RandomOverSampler:

    def __init__(self, random_state: int = 42):
        self.rng = np.random.default_rng(random_state)

    def __call__(self, X, y):

        X = np.asarray(X)
        y = np.asarray(y)

        classes, counts = np.unique(y, return_counts=True)
        max_count = counts.max()

        X_balanced = []
        y_balanced = []

        for cls in classes:

            idx = np.where(y == cls)[0]

            sampled = self.rng.choice(
                idx,
                size=max_count,
                replace=True,
            )

            X_balanced.append(X[sampled])
            y_balanced.append(y[sampled])

        return (
            np.concatenate(X_balanced),
            np.concatenate(y_balanced),
        )


class RandomUnderSampler:

    def __init__(self, random_state: int = 42):
        self.rng = np.random.default_rng(random_state)

    def __call__(self, X, y):

        X = np.asarray(X)
        y = np.asarray(y)

        classes, counts = np.unique(y, return_counts=True)
        min_count = counts.min()

        X_balanced = []
        y_balanced = []

        for cls in classes:

            idx = np.where(y == cls)[0]

            sampled = self.rng.choice(
                idx,
                size=min_count,
                replace=False,
            )

            X_balanced.append(X[sampled])
            y_balanced.append(y[sampled])

        return (
            np.concatenate(X_balanced),
            np.concatenate(y_balanced),
        )