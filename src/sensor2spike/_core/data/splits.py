"""Leak-free window splitting.

Splits are performed at the *group* (trial) level rather than the window level.
Overlapping windows from the same recording would otherwise appear in both the
training and the test set, inflating reported accuracy.
"""

from __future__ import annotations

import numpy as np

from ..contracts import WindowSet
from ..exceptions import ConfigurationError

__all__ = ["split_windows"]


def split_windows(
    windows: WindowSet,
    *,
    test_fraction: float = 0.3,
    seed: int = 0,
    stratify: bool = True,
) -> tuple[WindowSet, WindowSet]:
    """Split a :class:`WindowSet` into ``(train, test)`` with disjoint groups.

    Parameters
    ----------
    windows:
        Input windows.  When ``groups`` is ``None`` every window is treated as
        its own group.
    test_fraction:
        Fraction of *groups* assigned to the test split.
    seed:
        Seed for the deterministic shuffle.
    stratify:
        Preserve the class balance across splits (requires labels).
    """
    if not 0.0 < test_fraction < 1.0:
        raise ConfigurationError("test_fraction must be strictly between 0 and 1.")

    n_windows = windows.n_windows
    if n_windows < 2:
        raise ConfigurationError("At least two windows are required to build a train/test split.")

    groups = (
        windows.groups.copy()
        if windows.groups is not None
        else np.arange(n_windows, dtype=np.int64)
    )
    unique_groups, group_index = np.unique(groups, return_inverse=True)
    n_groups = unique_groups.size
    if n_groups < 2:
        raise ConfigurationError(
            "All windows belong to a single group; cannot create a leakage-free split. "
            "Provide trial/group identifiers."
        )

    labels = windows.labels
    if stratify and labels is not None:
        group_label = np.zeros(n_groups, dtype=np.int64)
        for index in range(n_groups):
            members = labels[group_index == index]
            values, counts = np.unique(members, return_counts=True)
            group_label[index] = values[np.argmax(counts)]
    else:
        group_label = None

    rng = np.random.default_rng(seed)
    test_flags = np.zeros(n_groups, dtype=bool)

    if group_label is not None:
        for label in np.unique(group_label):
            members = np.flatnonzero(group_label == label)
            members = members[rng.permutation(members.size)]
            n_test = int(round(members.size * test_fraction))
            n_test = max(1, min(n_test, members.size - 1)) if members.size > 1 else 0
            test_flags[members[:n_test]] = True
    else:
        order = rng.permutation(n_groups)
        n_test = max(1, min(int(round(n_groups * test_fraction)), n_groups - 1))
        test_flags[order[:n_test]] = True

    test_window_mask = test_flags[group_index]
    train_idx = np.flatnonzero(~test_window_mask)
    test_idx = np.flatnonzero(test_window_mask)

    if train_idx.size == 0 or test_idx.size == 0:
        raise ConfigurationError(
            "Split produced an empty train or test set. Reduce test_fraction or add more trials."
        )

    return windows.subset(train_idx), windows.subset(test_idx)
