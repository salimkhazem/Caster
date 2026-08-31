"""Statistics shared by the analysis scripts.

`spearman` lives here because it previously existed as three separate copies --
in the figure generator, the claim auditor and the figure-freshness checker --
and they disagreed. Two used ordinal ranks, which break ties by input position,
so the same 57 cells produced -0.77, -0.79, -0.81 and -0.86 depending on which
script asked and in what order the rows arrived. A statistic quoted in a paper
should have exactly one implementation.
"""

from __future__ import annotations

import numpy as np


def average_ranks(values) -> np.ndarray:
    """Ranks with ties averaged, as Spearman's rho requires.

    ``np.argsort(np.argsort(x))`` returns *ordinal* ranks: tied values receive
    distinct ranks assigned by position in the input. That makes any correlation
    computed from them depend on iteration order. It matters here because ties
    are common rather than incidental -- 41 of the 57 gate-specificity cells
    share an accept rate, 23 of them exactly zero, where the gate rejects every
    batch.
    """
    v = np.asarray(values, dtype=float)
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(len(v), dtype=float)
    ranks[order] = np.arange(len(v), dtype=float)
    sorted_v, i = v[order], 0
    while i < len(sorted_v):
        j = i
        while j + 1 < len(sorted_v) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def spearman(xs, ys) -> float:
    """Spearman rank correlation, tie-aware."""
    x, y = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    if x.size != y.size:
        raise ValueError(f"length mismatch: {x.size} vs {y.size}")
    if x.size < 2:
        return float("nan")
    return float(np.corrcoef(average_ranks(x), average_ranks(y))[0, 1])
