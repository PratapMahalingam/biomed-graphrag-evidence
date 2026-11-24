"""Rank helpers: top-k and reciprocal-rank fusion."""
from __future__ import annotations

import numpy as np


def topk(scores: np.ndarray, k: int, exclude: np.ndarray | None = None) -> np.ndarray:
    s = scores.astype(float).copy()
    if exclude is not None and len(exclude):
        s[exclude] = -np.inf
    k = min(k, int(np.isfinite(s).sum()))
    if k <= 0:
        return np.zeros(0, dtype=int)
    idx = np.argpartition(-s, k - 1)[:k]
    # stable tie-break on row index keeps results deterministic
    return idx[np.lexsort((idx, -s[idx]))]


def rrf(rankings: list[np.ndarray], n: int, k: int = 60, weights: list[float] | None = None) -> np.ndarray:
    """Reciprocal-rank fusion (Cormack et al. 2009). Returns a score per row (0 if unranked)."""
    out = np.zeros(n)
    weights = weights or [1.0] * len(rankings)
    for r, w in zip(rankings, weights):
        out[r] += w / (k + np.arange(1, len(r) + 1))
    return out
