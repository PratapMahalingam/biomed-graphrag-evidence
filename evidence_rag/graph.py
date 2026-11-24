"""One-hop expansion over the document-entity bipartite graph, and a degree-preserving null.

Expansion: entities are activated by (a) appearing in the claim and (b) appearing in the
first-stage seed documents (weighted by seed rank). Every document then scores the summed
activation of its entities, normalised by sqrt(#entities) so entity-rich abstracts do not win
by size alone. Weights are entity idf.

Null model: `curveball_rewire` shuffles which documents hold which entities while keeping every
document's entity count and every entity's document frequency fixed (Strona et al. 2014,
Nature Communications, "curveball" algorithm). If expansion over the rewired graph helps as
much as over the real graph, the gain comes from adding candidates, not from graph structure.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .entities import EntityIndex


def expansion_scores(index: EntityIndex, seed_rows: np.ndarray, query_entity_vec: np.ndarray,
                     alpha: float = 1.0, idf: np.ndarray | None = None) -> np.ndarray:
    """Score every doc row. `seed_rows` are row indices in rank order (best first)."""
    M = index.M
    w = index.idf() if idf is None else idf
    act = alpha * query_entity_vec * w
    if len(seed_rows):
        decay = 1.0 / np.log2(np.arange(len(seed_rows)) + 2.0)
        seed_ent = M[seed_rows]
        n_ent = np.maximum(np.asarray(seed_ent.sum(axis=1)).ravel(), 1.0)
        act = act + (sp.diags(decay / np.sqrt(n_ent)) @ seed_ent).sum(axis=0).A1 * w
    doc_norm = np.sqrt(np.maximum(np.asarray(M.sum(axis=1)).ravel(), 1.0))
    return (M @ act) / doc_norm


def curveball_rewire(M: sp.csr_matrix, rng: np.random.Generator, n_trades: int | None = None) -> sp.csr_matrix:
    """Degree-preserving random bipartite graph with the same row and column sums as M."""
    rows = [set(M.indices[M.indptr[i]:M.indptr[i + 1]].tolist()) for i in range(M.shape[0])]
    nonempty = [i for i, r in enumerate(rows) if r]
    if len(nonempty) < 2:
        return M.copy()
    n_trades = n_trades or 5 * len(nonempty)
    pairs = rng.choice(len(nonempty), size=(n_trades, 2))
    for a, b in pairs:
        if a == b:
            continue
        A, B = rows[nonempty[a]], rows[nonempty[b]]
        a_only, b_only = A - B, B - A
        if not a_only or not b_only:
            continue
        pool = list(a_only | b_only)
        rng.shuffle(pool)
        k = len(a_only)
        shared = A & B
        rows[nonempty[a]] = shared | set(pool[:k])
        rows[nonempty[b]] = shared | set(pool[k:])
    r_idx = np.repeat(np.arange(len(rows)), [len(r) for r in rows])
    c_idx = np.fromiter((c for r in rows for c in sorted(r)), dtype=np.int64, count=len(r_idx))
    return sp.csr_matrix((np.ones(len(r_idx)), (r_idx, c_idx)), shape=M.shape)
