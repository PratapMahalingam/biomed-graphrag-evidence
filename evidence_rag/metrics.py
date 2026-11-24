"""Ranking metrics and paired / clustered resampling statistics."""
from __future__ import annotations

from typing import Sequence

import numpy as np


def recall_at_k(ranked: Sequence[int], gold: set[int], k: int) -> float:
    return len(set(ranked[:k]) & gold) / len(gold) if gold else float("nan")


def hit_at_k(ranked: Sequence[int], gold: set[int], k: int) -> float:
    return float(bool(set(ranked[:k]) & gold))


def ndcg_at_k(ranked: Sequence[int], gold: set[int], k: int) -> float:
    if not gold:
        return float("nan")
    dcg = sum(1.0 / np.log2(i + 2) for i, d in enumerate(ranked[:k]) if d in gold)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / idcg


def mrr_at_k(ranked: Sequence[int], gold: set[int], k: int) -> float:
    for i, d in enumerate(ranked[:k]):
        if d in gold:
            return 1.0 / (i + 1)
    return 0.0


def rationale_recall(ranked_sents: Sequence[tuple[int, int]], gold: dict[int, set[int]], k: int) -> float:
    """Share of gold (doc, sentence) rationale pairs among the top-k retrieved sentences."""
    g = {(d, s) for d, ss in gold.items() for s in ss}
    return len(set(ranked_sents[:k]) & g) / len(g) if g else float("nan")


def doc_metrics(ranked: Sequence[int], gold: set[int], ks=(1, 3, 5, 10, 20)) -> dict[str, float]:
    out = {f"recall@{k}": recall_at_k(ranked, gold, k) for k in ks}
    out["ndcg@10"] = ndcg_at_k(ranked, gold, 10)
    out["mrr@10"] = mrr_at_k(ranked, gold, 10)
    return out


def _resample_index(n: int, clusters: np.ndarray | None, rng: np.random.Generator) -> np.ndarray:
    if clusters is None:
        return rng.integers(0, n, n)
    uniq = np.unique(clusters)
    members = {c: np.flatnonzero(clusters == c) for c in uniq}
    pick = rng.choice(uniq, size=len(uniq), replace=True)
    return np.concatenate([members[c] for c in pick])


def paired_bootstrap_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 0,
                        clusters: np.ndarray | None = None, level: float = 0.95) -> tuple[float, float, float]:
    """Mean of (b - a) with a percentile CI; resamples claims, or whole clusters if given."""
    d = np.asarray(b, float) - np.asarray(a, float)
    rng = np.random.default_rng(seed)
    boots = np.array([d[_resample_index(len(d), clusters, rng)].mean() for _ in range(n_boot)])
    lo, hi = np.quantile(boots, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(d.mean()), float(lo), float(hi)


def paired_sign_flip_p(a: np.ndarray, b: np.ndarray, n_perm: int = 10000, seed: int = 0) -> float:
    """Two-sided paired randomisation test on mean(b - a)."""
    d = np.asarray(b, float) - np.asarray(a, float)
    if not np.any(d):
        return 1.0
    rng = np.random.default_rng(seed)
    obs = abs(d.mean())
    flips = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
    null = np.abs((flips * d).mean(axis=1))
    return float((1 + (null >= obs - 1e-12).sum()) / (n_perm + 1))


def group_difference(values: np.ndarray, groups: np.ndarray, g1, g2, clusters: np.ndarray | None = None,
                     n_boot: int = 2000, seed: int = 0) -> dict[str, float]:
    """mean(g1) - mean(g2) with a bootstrap CI that resamples whole clusters.

    Clusters (e.g. the gold evidence document) keep claims that share a source abstract together,
    so a supported claim and its negated twin are not counted as independent evidence.
    """
    v, g = np.asarray(values, float), np.asarray(groups)
    obs = v[g == g1].mean() - v[g == g2].mean()
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = _resample_index(len(v), clusters, rng)
        gv, gg = v[idx], g[idx]
        if (gg == g1).any() and (gg == g2).any():
            boots.append(gv[gg == g1].mean() - gv[gg == g2].mean())
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {"diff": float(obs), "ci_low": float(lo), "ci_high": float(hi),
            f"n_{g1}": int((g == g1).sum()), f"n_{g2}": int((g == g2).sum())}


def matched_cluster_difference(values: np.ndarray, groups: np.ndarray, clusters: np.ndarray, g1, g2,
                               n_perm: int = 10000, seed: int = 0) -> dict[str, float]:
    """Within clusters holding both groups: mean over clusters of (mean g1 - mean g2), with a
    sign-flip p-value. This compares claims that share the same evidence abstract."""
    v, g, c = np.asarray(values, float), np.asarray(groups), np.asarray(clusters)
    diffs = []
    for cl in np.unique(c):
        m = c == cl
        a, b = v[m & (g == g1)], v[m & (g == g2)]
        if len(a) and len(b):
            diffs.append(a.mean() - b.mean())
    d = np.array(diffs)
    if not len(d):
        return {"n_clusters": 0, "diff": float("nan"), "p_sign_flip": float("nan")}
    p = paired_sign_flip_p(np.zeros_like(d), d, n_perm=n_perm, seed=seed)
    return {"n_clusters": int(len(d)), "diff": float(d.mean()), "p_sign_flip": p,
            "n_g1_better": int((d > 0).sum()), "n_g2_better": int((d < 0).sum())}
