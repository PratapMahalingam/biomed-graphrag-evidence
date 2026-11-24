import numpy as np
import pytest

from evidence_rag.experiment import graph_pool
from evidence_rag.fusion import rrf, topk
from evidence_rag.metrics import (group_difference, matched_cluster_difference, mrr_at_k, ndcg_at_k,
                                  paired_bootstrap_ci, paired_sign_flip_p, rationale_recall, recall_at_k)
from evidence_rag.rerank import CachedReranker, LexicalReranker


def test_topk_order_ties_and_exclude():
    s = np.array([0.1, 0.5, 0.5, 0.9, -1.0])
    assert topk(s, 3).tolist() == [3, 1, 2]
    assert topk(s, 2, exclude=np.array([3])).tolist() == [1, 2]
    assert topk(np.full(3, -np.inf), 2).tolist() == []


def test_rrf():
    f = rrf([np.array([0, 1]), np.array([1, 2])], 4, k=60)
    assert f[1] == pytest.approx(1 / 62 + 1 / 61)
    assert f[3] == 0 and np.argmax(f) == 1


def test_graph_pool_appends_unseen_graph_docs():
    h = np.array([5, 4, 3, 2, 1])
    g = np.array([4, 9, 8, 7])
    assert graph_pool(h, g, pool=3, n_extra=2).tolist() == [5, 4, 3, 9, 8]


def test_rank_metrics():
    assert recall_at_k([1, 2, 3], {3, 4}, 3) == 0.5
    assert ndcg_at_k([9, 3], {3}, 10) == pytest.approx(1 / np.log2(3))
    assert ndcg_at_k([3], {3}, 10) == 1.0
    assert mrr_at_k([9, 8, 3], {3}, 10) == pytest.approx(1 / 3)
    assert mrr_at_k([9, 8, 3], {3}, 2) == 0.0
    assert rationale_recall([(1, 0), (1, 2), (2, 1)], {1: {2, 3}}, 2) == 0.5


def test_paired_stats():
    a = np.zeros(50)
    b = np.ones(50) * 0.2
    mean, lo, hi = paired_bootstrap_ci(a, b)
    assert mean == pytest.approx(0.2) and lo == pytest.approx(0.2) and hi == pytest.approx(0.2)
    assert paired_sign_flip_p(a, a) == 1.0
    assert paired_sign_flip_p(a, b, n_perm=2000) < 0.01


def test_group_and_matched_cluster_difference():
    v = np.array([1, 1, 0, 0, 1, 0], float)
    g = np.array(["S", "S", "C", "C", "S", "C"])
    c = np.array([0, 1, 0, 1, 2, 2])
    gd = group_difference(v, g, "S", "C", clusters=c, n_boot=200)
    assert gd["diff"] == 1.0 and gd["n_S"] == 3
    mc = matched_cluster_difference(v, g, c, "S", "C", n_perm=500)
    assert mc["n_clusters"] == 3 and mc["diff"] == 1.0 and mc["n_g1_better"] == 3


def test_cached_reranker_counts_misses(tmp_path):
    r = CachedReranker(LexicalReranker(), tmp_path / "c.pkl")
    s1 = r.score("ampk lifespan", ["ampk extends lifespan", "statins"])
    assert s1.tolist() == [1.0, 0.0] and r.misses == 2
    r.score("ampk lifespan", ["statins", "new text"])
    assert r.misses == 3
    r.save()
    assert CachedReranker(LexicalReranker(), tmp_path / "c.pkl").cache == r.cache
