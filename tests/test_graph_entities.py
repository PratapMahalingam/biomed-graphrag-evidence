import numpy as np
import scipy.sparse as sp

from evidence_rag.entities import DictionaryExtractor, EntityIndex, normalize_entity, valid_entity
from evidence_rag.graph import curveball_rewire, expansion_scores


def test_normalize_entity():
    assert normalize_entity(" TNF-alpha, ") == "tnf alpha"
    assert normalize_entity("HMG-CoA/reductase") == "hmg coa reductase"
    assert not valid_entity("of") and not valid_entity("12") and valid_entity("ampk")


def test_dictionary_extractor_multiword():
    ex = DictionaryExtractor(["HMG-CoA reductase", "ampk", "the"])
    assert ex.extract(["They inhibit HMG CoA reductase and AMPK."]) == [{"hmg coa reductase", "ampk"}]


def test_entity_index_df_filter(tiny_docs):
    ents = [{"aspirin", "cox1"}, {"metformin", "ampk"}, {"ampk"}, {"statin"}]
    idx = EntityIndex.build(tiny_docs, ents, max_df_frac=1.0, min_df=2)
    assert idx.vocab == ["ampk"]  # only entity shared by >= 2 docs
    assert idx.M.shape == (4, 1) and idx.M.sum() == 2
    assert np.array_equal(idx.query_vector(["ampk", "unknown"]), [1.0])


def test_expansion_reaches_doc_via_shared_seed_entity():
    # doc0 (seed) shares rare entity 0 with doc2; doc1 shares nothing
    M = sp.csr_matrix(np.array([[1, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 1]], float))
    idx = EntityIndex(["e0", "e1", "e2"], M, [0, 1, 2, 3])
    s = expansion_scores(idx, np.array([0]), np.zeros(3), alpha=1.0)
    assert s[2] > 0 and s[3] > 0 and s[1] == 0
    s_q = expansion_scores(idx, np.array([], dtype=int), idx.query_vector(["e1"]), alpha=1.0)
    assert np.argmax(s_q) == 1


def test_curveball_preserves_degrees_and_mixes():
    rng = np.random.default_rng(0)
    M = sp.random(60, 40, density=0.1, random_state=1, format="csr")
    M.data[:] = 1
    R = curveball_rewire(M, rng)
    assert np.array_equal(np.asarray(M.sum(0)).ravel(), np.asarray(R.sum(0)).ravel())
    assert np.array_equal(np.asarray(M.sum(1)).ravel(), np.asarray(R.sum(1)).ravel())
    assert R.max() == 1
    overlap = M.multiply(R).sum() / M.sum()
    assert overlap < 0.6
