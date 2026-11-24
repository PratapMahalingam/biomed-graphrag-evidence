import numpy as np

from evidence_rag.dense import LSAEncoder, cached_doc_embeddings, corpus_fingerprint
from evidence_rag.sparse import BM25, tokenize


def test_tokenize_keeps_biomedical_tokens():
    assert tokenize("The IL-6 and p53 levels") == ["il-6", "p53", "levels"]


def test_bm25_prefers_rare_matching_term(tiny_docs):
    bm = BM25().fit([d.text for d in tiny_docs])
    s = bm.scores("hmg coa reductase")
    assert int(np.argmax(s)) == 3
    assert np.all(bm.idf > 0)
    assert np.all(bm.scores("zzzz unseen") == 0)


def test_bm25_query_terms_counted_once(tiny_docs):
    bm = BM25().fit([d.text for d in tiny_docs])
    assert np.allclose(bm.scores("statins"), bm.scores("statins statins statins"))


def test_lsa_encoder_and_cache(tiny_docs, tmp_path):
    enc = LSAEncoder(dim=3).fit(tiny_docs)
    E = cached_doc_embeddings(enc, tiny_docs, tmp_path)
    assert E.shape[0] == 4 and np.allclose(np.linalg.norm(E, axis=1), 1)
    assert (tmp_path / f"docemb_lsa_{corpus_fingerprint(tiny_docs)}.npy").exists()
    E2 = cached_doc_embeddings(enc, tiny_docs, tmp_path)
    assert np.array_equal(E, E2)
    q = enc.encode_queries(["statins cholesterol"])
    assert int(np.argmax(E @ q[0])) == 3
