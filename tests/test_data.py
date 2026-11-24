import json

import pytest

from evidence_rag.data import Doc, check_claims_against_corpus, load_claims, load_corpus, parse_claim


def test_parse_claim_unions_evidence_sets():
    c = parse_claim({"id": 3, "claim": " x ", "evidence": {
        "5": [{"sentences": [2, 5], "label": "SUPPORT"}, {"sentences": [7], "label": "SUPPORT"}]}, "cited_doc_ids": [5]})
    assert c.rationales == {5: {2, 5, 7}}
    assert c.evidence_sets == {5: [(2, 5), (7,)]}
    assert c.label == "SUPPORT" and c.text == "x" and c.gold_docs == {5}


def test_mixed_labels_rejected():
    with pytest.raises(ValueError):
        parse_claim({"id": 1, "claim": "x", "evidence": {
            "1": [{"sentences": [0], "label": "SUPPORT"}], "2": [{"sentences": [0], "label": "CONTRADICT"}]}})


def test_no_evidence_claims_filtered(tmp_path):
    p = tmp_path / "c.jsonl"
    p.write_text(json.dumps({"id": 1, "claim": "a", "evidence": {}}) + "\n" +
                 json.dumps({"id": 2, "claim": "b", "evidence": {"9": [{"sentences": [0], "label": "CONTRADICT"}]}}) + "\n")
    assert [c.claim_id for c in load_claims(p)] == [2]
    assert len(load_claims(p, evidence_only=False)) == 2


def test_corpus_duplicates_and_ranges(tmp_path):
    p = tmp_path / "corpus.jsonl"
    p.write_text(json.dumps({"doc_id": 1, "title": "t", "abstract": ["s0", "s1"]}) + "\n")
    docs = load_corpus(p)
    assert docs[0].text == "t s0 s1"
    good = parse_claim({"id": 1, "claim": "x", "evidence": {"1": [{"sentences": [1], "label": "SUPPORT"}]}})
    check_claims_against_corpus([good], docs)
    bad = parse_claim({"id": 2, "claim": "x", "evidence": {"1": [{"sentences": [2], "label": "SUPPORT"}]}})
    with pytest.raises(ValueError):
        check_claims_against_corpus([bad], docs)
    missing = parse_claim({"id": 3, "claim": "x", "evidence": {"7": [{"sentences": [0], "label": "SUPPORT"}]}})
    with pytest.raises(ValueError):
        check_claims_against_corpus([missing], docs)
    p.write_text(p.read_text() * 2)
    with pytest.raises(ValueError):
        load_corpus(p)


def test_doc_text():
    assert Doc(1, "T", ("a", "b")).text == "T a b"
