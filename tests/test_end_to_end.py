"""Synthetic end-to-end run of the experiment and the API (no model downloads)."""
from pathlib import Path

from fastapi.testclient import TestClient

from evidence_rag.api import create_app
from evidence_rag.experiment import Config, GraphParams, evaluate, select_first_stage, tune_graph
from make_synthetic import make
from run_experiment import build_resources


def test_pipeline_and_api(tmp_path: Path):
    make(tmp_path, n_docs=120, n_claims=40, seed=1)
    res, train, dev = build_resources(tmp_path, tmp_path / "cache", True, [], "lsa")
    cfg = Config(pool=10, graph=GraphParams(n_extra=5))
    gp, log = tune_graph(res, train, cfg, grid={"n_seeds": [3], "alpha": [0.0, 1.0], "n_extra": [5], "fuse_weight": [0.1]})
    assert len(log) == 2
    cfg.graph = gp
    out = evaluate(res, dev, cfg, n_null_first=3, n_null_rerank=2, seed=0, log=lambda *_: None,
                   budgets=(2, 4), n_null_curve=2)
    assert [r["budget"] for r in out["budget_curve"]] == [2, 4]
    assert out["budget_curve"][0]["rewired_hit"]["n_null"] == 2
    assert out["n_claims"] == len(dev)
    for s in ("bm25", "dense_lsa", "hybrid", "hybrid+graph", "hybrid>rerank", "hybrid>rerank(+5 graph)",
              "hybrid>rerank(+5 hybrid)"):  # cfg.first defaults to hybrid
        assert 0 <= out["doc_level"][s]["ndcg@10"] <= 1
    assert out["null_model"]["hybrid+graph ndcg@10"]["n_null"] == 3
    assert out["pool_hit"]["pool10+graph5"] >= out["pool_hit"]["pool10"]
    first, hits = select_first_stage(res, train, cfg, ["hybrid", "dense_lsa"])
    assert first in hits and set(hits) == {"hybrid", "dense_lsa"}
    cfg2 = Config(pool=10, first="dense_lsa", graph=gp)
    out2 = evaluate(res, dev, cfg2, n_null_first=1, n_null_rerank=1, log=lambda *_: None, budgets=(2,), n_null_curve=1)
    assert "dense_lsa>rerank(+5 graph)" in out2["doc_level"]
    assert out2["system_names"]["rerank"] == "dense_lsa>rerank"
    assert set(out["stance"]["bm25 recall@10"]) >= {"SUPPORT", "CONTRADICT", "unpaired", "same_abstract"}

    client = TestClient(create_app(res, cfg))
    assert client.get("/health").json()["n_docs"] == 120
    r = client.post("/search", json={"claim": dev[0].text, "k_docs": 3, "k_sentences": 2})
    assert r.status_code == 200
    body = r.json()
    assert len(body["docs"]) == 3 and len(body["evidence"]) == 2
    d = {x.doc_id: x for x in res.docs}
    for e in body["evidence"]:
        assert d[e["doc_id"]].sentences[e["sentence_index"]] == e["text"]
    assert client.post("/search", json={"claim": ""}).status_code == 422
