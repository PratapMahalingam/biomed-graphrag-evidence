"""Tune graph parameters on train claims, evaluate everything on dev claims, write results JSON.

Real data (SciFact + MedCPT + scispaCy entities, after scripts/build_index.py):
  python scripts/run_experiment.py --data-dir data/scifact --cache-dir cache --out results/scifact

Synthetic smoke test (no model downloads):
  python scripts/run_experiment.py --data-dir data/synthetic --out results/synthetic_smoke --smoke
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from evidence_rag.data import check_claims_against_corpus, load_claims, load_corpus  # noqa: E402
from evidence_rag.dense import LSAEncoder, cached_doc_embeddings, corpus_fingerprint  # noqa: E402
from evidence_rag.entities import DictionaryExtractor, EntityIndex  # noqa: E402
from evidence_rag.experiment import (Config, GraphParams, Resources, evaluate, select_first_stage,  # noqa: E402
                                     tune_graph)
from evidence_rag.pipeline import make_encoder  # noqa: E402
from evidence_rag.rerank import CachedReranker, LexicalReranker  # noqa: E402
from evidence_rag.sparse import BM25  # noqa: E402


def build_resources(data: Path, cache: Path, smoke: bool, encoders: list[str], hybrid: str) -> tuple[Resources, list, list]:
    docs = load_corpus(data / "corpus.jsonl")
    train = load_claims(data / "claims_train.jsonl")
    dev = load_claims(data / "claims_dev.jsonl")
    check_claims_against_corpus(train + dev, docs)
    bm25 = BM25().fit([d.text for d in docs])
    fp = corpus_fingerprint(docs)
    if smoke:
        enc = LSAEncoder(dim=32).fit(docs)
        dense = {"lsa": (enc, enc.encode_docs(docs))}
        hybrid = "lsa"
        vocab = (data / "entity_vocabulary.txt").read_text().split()
        doc_ents = DictionaryExtractor(vocab).extract([d.text for d in docs])
        claim_ents: dict[int, set[str]] = {}
        reranker = CachedReranker(LexicalReranker())
        index = EntityIndex.build(docs, doc_ents, max_df_frac=0.2)
    else:
        dense = {}
        for name in encoders:
            enc = make_encoder(name, docs)
            dense[name] = (enc, cached_doc_embeddings(enc, docs, cache))
        doc_ents = [set(e) for e in json.loads((cache / f"entities_docs_{fp}.json").read_text())]
        claim_ents = {}
        for split in ("train", "dev"):
            p = cache / f"entities_claims_{split}.json"
            if p.exists():
                claim_ents.update({int(k): set(v) for k, v in json.loads(p.read_text()).items()})
        from evidence_rag.rerank import MedCPTCrossEncoder

        reranker = CachedReranker(MedCPTCrossEncoder(), cache / "rerank_cache_medcpt-ce.pkl")
        index = EntityIndex.build(docs, doc_ents)
    res = Resources(docs, bm25, dense, hybrid, index, claim_ents, reranker)
    return res, train, dev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true", help="LSA + lexical reranker + dictionary entities")
    ap.add_argument("--encoders", nargs="*", default=["medcpt", "bge"])
    ap.add_argument("--hybrid", default="medcpt")
    ap.add_argument("--pool", type=int, default=10)
    ap.add_argument("--n-extra", type=int, default=5)
    ap.add_argument("--n-null-first", type=int, default=200)
    ap.add_argument("--n-null-rerank", type=int, default=5)
    ap.add_argument("--n-null-curve", type=int, default=100)
    ap.add_argument("--seed", type=int, default=2025)
    a = ap.parse_args()
    t = time.time()
    data, cache, out = Path(a.data_dir), Path(a.cache_dir), Path(a.out)
    res, train, dev = build_resources(data, cache, a.smoke, a.encoders, a.hybrid)
    print(f"{len(res.docs)} docs, {len(train)} train / {len(dev)} dev claims with evidence; "
          f"{len(res.index.vocab)} graph entities ({time.time() - t:.0f}s)")
    cfg = Config(pool=a.pool, graph=GraphParams(n_extra=a.n_extra))
    cands = ["hybrid", f"dense_{res.hybrid_dense}"]
    cfg.first, first_hits = select_first_stage(res, train, cfg, cands)
    print(f"first stage chosen on train: {cfg.first} (gold in top {cfg.pool}: {first_hits})")
    gp, tune_log = tune_graph(res, train, cfg)
    cfg.graph = gp
    print(f"tuned on train: {gp}")
    result = evaluate(res, dev, cfg, n_null_first=a.n_null_first, n_null_rerank=a.n_null_rerank, seed=a.seed,
                      n_null_curve=a.n_null_curve)
    if hasattr(res.reranker, "save"):
        res.reranker.save()
    result["tuning_log_train"] = tune_log
    result["first_stage_selection_train"] = first_hits
    result["seed"] = a.seed
    out.mkdir(parents=True, exist_ok=True)
    per_claim = result.pop("per_claim")
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    (out / "per_claim_dev.json").write_text(json.dumps(per_claim))
    for s, m in result["doc_level"].items():
        print(f"{s:38s} " + " ".join(f"{k}={v:.3f}" for k, v in m.items()))
    print(f"wrote {out / 'metrics.json'} ({time.time() - t:.0f}s)")


if __name__ == "__main__":
    np.set_printoptions(precision=4)
    main()
