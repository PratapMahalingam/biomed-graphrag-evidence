"""FastAPI service: claim in, ranked abstracts and evidence sentences out (with citations).

The service returns evidence only. It does not generate a verdict; anything downstream that
writes an answer should cite the returned (doc_id, sentence) pairs.
"""
from __future__ import annotations

import time

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .data import Claim
from .experiment import Config, Resources, first_stage, graph_pool, graph_rankings, rerank, select_sentences


class Query(BaseModel):
    claim: str = Field(min_length=3, max_length=1000)
    k_docs: int = Field(default=5, ge=1, le=50)
    k_sentences: int = Field(default=5, ge=1, le=50)
    use_graph: bool = True


class Sentence(BaseModel):
    doc_id: int
    sentence_index: int
    text: str


class DocHit(BaseModel):
    doc_id: int
    title: str
    rank: int
    via_graph: bool


class Answer(BaseModel):
    docs: list[DocHit]
    evidence: list[Sentence]
    latency_ms: float


def create_app(res: Resources, cfg: Config) -> FastAPI:
    app = FastAPI(title="Biomedical evidence retrieval", version="0.1.0")

    @app.get("/health")
    def health():
        return {"status": "ok", "n_docs": len(res.docs), "n_entities": len(res.index.vocab)}

    @app.post("/search", response_model=Answer)
    def search(q: Query):
        t = time.perf_counter()
        claim = Claim(-1, q.claim)
        hyb = first_stage(res, [claim], cfg)[cfg.first][0]
        pool = hyb[: cfg.pool]
        if q.use_graph:
            g = graph_rankings(res, res.index.M, [hyb], res.entities_for([claim]), cfg.graph, cfg.depth, res.index.idf())[0]
            pool = graph_pool(hyb, g, cfg.pool, cfg.graph.n_extra)
        base = set(hyb[: cfg.pool].tolist())
        ranked = rerank(res, claim, pool)
        sents = select_sentences(res, claim, ranked, cfg.sent_docs)[: q.k_sentences]
        by_id = {d.doc_id: d for d in res.docs}
        return Answer(
            docs=[DocHit(doc_id=res.docs[i].doc_id, title=res.docs[i].title, rank=r + 1, via_graph=int(i) not in base)
                  for r, i in enumerate(ranked[: q.k_docs])],
            evidence=[Sentence(doc_id=d, sentence_index=s, text=by_id[d].sentences[s]) for d, s in sents],
            latency_ms=(time.perf_counter() - t) * 1000,
        )

    return app
