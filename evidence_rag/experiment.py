"""The experiment: first-stage retrievers, graph expansion vs budget and rewired-graph controls,
cross-encoder reranking, sentence-level evidence selection, and a stance (SUPPORT vs CONTRADICT) audit.

All components are injected, so the same code runs on SciFact with MedCPT and scispaCy, and on the
synthetic smoke corpus with LSA, a lexical reranker and a dictionary entity extractor.
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from .data import Claim, Doc
from .entities import DictionaryExtractor, EntityIndex
from .fusion import rrf, topk
from .graph import curveball_rewire, expansion_scores
from .metrics import (doc_metrics, group_difference, hit_at_k, matched_cluster_difference,
                      paired_bootstrap_ci, paired_sign_flip_p, rationale_recall, recall_at_k)
from .rerank import Reranker
from .sparse import BM25, tokenize


@dataclass
class GraphParams:
    n_seeds: int = 5      # first-stage docs that activate entities
    alpha: float = 1.0    # weight of claim entities relative to seed entities
    n_extra: int = 5      # graph docs added to the rerank pool
    fuse_weight: float = 0.5  # weight of the graph ranking in first-stage RRF


@dataclass
class Config:
    depth: int = 100          # first-stage list length
    pool: int = 10            # rerank pool size before graph expansion
    first: str = "hybrid"     # first-stage list that seeds the graph and fills the rerank pool
    sent_docs: int = 3        # docs whose sentences are scored for rationale selection
    rrf_k: int = 60
    graph: GraphParams = field(default_factory=GraphParams)


@dataclass
class Resources:
    docs: list[Doc]
    bm25: BM25
    dense: dict[str, tuple[object, np.ndarray]]   # name -> (encoder, doc embeddings)
    hybrid_dense: str                              # which dense model is fused with BM25
    index: EntityIndex
    claim_entities: dict[int, set[str]]            # extra claim entities (e.g. NER); dictionary match is added
    reranker: Reranker

    def __post_init__(self):
        self.row_of = {d.doc_id: i for i, d in enumerate(self.docs)}
        self._dict = DictionaryExtractor(self.index.vocab)

    def entities_for(self, claims: list[Claim]) -> list[set[str]]:
        dict_hits = self._dict.extract([c.text for c in claims])
        return [self.claim_entities.get(c.claim_id, set()) | h for c, h in zip(claims, dict_hits)]


# ---------------------------------------------------------------- first stage

def first_stage(res: Resources, claims: list[Claim], cfg: Config) -> dict[str, list[np.ndarray]]:
    n = len(res.docs)
    out: dict[str, list[np.ndarray]] = {"bm25": [topk(res.bm25.scores(c.text), cfg.depth) for c in claims]}
    for name, (enc, E) in res.dense.items():
        Q = enc.encode_queries([c.text for c in claims])
        out[f"dense_{name}"] = [topk(E @ q, cfg.depth) for q in Q]
    hyb = []
    for r_b, r_d in zip(out["bm25"], out[f"dense_{res.hybrid_dense}"]):
        hyb.append(topk(rrf([r_b, r_d], n, cfg.rrf_k), cfg.depth))
    out["hybrid"] = hyb
    return out


def graph_rankings(res: Resources, M: sp.csr_matrix, hybrid: list[np.ndarray], q_ents: list[set[str]],
                   gp: GraphParams, depth: int, idf: np.ndarray) -> list[np.ndarray]:
    index = EntityIndex(res.index.vocab, M, res.index.doc_ids)
    out = []
    for r, ents in zip(hybrid, q_ents):
        # Seeds stay in the ranking (they share their own entities, so they score high): excluding
        # them would hand the graph's top RRF slots to non-seeds and demote the best first-stage docs.
        # graph_pool() skips anything already in the pool.
        s = expansion_scores(index, r[: gp.n_seeds], index.query_vector(ents), gp.alpha, idf=idf)
        s[s <= 0] = -np.inf
        out.append(topk(s, depth))
    return out


def fuse_graph(hybrid: list[np.ndarray], graph: list[np.ndarray], n: int, cfg: Config) -> list[np.ndarray]:
    return [topk(rrf([h, g], n, cfg.rrf_k, [1.0, cfg.graph.fuse_weight]), cfg.depth) for h, g in zip(hybrid, graph)]


def graph_pool(h: np.ndarray, g: np.ndarray, pool: int, n_extra: int) -> np.ndarray:
    base = list(h[:pool])
    seen = set(base)
    extra = [int(i) for i in g if int(i) not in seen][:n_extra]
    return np.array(base + extra, dtype=int)


# ---------------------------------------------------------------- reranking

def rerank(res: Resources, claim: Claim, rows: np.ndarray) -> np.ndarray:
    if not len(rows):
        return rows
    s = res.reranker.score(claim.text, [res.docs[i].text for i in rows])
    return rows[np.lexsort((np.arange(len(rows)), -s))]


def select_sentences(res: Resources, claim: Claim, rows: np.ndarray, n_docs: int) -> list[tuple[int, int]]:
    pairs, texts = [], []
    for i in rows[:n_docs]:
        d = res.docs[i]
        for j, s in enumerate(d.sentences):
            pairs.append((d.doc_id, j))
            texts.append(s)
    if not texts:
        return []
    s = res.reranker.score(claim.text, texts)
    return [pairs[i] for i in np.lexsort((np.arange(len(texts)), -s))]


# ---------------------------------------------------------------- tuning

def select_first_stage(res: Resources, claims: list[Claim], cfg: Config, candidates: list[str]) -> tuple[str, dict]:
    """Pick the list that fills the rerank pool: highest gold-hit rate within `cfg.pool` on these claims."""
    fs = first_stage(res, claims, cfg)
    gold = [c.gold_docs for c in claims]
    hits = {k: float(np.mean([hit_at_k(_ids(res, r), g, cfg.pool) for r, g in zip(fs[k], gold)])) for k in candidates}
    return max(candidates, key=lambda k: (hits[k], k == "hybrid")), hits


def tune_graph(res: Resources, claims: list[Claim], cfg: Config, grid: dict[str, list] | None = None) -> tuple[GraphParams, list[dict]]:
    """Choose graph parameters on training claims without the reranker (cheap).

    Objective for pool parameters: gold-hit rate of the expanded pool (first-stage top-`pool` +
    n_extra graph docs). Objective for fuse_weight: nDCG@10 of the fused hybrid ranking.
    """
    grid = grid or {"n_seeds": [3, 5, 10], "alpha": [0.0, 0.5, 1.0, 2.0], "n_extra": [cfg.graph.n_extra],
                    "fuse_weight": [0.05, 0.1, 0.25, 0.5, 1.0]}
    fs = first_stage(res, claims, cfg)
    q_ents = res.entities_for(claims)
    idf = res.index.idf()
    gold = [{res.row_of[d] for d in c.gold_docs} for c in claims]
    n = len(res.docs)
    log, best, best_key = [], None, None
    for n_seeds, alpha, n_extra in itertools.product(grid["n_seeds"], grid["alpha"], grid["n_extra"]):
        gp = GraphParams(n_seeds, alpha, n_extra)
        g1 = graph_rankings(res, res.index.M, fs[cfg.first], q_ents, gp, cfg.depth, idf)
        hit = np.mean([hit_at_k(list(graph_pool(h, gg, cfg.pool, n_extra)), go, 10**9)
                       for h, gg, go in zip(fs[cfg.first], g1, gold)])
        g = g1 if cfg.first == "hybrid" else graph_rankings(res, res.index.M, fs["hybrid"], q_ents, gp, cfg.depth, idf)
        rec = {"n_seeds": n_seeds, "alpha": alpha, "n_extra": n_extra, "pool_hit": float(hit)}
        for fw in grid["fuse_weight"]:
            cfg_fw = Config(cfg.depth, cfg.pool, cfg.first, cfg.sent_docs, cfg.rrf_k, GraphParams(n_seeds, alpha, n_extra, fw))
            fused = fuse_graph(fs["hybrid"], g, n, cfg_fw)
            rec[f"ndcg@10_fw{fw}"] = float(np.mean([doc_metrics(list(f), go)["ndcg@10"] for f, go in zip(fused, gold)]))
        log.append(rec)
        key = (rec["pool_hit"], -n_seeds)
        if best_key is None or key > best_key:
            best_key, best = key, rec
    fw_best = max(grid["fuse_weight"], key=lambda fw: best[f"ndcg@10_fw{fw}"])
    return GraphParams(best["n_seeds"], best["alpha"], best["n_extra"], fw_best), log


# ---------------------------------------------------------------- evaluation

def _ids(res: Resources, rows) -> list[int]:
    return [res.docs[int(i)].doc_id for i in rows]


def evaluate(res: Resources, claims: list[Claim], cfg: Config, n_null_first: int = 200,
             n_null_rerank: int = 10, seed: int = 0, log=print, budgets: tuple[int, ...] = (2, 4, 6, 10, 20),
             n_null_curve: int = 100) -> dict:
    t0 = time.time()
    n = len(res.docs)
    rng = np.random.default_rng(seed)
    gp = cfg.graph
    fs = first_stage(res, claims, cfg)
    q_ents = res.entities_for(claims)
    idf = res.index.idf()  # df is unchanged by degree-preserving rewiring
    F = cfg.first
    g_hyb = graph_rankings(res, res.index.M, fs["hybrid"], q_ents, gp, cfg.depth, idf)
    g_real = g_hyb if F == "hybrid" else graph_rankings(res, res.index.M, fs[F], q_ents, gp, cfg.depth, idf)
    systems: dict[str, list[list[int]]] = {k: [_ids(res, r) for r in v] for k, v in fs.items()}
    systems["hybrid+graph"] = [_ids(res, r) for r in fuse_graph(fs["hybrid"], g_hyb, n, cfg)]
    log(f"first stage done ({time.time() - t0:.0f}s)")

    big = cfg.pool + gp.n_extra
    pools = {
        f"pool{cfg.pool}": [h[: cfg.pool] for h in fs[F]],
        f"pool{big}": [h[:big] for h in fs[F]],
        f"pool{cfg.pool}+graph{gp.n_extra}": [graph_pool(h, g, cfg.pool, gp.n_extra) for h, g in zip(fs[F], g_real)],
    }
    names = {f"pool{cfg.pool}": f"{F}>rerank",
             f"pool{big}": f"{F}>rerank(+{gp.n_extra} {F})",
             f"pool{cfg.pool}+graph{gp.n_extra}": f"{F}>rerank(+{gp.n_extra} graph)"}
    final_rows = {}
    for key, plist in pools.items():
        ranked = [rerank(res, c, p) for c, p in zip(claims, plist)]
        final_rows[names[key]] = ranked
        systems[names[key]] = [_ids(res, r) for r in ranked]
    log(f"rerank done ({time.time() - t0:.0f}s)")

    gold = [c.gold_docs for c in claims]
    per_claim = {s: [doc_metrics(r, g) for r, g in zip(rank, gold)] for s, rank in systems.items()}
    pool_hit = {k: [hit_at_k(_ids(res, p), g, 10**9) for p, g in zip(v, gold)] for k, v in pools.items()}

    # sentence-level evidence selection from the top docs of selected systems
    sent_systems = list(dict.fromkeys(["bm25", "hybrid", F, names[f"pool{cfg.pool}"], names[f"pool{cfg.pool}+graph{gp.n_extra}"]]))
    row_lists = {**fs, **final_rows}
    sent = {}
    for s in sent_systems:
        sents = [select_sentences(res, c, r, cfg.sent_docs) for c, r in zip(claims, row_lists[s])]
        sent[s] = {f"rationale_recall@{k}": [rationale_recall(x, c.rationales, k) for x, c in zip(sents, claims)]
                   for k in (3, 5, 10)}
    log(f"sentences done ({time.time() - t0:.0f}s)")

    # null model: same expansion over degree-preserving rewired graphs
    null_first = {"hybrid+graph ndcg@10": [], "hybrid+graph recall@10": [], "pool+graph hit": []}
    null_rerank = {"ndcg@10": [], "recall@1": [], "recall@10": []}
    for r in range(max(n_null_first, n_null_rerank)):
        Mr = curveball_rewire(res.index.M, rng)
        g_null_h = graph_rankings(res, Mr, fs["hybrid"], q_ents, gp, cfg.depth, idf)
        g_null = g_null_h if F == "hybrid" else graph_rankings(res, Mr, fs[F], q_ents, gp, cfg.depth, idf)
        if r < n_null_first:
            fused = [_ids(res, x) for x in fuse_graph(fs["hybrid"], g_null_h, n, cfg)]
            m = [doc_metrics(f, g) for f, g in zip(fused, gold)]
            null_first["hybrid+graph ndcg@10"].append(float(np.mean([x["ndcg@10"] for x in m])))
            null_first["hybrid+graph recall@10"].append(float(np.mean([x["recall@10"] for x in m])))
            pl = [graph_pool(h, g, cfg.pool, gp.n_extra) for h, g in zip(fs[F], g_null)]
            null_first["pool+graph hit"].append(float(np.mean([hit_at_k(_ids(res, p), g, 10**9) for p, g in zip(pl, gold)])))
        if r < n_null_rerank:
            pl = [graph_pool(h, g, cfg.pool, gp.n_extra) for h, g in zip(fs[F], g_null)]
            rk = [_ids(res, rerank(res, c, p)) for c, p in zip(claims, pl)]
            m = [doc_metrics(x, g) for x, g in zip(rk, gold)]
            for k in null_rerank:
                null_rerank[k].append(float(np.mean([x[k] for x in m])))
            if hasattr(res.reranker, "save"):
                res.reranker.save()
            log(f"  rewired graph {r + 1}: rerank ndcg@10 {null_rerank['ndcg@10'][-1]:.4f} ({time.time() - t0:.0f}s)")
    log(f"null models done ({time.time() - t0:.0f}s)")

    curve = budget_curve(res, claims, fs[F], q_ents, cfg, idf, budgets, n_null_curve, rng)
    log(f"budget curve done ({time.time() - t0:.0f}s)")
    out = summarize(res, claims, cfg, per_claim, pool_hit, sent, null_first, null_rerank, names, q_ents)
    out["budget_curve"] = curve
    return out


def budget_curve(res: Resources, claims: list[Claim], hybrid: list[np.ndarray], q_ents: list[set[str]], cfg: Config,
                 idf: np.ndarray, budgets: tuple[int, ...], n_null: int, rng: np.random.Generator) -> list[dict]:
    """Fixed-size context: for budget B, compare first-stage top-B with first-stage top-(B - B//2)
    plus B//2 graph docs, over the real graph and over `n_null` rewired graphs. No reranker.
    (`hybrid` is whichever first-stage list cfg.first names.)"""
    gold = [c.gold_docs for c in claims]

    def score(graph_lists, B):
        m = B // 2
        pools = [graph_pool(h, g, B - m, m) for h, g in zip(hybrid, graph_lists)]
        hit = np.mean([hit_at_k(_ids(res, p), go, B) for p, go in zip(pools, gold)])
        rec = np.mean([len(set(_ids(res, p)) & go) / len(go) for p, go in zip(pools, gold)])
        return float(hit), float(rec)

    real = graph_rankings(res, res.index.M, hybrid, q_ents, cfg.graph, cfg.depth, idf)
    nulls = [graph_rankings(res, curveball_rewire(res.index.M, rng), hybrid, q_ents, cfg.graph, cfg.depth, idf)
             for _ in range(n_null)]
    rows = []
    for B in budgets:
        h_hit = float(np.mean([hit_at_k(_ids(res, h), go, B) for h, go in zip(hybrid, gold)]))
        h_rec = float(np.mean([recall_at_k(_ids(res, h), go, B) for h, go in zip(hybrid, gold)]))
        g_hit, g_rec = score(real, B)
        nh, nr = zip(*[score(n, B) for n in nulls]) if nulls else ((), ())
        rows.append({"budget": B, "graph_slots": B // 2, "hybrid_hit": h_hit, "hybrid_recall": h_rec,
                     "graph_hit": g_hit, "graph_recall": g_rec,
                     "rewired_hit": _null_summary(g_hit, list(nh)), "rewired_recall": _null_summary(g_rec, list(nr))})
    return rows


def _mean(rows: list[dict], key: str) -> float:
    return float(np.nanmean([r[key] for r in rows]))


def _null_summary(obs: float, null: list[float]) -> dict:
    a = np.array(null)
    if not len(a):
        return {"observed": obs}
    return {"observed": obs, "null_mean": float(a.mean()), "null_sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "null_max": float(a.max()), "n_null": int(len(a)),
            "p_upper": float((1 + (a >= obs - 1e-12).sum()) / (len(a) + 1))}


def summarize(res, claims, cfg, per_claim, pool_hit, sent, null_first, null_rerank, names, q_ents) -> dict:
    gp = cfg.graph
    keys = ["recall@1", "recall@3", "recall@10", "recall@20", "ndcg@10", "mrr@10"]
    table = {s: {k: _mean(v, k) for k in keys} for s, v in per_claim.items()}
    clusters = np.array([min(c.gold_docs) for c in claims])

    def arr(system, key):
        return np.array([r[key] for r in per_claim[system]], float)

    base, budget, graph = names[f"pool{cfg.pool}"], names[f"pool{cfg.pool + gp.n_extra}"], names[f"pool{cfg.pool}+graph{gp.n_extra}"]
    comparisons = {}
    F = cfg.first
    pairs = [("bm25", "hybrid"), (f"dense_{res.hybrid_dense}", "hybrid"), ("bm25", base), (F, base), ("hybrid", "hybrid+graph"),
             (base, graph), (budget, graph), (base, budget)]
    pairs = list(dict.fromkeys(pairs))
    for a, b in pairs:
        comparisons[f"{b} vs {a}"] = {}
        for k in ("ndcg@10", "recall@1", "recall@10"):
            x, y = arr(a, k), arr(b, k)
            mean, lo, hi = paired_bootstrap_ci(x, y, clusters=clusters)
            comparisons[f"{b} vs {a}"][k] = {"diff": mean, "ci_low": lo, "ci_high": hi,
                                             "p_sign_flip": paired_sign_flip_p(x, y)}
    nulls = {
        "hybrid+graph ndcg@10": _null_summary(table["hybrid+graph"]["ndcg@10"], null_first["hybrid+graph ndcg@10"]),
        "hybrid+graph recall@10": _null_summary(table["hybrid+graph"]["recall@10"], null_first["hybrid+graph recall@10"]),
        "pool+graph hit": _null_summary(float(np.mean(pool_hit[f"pool{cfg.pool}+graph{gp.n_extra}"])), null_first["pool+graph hit"]),
        f"{graph} ndcg@10": _null_summary(table[graph]["ndcg@10"], null_rerank["ndcg@10"]),
        f"{graph} recall@10": _null_summary(table[graph]["recall@10"], null_rerank["recall@10"]),
    }
    labels = np.array([c.label for c in claims])
    stance = {}
    for s in dict.fromkeys(["bm25", f"dense_{res.hybrid_dense}", "hybrid", base]):
        for k in ("recall@1", "recall@10", "ndcg@10"):
            v = arr(s, k)
            stance[f"{s} {k}"] = {
                "SUPPORT": float(v[labels == "SUPPORT"].mean()), "CONTRADICT": float(v[labels == "CONTRADICT"].mean()),
                "unpaired": group_difference(v, labels, "SUPPORT", "CONTRADICT", clusters=clusters),
                "same_abstract": matched_cluster_difference(v, labels, clusters, "SUPPORT", "CONTRADICT"),
            }
    # where can the graph help? claims whose gold doc the base pool misses
    miss = np.array([h == 0 for h in pool_hit[f"pool{cfg.pool}"]])
    rescued_graph = int(np.sum(miss & (np.array(pool_hit[f"pool{cfg.pool}+graph{gp.n_extra}"]) == 1)))
    rescued_budget = int(np.sum(miss & (np.array(pool_hit[f"pool{cfg.pool + gp.n_extra}"]) == 1)))
    n_ent = np.array([len(e) for e in q_ents])
    return {
        "n_claims": len(claims), "n_docs": len(res.docs), "config": {**cfg.__dict__, "graph": gp.__dict__},
        "system_names": {"first_stage": F, "rerank": base, "rerank_plus_first": budget, "rerank_plus_graph": graph},
        "entity_graph": {"n_entities": len(res.index.vocab), "n_edges": int(res.index.M.nnz),
                         "docs_with_entities": int((np.asarray(res.index.M.sum(axis=1)).ravel() > 0).sum()),
                         "claims_with_entities": int((n_ent > 0).sum()), "median_claim_entities": float(np.median(n_ent))},
        "doc_level": table,
        "pool_hit": {k: float(np.mean(v)) for k, v in pool_hit.items()},
        "pool_misses": {"base_pool_misses": int(miss.sum()), "rescued_by_graph": rescued_graph,
                        "rescued_by_budget": rescued_budget},
        "sentence_level": {s: {k: float(np.nanmean(v)) for k, v in d.items()} for s, d in sent.items()},
        "comparisons": comparisons, "null_model": nulls, "stance": stance,
        "per_claim": [{"claim_id": c.claim_id, "label": c.label, "n_gold_docs": len(c.gold_docs),
                       "n_entities": int(ne), **{f"{s}|{k}": per_claim[s][i][k] for s in per_claim for k in ("ndcg@10", "recall@10")}}
                      for i, (c, ne) in enumerate(zip(claims, n_ent))],
    }


def lexical_overlap(claim: Claim, docs_by_id: dict[int, Doc]) -> float:
    q = set(tokenize(claim.text))
    if not q:
        return 0.0
    best = 0.0
    for d in claim.gold_docs:
        best = max(best, len(q & set(tokenize(docs_by_id[d].text))) / len(q))
    return best
