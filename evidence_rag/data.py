"""Loading SciFact-format data (corpus.jsonl, claims_*.jsonl).

Format (SciFact release, Wadden et al. 2020):
  corpus.jsonl : {"doc_id": int, "title": str, "abstract": [sentence, ...], "structured": bool}
  claims.jsonl : {"id": int, "claim": str,
                  "evidence": {doc_id(str): [{"sentences": [int, ...], "label": "SUPPORT"|"CONTRADICT"}]},
                  "cited_doc_ids": [int, ...]}
Claims with an empty "evidence" dict have no annotated evidence in the corpus (NOT ENOUGH INFO).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Doc:
    doc_id: int
    title: str
    sentences: tuple[str, ...]

    @property
    def text(self) -> str:
        return " ".join((self.title, *self.sentences))


@dataclass
class Claim:
    claim_id: int
    text: str
    # doc_id -> set of gold rationale sentence indices (union over evidence sets)
    rationales: dict[int, set[int]] = field(default_factory=dict)
    # doc_id -> list of evidence sets (each a tuple of sentence indices)
    evidence_sets: dict[int, list[tuple[int, ...]]] = field(default_factory=dict)
    label: str | None = None  # "SUPPORT", "CONTRADICT", or None when no evidence

    @property
    def gold_docs(self) -> set[int]:
        return set(self.rationales)

    @property
    def has_evidence(self) -> bool:
        return bool(self.rationales)


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_corpus(path: str | Path) -> list[Doc]:
    docs = [
        Doc(int(r["doc_id"]), r["title"].strip(), tuple(s.strip() for s in r["abstract"]))
        for r in read_jsonl(path)
    ]
    ids = [d.doc_id for d in docs]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate doc_id in corpus")
    return docs


def parse_claim(r: dict) -> Claim:
    rationales: dict[int, set[int]] = {}
    sets: dict[int, list[tuple[int, ...]]] = {}
    labels = set()
    for doc_id, ev_list in (r.get("evidence") or {}).items():
        d = int(doc_id)
        for ev in ev_list:
            sents = tuple(int(s) for s in ev["sentences"])
            rationales.setdefault(d, set()).update(sents)
            sets.setdefault(d, []).append(sents)
            labels.add(ev["label"])
    if len(labels) > 1:
        # SciFact guarantees one label per claim; fail loudly if a file breaks that.
        raise ValueError(f"claim {r['id']} has mixed labels {labels}")
    return Claim(int(r["id"]), r["claim"].strip(), rationales, sets, labels.pop() if labels else None)


def load_claims(path: str | Path, evidence_only: bool = True) -> list[Claim]:
    claims = [parse_claim(r) for r in read_jsonl(path)]
    return [c for c in claims if c.has_evidence] if evidence_only else claims


def check_claims_against_corpus(claims: list[Claim], docs: list[Doc]) -> None:
    """Every gold doc must exist and every rationale index must be a valid sentence."""
    by_id = {d.doc_id: d for d in docs}
    for c in claims:
        for d, sents in c.rationales.items():
            if d not in by_id:
                raise ValueError(f"claim {c.claim_id}: gold doc {d} not in corpus")
            n = len(by_id[d].sentences)
            bad = [s for s in sents if not 0 <= s < n]
            if bad:
                raise ValueError(f"claim {c.claim_id}: sentence ids {bad} out of range for doc {d} ({n} sentences)")
