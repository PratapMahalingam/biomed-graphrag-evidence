"""Write a small synthetic corpus in SciFact format, for the smoke test.

The text is random and the entities are invented tokens. It checks that the pipeline runs end to
end; numbers computed on it say nothing about real retrieval quality.

python scripts/make_synthetic.py --out data/synthetic --n-docs 300 --n-claims 80
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FILLER = ("study results patients cohort analysis increased decreased associated observed measured "
          "levels expression response treatment model effect cells mice risk outcome trial").split()


def make(out: Path, n_docs: int, n_claims: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    n_ent = max(20, n_docs // 3)
    entities = [f"zx{i:03d}{'abcdefgh'[i % 8]}in" for i in range(n_ent)]  # e.g. "zx017bin"
    docs = []
    doc_ents = []
    for d in range(n_docs):
        ents = list(rng.choice(n_ent, size=3, replace=False))
        doc_ents.append(ents)
        sents = []
        for _ in range(int(rng.integers(4, 9))):
            words = list(rng.choice(FILLER, size=8))
            words.insert(int(rng.integers(0, 8)), entities[int(rng.choice(ents))])
            sents.append(" ".join(words).capitalize() + ".")
        docs.append({"doc_id": 1000 + d, "title": f"{entities[ents[0]]} and {entities[ents[1]]} in model systems",
                     "abstract": sents, "structured": False})
    claims = {"train": [], "dev": []}
    for c in range(n_claims):
        d = int(rng.integers(0, n_docs))
        s = int(rng.integers(0, len(docs[d]["abstract"])))
        words = docs[d]["abstract"][s].rstrip(".").lower().split()
        keep = [w for w in words if rng.random() < 0.6 or w.startswith("zx")]
        label = "SUPPORT" if rng.random() < 0.65 else "CONTRADICT"
        text = " ".join(keep + ([] if label == "SUPPORT" else ["not"])).capitalize() + "."
        split = "train" if c % 3 else "dev"
        claims[split].append({"id": c, "claim": text, "cited_doc_ids": [1000 + d],
                              "evidence": {str(1000 + d): [{"sentences": [s], "label": label}]}})
    claims["dev"].append({"id": n_claims, "claim": "Unrelated claim without evidence.", "evidence": {}, "cited_doc_ids": []})
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "corpus.jsonl", "w") as f:
        for r in docs:
            f.write(json.dumps(r) + "\n")
    for split, rows in claims.items():
        with open(out / f"claims_{split}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    (out / "entity_vocabulary.txt").write_text("\n".join(entities) + "\n")
    return {"docs": n_docs, "train": len(claims["train"]), "dev": len(claims["dev"])}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--n-docs", type=int, default=300)
    ap.add_argument("--n-claims", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    print(make(Path(a.out), a.n_docs, a.n_claims, a.seed))
