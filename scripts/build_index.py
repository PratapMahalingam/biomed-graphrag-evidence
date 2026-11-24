"""Embed the corpus and extract entities once; everything is cached under --cache-dir.

python scripts/build_index.py --data-dir data/scifact --cache-dir cache --encoders medcpt bge \
    --ner-models models/en_ner_bc5cdr_md models/en_ner_jnlpba_md
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evidence_rag.data import load_claims, load_corpus  # noqa: E402
from evidence_rag.dense import cached_doc_embeddings, corpus_fingerprint  # noqa: E402
from evidence_rag.pipeline import make_encoder  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--encoders", nargs="*", default=["medcpt"])
    ap.add_argument("--ner-models", nargs="*", default=[])
    ap.add_argument("--splits", nargs="*", default=["train", "dev"])
    a = ap.parse_args()
    data, cache = Path(a.data_dir), Path(a.cache_dir)
    docs = load_corpus(data / "corpus.jsonl")
    fp = corpus_fingerprint(docs)
    print(f"{len(docs)} docs, fingerprint {fp}")
    for name in a.encoders:
        t = time.time()
        enc = make_encoder(name, docs)
        E = cached_doc_embeddings(enc, docs, cache)
        print(f"{name}: {E.shape} in {time.time() - t:.0f}s")
    if a.ner_models:
        from evidence_rag.entities import ScispacyExtractor

        ex = ScispacyExtractor(a.ner_models)
        out = cache / f"entities_docs_{fp}.json"
        if not out.exists():
            t = time.time()
            ents = ex.extract([d.text for d in docs])
            out.write_text(json.dumps([sorted(e) for e in ents]))
            print(f"doc entities in {time.time() - t:.0f}s")
        for split in a.splits:
            claims = load_claims(data / f"claims_{split}.jsonl")
            ents = ex.extract([c.text for c in claims])
            (cache / f"entities_claims_{split}.json").write_text(
                json.dumps({str(c.claim_id): sorted(e) for c, e in zip(claims, ents)}))
            print(f"{split}: {len(claims)} claims with entities")


if __name__ == "__main__":
    main()
