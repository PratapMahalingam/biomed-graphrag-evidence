"""Serve the retriever. `--smoke` serves the synthetic corpus with no model downloads.

python scripts/serve.py --data-dir data/scifact --cache-dir cache          # real
python scripts/serve.py --data-dir data/synthetic --smoke                  # synthetic
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evidence_rag.api import create_app  # noqa: E402
from evidence_rag.experiment import Config, GraphParams  # noqa: E402
from run_experiment import build_resources  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--params", help="metrics.json from run_experiment.py; uses its first stage, pool and graph params")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    res, _, _ = build_resources(Path(a.data_dir), Path(a.cache_dir), a.smoke, ["medcpt"], "medcpt")
    cfg = Config()
    if a.params:
        c = json.loads(Path(a.params).read_text())["config"]
        cfg.graph = GraphParams(**c["graph"])
        cfg.first, cfg.pool = c["first"], c["pool"]
    import uvicorn

    uvicorn.run(create_app(res, cfg), host="127.0.0.1", port=a.port)


if __name__ == "__main__":
    main()
