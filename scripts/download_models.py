"""Download the two scispaCy NER models (v0.5.4) into models/ and make them load under spaCy 3.8.

The released configs store two booleans as strings ("True"/"False"); spaCy 3.8's config
validation rejects that, so the script rewrites them as real booleans. Nothing else is changed.
The Hugging Face models (MedCPT encoders and cross-encoder, bge-small) download on first use.
"""
from __future__ import annotations

import io
import re
import tarfile
import urllib.request
from pathlib import Path

BASE = "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/"
MODELS = ["en_ner_bc5cdr_md", "en_ner_jnlpba_md"]  # chemicals+diseases; DNA/RNA/protein/cell line/cell type


def main(out: Path = Path("models")):
    out.mkdir(exist_ok=True)
    for m in MODELS:
        dest = out / m
        if not dest.exists():
            print(f"downloading {m}")
            with urllib.request.urlopen(f"{BASE}{m}-0.5.4.tar.gz") as r:
                tar = tarfile.open(fileobj=io.BytesIO(r.read()), mode="r:gz")
            prefix = f"{m}-0.5.4/{m}/{m}-0.5.4/"
            members = [x for x in tar.getmembers() if x.name.startswith(prefix) and x.isfile()]
            for x in members:
                x.name = x.name[len(prefix):]
                tar.extract(x, dest, filter="data")
        cfg = dest / "config.cfg"
        text = cfg.read_text()
        cfg.write_text(re.sub(r'= "(True|False)"', lambda g: "= " + g.group(1).lower(), text))
        print(f"{m} ready at {dest}")


if __name__ == "__main__":
    main()
