"""Factories shared by scripts and the API."""
from __future__ import annotations

from .data import Doc


def make_encoder(name: str, docs: list[Doc] | None = None):
    from . import dense

    if name == "medcpt":
        return dense.MedCPTEncoder()
    if name == "bge":
        return dense.SentenceTransformerEncoder("BAAI/bge-small-en-v1.5")
    if name == "lsa":
        if docs is None:
            raise ValueError("lsa encoder needs docs to fit")
        return dense.LSAEncoder().fit(docs)
    raise ValueError(f"unknown encoder {name!r}")
