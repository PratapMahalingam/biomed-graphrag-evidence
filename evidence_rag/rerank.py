"""Cross-encoder rerankers with a persistent score cache."""
from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Protocol

import numpy as np

from .sparse import tokenize


class Reranker(Protocol):
    name: str

    def score(self, query: str, texts: list[str]) -> np.ndarray: ...


class LexicalReranker:
    """Query-term coverage; a cheap stand-in for tests and the synthetic smoke test."""

    name = "lexical"

    def score(self, query, texts):
        q = set(tokenize(query))
        if not q:
            return np.zeros(len(texts))
        return np.array([len(q & set(tokenize(t))) / len(q) for t in texts], dtype=float)


class MedCPTCrossEncoder:
    name = "medcpt-ce"
    MODEL = "ncbi/MedCPT-Cross-Encoder"

    def __init__(self, batch_size: int = 16, max_len: int = 512):
        self.batch_size, self.max_len = batch_size, max_len
        self._m = None

    def score(self, query, texts):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if self._m is None:
            self._m = (AutoTokenizer.from_pretrained(self.MODEL),
                       AutoModelForSequenceClassification.from_pretrained(self.MODEL).eval())
        tok, model = self._m
        out = []
        with torch.inference_mode():
            for i in range(0, len(texts), self.batch_size):
                chunk = texts[i : i + self.batch_size]
                enc = tok([query] * len(chunk), chunk, truncation=True, padding=True,
                          return_tensors="pt", max_length=self.max_len)
                out.append(model(**enc).logits.squeeze(-1).float().numpy().reshape(-1))
        return np.concatenate(out) if out else np.zeros(0)


class CachedReranker:
    """Memoises (query, text) -> score. Lets null-model reruns pay only for unseen pairs."""

    def __init__(self, inner: Reranker, path: str | Path | None = None):
        self.inner, self.name = inner, inner.name
        self.path = Path(path) if path else None
        self.cache: dict[str, float] = {}
        if self.path and self.path.exists():
            with open(self.path, "rb") as f:
                self.cache = pickle.load(f)
        self.misses = 0

    @staticmethod
    def _key(q: str, t: str) -> str:
        return hashlib.sha1(f"{q}\x00{t}".encode()).hexdigest()

    def score(self, query, texts):
        keys = [self._key(query, t) for t in texts]
        todo = [i for i, k in enumerate(keys) if k not in self.cache]
        if todo:
            s = self.inner.score(query, [texts[i] for i in todo])
            for i, v in zip(todo, s):
                self.cache[keys[i]] = float(v)
            self.misses += len(todo)
        return np.array([self.cache[k] for k in keys], dtype=float)

    def save(self):
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            with open(tmp, "wb") as f:
                pickle.dump(self.cache, f)
            tmp.replace(self.path)
