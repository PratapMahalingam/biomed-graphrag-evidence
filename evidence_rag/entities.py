"""Biomedical entity extraction and the document-entity incidence matrix.

Entities are keyed by normalised surface text (no ontology linking). This is a deliberate
simplification: it merges "TNF-alpha" and "tnf alpha" but not "TNF-alpha" and "cachectin".
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol

import numpy as np
import scipy.sparse as sp

from .data import Doc
from .sparse import STOPWORDS

_WS = re.compile(r"\s+")
_EDGE = re.compile(r"^[^\w]+|[^\w]+$")
_NGRAM_TOKEN = re.compile(r"[a-z0-9]+")


def normalize_entity(text: str) -> str:
    t = text.lower().replace("-", " ").replace("/", " ")
    t = _EDGE.sub("", _WS.sub(" ", t).strip())
    return t


def valid_entity(key: str, min_chars: int = 3) -> bool:
    return len(key) >= min_chars and key not in STOPWORDS and not key.isdigit()


class Extractor(Protocol):
    def extract(self, texts: list[str]) -> list[set[str]]: ...


class DictionaryExtractor:
    """Exact n-gram lookup of a fixed vocabulary of normalised entity keys."""

    def __init__(self, vocabulary: Iterable[str], max_n: int = 6):
        self.vocab = {normalize_entity(v) for v in vocabulary}
        self.vocab = {v for v in self.vocab if valid_entity(v)}
        self.max_n = max_n

    def extract(self, texts):
        out = []
        for t in texts:
            toks = _NGRAM_TOKEN.findall(normalize_entity(t))
            found = set()
            for n in range(1, self.max_n + 1):
                for i in range(len(toks) - n + 1):
                    g = " ".join(toks[i : i + n])
                    if g in self.vocab:
                        found.add(g)
            out.append(found)
        return out


class ScispacyExtractor:
    """Union of scispaCy NER models (default: BC5CDR chemicals/diseases + JNLPBA genes/proteins/cells)."""

    def __init__(self, model_paths: list[str | Path], batch_size: int = 64, n_process: int = 1):
        import spacy

        self.nlps = [spacy.load(str(p), exclude=["tagger", "attribute_ruler", "lemmatizer", "parser"])
                     for p in model_paths]
        self.batch_size, self.n_process = batch_size, n_process

    def extract(self, texts):
        out = [set() for _ in texts]
        for nlp in self.nlps:
            for i, doc in enumerate(nlp.pipe(texts, batch_size=self.batch_size, n_process=self.n_process)):
                for e in doc.ents:
                    k = normalize_entity(e.text)
                    if valid_entity(k):
                        out[i].add(k)
        return out


@dataclass
class EntityIndex:
    vocab: list[str]               # entity id -> key
    M: sp.csr_matrix               # docs x entities, binary
    doc_ids: list[int]

    @property
    def key_to_id(self) -> dict[str, int]:
        if not hasattr(self, "_k2i"):
            self._k2i = {k: i for i, k in enumerate(self.vocab)}
        return self._k2i

    @property
    def df(self) -> np.ndarray:
        return np.asarray(self.M.sum(axis=0)).ravel()

    def idf(self) -> np.ndarray:
        n = self.M.shape[0]
        return np.log((n + 1) / (self.df + 0.5))

    def query_vector(self, keys: Iterable[str]) -> np.ndarray:
        v = np.zeros(len(self.vocab))
        k2i = self.key_to_id
        for k in keys:
            if k in k2i:
                v[k2i[k]] = 1.0
        return v

    @classmethod
    def build(cls, docs: list[Doc], doc_entities: list[set[str]], max_df_frac: float = 0.05,
              min_df: int = 2) -> "EntityIndex":
        """Keep entities seen in >= min_df docs and <= max_df_frac of docs.

        Singletons cannot link two documents; very common entities ("patients", "cancer")
        link everything and carry little signal.
        """
        df = Counter(k for ents in doc_entities for k in ents)
        max_df = max(min_df, int(max_df_frac * len(docs)))
        vocab = sorted(k for k, c in df.items() if min_df <= c <= max_df)
        k2i = {k: i for i, k in enumerate(vocab)}
        rows, cols = [], []
        for r, ents in enumerate(doc_entities):
            for k in ents:
                if k in k2i:
                    rows.append(r)
                    cols.append(k2i[k])
        M = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(docs), len(vocab)))
        M.sum_duplicates()
        M.data[:] = 1.0
        return cls(vocab, M, [d.doc_id for d in docs])
