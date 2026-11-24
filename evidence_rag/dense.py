"""Dense encoders.

`LSAEncoder` is a dependency-light stand-in (TF-IDF + truncated SVD) used by tests and the
synthetic smoke test. `MedCPTEncoder` wraps NCBI's MedCPT query/article encoders
(Jin et al. 2023, Bioinformatics), which score by inner product of [CLS] embeddings.
`SentenceTransformerEncoder` wraps a general-domain model (default BAAI/bge-small-en-v1.5).
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol

import numpy as np

from .data import Doc


class Encoder(Protocol):
    name: str

    def encode_queries(self, texts: list[str]) -> np.ndarray: ...
    def encode_docs(self, docs: list[Doc]) -> np.ndarray: ...


class LSAEncoder:
    name = "lsa"

    def __init__(self, dim: int = 64, seed: int = 0):
        self.dim, self.seed = dim, seed
        self._fitted = False

    def fit(self, docs: list[Doc]) -> "LSAEncoder":
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        from .sparse import tokenize

        self.tfidf = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None, sublinear_tf=True)
        X = self.tfidf.fit_transform([d.text for d in docs])
        k = max(1, min(self.dim, X.shape[1] - 1, X.shape[0] - 1))
        self.svd = TruncatedSVD(n_components=k, random_state=self.seed).fit(X)
        self._fitted = True
        return self

    def _enc(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("call fit(docs) first")
        Z = self.svd.transform(self.tfidf.transform(texts))
        return Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)

    def encode_queries(self, texts):
        return self._enc(texts)

    def encode_docs(self, docs):
        return self._enc([d.text for d in docs])


def _torch_threads():
    import torch

    torch.set_num_threads(max(1, torch.get_num_threads()))
    return torch


class MedCPTEncoder:
    name = "medcpt"
    QUERY_MODEL = "ncbi/MedCPT-Query-Encoder"
    ARTICLE_MODEL = "ncbi/MedCPT-Article-Encoder"

    def __init__(self, batch_size: int = 16, max_doc_len: int = 512, max_query_len: int = 64):
        self.batch_size, self.max_doc_len, self.max_query_len = batch_size, max_doc_len, max_query_len
        self._q = self._a = None

    def _load(self, which: str):
        from transformers import AutoModel, AutoTokenizer

        name = self.QUERY_MODEL if which == "q" else self.ARTICLE_MODEL
        return AutoTokenizer.from_pretrained(name), AutoModel.from_pretrained(name).eval()

    def _run(self, tok, model, batch_args, max_len):
        torch = _torch_threads()
        out = []
        with torch.inference_mode():
            for i in range(0, len(batch_args), self.batch_size):
                chunk = batch_args[i : i + self.batch_size]
                if isinstance(chunk[0], (list, tuple)):
                    enc = tok([c[0] for c in chunk], [c[1] for c in chunk], truncation=True,
                              padding=True, return_tensors="pt", max_length=max_len)
                else:
                    enc = tok(chunk, truncation=True, padding=True, return_tensors="pt", max_length=max_len)
                out.append(model(**enc).last_hidden_state[:, 0, :].float().numpy())
        return np.concatenate(out) if out else np.zeros((0, 768), dtype=np.float32)

    def encode_queries(self, texts):
        if self._q is None:
            self._q = self._load("q")
        return self._run(*self._q, list(texts), self.max_query_len)

    def encode_docs(self, docs):
        if self._a is None:
            self._a = self._load("a")
        return self._run(*self._a, [[d.title, " ".join(d.sentences)] for d in docs], self.max_doc_len)


class SentenceTransformerEncoder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", batch_size: int = 32,
                 query_prefix: str = "Represent this sentence for searching relevant passages: "):
        self.model_name, self.batch_size, self.query_prefix = model_name, batch_size, query_prefix
        self.name = model_name.split("/")[-1]
        self._m = None

    def _model(self):
        if self._m is None:
            from sentence_transformers import SentenceTransformer

            self._m = SentenceTransformer(self.model_name, device="cpu")
        return self._m

    def encode_queries(self, texts):
        return self._model().encode([self.query_prefix + t for t in texts], batch_size=self.batch_size,
                                    normalize_embeddings=True)

    def encode_docs(self, docs):
        return self._model().encode([d.text for d in docs], batch_size=self.batch_size, normalize_embeddings=True)


def corpus_fingerprint(docs: list[Doc]) -> str:
    h = hashlib.sha1()
    for d in docs:
        h.update(f"{d.doc_id}\x00{d.text}\x01".encode())
    return h.hexdigest()[:12]


def cached_doc_embeddings(encoder: Encoder, docs: list[Doc], cache_dir: str | Path | None) -> np.ndarray:
    if cache_dir is None:
        return encoder.encode_docs(docs)
    path = Path(cache_dir) / f"docemb_{encoder.name}_{corpus_fingerprint(docs)}.npy"
    if path.exists():
        return np.load(path)
    E = encoder.encode_docs(docs)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, E)
    return E
