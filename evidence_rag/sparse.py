"""BM25 (Robertson/Okapi, Lucene-style idf) over a scipy sparse term matrix."""
from __future__ import annotations

import re

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

# Short English stop list; biomedical tokens like "p53" or "il-6" are kept.
STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the to was were with "
    "this these those which who whom than then there their them they not no but into via".split()
)
_TOKEN = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


class BM25:
    def __init__(self, k1: float = 0.9, b: float = 0.4):
        # k1/b defaults follow the Anserini/Pyserini BM25 defaults.
        self.k1, self.b = k1, b

    def fit(self, texts: list[str]) -> "BM25":
        self.vec = CountVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None)
        tf = self.vec.fit_transform(texts).tocsc().astype(np.float64)
        n_docs = tf.shape[0]
        dl = np.asarray(tf.sum(axis=1)).ravel()
        avgdl = dl.mean() if n_docs else 0.0
        df = np.diff(tf.indptr)  # csc: docs per term
        self.idf = np.log1p((n_docs - df + 0.5) / (df + 0.5))
        tf = tf.tocoo()
        denom = tf.data + self.k1 * (1 - self.b + self.b * dl[tf.row] / avgdl)
        w = tf.data * (self.k1 + 1) / denom * self.idf[tf.col]
        self.W = sp.csr_matrix((w, (tf.row, tf.col)), shape=tf.shape).T.tocsr()  # terms x docs
        self.n_docs = n_docs
        return self

    def scores(self, query: str) -> np.ndarray:
        q = self.vec.transform([query])
        q.data[:] = 1.0  # count each query term once
        return np.asarray((q @ self.W).todense()).ravel()
