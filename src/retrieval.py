"""Keyword retrieval: character n-gram BM25, zero dependencies.

Why char n-grams (2 and 3) instead of word tokenization?
Japanese has no spaces, so word retrieval needs a morphological analyzer
(MeCab/janome) — an extra dependency plus dictionary maintenance. Character
n-grams need no tokenizer and work well on short FAQ-sized documents; ASCII
runs (ATM, NISA, FAQ ids) are additionally indexed as whole tokens so exact
terms match strongly. This is the deliberate baseline (see DECISIONS.md);
embeddings would only be added if the eval suite showed retrieval failures.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

from .corpus import Doc

_K1 = 1.5
_B = 0.75

_ascii_word_re = re.compile(r"[A-Za-z0-9%]+")


def tokenize(text: str) -> list[str]:
    """Lowercased char 2/3-grams plus whole ASCII words."""
    lowered = text.lower()
    tokens: list[str] = list(_ascii_word_re.findall(lowered))
    compact = re.sub(r"\s+", "", lowered)
    for n in (2, 3):
        tokens.extend(compact[i : i + n] for i in range(len(compact) - n + 1))
    return tokens


class BM25Index:
    def __init__(self, docs: list[Doc]):
        self.docs = docs
        self.doc_tokens: list[list[str]] = [tokenize(d.text) for d in docs]
        self.doc_len = [len(t) for t in self.doc_tokens]
        self.avg_len = sum(self.doc_len) / len(docs) if docs else 0.0
        self.tf: list[dict[str, int]] = [{} for _ in docs]
        self.df: dict[str, int] = {}
        for i, tokens in enumerate(self.doc_tokens):
            counts: dict[str, int] = {}
            for t in tokens:
                counts[t] = counts.get(t, 0) + 1
            self.tf[i] = counts
            for term in counts:
                self.df[term] = self.df.get(term, 0) + 1
        self.n_docs = len(docs)

    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)

    def score(self, query_tokens: list[str], doc_index: int) -> float:
        s = 0.0
        counts = self.tf[doc_index]
        dl = self.doc_len[doc_index]
        norm = _K1 * (1 - _B + _B * dl / (self.avg_len or 1.0))
        for term in set(query_tokens):
            f = counts.get(term)
            if not f:
                continue
            idf = self._idf(term)
            s += idf * (f * (_K1 + 1)) / (f + norm)
        return s

    def search(self, query: str, k: int = 4) -> list[tuple[Doc, float]]:
        query_tokens = tokenize(query)
        scored = [
            (self.docs[i], self.score(query_tokens, i)) for i in range(self.n_docs)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [(doc, s) for doc, s in scored[:k] if s > 0]


@dataclass
class RetrievedDoc:
    id: str
    title: str
    text: str
    score: float


def format_context(retrieved: list[tuple[Doc, float]]) -> str:
    """Render retrieved docs for the prompt, with citation ids the model must
    repeat in its answer."""
    parts = []
    for doc, _score in retrieved:
        parts.append(f"[{doc.display_id()}] ({doc.title})\n{doc.text}")
    return "\n\n".join(parts)
