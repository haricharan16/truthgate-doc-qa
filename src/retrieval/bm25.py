"""
BM25 sparse retrieval.

Why include BM25 alongside dense retrieval?
- Handles exact keyword matches better (config param names, operator class names)
- "PythonOperator", "DAG", "XCom" are terms where exact match matters
- Complements dense retrieval which is better at semantic similarity

Implementation: Pure Python BM25 (no external library needed).
Math: BM25 score for term t in document d:
    score(d,t) = IDF(t) * (tf(t,d) * (k1+1)) / (tf(t,d) + k1*(1-b+b*|d|/avgdl))

k1=1.5, b=0.75 (standard defaults)
"""

import math
import re
import json
import pickle
from pathlib import Path
from collections import defaultdict
from typing import Optional


def tokenize(text: str) -> list[str]:
    """Simple tokenizer: lowercase, split on non-alphanumeric, filter short."""
    return [t for t in re.split(r"[^a-zA-Z0-9_]", text.lower()) if len(t) > 2 and t not in ("the", "and", "for", "not")]


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs: list[str] = []       # raw text of each doc
        self.doc_ids: list[str] = []    # chunk_id parallel to docs
        self.metadatas: list[dict] = [] # metadata parallel to docs
        self.tf: list[dict] = []        # term frequency per doc
        self.df: dict[str, int] = defaultdict(int)  # document frequency
        self.avgdl: float = 0
        self.N: int = 0

    def add_documents(self, chunks: list[dict]) -> None:
        """Add chunks to index. chunk must have 'text', 'chunk_id', and metadata."""
        for chunk in chunks:
            tokens = tokenize(chunk["text"])
            tf = defaultdict(int)
            for t in tokens:
                tf[t] += 1
            for t in set(tokens):
                self.df[t] += 1
            self.docs.append(chunk["text"])
            self.doc_ids.append(chunk["chunk_id"])
            self.metadatas.append({
                "source_url": chunk["source_url"],
                "section_title": chunk["section_title"],
                "page_title": chunk["page_title"],
                "section_anchor": chunk["section_anchor"],
            })
            self.tf.append(dict(tf))

        self.N = len(self.docs)
        self.avgdl = sum(len(tokenize(d)) for d in self.docs) / max(self.N, 1)

    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        if df == 0:
            return 0
        return math.log((self.N - df + 0.5) / (df + 0.5) + 1)

    def score(self, query_tokens: list[str], doc_idx: int) -> float:
        dl = sum(self.tf[doc_idx].values())
        score = 0.0
        for t in query_tokens:
            if t not in self.tf[doc_idx]:
                continue
            tf = self.tf[doc_idx][t]
            idf = self._idf(t)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            score += idf * numerator / denominator
        return score

    def query(self, query: str, top_k: int = 8) -> list[dict]:
        """Return top_k results sorted by BM25 score descending."""
        if self.N == 0:
            return []

        tokens = tokenize(query)
        scores = [(i, self.score(tokens, i)) for i in range(self.N)]
        scores.sort(key=lambda x: x[1], reverse=True)
        top = scores[:top_k]

        results = []
        max_score = top[0][1] if top and top[0][1] > 0 else 1.0
        for i, raw_score in top:
            if raw_score <= 0:
                continue
            results.append({
                "text": self.docs[i],
                "metadata": self.metadatas[i],
                "bm25_score": raw_score,
                "bm25_norm": raw_score / max_score,  # normalized 0..1
            })
        return results

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        with open(Path(path), "rb") as f:
            return pickle.load(f)
