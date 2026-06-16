"""
Hybrid retrieval: dense (ChromaDB) + sparse (BM25) with Reciprocal Rank Fusion.

Why RRF instead of score interpolation?
- RRF doesn't require normalizing scores across different scales
- More robust to outliers (a single high BM25 score doesn't dominate)
- RRF formula: score(d) = sum(1 / (k + rank(d, system_i)))
  where k=60 (standard), rank is 1-indexed position in each list

Dense weight: 0.6, Sparse weight: 0.4
These weights were tuned on a 10-question dev set. See DECISIONS.md.

After fusion: top-K_RERANK results are "reranked" by re-scoring them with
cosine similarity against the query. This is a naive reranker (no cross-encoder).
"""

import os
import logging
from typing import Optional

from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25 import BM25Index
from src.retrieval.embedder import Embedder

logger = logging.getLogger(__name__)

RRF_K = 60
DENSE_WEIGHT = 0.6
SPARSE_WEIGHT = 0.4
TOP_K = int(os.getenv("TOP_K_RETRIEVAL", 8))
TOP_K_RERANK = int(os.getenv("TOP_K_RERANK", 3))


class HybridRetriever:
    def __init__(self, vector_store: VectorStore, bm25_index: BM25Index, embedder: Embedder):
        self.vs = vector_store
        self.bm25 = bm25_index
        self.embedder = embedder

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K,
        top_k_rerank: int = TOP_K_RERANK,
        synthesis_mode: bool = False,
    ) -> list[dict]:
        """
        Full hybrid retrieval pipeline.
        
        synthesis_mode: if True, doubles top_k to improve multi-section synthesis.
        Triggered when question contains synthesis-indicator words.
        
        Returns top_k_rerank results, each with:
        {text, metadata, score, bm25_score, dense_score, rrf_score, final_rank}
        """
        if synthesis_mode:
            top_k = min(top_k * 2, 16)

        # 1. Embed query
        query_embedding = self.embedder.embed_query(query)

        # 2. Dense retrieval
        dense_results = self.vs.query(query_embedding, top_k=top_k)

        # 3. Sparse retrieval
        sparse_results = self.bm25.query(query, top_k=top_k)

        # 4. RRF fusion
        fused = self._rrf_fusion(dense_results, sparse_results)

        # 5. Re-score by cosine similarity (naive reranker)
        # Keeps semantic ordering as final signal
        reranked = sorted(fused, key=lambda x: x["score"], reverse=True)

        top = reranked[:top_k_rerank]
        for i, r in enumerate(top):
            r["final_rank"] = i + 1

        return top

    def _rrf_fusion(self, dense: list[dict], sparse: list[dict]) -> list[dict]:
        """Combine dense and sparse results using RRF."""
        # Build lookup by text fingerprint (first 100 chars as key)
        def fp(text):
            return text[:100]

        scores = {}  # fp -> accumulated score + data
        
        for rank, result in enumerate(dense):
            key = fp(result["text"])
            rrf = DENSE_WEIGHT / (RRF_K + rank + 1)
            if key not in scores:
                scores[key] = {**result, "rrf_score": 0, "bm25_score": 0}
            scores[key]["rrf_score"] += rrf
            scores[key]["dense_score"] = result.get("score", 0)

        for rank, result in enumerate(sparse):
            key = fp(result["text"])
            rrf = SPARSE_WEIGHT / (RRF_K + rank + 1)
            if key not in scores:
                scores[key] = {
                    "text": result["text"],
                    "metadata": result["metadata"],
                    "score": result["bm25_norm"],
                    "rrf_score": 0,
                    "dense_score": 0,
                }
            scores[key]["rrf_score"] += rrf
            scores[key]["bm25_score"] = result.get("bm25_norm", 0)

        # Sort by rrf_score, but use dense score as tiebreaker
        all_results = list(scores.values())
        all_results.sort(key=lambda x: (x["rrf_score"], x.get("dense_score", 0)), reverse=True)
        return all_results


SYNTHESIS_INDICATORS = {
    "when", "what happens", "interaction", "relationship", "affect",
    "difference between", "compare", "both", "also", "together", "combination"
}


def is_synthesis_query(query: str) -> bool:
    """Heuristic: does this query likely require multi-section synthesis?"""
    q = query.lower()
    return any(ind in q for ind in SYNTHESIS_INDICATORS)
