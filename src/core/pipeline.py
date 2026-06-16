"""
TruthGate Pipeline — orchestrates all components (Gemini version).

Query flow:
1. FalsePremiseClassifier → if FALSE_PREMISE: return immediately
2. HybridRetriever        → fetch top-k chunks
3. AnswerabilityGate      → refuse if below threshold or LLM says no
4. AnswerGenerator        → generate cited answer
5. Hedging check + cost log
"""

import os
import time
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal

from src.core.classifier import FalsePremiseClassifier
from src.core.answerability import AnswerabilityGate
from src.core.generator import AnswerGenerator
from src.retrieval.hybrid import HybridRetriever, is_synthesis_query

logger = logging.getLogger(__name__)

COST_LOG = Path(os.getenv("COST_LOG_FILE", "./logs/cost_log.jsonl"))
COST_LOG.parent.mkdir(parents=True, exist_ok=True)

ResponseType = Literal["ANSWERED", "NOT_IN_DOCS", "FALSE_PREMISE", "ERROR"]


@dataclass
class QueryResponse:
    response_type: ResponseType
    question: str
    answer: str = ""
    citations: list = None
    has_citations: bool = False
    has_hedging: bool = False
    refusal_reason: str = ""
    best_retrieval_score: float = 0.0
    correction: str = ""
    fp_method: str = ""
    total_cost_usd: float = 0.0
    total_latency_ms: float = 0.0
    chunks_retrieved: int = 0

    def __post_init__(self):
        if self.citations is None:
            self.citations = []

    def to_dict(self) -> dict:
        return asdict(self)


class TruthGatePipeline:
    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever
        self.fp_classifier = FalsePremiseClassifier()
        self.answerability_gate = AnswerabilityGate()
        self.generator = AnswerGenerator()

    def query(self, question: str) -> QueryResponse:
        t_start = time.time()
        cost = 0.0

        try:
            # Step 1: False-premise check
            fp = self.fp_classifier.classify(question)
            if fp.is_false_premise:
                return self._respond(QueryResponse(
                    response_type="FALSE_PREMISE",
                    question=question,
                    correction=fp.correction,
                    fp_method=fp.method,
                    total_cost_usd=0.0,
                    total_latency_ms=(time.time() - t_start) * 1000,
                ))

            # Step 2: Retrieve
            chunks = self.retriever.retrieve(question, synthesis_mode=is_synthesis_query(question))
            print("\nRETRIEVED CHUNKS")
            for i, c in enumerate(chunks):
                title = c.get("metadata", {}).get("section_title")
                print(f"\n--- Chunk {i+1} ---")
                print("TITLE:", title)
                print(c["text"][:500])

            # Step 3: Answerability gate
            ans = self.answerability_gate.check(question, chunks)
            if not ans.is_answerable:
                return self._respond(QueryResponse(
                    response_type="NOT_IN_DOCS",
                    question=question,
                    refusal_reason=ans.reason,
                    best_retrieval_score=ans.best_score,
                    chunks_retrieved=len(chunks),
                    total_cost_usd=0.0,
                    total_latency_ms=(time.time() - t_start) * 1000,
                ))

            # Step 4: Generate
            gen = self.generator.generate(question=question, chunks=chunks)
            cost += gen.cost_usd

            return self._respond(QueryResponse(
                response_type="ANSWERED",
                question=question,
                answer=gen.answer,
                citations=gen.citations,
                has_citations=gen.has_citations,
                has_hedging=self.answerability_gate.has_hedging(gen.answer),
                chunks_retrieved=len(chunks),
                best_retrieval_score=ans.best_score,
                total_cost_usd=cost,
                total_latency_ms=(time.time() - t_start) * 1000,
            ))

        except Exception as e:
            import traceback
            print("\n" + "="*80)
            print("PIPELINE ERROR")
            traceback.print_exc()
            print("="*80 + "\n")
            logger.error(f"Pipeline error: {e}", exc_info=True)
            return QueryResponse(
                response_type="ERROR",
                question=question,
                answer=str(e),
                total_latency_ms=(time.time() - t_start) * 1000,
            )

    def _respond(self, resp: QueryResponse) -> QueryResponse:
        try:
            with open(COST_LOG, "a") as f:
                f.write(json.dumps({
                    "type": resp.response_type,
                    "cost": resp.total_cost_usd,
                    "latency_ms": resp.total_latency_ms,
                    "question": resp.question[:100],
                }) + "\n")
        except Exception:
            pass
        return resp
