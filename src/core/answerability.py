"""
AnswerabilityGate — core refusal logic
"""

import os
import re
import time
import logging
from dataclasses import dataclass
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.35"))

_SYSTEM = (
    "You are a strict document QA system for Apache Airflow documentation.\n"
    "You may ONLY use the provided context to answer. Your training knowledge is DISABLED.\n"
    "If the context does NOT contain enough information to answer, respond with exactly: CANNOT_ANSWER\n"
    "Do not infer, extrapolate, or use knowledge not present in the context."
)

_HEDGING = [
    r"\bI think\b", r"\bprobably\b", r"\bmight be\b", r"\bI believe\b",
    r"\bI'm not sure\b", r"\bI'm not certain\b", r"\bseems like\b", r"\bI assume\b",
    r"\bnot entirely clear\b",
]


@dataclass
class AnswerabilityResult:
    is_answerable: bool
    best_score: float
    reason: str  
    raw_response: str = ""


class AnswerabilityGate:
    def __init__(self):
        key = os.environ.get("GEMINI_API_KEY")

        if not key:
            logger.warning(
                "GEMINI_API_KEY not set. "
                "Answerability LLM check disabled."
            )
            self.client = None
            self.model = None
            return

        self.client = genai.Client(api_key=key)
        self.model = os.getenv(
            "GEMINI_CHAT_MODEL",
            "models/gemini-3.1-flash-lite"
        )

    def check(self, question: str, retrieved_chunks: list) -> AnswerabilityResult:
        if not retrieved_chunks:
            return AnswerabilityResult(is_answerable=False, best_score=0.0, reason="no_chunks")

        best_score = max(c.get("score", 0) for c in retrieved_chunks)
        logger.debug(f"Best retrieval score: {best_score:.3f}")

        # Layer 1: similarity threshold 
        if best_score < SIMILARITY_THRESHOLD:
            logger.info(f"Refusing: score {best_score:.3f} < threshold {SIMILARITY_THRESHOLD}")
            return AnswerabilityResult(is_answerable=False, best_score=best_score, reason="below_threshold")
        if self.client is None:
            return AnswerabilityResult(
                is_answerable=True,
                best_score=best_score,
                reason="llm_disabled"
                )
        print("ANSWERABILITY CLIENT:", self.client is not None)
        # Layer 2: Gemini check
        context = self._format_context(retrieved_chunks)
        prompt = (
            f"Context from Airflow documentation:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Answer (or CANNOT_ANSWER if context is insufficient):"
        )

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=_SYSTEM,
                    max_output_tokens=700,
                    temperature=0.0,
                ),
            )
            raw = response.text.strip()
            # print("\nANSWERABILITY RAW RESPONSE:")
            # print(raw)
            # print()

            if raw.startswith("CANNOT_ANSWER"):
                return AnswerabilityResult(
                    is_answerable=False, best_score=best_score,
                    reason="llm_refused", raw_response=raw,
                )
            return AnswerabilityResult(
                is_answerable=True, best_score=best_score,
                reason="answerable", raw_response=raw,
            )
        except Exception as e:
            logger.error(f"Answerability check failed: {e}")
            # Fail open on error
            return AnswerabilityResult(is_answerable=True, best_score=best_score, reason="check_failed")

    def has_hedging(self, answer: str) -> bool:
        return any(re.search(p, answer, re.IGNORECASE) for p in _HEDGING)

    def _format_context(self, chunks: list) -> str:
        parts = []
        for i, chunk in enumerate(chunks):
            meta = chunk.get("metadata", {})
            title = meta.get("section_title", "Unknown Section")
            url = meta.get("source_url", "")
            parts.append(f"[Chunk {i+1}] {title}\nSource: {url}\n{chunk['text']}")
        return "\n\n---\n\n".join(parts)
