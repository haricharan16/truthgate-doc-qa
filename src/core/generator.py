"""
Answer generator with mandatory citations — using Gemini 1.5 Flash (FREE).

Free tier: 15 RPM, 1M TPM/day — sufficient for all 60 eval questions.
Cost: $0.00 on free tier.

Gemini 1.5 Flash paid pricing (if you exceed free tier):
  Input:  $0.075 per 1M tokens
  Output: $0.30  per 1M tokens
  → ~$0.0001 per query (still well under $0.02 budget)
"""

import os
import time
import logging
from dataclasses import dataclass
from google import genai
from google.genai import types
import re

logger = logging.getLogger(__name__)

# Gemini 1.5 Flash pricing (if paid tier needed)
INPUT_COST_PER_M  = 0.075   # $0.075 per 1M input tokens
OUTPUT_COST_PER_M = 0.30    # $0.30  per 1M output tokens

_SYSTEM = """You are a precise technical documentation assistant for Apache Airflow.

STRICT RULES:
1. Answer ONLY from the provided documentation context — never from memory.
2. Every factual claim MUST include a citation: [Source 1], [Source 2], etc.
3. If multiple sources support a claim, cite all: [Source 1, Source 2].
4. Be concise but complete. 2-5 sentences for simple questions; up to 10 for complex ones.
5. Include exact config values, parameter names, or code snippets if present in the docs.
6. End your answer with:
   Sources: [Source N — section title — URL]

Format example:
The scheduler runs tasks based on the DAG's schedule_interval [Source 1]. It polls the
metadata database every heartbeat_interval seconds [Source 2].
Sources: [Source 1 — Scheduler Overview — https://...] [Source 2 — Configuration — https://...]"""


@dataclass
class GenerationResult:
    answer: str
    citations: list
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    has_citations: bool


class AnswerGenerator:
    def __init__(self):
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY not set.")
        self.client = genai.Client(api_key=key)
        self.model = os.getenv("GEMINI_CHAT_MODEL", "models/gemini-2.5-flash")

    def generate(self, question: str, chunks: list, preflight_answer: str = "") -> GenerationResult:
        context, source_map = self._build_context(chunks)
        prompt = f"Documentation context:\n{context}\n\nQuestion: {question}\n\nAnswer with citations:"

        t0 = time.time()
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM,
                max_output_tokens=900,
                temperature=0.1,
            ),
        )
        latency_ms = (time.time() - t0) * 1000
        raw = response.text.strip()

        # Token counts from usage metadata
        usage = getattr(response, "usage_metadata", None)
        input_tokens  = getattr(usage, "prompt_token_count",     0) if usage else 0
        output_tokens = getattr(usage, "candidates_token_count", 0) if usage else 0
        cost = self._calc_cost(input_tokens, output_tokens)

        has_citations = "[Source" in raw
        citations = self._extract_citations(raw, source_map)

        return GenerationResult(
            answer=raw,
            citations=citations,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            latency_ms=latency_ms,
            has_citations=has_citations,
        )

    def _build_context(self, chunks: list) -> tuple:
        parts = []
        source_map = {}
        max_chars = 2400   # ~600 tokens per chunk

        for i, chunk in enumerate(chunks):
            n = i + 1
            meta = chunk.get("metadata", {})
            title = meta.get("section_title", f"Section {n}")
            url   = meta.get("source_url", "")
            anchor = meta.get("section_anchor", "")
            full_url = f"{url}#{anchor}" if anchor else url

            text = chunk["text"]
            if len(text) > max_chars:
                text = text[:max_chars] + "..."

            parts.append(f"[Source {n}] {title}\nURL: {full_url}\n{text}")
            source_map[n] = {"title": title, "url": full_url}

        return "\n\n---\n\n".join(parts), source_map

    def _extract_citations(self, answer: str, source_map: dict) -> list:
        cited = []
        seen = set()
        for m in re.finditer(r"\[Source (\d+)\]", answer):
            n = int(m.group(1))
            if n in source_map and n not in seen:
                cited.append({"source_n": n, **source_map[n]})
                seen.add(n)
        return cited

    def _calc_cost(self, inp: int, out: int) -> float:
        # On free tier this is effectively $0; formula for paid tier:
        return (inp * INPUT_COST_PER_M + out * OUTPUT_COST_PER_M) / 1_000_000
