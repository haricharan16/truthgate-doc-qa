"""
Unit tests for TruthGate core logic (Gemini version).
All LLM calls are mocked — no API key needed for tests.

Run: python -m pytest tests/ -v
"""

import sys
import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ["GEMINI_API_KEY"] = "AIza-test-key"

# ── BM25 Tests (no API) ──────────────────────────────────────────────────────

from src.retrieval.bm25 import BM25Index, tokenize


class TestBM25:
    def test_tokenize_basic(self):
        tokens = tokenize("Hello World foo")
        assert "hello" in tokens
        assert "world" in tokens

    def test_tokenize_filters_short_and_stopwords(self):
        tokens = tokenize("a to is the")
        assert tokens == []

    def test_empty_index_returns_empty(self):
        assert BM25Index().query("anything") == []

    def _make_doc(self, cid, text, title="S"):
        return {"chunk_id": cid, "text": text, "source_url": "u",
                "section_title": title, "page_title": "P", "section_anchor": ""}

    def test_basic_retrieval(self):
        b = BM25Index()
        b.add_documents([
            self._make_doc("c1", "Airflow DAG scheduling uses cron expressions", "Scheduling"),
            self._make_doc("c2", "Python operators run Python callables", "Operators"),
        ])
        results = b.query("DAG scheduling")
        assert results[0]["metadata"]["section_title"] == "Scheduling"

    def test_exact_keyword_ranks_higher(self):
        b = BM25Index()
        b.add_documents([
            self._make_doc("c1", "XCom cross-communication between tasks", "XCom"),
            self._make_doc("c2", "Tasks share data using various mechanisms", "Tasks"),
        ])
        assert b.query("XCom cross-communication")[0]["metadata"]["section_title"] == "XCom"

    def test_positive_score(self):
        b = BM25Index()
        b.add_documents([self._make_doc("c1", "scheduler runs tasks every minute")])
        assert b.query("scheduler tasks")[0]["bm25_score"] > 0

    def test_irrelevant_zero_score(self):
        b = BM25Index()
        b.add_documents([self._make_doc("c1", "airflow dag scheduling tasks operators")])
        results = b.query("quantum physics neutron star")
        if results:
            assert results[0]["bm25_score"] == 0


# ── Chunker Tests (no API) ───────────────────────────────────────────────────

from src.ingestion.chunker import chunk_section, count_tokens, MAX_TOKENS


class TestChunker:
    def _sec(self, content, title="Test Section"):
        return {"title": title, "content": content, "url": "http://ex.com/docs",
                "section_anchor": "test", "page_title": "Test Page"}

    def test_short_section_one_chunk(self):
        assert len(chunk_section(self._sec("Short content."))) == 1

    def test_empty_no_chunks(self):
        assert chunk_section(self._sec("")) == []

    def test_whitespace_no_chunks(self):
        assert chunk_section(self._sec("   \n\n\t  ")) == []

    def test_long_section_splits(self):
        chunks = chunk_section(self._sec("This is a test sentence about Airflow. " * 200))
        assert len(chunks) > 1

    def test_chunk_respects_max_tokens(self):
        chunks = chunk_section(self._sec("Word " * 2000))
        for c in chunks:
            assert c.token_count <= MAX_TOKENS + 60

    def test_metadata_preserved(self):
        chunks = chunk_section(self._sec("Some content.", title="Page > Section"))
        assert chunks[0].source_url == "http://ex.com/docs"
        assert chunks[0].section_title == "Page > Section"
        assert chunks[0].section_anchor == "test"

    def test_chunk_ids_unique(self):
        chunks = chunk_section(self._sec("Para about airflow.\n\n" * 100))
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_citation_contains_url(self):
        chunks = chunk_section(self._sec("Some content."))
        assert "http://ex.com/docs" in chunks[0].citation


# ── False-Premise Rule Tests (no API) ────────────────────────────────────────

from src.core.classifier import check_rules


class TestFalsePremiseRules:
    def test_json_indexes(self):
        assert check_rules("Why does Airflow store all indexes in JSON format?").is_false_premise

    def test_ruby(self):
        assert check_rules("Since Airflow is written in Ruby, how do I use gems?").is_false_premise

    def test_mongodb(self):
        assert check_rules("Why does Airflow use MongoDB for metadata storage?").is_false_premise

    def test_yaml_dags(self):
        assert check_rules("How do I write DAGs written in YAML in Airflow?").is_false_premise

    def test_webserver_schedules(self):
        assert check_rules("How do I configure the webserver to schedule DAG runs?").is_false_premise

    def test_xcom_large_files(self):
        assert check_rules("Since XCom is designed for large file transfers, what is the limit?").is_false_premise

    def test_valid_passes(self):
        assert not check_rules("What is the default executor in Airflow?").is_false_premise

    def test_valid_python_passes(self):
        assert not check_rules("How do I write a Python operator in Airflow?").is_false_premise

    def test_valid_scheduler_passes(self):
        assert not check_rules("How does the Airflow scheduler determine when to run a DAG?").is_false_premise

    def test_correction_nonempty(self):
        r = check_rules("Airflow is written in Ruby right?")
        assert r.is_false_premise and len(r.correction) > 10


# ── Answerability Gate Tests (mocked Gemini) ─────────────────────────────────

from src.core.answerability import AnswerabilityGate, SIMILARITY_THRESHOLD


class TestAnswerabilityGate:
    def _chunks(self, score):
        return [{"text": "Some text", "metadata": {"section_title": "S", "source_url": "u"}, "score": score}]

    def test_below_threshold_refuses(self):
        gate = AnswerabilityGate()
        result = gate.check("Any question?", self._chunks(SIMILARITY_THRESHOLD - 0.05))
        assert not result.is_answerable
        assert result.reason == "below_threshold"

    def test_empty_chunks_refuses(self):
        gate = AnswerabilityGate()
        result = gate.check("Any question?", [])
        assert not result.is_answerable
        assert result.reason == "no_chunks"

    def test_llm_cannot_answer(self):
        gate = AnswerabilityGate()
        mock_resp = MagicMock()
        mock_resp.text = "CANNOT_ANSWER — context insufficient"
        gate.client = MagicMock()
        gate.client.models.generate_content.return_value = mock_resp

        result = gate.check("Question not in docs?", self._chunks(SIMILARITY_THRESHOLD + 0.1))
        assert not result.is_answerable
        assert result.reason == "llm_refused"

    def test_llm_answers(self):
        gate = AnswerabilityGate()
        mock_resp = MagicMock()
        mock_resp.text = "The default executor is SequentialExecutor [Source 1]."
        gate.client = MagicMock()
        gate.client.models.generate_content.return_value = mock_resp

        result = gate.check("What is the default executor?", self._chunks(SIMILARITY_THRESHOLD + 0.2))
        assert result.is_answerable
        assert result.reason == "answerable"

    def test_hedging_detected(self):
        gate = AnswerabilityGate()
        assert gate.has_hedging("I think the answer might be X")
        assert gate.has_hedging("I believe this is correct")
        assert gate.has_hedging("Probably around 32 workers")

    def test_no_hedging_confident(self):
        gate = AnswerabilityGate()
        assert not gate.has_hedging("The CeleryExecutor distributes tasks across worker nodes.")


# ── Pipeline Integration Tests (fully mocked) ────────────────────────────────

from src.core.pipeline import TruthGatePipeline


class TestPipeline:
    def _pipeline(self, retrieval_score=0.1):
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [
            {"text": "chunk", "metadata": {"section_title": "S", "source_url": "u"}, "score": retrieval_score}
        ]
        p = TruthGatePipeline(mock_retriever)
        p.fp_classifier = MagicMock()
        p.fp_classifier.classify.return_value = MagicMock(is_false_premise=False)
        return p

    def test_low_score_returns_not_in_docs(self):
        result = self._pipeline(retrieval_score=0.1).query("Some question?")
        assert result.response_type == "NOT_IN_DOCS"
        assert result.refusal_reason == "below_threshold"

    def test_false_premise_skips_retrieval(self):
        p = self._pipeline(retrieval_score=0.9)
        p.fp_classifier.classify.return_value = MagicMock(
            is_false_premise=True,
            correction="Airflow does not use Ruby.",
            method="rules"
        )
        result = p.query("Since Airflow is written in Ruby...")
        assert result.response_type == "FALSE_PREMISE"
        p.retriever.retrieve.assert_not_called()

    def test_answered_response(self):
        p = self._pipeline(retrieval_score=0.8)
        p.answerability_gate = MagicMock()
        p.answerability_gate.check.return_value = MagicMock(
            is_answerable=True, best_score=0.8, reason="answerable", raw_response="Good"
        )
        p.answerability_gate.has_hedging.return_value = False

        mock_gen = MagicMock()
        mock_gen.answer = "The default executor is SequentialExecutor [Source 1]."
        mock_gen.citations = [{"source_n": 1, "title": "S", "url": "u"}]
        mock_gen.cost_usd = 0.0
        mock_gen.latency_ms = 900
        mock_gen.has_citations = True
        p.generator = MagicMock()
        p.generator.generate.return_value = mock_gen

        result = p.query("What is the default executor?")
        assert result.response_type == "ANSWERED"
        assert "SequentialExecutor" in result.answer


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
