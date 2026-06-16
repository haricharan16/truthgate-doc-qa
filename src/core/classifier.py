"""
False-premise detection using Gemini.

Two-layer approach:
1. Rule-based: fast regex against known Airflow facts (free, instant)
2. Gemini LLM: for suspicious questions that pass rule layer

Free tier: gemini-1.5-flash — 15 RPM, 1M TPM/day.
"""

import os
import re
import logging
import time
from dataclasses import dataclass
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    is_false_premise: bool
    correction: str = ""
    confidence: float = 0.0
    method: str = ""  # "rules" or "llm"


# ── Known Airflow facts (regex pattern, correction message) ──────────────────
KNOWN_AIRFLOW_FACTS = [
    (r"store[s]?\s+.*\bindexes?\b.*\bJSON\b",
     "Airflow does not store indexes in JSON. It uses a relational SQL database (SQLite/MySQL/PostgreSQL) for metadata storage."),

    (r"uses?\s+(MongoDB|Redis|Cassandra|DynamoDB)\s+(for\s+)?(its\s+)?(metadata|data|storage|backend)",
     "Airflow uses a relational SQL database for its metadata store, not NoSQL databases."),

    (r"stores?\s+.*\bDAGs?\b.*\bdatabase\b",
     "Airflow stores DAG definitions as Python files on the filesystem (dag_folder), not in the database."),

    (r"written\s+in\s+(Ruby|Java|Go|Rust|JavaScript|TypeScript|PHP|Scala)",
     "Airflow is written in Python, not Ruby/Java/Go/etc."),

    (r"DAGs?\s+written\s+in\s+(YAML|JSON|XML)",
     "Airflow DAGs are written in Python. YAML-based DAGs are a community plugin, not core Airflow."),

    (r"uses?\s+cron\s+daemon",
     "Airflow has its own scheduler process; it does not use the system cron daemon."),

    (r"webserver\s+.*\bschedule[s]?\b|schedule[s]?\b.*\bwebserver\b",
     "The Airflow webserver does NOT schedule tasks. The scheduler is a separate component."),

    (r"XCom.*large.file|large.file.*XCom|XCom.*designed.*large",
     "XCom is for small data (IDs, strings, small dicts), NOT large file transfers. Use S3/GCS for files."),

    (r"BashOperator\s+.*\bDocker\b|Docker\b.*\bBashOperator",
     "BashOperator runs shell commands directly on the host, not in Docker. Use DockerOperator for Docker."),

    (r"real.?time\s+streaming\s+mode|streaming\s+mode",
     "Airflow is a batch workflow orchestrator, not a real-time streaming platform."),
]

_SUSPICIOUS_PATTERNS = [
    r"does airflow (use|support|require|have)",
    r"is (it true|correct) that airflow",
    r"why does airflow (use|store|run|require)",
    r"since airflow (is|uses|has)",
    r"how does airflow.*unlike",
]

_FP_SYSTEM = (
    "You are a precise Apache Airflow expert who detects false premises in questions.\n"
    "A false premise is a factual error embedded in the question itself.\n"
    "Respond ONLY with one of:\n"
    "  FALSE_PREMISE: <one sentence correcting the error>\n"
    "  NOT_FALSE_PREMISE\n"
    "No other output."
)


def check_rules(question: str) -> ClassificationResult:
    """Fast regex-based false-premise check. No API call."""
    for pattern, correction in KNOWN_AIRFLOW_FACTS:
        if re.search(pattern, question, re.IGNORECASE):
            return ClassificationResult(
                is_false_premise=True,
                correction=correction,
                confidence=0.9,
                method="rules",
            )
    return ClassificationResult(is_false_premise=False, method="rules")


class FalsePremiseClassifier:
    def __init__(self):
        key = os.environ.get("GEMINI_API_KEY")

        if not key:
            logger.warning(
                "GEMINI_API_KEY not set. "
                "False-premise detection will use rules only."
            )
            self.client = None
            self.model = None
            return

        self.client = genai.Client(api_key=key)
        self.model = os.getenv(
            "GEMINI_CHAT_MODEL",
            "models/gemini-2.5-flash"
        )

    def classify(self, question: str) -> ClassificationResult:
        rule_result = check_rules(question)

        if rule_result.is_false_premise:
            return rule_result

        if self.client is None:
            return ClassificationResult(
                is_false_premise=False,
                method="rules_only"
            )

        if not self._looks_suspicious(question):
            return ClassificationResult(
                is_false_premise=False,
                method="rules_pass"
            )

        return self._llm_classify(question)

    def _looks_suspicious(self, question: str) -> bool:
        q = question.lower()
        return any(re.search(p, q) for p in _SUSPICIOUS_PATTERNS)

    def _llm_classify(self, question: str) -> ClassificationResult:
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=question,
                config=types.GenerateContentConfig(
                    system_instruction=_FP_SYSTEM,
                    max_output_tokens=80,
                    temperature=0.0,
                ),
            )
            text = response.text.strip()
            if text.startswith("FALSE_PREMISE:"):
                correction = text[len("FALSE_PREMISE:"):].strip()
                return ClassificationResult(
                    is_false_premise=True,
                    correction=correction,
                    confidence=0.75,
                    method="llm",
                )
            return ClassificationResult(is_false_premise=False, confidence=0.8, method="llm")
        except Exception as e:
            logger.warning(f"LLM false-premise check failed: {e}")
            return ClassificationResult(is_false_premise=False, method="llm_failed")
