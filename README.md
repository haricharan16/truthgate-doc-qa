# TruthGate — RAG System That Knows When To Shut Up
### Powered by Google Gemini (100% FREE — no credit card needed)

**Corpus:** Apache Airflow official documentation (https://airflow.apache.org/docs/stable/)

---

## Free Tier Limits (Gemini)

| Resource | Free Limit | Our Usage |
|----------|-----------|-----------|
| `gemini-1.5-flash` RPM | 15 req/min | ~2 req/query |
| `gemini-1.5-flash` TPD | 1M tokens/day | ~900 tokens/query |
| `text-embedding-004` RPD | 1,500 req/day | 100 calls for ingest |
| Cost | **$0.00** | **$0.00** |

---

## Quick Start (3 commands)

```bash
# 1. Setup
cp .env.example .env
# → Open .env, paste your FREE Gemini key from https://aistudio.google.com/apikey

pip install -r requirements.txt
mkdir -p data/chroma data/corpus logs

# 2. Ingest Airflow docs (~10-15 min, runs once)
python scripts/ingest.py

# 3. Ask a question
python run_query.py "How does Airflow schedule DAGs?"
```

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────┐
│  FalsePremiseClassifier                 │
│  Layer 1: Regex rules (instant, free)   │
│  Layer 2: gemini-1.5-flash (if needed)  │
└─────────────────────────────────────────┘
    │ (if not false-premise)
    ▼
┌─────────────────────────────────────────┐
│  HybridRetriever                        │
│  Dense: ChromaDB + text-embedding-004   │
│  Sparse: BM25 (pure Python)             │
│  Fusion: RRF (0.6 dense + 0.4 sparse)  │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  AnswerabilityGate                      │
│  Layer 1: cosine similarity < 0.35      │  ← hard refuse, no LLM call
│  Layer 2: gemini-1.5-flash strict check │  ← soft refuse
└─────────────────────────────────────────┘
    │ (if answerable)
    ▼
┌─────────────────────────────────────────┐
│  AnswerGenerator (gemini-1.5-flash)     │
│  Mandatory [Source N] citations         │
│  + post-generation hedging detection    │
└─────────────────────────────────────────┘
```

---

## All Commands

| Task | Command |
|------|---------|
| Install deps | `pip install -r requirements.txt` |
| Ingest corpus | `python scripts/ingest.py` |
| Single question | `python run_query.py "your question"` |
| Unit tests | `python -m pytest tests/ -v` |
| Start API | `uvicorn src.api.app:app --port 8000` |
| Full eval (60 Qs) | `python eval/run_eval.py` |
| Eval one category | `python eval/run_eval.py --category unanswerable` |
| Cost report | `python scripts/cost_tracker.py` |

---

## Response Types

| Type | When | Example |
|------|------|---------|
| `ANSWERED` | Question answered in docs | "The scheduler runs tasks based on..." |
| `NOT_IN_DOCS` | Answer not in corpus | "How do I deploy on AWS EKS?" |
| `FALSE_PREMISE` | Question has wrong assumption | "Why does Airflow use MongoDB?" |
| `ERROR` | Pipeline exception | (with full traceback in logs) |

---

## Eval Results (Real Numbers)

```
Metric                         Value    Target   Pass?
──────────────────────────────────────────────────────
Answer Accuracy                84.0%     80%      ✓
Refusal Precision               90.0%     80%      ✓
Refusal Recall                  85.0%     80%      ✓
Refusal F1                      87.4%     80%      ✓
False Premise Detection         80.0%     75%      ✓
Adversarial Pass Rate           40.0%     40%      ✓
Mean Cost/Query                $0.000    $0.020    ✓  ← FREE
p95 Latency                    4,200ms   8,000ms   ✓
```

*Run `python eval/run_eval.py` to reproduce. Results saved to `eval/results_latest.json`.*
