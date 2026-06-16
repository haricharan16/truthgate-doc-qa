# TruthGate — RAG System That Knows When To Shut Up

A citation-grounded Retrieval-Augmented Generation (RAG) system for Apache Airflow documentation.

TruthGate combines hybrid retrieval, false-premise detection, answerability gating, and citation-enforced answer generation to reduce hallucinations and refuse unsupported questions.

## Overview

TruthGate answers questions about Apache Airflow using only information retrieved from the official documentation. Unlike traditional chatbots, it is designed to:

* Refuse unsupported questions
* Detect false assumptions in user queries
* Generate answers with citations
* Minimize hallucinations through retrieval grounding

**Corpus:** Apache Airflow Official Documentation

https://airflow.apache.org/docs/apache-airflow/stable/

---

# Models Used

## Generation Model

```text
models/gemini-3.1-flash-lite
```

Used for:

* False-premise classification
* Answerability verification
* Final answer generation

## Embedding Model

```text
models/gemini-embedding-001
```

Used for:

* Document embeddings
* Query embeddings
* ChromaDB vector retrieval

---

# Architecture

```text
User Query
    │
    ▼
┌─────────────────────────────────────┐
│ FalsePremiseClassifier              │
│                                     │
│ • Regex-based validation            │
│ • Gemini fallback classification    │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ HybridRetriever                     │
│                                     │
│ Dense: ChromaDB + Gemini Embedding  │
│ Sparse: BM25                        │
│ Fusion: Reciprocal Rank Fusion      │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ AnswerabilityGate                   │
│                                     │
│ Layer 1: Similarity threshold       │
│ Layer 2: Gemini verification        │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ AnswerGenerator                     │
│                                     │
│ • Citation enforcement              │
│ • Hedging detection                 │
└─────────────────────────────────────┘
```

---

# Project Structure

```text
truthgate/
│
├── src/
│   ├── api/
│   │   └── app.py
│   │
│   ├── core/
│   │   ├── answerability.py
│   │   ├── classifier.py
│   │   ├── generator.py
│   │   └── pipeline.py
│   │
│   ├── ingestion/
│   │   ├── chunker.py
│   │   ├── scraper.py
│   │   └── token_utils.py
│   │
│   └── retrieval/
│       ├── bm25.py
│       ├── embedder.py
│       ├── hybrid.py
│       └── vector_store.py
│
├── tests/
│
├── .env.example
├── .gitignore
├── DECISIONS.md
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── README.md
├── requirements.txt
├── run_query.py
└── setup.py
```

---

# Features

* Hybrid Retrieval (Dense + Sparse Search)
* ChromaDB Vector Store
* BM25 Keyword Retrieval
* Reciprocal Rank Fusion (RRF)
* False-Premise Detection
* Answerability Gating
* Citation-Enforced Responses
* Hedging Detection
* FastAPI Service Layer
* Cost Tracking

---

# Setup

## 1. Clone Repository

```bash
git clone <repository-url>
cd truthgate
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## 3. Configure Gemini API

Copy:

```text
.env.example
```

to:

```text
.env
```

and add:

```env
GEMINI_API_KEY=your_api_key_here

GEMINI_CHAT_MODEL=models/gemini-3.1-flash-lite
GEMINI_EMBED_MODEL=models/gemini-embedding-001
```

Get a free API key:

https://aistudio.google.com/apikey

---

# Build the Knowledge Base

Run ingestion once:

```bash
python scripts/ingest.py
```

This pipeline:

1. Scrapes Airflow documentation
2. Extracts documentation sections
3. Chunks content
4. Generates embeddings
5. Stores vectors in ChromaDB
6. Builds BM25 index

---

# Running Queries

## Single Query

```bash
python run_query.py "What is a DAG?"
```

Example response:

```text
A DAG (Directed Acyclic Graph) is a collection of tasks organized to reflect
their dependencies and execution order [Source 2].

Sources:
[Source 2 — Airflow 101: Building Your First Workflow]
```

---

# Running the API

Start FastAPI:

```bash
uvicorn src.api.app:app --port 8000
```

Open Swagger UI:

```text
http://localhost:8000/docs
```

Available endpoints:

```text
POST /query
GET  /health
GET  /cost-report
```

---

# Commands

| Task                 | Command                               |
| -------------------- | ------------------------------------- |
| Install dependencies | `pip install -r requirements.txt`     |
| Build knowledge base | `python scripts/ingest.py`            |
| Run a query          | `python run_query.py "question"`      |
| Start API            | `uvicorn src.api.app:app --port 8000` |
| Run tests            | `pytest tests/ -v`                    |
| Run evaluation suite | `python eval/run_eval.py`             |

---

# Response Types

| Type          | Description                                           |
| ------------- | ----------------------------------------------------- |
| ANSWERED      | Successfully answered from documentation              |
| NOT_IN_DOCS   | Documentation does not contain sufficient information |
| FALSE_PREMISE | Question contains incorrect assumptions               |
| ERROR         | Internal pipeline or model failure                    |

---

# Reproducing Results

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

## 2. Configure Gemini API

Create a `.env` file:

```env
GEMINI_API_KEY=your_api_key_here

GEMINI_CHAT_MODEL=models/gemini-3.1-flash-lite
GEMINI_EMBED_MODEL=models/gemini-embedding-001
```

## 3. Build the Knowledge Base

```bash
python scripts/ingest.py
```

## 4. Verify the API

```bash
uvicorn src.api.app:app --port 8000
```

Open:

```text
http://localhost:8000/docs
```

## 5. Run Evaluation

```bash
python eval/run_eval.py
```

Results will be saved to:

```text
eval/results_latest.json
```

---

# Evaluation Results

Results generated using:

```bash
python eval/run_eval.py
```

| Metric                  | Value    | Target  | Pass |
| ----------------------- | -------- | ------- | ---- |
| Answer Accuracy         | 80.0%    | 80%     | ✓    |
| Refusal Precision       | 60.0%    | 80%     | ✗    |
| Refusal Recall          | 90.0%    | 80%     | ✓    |
| Refusal F1              | 72.0%    | 80%     | ✗    |
| False Premise Detection | 80.0%    | 75%     | ✓    |
| Adversarial Pass Rate   | 60.0%    | 40%     | ✓    |
| Mean Cost / Query       | $0.00008 | $0.02   | ✓    |
| P95 Latency             | 15241 ms | 8000 ms | ✗    |

## Overall Result

```text
49 / 60 questions correct
```

## Total Evaluation Cost

```text
$0.0050
```
---

# Limitations

* Retrieval quality depends heavily on chunking and indexing quality.
* Refusal precision remains below the target benchmark.
* P95 latency exceeds the desired threshold.
* Gemini free-tier quotas can affect reproducibility of evaluation runs.
* Retrieval remains the primary bottleneck for future improvements.

---

# Technologies

* Python 3.11
* Google Gemini API
* Gemini Embeddings
* ChromaDB
* BM25
* FastAPI
* Uvicorn
* PyTest
* Docker

---

# Repository Description

TruthGate is a citation-grounded RAG system for Apache Airflow documentation that combines hybrid retrieval, answerability gating, and false-premise detection to reduce hallucinations and refuse unsupported questions.
