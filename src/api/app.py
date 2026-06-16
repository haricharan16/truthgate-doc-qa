import os
import json
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from dotenv import load_dotenv
load_dotenv()
# print("GEMINI_API_KEY exists:", bool(os.getenv("GEMINI_API_KEY")))
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

BM25_PATH = Path("./data/bm25_index.pkl")
pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    logger.info("Loading TruthGate pipeline (Gemini)...")

    from src.retrieval.vector_store import VectorStore
    from src.retrieval.bm25 import BM25Index
    from src.retrieval.embedder import Embedder
    from src.retrieval.hybrid import HybridRetriever
    from src.core.pipeline import TruthGatePipeline

    vs = VectorStore()
    if vs.is_empty():
        raise RuntimeError("Vector store empty — run `python scripts/ingest.py` first.")
    if not BM25_PATH.exists():
        raise RuntimeError("BM25 index missing — run `python scripts/ingest.py` first.")

    bm25 = BM25Index.load(BM25_PATH)
    embedder = Embedder()
    retriever = HybridRetriever(vs, bm25, embedder)
    pipeline = TruthGatePipeline(retriever)
    logger.info(f"Ready. {vs.count()} chunks indexed.")
    yield


app = FastAPI(
    title="TruthGate (Gemini)",
    description="RAG QA over Airflow docs — powered by free Gemini API",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class QueryRequest(BaseModel):
    question: str
    class Config:
        json_schema_extra = {"example": {"question": "How does Airflow schedule DAGs?"}}


class QueryResponse(BaseModel):
    response_type: str
    question: str
    answer: str = ""
    citations: list = []
    correction: str = ""
    refusal_reason: str = ""
    best_retrieval_score: float = 0.0
    has_citations: bool = False
    has_hedging: bool = False
    total_cost_usd: float = 0.0
    total_latency_ms: float = 0.0
    chunks_retrieved: int = 0


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty")
    if len(req.question) > 1000:
        raise HTTPException(400, "Question too long (max 1000 chars)")
    result = pipeline.query(req.question)
    return QueryResponse(**result.to_dict())


@app.get("/health")
async def health():
    from src.retrieval.vector_store import VectorStore
    vs = VectorStore()
    return {
        "status": "ok",
        "model": "models/gemini-2.5-flash",
        "embed_model": "models/gemini-embedding-001",
        "index_chunks": vs.count(),
        "bm25_ready": BM25_PATH.exists(),
        "pipeline_ready": pipeline is not None,
    }


@app.get("/cost-report")
async def cost_report():
    log_path = Path(os.getenv("COST_LOG_FILE", "./logs/cost_log.jsonl"))
    if not log_path.exists():
        return {"message": "No queries logged yet."}

    lines = [json.loads(l) for l in log_path.read_text().splitlines() if l.strip()]
    if not lines:
        return {"message": "No queries logged yet."}

    costs = [l["cost"] for l in lines]
    latencies = sorted(l["latency_ms"] for l in lines)
    by_type = {}
    for l in lines:
        by_type.setdefault(l["type"], 0)
        by_type[l["type"]] += 1

    p95 = latencies[int(len(latencies) * 0.95)]
    return {
        "total_queries": len(lines),
        "by_response_type": by_type,
        "mean_cost_usd": round(sum(costs) / len(costs), 6),
        "note": "Free Gemini tier = $0.00 cost",
        "mean_latency_ms": round(sum(latencies) / len(latencies), 1),
        "p95_latency_ms": round(p95, 1),
    }
