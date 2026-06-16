"""
One-time ingestion script (Gemini version).
Run: python scripts/ingest.py

Steps:
1. Scrape Airflow docs → cached to ./data/corpus/
2. Chunk by sections
3. Embed with Gemini text-embedding-004 (FREE)
4. Store in ChromaDB
5. Build BM25 index → ./data/bm25_index.pkl

FREE tier note: 1,500 embed requests/day, 100 RPM.
Script auto-throttles at 0.65s between batches.
Typical corpus: ~2,000 chunks → ~100 API calls → ~70 seconds total.
"""

import os
import sys
import logging
from pathlib import Path
from collections import Counter

# from truthgate.src.retrieval import embedder

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()

from src.ingestion.scraper import AirflowDocsScraper
from src.ingestion.chunker import chunk_sections
from src.retrieval.embedder import Embedder
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25 import BM25Index

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BM25_PATH    = Path("./data/bm25_index.pkl")
MAX_PAGES    = int(os.getenv("INGEST_MAX_PAGES", "300"))
# EMBED_BATCH  = 20


def main():
    logger.info("=== TruthGate Ingestion (Gemini / FREE) ===")

    # Step 1: Scrape
    logger.info("Step 1/4: Scraping Airflow docs...")
    sections = list(AirflowDocsScraper().scrape(max_pages=MAX_PAGES))
    logger.info(f"  → {len(sections)} sections scraped")
    if not sections:
        logger.error("No sections scraped. Check your network.")
        sys.exit(1)

    # Step 2: Chunk
    logger.info("Step 2/4: Chunking...")
    chunks = chunk_sections(sections)
    logger.info(f"  → {len(chunks)} chunks")

    # Step 3: Embed + store
    logger.info("Step 3/4: Generating local embeddings...")
    embedder = Embedder()
    vs = VectorStore()
    chunk_dicts = [c.to_dict() for c in chunks]

    texts = [c["text"] for c in chunk_dicts]

    logger.info("  → Generating embeddings...")
    all_embeddings = embedder.embed_texts(
    texts,
    input_type="document"
    )


    ids = [c["chunk_id"] for c in chunk_dicts]

    duplicates = [k for k, v in Counter(ids).items() if v > 1]

    print(f"Total chunks: {len(ids)}")
    print(f"Unique IDs: {len(set(ids))}")
    print(f"Duplicate IDs: {len(duplicates)}")

    for d in duplicates[:20]:
        print("DUPLICATE:", d)
    added = vs.add_chunks(chunk_dicts, all_embeddings)
    logger.info(f"  → {added} chunks added to ChromaDB (total: {vs.count()})")

    # Step 4: BM25
    logger.info("Step 4/4: Building BM25 index...")
    bm25 = BM25Index()
    bm25.add_documents(chunk_dicts)
    bm25.save(BM25_PATH)
    logger.info(f"  → Saved to {BM25_PATH}")

    logger.info("=== Done! Cost: $0.00 (Local embeddings) ===")
    logger.info("Next: python run_query.py 'What is a DAG?'")
    logger.info("      python eval/run_eval.py")


if __name__ == "__main__":
    main()
