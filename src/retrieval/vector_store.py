"""
ChromaDB vector store interface.

Why ChromaDB:
- Local persistence (no external service needed)
- Built-in cosine similarity
- Metadata filtering (we use this for source filtering)
- Free

Why not Pinecone/Weaviate: overkill for a 200-page corpus (~2,000 chunks).
ChromaDB handles this size fine in-process.

Known limitation: ChromaDB's built-in embeddings are less accurate than
Anthropic/OpenAI embeddings. We override with our own embedding function.
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

CHROMA_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", "./data/chroma"))
COLLECTION_NAME = "airflow_docs"


class VectorStore:
    def __init__(self, persist_dir: Path = CHROMA_DIR):
        self.persist_dir = persist_dir
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}  # cosine similarity
        )

    def add_chunks(self, chunks: list[dict], embeddings: list[list[float]]) -> int:
        """
        Add chunks with precomputed embeddings.
        
        chunks: list of Chunk.to_dict()
        embeddings: parallel list of embedding vectors
        
        Returns number of chunks added.
        """
        if not chunks:
            return 0

        ids = [c["chunk_id"] for c in chunks]
        texts = [c["text"] for c in chunks]
        metadatas = [{
            "source_url": c["source_url"],
            "section_title": c["section_title"],
            "page_title": c["page_title"],
            "section_anchor": c["section_anchor"],
            "token_count": c["token_count"],
            "chunk_index": c["chunk_index"],
        } for c in chunks]

        # ChromaDB has a batch size limit; chunk in batches of 100
        batch_size = 100
        added = 0
        for i in range(0, len(chunks), batch_size):
            batch_ids = ids[i:i+batch_size]
            # Skip already-existing IDs
            existing = set(self.collection.get(ids=batch_ids)["ids"])
            new_mask = [j for j, bid in enumerate(batch_ids) if bid not in existing]
            if not new_mask:
                continue

            self.collection.add(
                ids=[batch_ids[j] for j in new_mask],
                embeddings=[embeddings[i+j] for j in new_mask],
                documents=[texts[i+j] for j in new_mask],
                metadatas=[metadatas[i+j] for j in new_mask],
            )
            added += len(new_mask)

        logger.info(f"Added {added} new chunks to vector store (total: {self.collection.count()})")
        return added

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 8,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """
        Query by embedding vector.
        
        Returns list of {text, metadata, distance, score} sorted by score desc.
        distance is cosine distance (0=identical, 2=opposite)
        score is 1 - distance/2 (0..1, higher=better)
        """
        kwargs = {"query_embeddings": [query_embedding], "n_results": min(top_k, self.collection.count())}
        if where:
            kwargs["where"] = where

        results = self.collection.query(**kwargs, include=["documents", "metadatas", "distances"])

        output = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            score = 1.0 - dist / 2.0  # convert cosine distance to similarity
            output.append({
                "text": doc,
                "metadata": meta,
                "distance": dist,
                "score": score,
            })

        return output

    def count(self) -> int:
        return self.collection.count()

    def is_empty(self) -> bool:
        return self.count() == 0
