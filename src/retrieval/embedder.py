# """
# Embedding wrapper using Google Gemini text-embedding-004.

# FREE tier limits (as of 2024):
#   - 1,500 requests/day
#   - 100 requests/minute
#   - Model: text-embedding-004 (768 dimensions)

# Why text-embedding-004:
#   - Free, no credit card needed
#   - 768-dim embeddings, strong for technical docs
#   - Supports task_type: RETRIEVAL_DOCUMENT vs RETRIEVAL_QUERY
#     (different representations, improves retrieval accuracy)

# Rate limiting: we sleep 0.6s between batches to stay under 100 RPM.
# """

# import os
# import time
# import logging
# from google import genai
# from google.genai import types

# logger = logging.getLogger(__name__)

# EMBED_MODEL   = os.getenv("GEMINI_EMBED_MODEL", "models/gemini-embedding-001")
# EMBED_BATCH   = 20   # Gemini embed allows up to 100 texts per call; we use 20 to be safe


# def _make_client():
#     key = os.environ.get("GEMINI_API_KEY")
#     if not key:
#         raise RuntimeError("GEMINI_API_KEY not set. Get a free key at https://aistudio.google.com/apikey")
#     return genai.Client(api_key=key)


# class Embedder:
#     def __init__(self):
#         self.client = _make_client()

#     def embed_texts(self, texts: list, input_type: str = "document") -> list:
#         """
#         Embed a list of texts.
#         input_type: "document" for corpus chunks, "query" for search queries.
#         Maps to Gemini task_type: RETRIEVAL_DOCUMENT or RETRIEVAL_QUERY.
#         """
#         task_type = "RETRIEVAL_DOCUMENT" if input_type == "document" else "RETRIEVAL_QUERY"
#         all_embeddings = []

#         for i in range(0, len(texts), EMBED_BATCH):
#             batch = texts[i : i + EMBED_BATCH]
#             retries = 0
#             while retries < 4:
#                 try:
#                     # logger.info(f"EMBED_MODEL={EMBED_MODEL}")
#                     response = self.client.models.embed_content(
#                         model=EMBED_MODEL,
#                         contents=batch,
#                         config=types.EmbedContentConfig(task_type=task_type),
#                     )
#                     all_embeddings.extend([e.values for e in response.embeddings])
#                     # Polite delay to respect free-tier rate limits
#                     if i + EMBED_BATCH < len(texts):
#                         time.sleep(0.65)
#                     break
#                 except Exception as e:
#                     wait = 2 ** retries
#                     logger.warning(f"Embed error (retry {retries}): {e}. Waiting {wait}s...")
#                     time.sleep(wait)
#                     retries += 1
#             else:
#                 raise RuntimeError(f"Embedding failed after retries for batch starting at {i}")

#         return all_embeddings

#     def embed_query(self, query: str) -> list:
#         """Embed a single query string with RETRIEVAL_QUERY task type."""
#         results = self.embed_texts([query], input_type="query")
#         return results[0]

#     def estimate_cost(self, texts: list) -> float:
#         """Gemini embedding is FREE on the free tier. Always returns 0."""
#         return 0.0

import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

MODEL_NAME = "BAAI/bge-small-en-v1.5"


class Embedder:
    def __init__(self):
        logger.info(f"Loading embedding model: {MODEL_NAME}")
        self.model = SentenceTransformer(MODEL_NAME)

    def embed_texts(self, texts: list, input_type: str = "document") -> list:
        """
        Embed a list of texts.

        input_type is ignored because BGE uses the same encoder
        for documents and queries.
        """

        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        return embeddings.tolist()

    def embed_query(self, query: str) -> list:
        """
        Embed a single query.
        """

        embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        return embedding[0].tolist()

    def estimate_cost(self, texts: list) -> float:
        return 0.0