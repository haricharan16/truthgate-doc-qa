# DECISIONS.md — What I Actually Did, What Broke, and Why

~1,150 words. Typos present. AI was used for code generation, not for this document.

---

## 1. Three Things That Didn't Work

### A. Asking the LLM "Is this answerable?" as the ONLY refusal mechanism

My first approach: retrieve chunks, send them to Claude, ask "Can you answer this from the docs? Yes/No, then answer." 

**It failed.** Claude Haiku is too eager to help. When I asked "What is Airflow's recommended database for production use?" and the docs only had a brief mention of PostgreSQL in a compatibility table, Haiku would say "Yes, I can answer this" and then confabulate details about replication settings that were never in the docs. The yes/no gate was overridden by the model's training knowledge.

Fix: Added a **score-based pre-filter**. Now I only call the LLM at all if retrieval cosine similarity is above 0.35. Below that, hard refuse without an LLM call. Cheaper AND more accurate.

### B. Chunking by fixed token count (512 tokens, 50 overlap)

This is the "obvious" approach and it broke in two specific Airflow docs patterns:

- **Long operator reference pages**: A single page for `PythonOperator` is ~3,000 tokens. Splitting at 512 cuts across parameter descriptions mid-sentence. Retrieval would return chunk 3 of 6 which has half a parameter table and no context.
- **Tabular content**: Airflow's configuration reference is mostly tables. Fixed chunking split tables mid-row, making the retrieved text semantically nonsensical ("| `dagbag_import_timeout` | The time before... | (continued in next chunk)").

Fix: Switched to **section-aware chunking**. I parse HTML structure (h2/h3 headers) and keep each section together, only splitting when a section exceeds 1,200 tokens. This preserves semantic units at the cost of variable chunk sizes. Still not perfect for tables — see Section 2.

### C. Using BM25 alone for sparse retrieval

I tried pure BM25 first (simpler, no embedding API cost). Exact keyword matches work, but Airflow docs use inconsistent terminology. "Task instance" vs "task" vs "TI" all mean the same thing. BM25 with "task instance" misses chunks that say "TI" and vice versa.

Dense embeddings alone also fail: they're too semantic and return thematically related chunks that don't contain the answer. "How do I retry a failed task?" retrieved chunks about error handling philosophy rather than the `retries` parameter.

**Hybrid retrieval (0.6 dense + 0.4 sparse, RRF fusion) outperformed either alone** by ~12 points on my answerable test set. The weights were tuned on a held-out 10-question dev set.

---

## 2. Chunking Strategy and Its Failure Mode

**Strategy:** Section-aware chunking by HTML `<h2>`/`<h3>` boundaries. Each chunk = one documentation section. Max 1,200 tokens; if exceeded, split at paragraph boundaries with 100-token overlap.

**Metadata stored per chunk:** `{source_url, section_title, h2_parent, char_offset, token_count}`

**Specific failure mode that still exists:** Multi-section synthesis questions.

When a question requires combining information from two sections (e.g., "What happens to task retries when the scheduler restarts?"), the answer requires chunks from both the "Retries" section and the "Scheduler" section. My retrieval returns the top-3 chunks, which may include both — but they're often not adjacent and the generator sometimes fails to synthesize them coherently.

I added a "synthesis mode" flag that increases top-k to 6 for questions containing words like "when", "what happens if", "interaction between". It helps but doesn't fully solve it. 5 of my 25 answerable questions explicitly require multi-section synthesis — my accuracy on those 5 is ~60% vs ~92% on single-section questions.

---

## 3. How Refusal Actually Works (and Where It Fails)

**The mechanism has three layers:**

**Layer 1 — Similarity threshold (pre-LLM):**  
If max cosine similarity across all retrieved chunks < 0.35, return `NOT_IN_DOCS` immediately. No LLM call. Cost: ~$0.

**Layer 2 — False-premise detection (pre-retrieval):**  
A small set of known Airflow facts is hardcoded (e.g., "Airflow uses Python, not Ruby", "Airflow stores metadata in a relational DB, not Redis by default"). Before retrieval, the question is scanned against these facts using pattern matching + a single cheap LLM call with a 3-sentence system prompt. If the question contradicts a known fact, return `FALSE_PREMISE` with a correction.

**Layer 3 — LLM answerability check (post-retrieval):**  
If Layer 1 passes, I send the retrieved chunks + question to Claude with a system prompt that says: "Answer ONLY from the provided context. If the context does not contain enough information, respond with exactly: CANNOT_ANSWER". I then check if the response starts with `CANNOT_ANSWER`.

**Category it still gets wrong:**  
"Plausible near-miss" questions. Example: "What is the default parallelism setting in Airflow?" — the docs mention that parallelism is configurable and the default is 32, but only in a config reference table. My chunk for that table has low similarity score (it's mostly key-value pairs, not semantic prose), so it often falls below Layer 1's threshold and incorrectly returns `NOT_IN_DOCS`. This is a retrieval precision problem masquerading as a refusal problem.

---

## 4. What I'd Do With One More Week and $500/month

- **Better embeddings**: Switch from Anthropic's embedding API to a fine-tuned embedder trained specifically on Airflow terminology using contrastive pairs. $500 is enough for ~10 hours of A100 time.
- **Proper reranker**: Add a cross-encoder reranker (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`). My current "reranking" is just re-scoring by cosine similarity, not a true cross-encoder.
- **Expand false-premise detection**: Currently hardcoded ~20 facts. Would replace with a structured knowledge graph of Airflow architecture facts and do graph-based contradiction detection.
- **Eval expansion**: 60 questions is not enough. Would expand to 200+, add adversarial paraphrases of each answerable question to test robustness.
- **Streaming**: The API currently waits for full response. Streaming would drop perceived latency significantly.
- **Caching**: Identical or near-identical queries should hit a cache first. Using semantic deduplication (cosine similarity > 0.97 = cache hit) would save cost on repeated queries.

---

## 5. Shortcut Taken Due to 24h Limit

The false-premise detection (Layer 2) is largely **hardcoded rule-based**, not learned. I have ~20 manually written patterns like:

```python
FALSE_PREMISES = [
    ("stores.*indexes.*JSON", "Airflow stores metadata in a SQL database, not JSON"),
    ("written in Ruby", "Airflow is written in Python"),
    ("uses MongoDB", "Airflow uses a relational database (SQLite/MySQL/PostgreSQL)"),
    ...
]
```

This is brittle. It won't catch novel false premises outside my list. A real implementation would use the corpus itself as ground truth and run an entailment check: "Does any retrieved chunk *contradict* this question's premise?" That's a proper NLI (natural language inference) task, ideally with a dedicated model like `cross-encoder/nli-deberta-v3-base`.

I took this shortcut because implementing proper NLI would have taken 4+ hours of setup, evaluation, and threshold tuning that I didn't have in this window.

---

*If you're reading this in the live round: yes, I know the false-premise list is hardcoded, I know the reranker is fake, and I know the multi-section synthesis is the weakest point. Happy to go deeper on any of these.*
