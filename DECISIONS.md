# DECISIONS.md 

---

## 1. Three Things That Didn't Work

### A. Asking the LLM "Is this answerable?" as the ONLY refusal mechanism

My first approach was simple: retrieve chunks, send them to Gemini, ask "Can you answer this from the docs? Yes/No, then answer."

**It failed.**

Even with explicit instructions, Gemini was still willing to answer questions using its training knowledge when retrieval quality was poor. For example, when a question touched on production database recommendations and the retrieved context only contained partial information, the model would often fill in missing details that were not actually present in the documentation.

Fix: I have added a **score-based pre-filter**. Now I only call the answerability LLM after retrieval if cosine similarity is above 0.35. Below that threshold the system refuses immediately without making an LLM call. This reduced hallucinations and lowered cost.

### B. Chunking by fixed token count (512 tokens, 50 overlap)

This is the obvious approach and it broke in two specific Airflow documentation patterns:

* **Long operator reference pages**: A single operator page can exceed several thousand tokens. Fixed-size chunks often split parameter descriptions in the middle of explanations.
* **Tabular content**: Airflow configuration documentation contains many tables. Fixed chunking split rows across chunks, producing retrieval results that were difficult for the generator to interpret correctly.

Fix: Switched to **section-aware chunking**. I parse HTML structure (`h2`/`h3` boundaries) and keep sections intact whenever possible. Sections are only split when they exceed 1,200 tokens.

### C. Using BM25 alone for sparse retrieval

I initially tried pure BM25 retrieval because it was simple and free.

The problem is that Airflow documentation uses multiple terms for the same concept. For example:

* Task Instance
* TI
* Task

BM25 often misses relevant chunks when terminology differs.

Dense retrieval alone also struggled because it sometimes retrieved semantically related content that did not contain the actual answer.

**Hybrid retrieval (dense + sparse with Reciprocal Rank Fusion) consistently outperformed either approach alone during development testing.**

### D. Chunk ID Collisions During Ingestion

A bug I did not anticipate appeared during ingestion.

Chunk IDs were generated using a hash of the source URL and section anchor. In practice, multiple chunks occasionally generated identical IDs, which caused ChromaDB to raise `DuplicateIDError` exceptions which took majority of the time ton understand the duplicate IDs.

The symptom appeared as ingestion succeeding for scraping and embedding but failing during vector-store insertion.

Fix:

* Added duplicate detection during ingestion.
* Removed duplicate chunks before indexing.
* Then I updated chunk ID generation to guarantee uniqueness across the corpus.

This issue was responsible for several hours of debugging because it initially appeared to be a ChromaDB problem rather than a chunk-generation problem.


---

## 2. Chunking Strategy and Its Failure Mode

**Strategy:** Section-aware chunking using HTML `h2` and `h3` boundaries.

Each chunk corresponds to a documentation section whenever possible.

Rules:

* Maximum chunk size: 1,200 tokens
* Split only when necessary
* Paragraph-aware splitting
* 100-token overlap for oversized sections

**Metadata stored per chunk:**

```python
{
    "source_url",
    "section_title",
    "page_title",
    "section_anchor",
    "token_count",
    "chunk_index"
}
```

### Specific failure mode that still exists

Multi-section synthesis questions.

Example:

> What happens to task retries when the scheduler restarts?

The answer requires information from multiple documentation sections. Retrieval often finds the relevant chunks, but the generator occasionally fails to synthesize information across those sections correctly.

To partially address this, I added a synthesis mode that increases retrieval depth for questions containing phrases such as:

* what happens
* interaction
* compare
* difference between
* together

This helps but does not fully solve the problem.

---

## 3. How Refusal Actually Works (and Where It Fails)

The mechanism has three layers.

### Layer 1 — Similarity Threshold

Here in this step I chose, if the maximum retrieval similarity score is below 0.35, return: NOT_IN_DOCS.

Then, No answerability LLM call is made.

### Layer 2 — False-Premise Detection

Before retrieval, the system checks for known false assumptions that were provided.

If a false premise is detected, the system returns: FALSE_PREMISE along with a correction.

### Layer 3 — LLM Answerability Check

If retrieval passes Layer 1, the retrieved chunks and question are sent to Gemini with a strict instruction:

> Answer ONLY from the provided context. If the context does not contain enough information, respond with exactly CANNOT_ANSWER.

If the response begins with: CANNOT_ANSWER

the system refuses the query.

### Category It Still Gets Wrong

Plausible near-miss questions.

Example:

> What is the default parallelism setting in Airflow? The answer was given as cannot answer.

Actually, the answer exists in the documentation, but only inside a configuration table.

Those chunks often receive lower retrieval scores than narrative text sections and occasionally fall below the refusal threshold.

This is fundamentally a retrieval problem that manifests as a refusal error.

---

## 4. What I'd Do With One More Week and $500/Month

* Train a domain-specific embedding model for Airflow terminology.
* Add a real cross-encoder reranker instead of cosine-similarity reranking.
* Expand false-premise detection beyond hardcoded rules.
* Increase the evaluation suite from 60 questions to 200+ questions.
* Actually, as Ihave used free tier Gemini model, the rate limits are low and while I was in the evaluation phase, instead of response, I got "ERROR" message, this issue can be resolved with monetisation and purchasing better models.

The two metrics that most need improvement are refusal precision and latency.

---

## 5. Shortcut Taken Due to the Time Limit

False-premise detection is still largely rule-based.

The system currently contains a manually written list of patterns such as:

```python
FALSE_PREMISES = [
    ("stores.*indexes.*JSON",
     "Airflow stores metadata in a SQL database, not JSON"),
    ("written in Ruby",
     "Airflow is written in Python"),
    ("uses MongoDB",
     "Airflow uses a relational database")
]
```

This is effective for common misconceptions but does not generalize well to unseen false premises.

A more principled solution would retrieve evidence from the corpus and perform contradiction detection using a dedicated NLI model.

I chose the rule-based approach because implementing, evaluating, and tuning an NLI pipeline would have required significantly more time than was available.

---

## Final Thoughts

The final evaluation set contained 60 questions across answerable, unanswerable, false-premise, and adversarial categories.

The strongest part of the system is retrieval-grounded answering with citation enforcement.

The weakest areas are:

* Multi-section synthesis
* Refusal precision
* Latency

If I had additional time, those are the areas I would prioritize first.
