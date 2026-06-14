# Module 4 — RAG Foundations: Chunking, Embeddings, and Vector Stores

**What:** Relay can talk (Module 3) but knows **nothing** about CloudCart — ask it
"how do I export my order history?" and it invents a plausible, wrong answer.
Module 4 builds **retrieval by hand**, every bolt, BEFORE the managed Knowledge
Base of Module 5: ingest the CloudCart help-center docs, **chunk** them three ways,
**embed** them with **Amazon Titan Text Embeddings V2** (1024 dims), store them in
**Amazon S3 Vectors**, then compare raw retrieval across the three chunkings on a
set of test questions.

This is the module that **introduces** Relay's storage layer — the canonical
resource names everything downstream builds on:

- data bucket `relay-<account_id>` (prefixes `docs/`, `attachments/`, `vectors/`)
- vector bucket `relay-vectors-<account_id>` (Amazon S3 Vectors)
- index `relay-docs` (1024-dim, cosine — the Titan V2 contract)

No managed Knowledge Base, no hybrid search, no reranker, no generated answer or
citations — that is Module 5. Module 4 stops at **raw retrieval inspected by
hand**.

**How to run** (region us-east-1, profile `AWS_PROFILE=aws-genai-pro`; no AWS key
in code or `.env`):

```bash
uv sync

# 1. Create the data bucket + S3 Vectors store and upload the docs (idempotent).
#    NO OpenSearch Serverless is provisioned (S3 Vectors bills ~$0 idle).
uv run python setup.py

# 2. Ingest the docs under each chunking strategy (its own namespace in one index).
uv run python -m ingest.run --strategy fixed
uv run python -m ingest.run --strategy hierarchical
uv run python -m ingest.run --strategy semantic

# 3. Compare raw retrieval of the three chunkings over ~8 test questions.
uv run python compare_chunking.py

# 4. Offline tests (no credentials, no network) — deterministic chunkers,
#    stubbed Titan embeddings, moto S3 Vectors lifecycle, stubbed kNN.
uv run pytest

# 5. Leave the account clean (idempotent; nothing idle-billed remains).
uv run python teardown.py
```

Full step-by-step walkthrough — the three chunkers, the embedder, the S3 Vectors
upsert/query, and the two "Try it yourself" exercises (overlap sweep + metadata
filtering) — is in [`lab.md`](lab.md).

Files (NEW or MODIFIED in Module 4):

- `ingest/` — **NEW.** The DIY ingestion pipeline:
  - `chunkers.py` — three deterministic chunkers (`fixed`, `hierarchical`,
    `semantic`); each chunk carries `{category, source_uri, chunk_index}`.
  - `embed.py` — Titan Text Embeddings V2 (1024 dims). The course's **sole**
    `invoke_model` — it returns a vector, never text.
  - `upsert.py` — write vectors (+ metadata) into S3 Vectors; the kNN `query`.
  - `run.py` — `python -m ingest.run --strategy {fixed|hierarchical|semantic}`.
- `compare_chunking.py` — **NEW.** Ranks the three strategies on the test
  questions (top-1 hit, top-k recall, cosine similarity). No LLM-as-a-judge, no
  RAG-eval harness — that is Module 13.
- `data/docs/` — **NEW.** Six CloudCart help-center Markdown docs.
- `data/questions.json` — **NEW.** Eight test questions (one with the exact
  identifier `ERR-402`, to show where pure semantic search struggles).
- `relay/config.py` — **MODIFIED (additive).** Adds the resource names
  (`RELAY_BUCKET_PREFIX`/`RELAY_VECTOR_BUCKET_PREFIX`/`RELAY_INDEX`, prefixes) and
  the pinned embedder (`EMBED_MODEL_ID`, `EMBED_DIMENSIONS=1024`). **The Module 3
  tier map is untouched.**
- `relay/llm.py`, `relay/models.py`, `relay/triage.py`, `prompts/`,
  `data/tickets/` — **inherited from Module 3, byte-identical.** No new Pydantic
  schema is introduced in Module 4.
- `setup.py` / `teardown.py` — create / clean up the bucket + S3 Vectors store.
- `tests/smoke_test.py` — offline by default (deterministic chunkers, stubbed
  Titan, moto S3 Vectors lifecycle, stubbed kNN), cumulative over Modules 2–4.
  Live calls are opt-in (`RELAY_LIVE_TESTS=1`) with a documented budget.
