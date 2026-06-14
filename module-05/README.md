# Module 5 — Managed RAG: Bedrock Knowledge Bases, Hybrid Search, and Rerankers

**What:** Module 4 built retrieval **by hand** — chunk, embed, upsert, kNN — and
you know every bolt. Module 5 hands that pipeline to a managed **Bedrock Knowledge
Base** (`relay-kb`) that ingests the CloudCart docs, keeps the index in sync,
searches in **hybrid** mode, **reranks**, and returns **cited, grounded** answers.
The KB reuses Module 4's **Amazon S3 Vectors** index `relay-docs` (the Titan V2,
1024-dim contract) as its vector store — **no** always-on serverless search
cluster. Relay gains `relay/kb.py`: `retrieve()` (the **Retrieve** access pattern)
and `answer()` (the **RetrieveAndGenerate** access pattern → an `Answer` with
`Citation`s and `grounded`).

This module **freezes** two schemas in `relay/models.py` (06 §2 / bible §3.1),
reproduced field-for-field, with **no** `score` / `confidence` field, ever:

```python
class Citation(BaseModel):   # frozen M5
    source_uri: str
    snippet: str

class Answer(BaseModel):     # frozen M5
    text: str
    citations: list[Citation]
    grounded: bool           # M5: bool(citations) heuristic; M9 recomputes it
```

`Ticket` and `Triage` are **untouched**. No `Attachment` (M6), no agent or
KB-search tool (M7), no guardrail / contextual grounding check (M9), no LLM-judge
or RAG-eval harness (M13). Module 5 stops at **a managed KB with cited answers and
a measured hybrid/rerank/freshness benchmark**.

**How to run** (region us-east-1, profile `AWS_PROFILE=aws-genai-pro`; no AWS key
in code or `.env`):

```bash
uv sync

# 0. Module 5 REUSES Module 4's storage layer. If you tore Module 4 down, run its
#    setup + ingestion first (data bucket relay-<account_id>/docs/ + index relay-docs).

# 1. Create the Knowledge Base relay-kb over the EXISTING S3 Vectors index,
#    attach the docs/ data source, run the first ingestion job to COMPLETE
#    (idempotent). NO always-on search cluster is provisioned.
uv run python setup.py

# 2. Ask a question — generated answer + numbered citations + grounded: True.
uv run python -m relay.kb "How do I change my CloudCart subscription plan?"

# 3. Benchmark four configurations on 8 questions: M4 DIY vs KB semantic vs
#    KB hybrid vs KB hybrid+rerank (two questions carry exact identifiers).
uv run python compare_retrieval.py

# 4. Prove the KB re-syncs after a doc edit (skill 1.4.5): edit a plan price,
#    re-ingest incrementally, show the answer before/after, then restore.
uv run python freshness_test.py

# 5. Offline tests (no credentials, no network) — Citation/Answer contract,
#    retrieve/answer over stubbed Retrieve/RetrieveAndGenerate, hybrid+rerank
#    request wiring, query decomposition, setup/teardown idempotency. Cumulative
#    over Modules 2–5.
uv run pytest

# 6. Remove Module 5's KB + data source + IAM role. KEEPS the S3 Vectors index
#    and docs bucket (Module 7 reuses them; idle ~$0).
uv run python teardown.py
```

Full step-by-step walkthrough — the KB anatomy, the two access patterns, hybrid
search and the reranker, query decomposition, the freshness re-sync, and the two
"Try it yourself" exercises (metadata filtering + the reranker's latency cost) — is
in [`lab.md`](lab.md).

Files (NEW or MODIFIED in Module 5):

- `relay/kb.py` — **NEW.** `retrieve()` (Retrieve, top-k, `HYBRID`/`SEMANTIC`,
  optional Bedrock reranker, `category` filter) and `answer()` (RetrieveAndGenerate
  on the **smart** tier via `config.model_arn()`, optional `QUERY_DECOMPOSITION`)
  → the frozen `Answer`/`Citation`. `grounded = bool(citations)` with a comment to
  Module 9's real grounding check. No bare model ID, no single-prompt model call.
- `compare_retrieval.py` — **NEW.** The four-way benchmark (M4 DIY vs KB
  semantic/hybrid/hybrid+rerank) over `data/kb_questions.json`. Human inspection —
  **no** LLM-as-a-judge, **no** RAG-eval harness (that is Module 13).
- `freshness_test.py` — **NEW.** Edit `data/docs/billing-plans.md`, re-upload,
  `StartIngestionJob` (incremental sync), show the answer before/after, restore.
- `data/kb_questions.json` — **NEW.** Eight KB questions; two carry exact
  identifiers (`ERR-402`, the `Growth` plan name) and one is compound (for
  decomposition).
- `data/docs/billing-plans.md` — **NEW.** The CloudCart plan/pricing doc the
  headline question and the freshness test use.
- `relay/models.py` — **MODIFIED (additive).** Adds `Citation` and `Answer`.
  `Ticket`/`Triage` unchanged.
- `relay/config.py` — **MODIFIED (additive).** Adds `RELAY_KB_NAME="relay-kb"`,
  the data-source name, the answer tier, `model_arn()`, the reranker
  (`RERANK_MODEL_ID="amazon.rerank-v1:0"`, `rerank_model_arn()`), and reranker
  pricing. **The Module 3 tier map and the Module 4 embedder are untouched.**
- `relay/__init__.py` — **MODIFIED (additive).** Tracks the new `kb` submodule.
- `setup.py` / `teardown.py` — **NEW (M5 versions).** Create / clean up the KB,
  data source, and IAM role; teardown keeps the S3 Vectors index + docs for M7.
- `relay/llm.py`, `relay/triage.py`, `ingest/`, `prompts/`, `data/tickets/`,
  `data/questions.json`, the six Module 4 docs, `compare_chunking.py` —
  **inherited from Module 4, byte-identical.**
- `tests/smoke_test.py` — offline by default (cumulative Modules 2–5); live calls
  opt-in (`RELAY_LIVE_TESTS=1`) with a documented budget (≤5 calls; one is a real
  RetrieveAndGenerate that skips cleanly if the KB is not set up).
