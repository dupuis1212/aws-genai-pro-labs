# Module 5 lab — managed RAG: Bedrock Knowledge Bases, hybrid search, rerankers

> **This lab cost me $0.20 on June 2026 prices** (a clean run-through is well under
> this; the syllabus budget for Module 5 is < $2). Every token figure below is read
> from the API response, never guessed. The spend is a handful of
> **RetrieveAndGenerate** and **Retrieve** calls, the **Cohere Rerank 3.5** fee, and
> one Titan ingestion job, all over a tiny corpus:
>
> - **Ingestion** — each ingestion job embeds ~7 small docs with **Amazon Titan
>   Text Embeddings V2** at $0.02 / million tokens: a few thousand tokens, well
>   under **$0.001** per sync. The freshness test re-syncs **one** changed doc
>   twice (incremental — `numberOfModifiedDocumentsIndexed: 1`), another fraction
>   of a cent.
> - **Answers** — `relay.kb` and `freshness_test.py` make a handful of
>   **RetrieveAndGenerate** calls on the **smart** tier (`us.amazon.nova-2-lite-v1:0`,
>   ~$0.30 in / ~$2.50 out per million tokens) — a few hundred tokens each, a few
>   cents total. `compare_retrieval.py` makes ~24 **Retrieve** calls (8 questions ×
>   3 runnable KB configs) — Retrieve embeds the query and searches; no generation —
>   plus 8 **HYBRID** attempts S3 Vectors rejects immediately (no charge, caught as n/a).
> - **Reranker** — reranked retrievals/answers add the **Cohere Rerank 3.5** fee
>   (~$2.00 / 1,000 queries). This is the single biggest line if you rerank a lot:
>   ~60 reranked calls during a full run + the "Try it yourself" ≈ **$0.12**. (Watch
>   it — it dwarfs the generation cost on this tiny corpus.)
> - **S3 Vectors** storage for ~90 vectors and the queries is fractions of a cent
>   and the store bills **~$0 idle**. The KB itself and the IAM role bill **~$0
>   idle**.
>
> Prices are **as of June 2026** — re-verify on the
> [Bedrock pricing page](https://aws.amazon.com/bedrock/pricing/), the
> [S3 pricing page](https://aws.amazon.com/s3/pricing/), and the **Cohere Rerank**
> line on the Bedrock pricing page before you run it.
>
> **Teardown reminder:** run `uv run python teardown.py` when you're done. It
> deletes the Knowledge Base `relay-kb`, its data source, and the `relay-kb-role`
> IAM role, and **keeps** the S3 Vectors indexes (`relay-kb-docs`, the KB's own,
> and Module 4's `relay-docs`) and the docs bucket — Module 7's agent retrieves
> from the KB built on `relay-kb-docs`. The KB and role bill ~$0 idle, so this is
> hygiene more than cost rescue — but the course rule is one tested teardown per
> setup. The M1 $5 budget stays; it backstops the course.

**Goal:** create the managed Knowledge Base `relay-kb` over the CloudCart docs,
give Relay an `answer()` that produces **cited, grounded** answers
(`Answer`/`Citation`), and **measure** hybrid search and the reranker against
Module 4's DIY retrieval — freshness included.

Region for the whole course: **us-east-1**. Profile: `AWS_PROFILE=aws-genai-pro`
(any profile that resolves to your course account works). No AWS key in code or
`.env` — credentials come from the profile.

---

## Step 1 — Carry the cumulative state forward

Module 5 starts from Module 4's `relay/` package byte-for-byte (`models.py`,
`config.py`, `llm.py`, `triage.py`) **plus** Module 4's `ingest/` pipeline and its
storage layer: the data bucket `relay-<account_id>` with `docs/` populated, and the
S3 Vectors index `relay-docs`. Module 5 **reuses** all of it — it never re-ingests
by hand and never recreates Module 4's bucket or index.

If you tore Module 4 down between modules, rebuild that storage layer first (run
Module 4's `setup.py` + `ingest.run`), then come back here. Module 5's `setup.py`
**prechecks** the prerequisites and tells you exactly what is missing rather than
silently rebuilding someone else's resource.

```bash
uv sync   # boto3 + pydantic; no new runtime dependency for Module 5
```

Confirm the prerequisites and your identity:

```bash
aws sts get-caller-identity            # the account the bucket name is suffixed with
aws s3 ls s3://relay-$(aws sts get-caller-identity --query Account --output text)/docs/
```

---

## Step 2 — Stand up the Knowledge Base (`setup.py`)

`setup.py` is idempotent and verbose. It:

1. **Prechecks** the Module 4 storage layer (data bucket + `docs/` + the
   `relay-docs` index) exists.
2. Creates the **least-privilege IAM role** `relay-kb-role` the KB assumes (trust
   to `bedrock.amazonaws.com`; explicit-ARN access to invoke Titan, read `docs/`,
   and use the S3 Vectors index — **zero wildcard resources**).
3. Creates the **Knowledge Base `relay-kb`** with **S3 Vectors** storage pointed at
   a **dedicated, KB-owned** index `relay-kb-docs` (1024 dims, cosine, Titan V2) in
   the same vector bucket — **not** a Quick-Create always-on serverless collection
   (that bills ~$174/month idle; S3 Vectors bills ~$0 idle). The KB gets its **own**
   index, kept separate from Module 4's `relay-docs` DIY index: a Bedrock KB writes
   Bedrock-schema vectors it alone can read back as content, so mixing it with
   Module 4's raw vectors would return half-empty results. `relay-docs` stays the
   DIY baseline `compare_retrieval.py` benchmarks against.
4. Uploads a `<doc>.md.metadata.json` **sidecar** per doc so the KB indexes
   `category` as a **filterable** metadata attribute (the multi-tenant /
   scoped-retrieval lever, and "Try it yourself" #1).
5. Attaches the **S3 data source** over `s3://relay-<account_id>/docs/`.
6. Starts the **first ingestion job** and waits for `COMPLETE` (parse → chunk →
   embed with Titan → write vectors into `relay-kb-docs`).
7. Records the KB id and data-source id in `.kb_id` / `.kb_data_source_id` so the
   rest of the lab finds them without an env var.

```bash
uv run python setup.py
# ... data source 'relay-docs-source': CREATED ...
# ... ingestion job <id>: COMPLETE — docs/ embedded into 'relay-kb-docs'.
```

The **integration component** here is the KB itself: it connects a documentary
system (S3 `docs/`) to the foundation model, managing parsing, chunking,
embedding, the vector store, and ingestion jobs for you (skill 1.4.4). Wikis, DMS,
and other connectors plug in the same way — survey only; the lab uses S3.

---

## Step 3 — Cited answers: `relay/kb.py` (`Retrieve` vs `RetrieveAndGenerate`)

`relay/kb.py` exposes the **two standardized access patterns** (skill 1.5.6,
access-pattern slice):

- **`retrieve(query, ...)`** → the **Retrieve** API: you get the raw passages back
  and own the prompt and model yourself. This is the pattern the agent will use in
  Module 7. Hybrid search and the reranker are toggled here.
- **`answer(query, ...)`** → the **RetrieveAndGenerate** API: the KB retrieves **and**
  generates a grounded answer with **turnkey citations**, on the **smart** tier
  resolved through `relay/config.py` (`model_arn("smart")` — never a hard-coded ID).
  We map the response into the frozen `Answer`/`Citation` schemas.

Relay uses `RetrieveAndGenerate` here for the turnkey citations; it switches to
`Retrieve` when the agent arrives (Module 7).

```bash
uv run python -m relay.kb "How do I change my CloudCart subscription plan?"
# Open Billing -> Subscription and click Change plan. ...
#
# Citations (1):
#   [1] s3://relay-<account_id>/docs/billing-plans.md
#       Changing your plan ... Open Billing -> Subscription and click Change plan. ...
#
# grounded: True
```

`grounded` is the heuristic **`bool(citations)`** at this stage: an answer that
cited at least one retrieved source is treated as grounded. **Module 9** keeps this
exact field but recomputes it from a real **contextual grounding check** (a
guardrail) and escalates ungrounded answers — same field name and type, different
computation.

> ⚠️ **A reranker fixes bad retrieval.** It does not. The reranker reorders what the
> retriever **already** returned. If the right chunk is not in the candidates (bad
> chunking, a malformed query, a stale index), no reranker conjures it. **Coverage
> first** (hybrid, sync), **order second**.

---

## Step 4 — Hybrid search, the reranker, and query decomposition

Pure semantic search can miss **exact identifiers** — a SKU, a CloudCart error code
like `ERR-402`, a plan name like `Growth` — because the token has no semantic
neighbourhood. **Hybrid search** (`search_type="HYBRID"`, the `overrideSearchType`
field) combines keyword and vector matching to pin them.

**Hybrid search is a property of the vector store, not a free switch.** Live-verified
June 2026: Bedrock Knowledge Bases run hybrid search only on hybrid-capable stores
(Aurora PostgreSQL, MongoDB Atlas, or the always-on serverless search cluster the
article prices out at ~$174/month). On **Amazon S3 Vectors** — the course's ~$0-idle
store — the API rejects HYBRID ("HYBRID search type is not supported for search
operation on index ..."). So the lab runs **semantic** retrieval and uses the
**reranker** as the precision lever instead; the `kb_hybrid` column shows `n/a`. That
trade-off (hybrid vs ~$0 idle) is exactly what Domain 1 tests — see the exam corner.

A **Bedrock reranker** (`cohere.rerank-v3-5:0`, in `relay/config.py` — the only
reranker in the us-east-1 catalogue as of June 2026; Amazon Rerank is the documented
alternative for Regions that carry it) re-scores the retriever's candidates for
precision of the top-k. **Query decomposition** (`answer(decompose=True)`, the
`QUERY_DECOMPOSITION` orchestration) breaks a compound question into sub-queries.

`compare_retrieval.py` runs **four configurations** over the 8 questions in
`data/kb_questions.json` and prints, per question and in aggregate, the top-1 hit
rate and recall:

```bash
uv run python compare_retrieval.py
#                  M4 DIY      KB sem      KB sem+rr   KB hyb*
#   top-1 hit      Y           Y           Y           n/a
#   recall         1/1         1/1         1/1         n/a
# (KB hyb* = HYBRID, n/a on S3 Vectors — needs a hybrid-capable store)
```

Read it **by hand**: the reranker (`KB sem+rr`) reorders the semantic candidates for
precision but never surfaces a doc the retriever missed — coverage first, order
second. Relevance is the **hand label** in `data/kb_questions.json` — there is **no**
LLM-as-a-judge and **no** RAG-eval harness here (that is Module 13).

Try the compound question with decomposition in a Python shell:

```python
from relay import kb
a = kb.answer("How do I downgrade my plan without losing my order history?",
              decompose=True)
print(a.text); print([c.source_uri for c in a.citations])
# decomposition retrieves for "downgrade" AND "order history" -> cites both docs.
```

---

## Step 5 — Keeping the knowledge fresh (`freshness_test.py`)

A managed KB does **not** magically know the docs changed — you re-run an
**ingestion job**, and the KB does an **incremental sync**: it detects which objects
changed (by S3 ETag / last-modified) and re-embeds only those, not the whole corpus
(skill 1.4.5).

```bash
uv run python freshness_test.py
# 1. Answer BEFORE: "... the Growth plan is $79 per month ..."  cited billing-plans.md
# 2. Editing billing-plans.md (Growth price -> $99) and re-uploading ...
# 3. Re-syncing the Knowledge Base (incremental) ... COMPLETE.
# 4. Answer AFTER: "... the Growth plan is $99 per month ..."
#    PROVEN: the new price $99 appears in the answer — the KB re-synced.
# 5. Restoring the original price and re-syncing (repeatable lab).
```

In production you would not edit a doc by hand — an **EventBridge** schedule or an
S3 event triggers the ingestion job (the event-driven version is Module 11); the
incremental-sync mechanism is the same one you watch here.

---

## Step 6 — Offline tests, then teardown

```bash
uv run pytest                          # offline: no creds, no network (Modules 2–5)
RELAY_LIVE_TESTS=1 uv run pytest -m live  # up to 5 sub-cent real calls (budgeted)
uv run python teardown.py              # delete KB + data source + role; KEEP vectors
```

The offline tests cover the frozen `Citation`/`Answer` contract, `retrieve()` /
`answer()` over stubbed Retrieve / RetrieveAndGenerate (the data plane moto does
not implement), the hybrid+rerank request wiring, query decomposition, the
four-way `compare_retrieval` scoring, and `setup.py`/`teardown.py` idempotency. The
live marker makes at most **five** real calls total (two Module 2/3 `converse`, two
Module 4 Titan embeddings, and one Module 5 `RetrieveAndGenerate` that **skips
cleanly** if the KB is not set up).

Teardown keeps the S3 Vectors indexes (`relay-kb-docs`, the KB's own, and Module
4's `relay-docs`) and the docs bucket on purpose — Module 7's agent retrieves from
the KB built on `relay-kb-docs`, and `setup.py` rebuilds the KB over the kept index.
Pass `--delete-vectors` only if you want to drop Module 4's `relay-docs` DIY index
for a clean slate (Module 4's setup would then need a re-run).

---

## Try it yourself

1. **Metadata filtering for multi-tenant retrieval.** `retrieve()` and `answer()`
   take a `category` argument that scopes retrieval to one CloudCart category
   (`billing`, `technical`, ...). Ask a `billing` question with
   `category="technical"`: the filter scopes retrieval to the technical docs, so
   you get technical chunks back — none answering the billing question — and the
   grounded answer comes back empty or refuses for lack of support. That filter is
   the lever a multi-tenant deployment uses to keep one customer's docs out of
   another's answers.
2. **Is the reranker worth its latency?** Time `retrieve(..., rerank=False)` vs
   `retrieve(..., rerank=True)` on the same query (Python's `time.perf_counter`).
   The reranker adds a cross-encoder pass and a per-query fee. On CloudCart's short,
   well-sectioned docs, does the precision gain justify the added latency and cost,
   or is hybrid-without-rerank already good enough? Decide with the numbers, not a
   vibe.
