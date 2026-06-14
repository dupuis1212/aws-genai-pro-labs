# Module 4 lab — RAG foundations: chunking, embeddings, and S3 Vectors

> **This lab cost me $0.01 on June 2026 prices.** A clean run-through as written —
> `setup.py`, ingesting all three chunking strategies, `compare_chunking.py` over
> the eight test questions, and the live smoke test — embedded **8,296 input
> tokens** total (8,147 ingesting the three strategies + 116 for the eight
> questions + 33 in the live test), on **Amazon Titan Text Embeddings V2** at
> $0.02 / million tokens: about **$0.0002** of embeddings. The two live `converse()`
> smoke calls (one Nova Micro stream, one Nova 2 Lite, capped at 64 output tokens)
> add roughly **$0.0003**, for a measured Bedrock total of **~$0.0005**. Everything
> else is rounding error — **Amazon S3 Vectors** storage for the 66 small vectors
> and ~26 queries is fractions of a cent, and the store bills **~$0 idle**. The
> whole lab lands well under one cent; I round up to a penny to be honest about S3
> request charges. Every token figure is read from the Titan/Converse response,
> never guessed. Prices are as of June 2026 — re-verify on the
> [Bedrock pricing page](https://aws.amazon.com/bedrock/pricing/) and the
> [S3 pricing page](https://aws.amazon.com/s3/pricing/).
>
> **Teardown reminder:** run `uv run python teardown.py` when you're done. It
> deletes the `relay-docs` index and the `relay-vectors-<account_id>` bucket, then
> empties and deletes the data bucket (`--keep-data` to retain it). S3 Vectors is
> idle ~$0, so this is hygiene more than cost rescue — but the course rule is one
> tested teardown per setup. The M1 $5 budget stays; it backstops the course.

**Goal:** build retrieval **by hand**. Ingest the CloudCart docs, **chunk** them
three ways, **embed** them with Titan Text Embeddings V2, store them in **Amazon
S3 Vectors**, and **compare** raw retrieval of the three chunkings on real
questions. This is the DIY pipeline Module 5 will hand to a managed Bedrock
Knowledge Base — you build it first so you understand every bolt the managed
service hides.

Region for the whole course: **us-east-1**. Profile: `AWS_PROFILE=aws-genai-pro`
(any profile that resolves to your course account works). No AWS key in code or
`.env` — credentials come from the profile.

---

## Step 1 — Carry the cumulative state forward

Module 4 starts from Module 3's `relay/` package byte-for-byte (`models.py`,
`llm.py`, `config.py`, `triage.py`) and **adds** the `ingest/` pipeline. No
Pydantic schema is introduced or changed in this module — `Ticket` and `Triage`
stay exactly as Module 2 froze them. The only edit to `relay/` is **additive**:
`config.py` gains the bucket/index names and the embedder ID.

```bash
uv sync
```

Confirm the inherited LLM layer is in place and the model-ID containment law still
holds (nothing leaks out of `config.py`):

```bash
grep -rE '(us|global|eu)\.(amazon|anthropic)\.' relay/ | grep -v config.py
# -> empty
```

## Step 2 — Why RAG, not fine-tuning (the one-minute version)

Relay's knowledge is CloudCart's **docs**, and docs change. Three reasons retrieval
beats fine-tuning here: **cost** (no retraining on every product update),
**freshness** (edit a doc, re-ingest — the weights never move), and **attribution**
(you can cite the exact source, which a support agent must). Fine-tuning answers
different problems (style, format, a skill the base model lacks). RAG is a pipeline:
**ingest -> retrieve -> generate**. Module 4 builds the **ingest + retrieve** half;
Module 5 adds cited **generation**.

## Step 3 — `setup.py`: the data bucket and the S3 Vectors store

```bash
uv run python setup.py
```

It creates, idempotently:

- the data bucket `relay-<account_id>` with the three prefixes `docs/`,
  `attachments/`, `vectors/` (the `attachments/` prefix is filled by the
  multimodal intake in Module 6 — created empty now);
- the docs uploaded under `docs/`;
- the vector bucket `relay-vectors-<account_id>` and the index `relay-docs`
  (**1024 dimensions, cosine** — the Titan V2 contract).

It does **not** create OpenSearch Serverless. That is the #1 cost trap of pre-2026
RAG tutorials — an always-on cluster billed ~$174/month. **Amazon S3 Vectors**
(GA December 2025) bills ~$0 idle: $0.06/GB-month storage and $2.50 per million
queries (as of June 2026). For a course-scale corpus, it wins before the first
query. Run `setup.py` twice — it reports "already exists. Reusing." and changes
nothing.

## Step 4 — Chunking: the lever nobody tunes (`ingest/chunkers.py`)

A foundation model does not digest a 40-page doc in one shot, and retrieval returns
a **chunk**, not a file — so how you cut the docs decides what retrieval can ever
find. Module 4 builds three strategies, all **deterministic** (same doc -> same
chunks, which is what makes the comparison fair and the offline test exact):

- **fixed** — cut every N characters with an **overlap** window. Ignores structure;
  the baseline.
- **hierarchical** — split on Markdown headings; each section becomes a
  self-contained chunk that keeps its heading trail.
- **semantic** — group whole sentences up to a budget, never splitting
  mid-sentence (a cheap, dependency-free stand-in for embedding-similarity
  grouping).

Each chunk carries the canonical metadata `{category, source_uri, chunk_index}`.

```python
from ingest.chunkers import chunk_document
chunks = chunk_document(open("data/docs/orders-export.md").read(),
                        "s3://relay-…/docs/orders-export.md", "hierarchical")
# -> each chunk's text starts with its heading trail; metadata() carries category etc.
```

My opinion, to be tested in Step 6: for CloudCart's short, well-sectioned help
articles, **hierarchical** chunking on Markdown headings should beat fixed-size —
but you only know after you measure, which is exactly what the lab does.

## Step 5 — Embed and upsert (`ingest/embed.py`, `ingest/upsert.py`, `ingest/run.py`)

`ingest/embed.py` embeds chunks with **Titan Text Embeddings V2** at 1024 dims.
This is the **one** place in the whole course that uses the bedrock-runtime
embeddings path instead of `converse()` — the Converse API cannot embed. It
returns a **vector, never text**; all generation stays on the `converse()` layer.

`ingest/run.py` ties it together — chunk, embed in batch, upsert into `relay-docs`
under a namespace for the strategy, then print the chunk count, embeddings cost
(from the Titan token count), and the upsert confirmation:

```bash
uv run python -m ingest.run --strategy fixed
uv run python -m ingest.run --strategy hierarchical
uv run python -m ingest.run --strategy semantic
```

Expected tail (numbers vary with your corpus):

```text
Ingested 6 docs with the 'hierarchical' chunker:
  account-password-reset.md            5 chunks
  ...
  chunks total      : 35
  vectors upserted  : 35 -> index 'relay-docs' (bucket relay-vectors-<account_id>)
  embeddings        : Titan Text Embeddings V2, 1024 dims
  embed tokens      : 3112
  embed cost        : $0.000062 (as of June 2026 — re-verify pricing)
```

Each vector's key is namespaced `strategy#doc#chunk_index` (e.g.
`hierarchical#orders-export#2`), so all three strategies live in **one** index and
the comparison filters by the `strategy` metadata key.

## Step 6 — Compare raw retrieval (`compare_chunking.py`)

```bash
uv run python compare_chunking.py
```

For each question in `data/questions.json` it embeds the question with the **same**
Titan model the docs were embedded with, runs a top-k kNN query **per strategy**,
and scores the hits against the **hand-labeled** relevant docs. Example (truncated):

```text
Question: "How do I export my order history?"
                 fixed         hierarchical  semantic
  top-1 hit      Y             Y             Y
  top-k recall   1/1           1/1           1/1
  best sim       0.824         0.867         0.771
```

The "relevance" is the human label in the questions file and the score is plain
**cosine similarity** — there is **no LLM-as-a-judge and no RAG-evaluation
harness** here (that is Module 13). You inspect the chunks yourself; that is the
point of building retrieval by hand.

Watch the `ERR-402` question. Pure semantic similarity often struggles to pin an
**exact identifier** that a keyword search would nail instantly — which is the
motivation for the **hybrid search** Module 5 adds. The fix for a passage that is
in the index but not retrievable is to **re-chunk**, not to bump `k` or add a
reranker (the reranker is Module 5; here, you re-chunk).

## Step 7 — Try it yourself

1. **Sweep the fixed-size overlap and re-measure recall.** In
   `ingest/chunkers.py`, change `FIXED_OVERLAP_CHARS` (try 0, 100, 300), re-ingest
   `--strategy fixed`, and re-run `compare_chunking.py`. More overlap rescues facts
   that straddle a boundary — at the cost of more chunks (more embeddings). Find
   the knee.
2. **Filter retrieval by metadata `category`.** `compare_chunking.py` already
   passes each question's `category` as an S3 Vectors metadata filter. Drop the
   `category` from a billing question in `data/questions.json` and watch
   cross-category chunks compete; add it back and watch the filter sharpen recall.
   This is the metadata-filtering skill (multi-tenant, freshness) made concrete.

## Step 8 — Teardown

```bash
uv run python teardown.py            # delete index + vector bucket + data bucket
uv run python teardown.py --keep-data  # keep the data bucket and its docs
```

Idempotent and tested. It deletes the `relay-docs` index (and all its vectors),
the `relay-vectors-<account_id>` bucket, and — unless `--keep-data` — empties and
deletes the data bucket, then confirms nothing idle-billed remains. To rebuild for
Module 5, just run `setup.py` and `ingest.run` again.

---

## Run the tests (no credentials needed)

```bash
uv run pytest          # offline by default — deterministic chunkers, stubbed Titan,
                       # moto S3 Vectors lifecycle, stubbed kNN; cumulative M2–M4
RELAY_LIVE_TESTS=1 uv run pytest -m live   # opt-in real calls (budget documented)
```

The offline suite guards: the **frozen `converse()` signature** and the model-ID
containment grep gate (inherited from Module 3); that **no new Pydantic schema** is
introduced; the resource names **field-for-field**; the embedder **pinned at Titan
V2 / 1024 dims**; the three **deterministic** chunkers and their canonical
metadata; the Titan embeddings path returning a **1024-dim vector**; the S3 Vectors
upsert lifecycle on moto and the kNN query via a Stubber; **idempotent** setup and
teardown; and the boundary grep gates (**exactly one** embeddings call, **no**
OpenSearch in code, **no** managed-KB/agent tokens). The live tests make at most
four sub-cent calls (two inherited Converse + two Titan embeddings) — no live S3
Vectors writes.
