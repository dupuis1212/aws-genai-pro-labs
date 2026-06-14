# Module 12 lab — the token economy: cost & performance optimization for GenAI

> **This lab cost me $0.02 on June 2026 prices** (the syllabus budget for Module 12 is
> < $2). Everything Module 12 adds is **on-demand / ~$0 idle**: the `relay-cache` DynamoDB
> table is **PAY_PER_REQUEST** (no idle capacity), the batch IAM role is free, and the only
> spend is a few **fast/smart-tier Nova** Converse calls for the before/after report plus the
> capped live tests. **Caching REDUCES the net cost** — a repeated question hits the
> semantic cache (cost ≈ 0) and a reused prompt prefix bills at ~10%. The figures below are
> the MEASURED usage of one full live run (`setup.py` run twice, `cost_report.py` baseline +
> optimized, one `--submit-batch` job, the capped live tests, and `teardown.py`),
> cross-checked against the CloudWatch `AWS/Bedrock` token metrics for the run window
> (**53,367 input / 5,010 output tokens** total) and the pricing pages — never guessed
> (re-verify on the [Bedrock pricing page](https://aws.amazon.com/bedrock/pricing/) —
> **prompt caching, batch, and service tiers** sections — and the
> [DynamoDB pricing page](https://aws.amazon.com/dynamodb/pricing/on-demand/), **as of June
> 2026**):
>
> | Item | Usage | Cost |
> |---|---|---|
> | `cost_report.py` baseline (smart-tier, full price, no cache) | 10 tickets | ~$0.0031 |
> | `cost_report.py` optimized (router + prompt cache + semantic cache) | 9 misses + 1 real cache hit | ~$0.0006 |
> | KB live test (1 real RetrieveAndGenerate on `relay-kb`, smart) | ~1.2k in / ~0.2k out tok | ~$0.0009 |
> | inherited live tests (fast stream / smart / Titan embed / vision / Comprehend PII) | small | ~$0.0006 |
> | Module 12 live test (3 fast-tier Converse, prompt caching on) | ~1.4k in / ~0.2k out tok | ~$0.0001 |
> | batch backfill (`--submit-batch`, Nova Micro, −50%) — submitted then **stopped** | 100 records (padded), 0 processed | ~$0.00 |
> | `relay-cache` DynamoDB table (on-demand + TTL) + Titan cache embeds | a few dozen writes/reads | ~$0.00 (~$0 idle) |
> | CloudFormation / IAM / S3 batch artifacts | — | ~$0.00 |
> | **Total (measured, CloudWatch-cross-checked)** | | **≈ $0.02** |
>
> The point of the module: the **prompt caching** and **semantic cache** levers make the
> *optimized* run far cheaper than the *baseline* run on the SAME tickets — the before/after
> table the live `cost_report.py` printed shows **$/ticket down ~82%** (0.0313c → 0.0055c)
> with **1 semantic-cache hit** at cost ≈ 0. **Teardown reminder:** `uv run python
> teardown.py` drops the `relay-cache` table, the `relay-batch-role`, and the `batch/` S3
> artifacts (B5), on top of the inherited M11 front-door / pipeline cleanup. Nothing Module 12
> created is left idle-billed.

---

## 0. Prerequisites

```bash
export AWS_PROFILE=aws-genai-pro      # us-east-1 everywhere; never an API key in code/.env
uv sync                               # Module 12 adds NO new runtime dependency
```

You inherit the full Relay from Module 11 — the deployed API, the agent, the Knowledge Base,
the guardrail, PII/IAM, and the `converse()` layer. The **M1 $5 budget alarm** is still in
place (never torn down until the course ends).

## 1. Copy the cumulative state

`module-12/` carries `module-11/relay/` **byte-identical**, then applies exactly the M12
increment: `relay/cache.py` (NEW), `cost_report.py` (NEW), and additive edits to
`relay/llm.py`, `relay/config.py`, `relay/api/worker_handler.py`, `setup.py`, `teardown.py`.
The inherited `relay/models.py` is **unchanged** — `cost_cents` has existed since M7.

## 2. Instrument `cost_cents` — measure before you optimize

A ticket is **several** `converse()` calls (triage on the fast tier + answer generation on
the smart tier + agent tool loops). Its real cost is the **sum** over all of them. Module 12
adds a per-ticket **`CostMeter`** to `relay/llm.py` that records every call's token usage and
totals it through the **M3 per-tier price map** (`config.estimate_cost_discounted` — the API
usage block is the source of truth, never a guess):

```python
from relay import llm

with llm.CostMeter() as meter:
    run_relay(payload)            # any number of converse() calls, any tier
record.cost_cents = meter.cost_cents
```

The **worker** (`relay/api/worker_handler.py`) does exactly this and writes the metered
`cost_cents` onto the persisted `TicketRecord` — the M7 placeholder field is finally real,
**by addition**, without touching the frozen schema or rewriting the agent.

## 3. Prompt caching — a cached input prefix (≈ −90%, no staleness)

Relay's long system prompt is identical on every ticket — a perfect **reused input prefix**
to cache provider-side. It rides through `converse()`'s existing `**params`:

```python
res = llm.converse(messages, tier="auto", system=[{"text": SYSTEM_PROMPT}],
                   cache_prompt=True)   # inserts a Converse cache point on the system prefix
res.usage["cacheReadInputTokens"]      # input served from cache -> bills at ~10%
```

This is the **only** lever wired onto the **interactive** path — caching *input* can never
serve a stale answer (the prompt-vs-semantic-cache distinction the exam tests).

## 4. The semantic cache — `relay/cache.py` (NEW)

Wire it in **front** of the answer path for frequent questions:

```python
from relay import cache as cache_module

cache = cache_module.SemanticCache()          # DynamoDB relay-cache + Titan V2 embeddings
hit = cache.lookup("where is my order 1042?")
if hit.hit:
    return hit.answer                         # cost ≈ 0, cache_hit=True
answer = kb.answer(question)                  # the MISS path: a real converse()
cache.store(question, answer)                 # so the next near-duplicate hits
```

It tries the **exact hash** first (deterministic request hashing — zero risk), then a
**semantic** match by Titan-V2 cosine similarity ≥ a **strict 0.95 threshold**. The real risk
is a **false hit**, so it is *never* blind: the threshold gates every semantic hit, a **TTL**
(`expires_at`, DynamoDB native TTL) ages entries out (passive invalidation), and
`cache.invalidate(question)` drops one on a **known** doc change (active invalidation).

The table is **DynamoDB on-demand** (`relay-cache`, ~$0 idle), created by `setup.py`.

## 5. Batch inference — the eval-backfill path (−50%, NEVER interactive)

A latency-tolerant job — re-scoring a whole golden set offline — is perfect for **batch
inference** (`CreateModelInvocationJob`, asynchronous, −50%). `setup.py` builds a small JSONL
backfill (padded to the 100-record floor Bedrock requires), uploads it under
`s3://relay-<account>/batch/input/`, and prints the exact submit call:

```bash
uv run python setup.py                 # uploads the JSONL, prints the submit command
uv run python setup.py --submit-batch  # actually submits the (minutes-long) -50% job
```

> The eval **harness** that backfills through this path is **Module 13** — Module 12 provides
> only the batch road. Batch is **never** on Relay's interactive traffic (a customer is
> waiting).

## 6. The Flex tier — −50%, latency-tolerant (eval/backfill only)

The same eval/backfill jobs can run on the **Flex service tier** (−50%, re:Invent 2025) via
the same `**params` channel:

```python
llm.converse(messages, tier="fast", service_tier="flex")   # eval/backfill ONLY
```

Relay's interactive path stays on **Standard** (`config.DEFAULT_SERVICE_TIER`). The agent, the
worker, and the API handlers **never** pass `service_tier="flex"` — enforced in code and
asserted by the smoke test (`test_m12_flex_and_batch_never_on_the_interactive_path`).

## 7. `cost_report.py` — the before/after table (the graded deliverable)

```bash
uv run python cost_report.py            # live: baseline vs optimized, $/ticket AND p95
uv run python cost_report.py --offline  # the same shape with NO AWS / NO cost
```

It replays a reference ticket set twice — **baseline** (every ticket pays a full-price
smart-tier answer) vs **optimized** (the M3 router picks the tier + prompt caching + the
semantic cache) — and prints the before/after **$/ticket AND p95** table. A planted
near-duplicate question hits the semantic cache and reports `cost_cents ≈ 0`
(`cache_hit=True`). This is exactly the format the AIP-C01 Domain-4 questions ask for.

## 8. Try it yourself

1. **Tune the similarity threshold.** Run `cost_report.py --offline --threshold 0.85` then
   `--threshold 0.98` and watch the **hit rate vs false hits** trade-off. The product owner,
   not just the engineer, owns this number — a false hit is a *wrong answer to a customer*.
2. **Batch vs synchronous.** Submit the backfill with `setup.py --submit-batch` and compare
   the **total cost** of the 100-record job to running the same work synchronously (−50%).

## 9. Run the tests

```bash
uv run pytest                              # offline cumulative suite (Modules 2–12), no creds
RELAY_LIVE_TESTS=1 uv run pytest -m live   # opt-in, capped (a few sub-cent fast-tier calls)
```

The offline suite proves: the cost meter sums real usage through the M3 price map, prompt
caching inserts a cache point + surfaces cache-read tokens, the semantic cache's exact /
semantic / miss decisions on a moto table with a stubbed embedder, the worker populates
`cost_cents`, and `cost_report.py --offline` prints the before/after table with a cache hit.

## 10. Teardown (leave nothing idle-billed — B5)

```bash
uv run python teardown.py                  # drops relay-cache + relay-batch-role + batch/ artifacts
                                           #   (+ the inherited M11 front door / pipeline cleanup)
```

After teardown, **zero** resource Module 12 added is billed at idle: the `relay-cache`
on-demand table is deleted, the batch role is removed, and the `batch/` S3 artifacts are
purged. The inherited tables / KB / guardrail are kept (downstream modules reuse them); the
**M1 $5 budget alarm stays** on purpose.

## Boundaries

- No **dashboard / cost-anomaly alarm** (Module 14 — M12 instruments `cost_cents`, M14
  displays + alarms on it).
- No **eval harness** (golden set / judge / regression gate) — Module 13. M12 ships only the
  batch backfill road.
- No **provisioned throughput / SageMaker endpoint** built (Module 11 — theory here, viewed
  from the cost/Flex angle); no **param tuning** (Module 2); no **router / streaming /
  cross-Region** rebuild (Module 3 — consumed and measured under cost).
- No new **schema field**; no **OpenSearch**; no idle-billed resource.
