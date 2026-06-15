# Module 12 — The Token Economy: Cost and Performance Optimization for GenAI

**What:** Relay runs in production (Module 11) — but its cost is a **black box**. Every
ticket maybe pays twice what it needs, and the last prompt change maybe doubled the bill
without anyone seeing it. The `TicketRecord.cost_cents` field has read `0.0` since Module 7;
it was never wired. *You can't cut what you can't see.*

Module 12 **instruments the $/ticket** (`cost_cents` finally measured, summed across every
`converse()` call in a ticket), then attacks it on four fronts — **prompt caching**
(≈ −90% on the cached input prefix), a **semantic cache** (DynamoDB + Titan embeddings) for
frequent questions, **batch inference** (−50%) for eval backfills, and the **Flex tier**
(−50%) for latency-tolerant jobs — and proves it with a **before/after table (cost AND p95)**.

```bash
uv run python cost_report.py            # the before/after $/ticket + p95 table (live)
uv run python cost_report.py --offline  # the same table shape, no AWS, no cost
```

```
================================================================
  Relay cost & latency — before / after (Module 12)
================================================================
  reference tickets : 10
  semantic-cache hits (optimized) : 1

  metric                      baseline     optimized       delta
  ------------------------------------------------------------
  $/ticket (cents)              0.0546        0.0011     -98.0% down
  total cost (cents)            0.5460        0.0112     -98.0% down
  p95 latency (ms)             15.0457        5.2901     -64.8% down

  Flex/batch on the EVAL path (latency-tolerant, never interactive):
    same volume as a batch backfill -> 0.2730c (-50% vs 0.5460c)
================================================================
```

## What this module builds (on top of Module 11)

- **`relay/cache.py` (NEW)** — the **semantic cache**. It keeps the three caches the exam
  keeps separate distinct:
  - **deterministic request hashing / result fingerprinting** — an exact-match SHA-256 key
    (zero risk: a byte-identical question gets the byte-identical stored answer);
  - **semantic caching** — match a *near-duplicate* question by **Titan V2 embedding
    similarity** ≥ a **strict threshold (0.95)** and serve the stored answer (cost ≈ 0). The
    real risk is a **false hit**, so it is *never* a blind cache: a strict threshold + a
    **TTL** (passive invalidation) + `invalidate()` (active, on a known doc change);
  - it leaves **prompt caching** to `relay/llm.py` (a cached *input prefix*, not output).
  Storage is a **DynamoDB on-demand** table `relay-cache` (~$0 idle). No model ID, no
  generation here — embeddings reuse the Module 4 Titan path; the answer comes from the
  caller on a miss.
- **`cost_report.py` (NEW)** — the graded deliverable: replays a reference ticket set
  **baseline vs optimized** and prints the **$/ticket AND p95** before/after table, with a
  semantic-cache hit (`cost_cents ≈ 0`).
- **`relay/llm.py` (MODIFIED, by addition)** — `converse()`'s signature is **byte-identical**.
  Two cost levers ride through the existing `**params`: `cache_prompt=True` (a Converse
  **cache point** on the system prefix) and `service_tier="flex"` (the −50% tier). A
  per-ticket **`CostMeter`** sums every call's real token usage through the M3 price map.
- **`relay/config.py` (MODIFIED, by addition)** — the `relay-cache` table name + key + TTL,
  the similarity threshold + TTL, the Flex service tier, the batch/Flex/prompt-cache
  discounts. **The per-tier price map already existed since Module 3** — M12 only *consumes*
  it.
- **`relay/api/worker_handler.py` (MODIFIED, by addition)** — wraps the agent run in a
  `CostMeter` and writes the metered **`cost_cents`** onto the persisted `TicketRecord` (the
  M7 placeholder field finally real).
- **`setup.py` / `teardown.py` (MODIFIED)** — `setup.py` creates the `relay-cache` table
  (on-demand + TTL) and submits a demo **batch backfill** job; `teardown.py` drops the cache
  table, the batch role, and the batch S3 artifacts (B5).

## Frozen contracts (no schema change — bible §3.1)

Module 12 adds **no Pydantic field**. `cost_cents` was frozen at **M7** (a `0.0` float
placeholder) and is **finally populated** here — same field, same type, never "new".
`converse(messages, *, tier="auto", stream=False, **params)` is **byte-identical M3→M15**;
prompt caching and the Flex tier are `**params` keys, never a new parameter and never a
parallel `bedrock-runtime` client. `feedback_rating` / `POST /feedback` stay **Module 13**.

## The three caches — different things, different risks (the exam's core misconception)

| Cache | Caches | Risk | Where in Relay |
|---|---|---|---|
| **prompt caching** | a reused **input prefix** (provider-side) | none (it's input — can't go stale) | `converse(cache_prompt=True)` on the system prompt |
| **semantic caching** | a **stored answer** for a *similar* question | a **false hit** → a stale/wrong answer | `relay/cache.py` (threshold + TTL) |
| **deterministic hashing** | a **stored answer** for the *exact* question | none (exact key) | `relay/cache.py` (the hash lookup) |

## Flex / batch — −50%, but latency-tolerant ONLY

The Flex tier and batch inference are **eval/backfill** levers (the batch job Module 13's
eval harness backfills through). They are **never** wired onto Relay's interactive traffic —
a customer is waiting, and −50% never justifies a slower answer. The lab enforces this in
code (the agent / worker / API never pass `service_tier="flex"`) and the report shows the
Flex/batch saving as a *separate eval-path line*, not on the interactive tickets.

## Run it

```bash
export AWS_PROFILE=aws-genai-pro          # us-east-1 everywhere; no keys in code/.env
uv sync                                   # M12 adds NO new runtime dep (boto3 covers it all)
uv run python setup.py                    # upstream (M5–M10) + the relay-cache table + batch demo
uv run python cost_report.py              # the before/after $/ticket + p95 table (live)
uv run python cost_report.py --offline    # the same shape with no AWS / no cost
uv run pytest                             # offline cumulative suite (Modules 2–12)
RELAY_LIVE_TESTS=1 uv run pytest -m live  # opt-in, capped (a few sub-cent fast-tier calls)
uv run python teardown.py                 # drops the cache table + batch role + artifacts (+ M11 front door)
```

`setup.py --submit-batch` actually submits the (minutes-long) batch backfill job; by default
it only uploads the JSONL and prints the exact `CreateModelInvocationJob` call.

## Boundaries (what this module does NOT do)

- No **dashboard / cost-anomaly alarm** on `$/ticket` — Module 14 (M12 instruments it,
  M14 displays + alarms on it).
- No **eval harness** (golden set / LLM-as-a-judge / regression gate) — Module 13. M12
  provides *only* the batch path the eval backfill rides.
- No **provisioned throughput / SageMaker endpoint** built — Module 11 (theory here, looked
  at from the cost/Flex angle only).
- No **temperature / top-k / top-p** tuning (Module 2); no **router / streaming /
  cross-Region** rebuild (Module 3 — consumed and *measured under cost*, not re-built).
- No new **schema field**; no **OpenSearch**; no idle-billed resource (the cache table is
  on-demand, and torn down anyway).

See `lab.md` for the full step-by-step, the measured cost, and "Try it yourself".
