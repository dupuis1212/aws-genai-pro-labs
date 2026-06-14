# Module 15 lab — Capstone: ship Relay v1.0, then tear it down to ~$0

> **This lab cost me $0.41 on June 2026 prices** (the syllabus budget for the capstone is
> < $3). The headline `demo_capstone.py` run — 20 tickets end to end through the assembled
> agent, KB, guardrail, and tables — measured **7.46¢** of real Bedrock tokens (0.37¢/ticket,
> all on the `smart` tier via `relay.llm`'s metered usage). The rest is the live `-m live`
> smoke calls, the idempotent `setup.py` (one small Titan KB ingestion + the seed/eval/batch
> uploads — no eval/batch JOB submitted, so token-only), and a handful of re-runs while
> building the recap. **AgentCore long-term Memory** (~$0.75 / 1K records / month — the only
> monthly idle item) was created by `setup.py` and **purged at teardown the same day**, so it
> prorates to ~$0. Everything else (DynamoDB on-demand, S3 Vectors, the MCP Lambda + Function
> URL, the `relay-ops` dashboard + 4 alarms, invocation logs) is idle ~$0 and torn down. The
> figures below are the MEASURED usage; re-verify on the
> [Bedrock pricing page](https://aws.amazon.com/bedrock/pricing/) — **as of June 2026**:
>
> | Item | Usage | Cost |
> |---|---|---|
> | `demo_capstone.py` — 20 tickets end to end (triage + agent + KB + guardrail + vision + handoff), metered smart-tier | `total_cost_cents=7.46` measured | $0.07 |
> | Re-runs while building the recap (~4 partial/full demo runs + probes) | metered, smart tier | ~$0.25 |
> | `-m live` capped smoke calls (M2–M15: fast/smart/Titan/KB/vision/agent/handoff/Comprehend/guardrail/cost-meter/judge×2/metric/capstone ticket) | 14 capped sub-cent calls | ~$0.06 |
> | `setup.py` ×2 (idempotent) — Titan KB ingestion + seed/eval/batch dataset uploads (no JOB submitted) | a few KB-ingestion + embed cents | ~$0.03 |
> | AgentCore long-term Memory (`relay-memory`) — the only monthly idle item | created then **purged same day** | ~$0.00 |
> | DynamoDB on-demand / S3 Vectors / MCP Lambda + URL / `relay-ops` dashboard + 4 alarms / invocation logs | idle ~$0, torn down | ~$0.00 |
> | **Total (measured)** | | **$0.41** |
>
> The point of the capstone: a pile of fourteen green modules is **not a system**. After this
> lab Relay is **assembled** (one diagram, every block tagged by module), **hardened**
> (idempotency / timeouts / quotas, by addition), **reviewed** (the Generative AI Lens, pillar
> by pillar), **demonstrated** (20 costed tickets), **tagged `v1.0`** — and then **destroyed**,
> the account back to **~$0/month**.
>
> **Teardown reminder (the point of orgue, B5):** `uv run python teardown.py` is EXHAUSTIVE —
> it deletes the `relay-ops` dashboard + 4 alarms + SNS topic, disables invocation logging,
> deletes the cache table + MCP Lambda + Function URL + bounded role, **purges the AgentCore
> long-term Memory** (the only monthly idle item), deletes the guardrail, the KB + its role,
> the eval/batch roles + S3 artifacts, and (when deployed) the CDK API + pipeline stacks. The
> M1 $5 budget alarm is kept on purpose (Module 1 owns it). After teardown nothing M1→M15
> created is left idle-billed.

---

## 0. Prerequisites

- `AWS_PROFILE=aws-genai-pro`, **us-east-1** everywhere, the M1 budget alarm in place.
- Modules 5–14 set up (`uv run python setup.py`): the KB `relay-kb`, the agent tables +
  MCP Lambda, the guardrail, the cache table, the eval harness + committed baseline, and the
  `relay-ops` observability layer. `setup.py` is idempotent — run it twice, it reuses every
  resource and never duplicates.
- Optional: `export RELAY_BUDGET_EMAIL=you@example.com` for the SNS alarm emails.

## 1. Copy the cumulative state (nothing is rewritten)

`module-15/` carries `module-14/` byte-identical — the whole `relay/` package, `evals/`,
`observability/`, the CDK app, the committed baseline, `docs/runbook.md` — and **adds only**
the capstone increment: `demo_capstone.py`, `docs/genai-lens-review.md`, the hardening by
addition (idempotency on the front-door write), and this README/lab. `uv sync` installs **no
new dependency**.

## 2. Assemble — one system, fourteen modules (skill 1.1.1)

The architecture diagram (README + the article's Mermaid) tags every block with the module
that built it: API/CDK (M11), intake + PII (M6/M10), the Strands agent + MCP (M7), AgentCore
Runtime + Memory + HITL (M8), the KB on S3 Vectors (M4/M5), DynamoDB (M7), the guardrail (M9),
caches (M12), evals + gate (M13), observability (M14), the `relay-events` bus (M11). The
assembled system **is** the solution design — the choices hold together (S3 Vectors over
OpenSearch for ~$0 idle, `converse()` as the single call site, async via SQS, HITL on refunds).

## 3. Deploy the system

```bash
uv run python setup.py          # idempotent: KB + tables + MCP Lambda + guardrail + cache +
                                # AgentCore Memory + relay-ops dashboard/alarms + invocation logs
uv run python setup.py          # run it AGAIN — every resource "already exists. Reusing." (no dup)
# (optional) the deployed front door for the --api-url demo path:
uv run cdk deploy RelayApiStack # API Gateway + 4 Lambda + SQS + relay-events
```

`setup.py` launches **no AgentCore Runtime** to create (the `agentcore` CLI does that; idle is
FREE). The MCP server runs as a Lambda (URL recorded in `.mcp_url`). For a fully local run you
can instead serve it on your laptop:

```bash
uv run python -m mcp_server     # serves http://127.0.0.1:8000/mcp
export RELAY_MCP_URL=http://127.0.0.1:8000/mcp
```

## 4. Harden — idempotency, timeouts, quotas (by addition, no signature touched)

| Failure mode | Hardening | Where |
|---|---|---|
| A CloudCart webhook delivers the SAME ticket twice | **Idempotency** key on the front-door `received` write — a conditional `attribute_not_exists(ticket_id)` PutItem; a duplicate is an idempotent no-op (202 `duplicate:true`), **never a second pipeline or refund** | `mcp_server/store.create_ticket_first_seen`, `relay/api/post_handler` (M15) |
| The agent loops on a failing tool | **Timeout** + stop conditions — `AGENT_TIMEOUT_S=60`, `MAX_ITERATIONS=6`, bedrock-runtime read/connect timeouts; the run ends `failed`, not hung | `relay/agent.py` (M7, APPLIED here) |
| A burst of tickets exhausts the Bedrock quota | **Quotas / throttling** — Lambda reserved concurrency + SQS buffer; `relay.llm.converse()` exponential backoff + jitter on throttling | `cdk/`, `relay/config.py`, `relay/llm.py` (M11/M3) |
| A tool (DynamoDB) errors mid-run | **Partial-failure handling** — the tool returns a model-readable error (no crash); a poison SQS job DLQs after `RELAY_QUEUE_MAX_RECEIVE=3` | `mcp_server/store`, `relay/api/worker_handler` (M7/M11) |
| A refund is proposed but not approved | Stays `awaiting_approval` — **nothing executes** without `POST /tickets/{id}/approve` | `relay/agent.py`, `relay/approve.py` (M8) |
| A cost spike | The M1 budget alarm + the `relay-ops` cost-anomaly alarm (learned band) | M1 / M14 |

**Hardening ≠ adding features.** It is RESILIENCE around what already exists. Prove the
idempotency in the smoke test (two submissions → one `TicketRecord`).

## 5. Review — the Well-Architected Generative AI Lens (skill 1.1.3)

`docs/genai-lens-review.md` walks the **six pillars** (Operational Excellence, Security,
Reliability, Performance Efficiency, Cost Optimization, Sustainability) + the GenAI-specific
considerations (model selection, RAG grounding, responsible AI, HITL, evaluation), each with a
status (✅/⚠️), the concrete Relay control, and the remaining gap. The ⚠️ rows ARE the
"In production" frontier (multi-Region, formal SLA, KMS CMKs, a third-party audit). It is a
checklist a reviewer ticks — not paperwork.

## 6. Demonstrate — the 20-ticket costed run

```bash
uv run python demo_capstone.py --list     # the 20 demo tickets (no AWS)
uv run python demo_capstone.py            # LOCAL path: the frozen run_relay seam, metered
uv run python demo_capstone.py --api-url $RELAY_API_URL   # DEPLOYED-API path (POST -> poll GET)
```

The run drives 20 varied tickets (16 nominal + 2 adversarial reusing the M9 attacks + 2
multimodal reusing the M6 screenshots) end to end and prints, per ticket, its triage and — by
outcome — a cited answer, an action, an escalation, or `awaiting_approval`. The recap:

```
RECAP
  tickets               : 20
  by status             : {'answered': 13, 'failed': 5, 'awaiting_approval': 2}
  by category           : {'nominal': 16, 'adversarial': 2, 'multimodal': 2}
  escalation rate       : 0.0%
  awaiting approval     : 2
  total $/ticket        : 0.0746 USD (7.457¢ over 20 tickets)
  cost per ticket       : 0.3729¢
  p95 latency           : 7327 ms
  golden-set grounding  : 0.963  (floor 0.8)
```

The two refunds park on the HITL gate; the attacks are handled safely (no PII exfiltration);
the vision tickets answer with the screenshot's content. The `failed` tickets are the
**hardening working** — the turn cap stopping a KB-search runaway, not a crash. `cost_cents`
is the real metered smart-tier token cost (the M12 instrumentation, consumed); the grounding
score is a re-run of the M13 golden set (the same `run_evals.py` contract, gate floor 0.8).

## 7. Release — tag v1.0

```bash
git add -A && git commit -m "Relay v1.0 — capstone: assembled, hardened, reviewed, demonstrated"
git tag v1.0                              # the contract deliverable (06 §3)
```

The README carries the pitch + the architecture + the run numbers + the teardown command — a
portfolio-ready repo, not a tutorial.

## 8. Tear down — prove ~$0/month (the point of orgue, B5)

```bash
uv run python teardown.py
```

EXHAUSTIVE: the `relay-ops` dashboard + 4 alarms + SNS topic, invocation logging disabled +
log groups purged, the cache table, the MCP Lambda + Function URL + bounded role, **the
AgentCore long-term Memory purged** (the only monthly idle item), the guardrail, the KB + its
role, the eval/batch roles + S3 artifacts, the agent tables, and (when deployed) the CDK API +
pipeline stacks. Then verify (Cost Explorer / a resource inventory) that nothing is idle-billed
— **account back to ~$0/month**. The M1 $5 budget alarm is kept (Module 1 owns it).

## 9. Run the tests

```bash
uv run pytest                                  # offline cumulative suite (Modules 2–15)
RELAY_LIVE_TESTS=1 uv run pytest -m live        # opt-in, capped (one real capstone ticket, < $0.02)
```

The M15 tests assert: `demo_capstone.py --list` returns exactly 20 tickets with the
nominal/adversarial/multimodal mix; idempotency (two submissions of the same ticket → one
`TicketRecord`, the second an `IdempotentReplay`, no second enqueue); the recap math (status
counts, escalation rate, $/ticket, p95) on a scripted offline runner; `docs/genai-lens-review.md`
covers all six pillars; the README documents `v1.0`; and `converse()`'s signature is still
byte-identical M3→M15.

## 10. Try it yourself

- **A 21st ticket.** Add a ticket of your own to `demo_capstone.py` (e.g. a partial-refund
  edge case) and explain where it lands in the recap.
- **One Lens row.** Add a row to `docs/genai-lens-review.md` for the gap you would close FIRST
  in production — and say why (blast radius or the cost of being wrong should drive the order).
