# Module 15 — Capstone: Ship Relay, a Production-Grade GenAI Support Agent

**Relay v1.0.** Fourteen modules built Relay piece by piece: triage (M2), the `converse()`
layer (M3), RAG + Knowledge Base (M4/M5), multimodal intake (M6), the Strands agent + MCP
(M7), AgentCore Runtime + Memory + HITL (M8), the guardrail (M9), PII redaction + IAM (M10),
the serverless API + CDK + pipeline (M11), caches + `cost_cents` (M12), the golden-set evals +
gate (M13), and the `relay-ops` observability layer (M14). Every piece passed in isolation —
but a pile of fourteen green modules **is not a system**. A system is what survives a real
ticket end to end, a duplicated webhook, a runaway agent, and a cost spike — and what you can
**destroy and rebuild on demand**.

This capstone does five things and adds **no new concept and no new service**:

1. **Assemble** — one cohesive architecture diagram, every block tagged with the module that
   built it (skill **1.1.1**, solution design). The assembled system *is* the design.
2. **Harden** (by addition, no signature touched) — **idempotency** on the front-door write (a
   conditional `attribute_not_exists(ticket_id)` PutItem, so a duplicated CloudCart webhook is
   an idempotent no-op, never a second refund), **timeouts** + stop conditions applied on the
   agent (`AGENT_TIMEOUT_S`, `MAX_ITERATIONS`), and **quotas / throttling** (Lambda reserved
   concurrency + SQS buffer + the M3 `converse()` backoff).
3. **Review** — `docs/genai-lens-review.md`, the **AWS Well-Architected Generative AI Lens**
   applied pillar by pillar as an actionable checklist that surfaces the gaps (skill **1.1.3**).
4. **Demonstrate** — `demo_capstone.py` runs **20 varied tickets** (≈16 nominal + 2 adversarial
   reusing the M9 attacks + 2 multimodal reusing the M6 screenshots) end to end and prints a
   **costed recap**: count by status, escalation rate, total **$/ticket**, **p95** latency, and
   the **golden-set grounding** score (a re-run of the M13 harness).
5. **Release + tear down** — tag **`v1.0`**, then **`teardown.py`** destroys everything,
   **including the AgentCore long-term Memory** (the only monthly idle item), and verifies the
   account is back to **~$0/month** (decision B5 — the trust differentiator of the course).

## Run it

```bash
export AWS_PROFILE=aws-genai-pro              # us-east-1 everywhere; no keys in code/.env
export RELAY_BUDGET_EMAIL=you@example.com     # optional — the SNS alarm email (M1 convention)
uv sync                                       # M15 adds NO new runtime dep
uv run pytest                                 # offline cumulative suite (Modules 2–15)

# (optional) deploy the full front door for the DEPLOYED-API demo path:
uv run python setup.py                        # upstream (M5–M14) + cdk deploy + AgentCore
# then point the demo at the stage:  --api-url $RELAY_API_URL

# the headline deliverable — 20 tickets end to end, costed:
uv run python demo_capstone.py --list         # the 20 demo tickets (no AWS)
uv run python demo_capstone.py                # LOCAL path: the frozen run_relay seam, metered
uv run python demo_capstone.py --api-url $RELAY_API_URL   # DEPLOYED-API path (POST -> poll GET)

RELAY_LIVE_TESTS=1 uv run pytest -m live      # opt-in, capped (one real capstone ticket, < $0.02)

git tag v1.0                                  # the contract deliverable (06 §3)
uv run python teardown.py                     # EXHAUSTIVE — incl. AgentCore long-term Memory; ~$0/month
```

The demo has **two execution paths, one recap**. `--api-url` is the brief's deployed-API path
(`POST /tickets` → poll `GET /tickets/{id}` against the `cdk deploy`-ed stage). The default
**local** path drives `relay.run.run_relay()` — the *exact* frozen seam the deployed worker
invokes — wrapped in a `CostMeter`, so the capstone is runnable and the recap reproducible
without standing up the whole stack. Same agent, same KB, same guardrail, same Bedrock calls;
it just skips the API Gateway + SQS hop. **No new model client, no new generation path** — it
consumes `run_relay`.

## What this module builds (on top of Module 14)

- **`demo_capstone.py` (NEW)** — the 20-ticket costed end-to-end run + the recap (`--list`,
  `--api-url`, the local `run_relay` path, `--live-eval`).
- **`docs/genai-lens-review.md` (NEW)** — the Generative AI Lens checklist, six pillars + the
  GenAI-specific considerations, each with a status (✅/⚠️), the Relay control, and the gap.
- **Hardening by addition** — `mcp_server/store.py` gains `create_ticket_first_seen()` (the
  conditional idempotent first-write) + `IdempotentReplay`; `relay/api/post_handler.py` uses it
  for the `received` write and treats a duplicate as a no-op. **No signature, schema, or
  resource name changed.** The agent's timeouts/stop-conditions (M7) and the `converse()`
  backoff (M3) are *applied*, not re-specified.
- **`README.md` / `lab.md` (rewritten)** — the v1.0 pitch + the measured cost + the exhaustive
  teardown reminder.

## Frozen contracts (bible §3)

**Zero contract change.** The capstone CONSUMES the Pydantic schemas
(`Ticket`/`Triage`/`Citation`/`Answer`/`AgentAction`/`TicketRecord` with its 7 statuses), the
`converse(messages, *, tier="auto", stream=False, **params)` signature (byte-identical M3→M15),
the API (`POST /tickets`, `GET /tickets/{id}`, `POST /tickets/{id}/approve`), the `relay-events`
bus, and the évals contract — all field-for-field. The hardening adds idempotency/timeouts/
quotas *around* them; it mutates nothing.

## Boundaries (what this module does NOT do)

- **No new concept and no new service** — if it taught something not seen in M1–M14, it would
  be a content module, not a capstone.
- **No exam strategy / mock exam / revision grid / debrief** — that is **Module 16** (a sharp
  boundary; M16 ships no Relay code).
- **No real multi-Region, no formal SLA, no load testing at scale, no fine-tuning** — theory,
  pointed to by the Lens review and the "In production" section.
- **No idle-billed resource left behind** — the verified teardown is the point.

See `lab.md` for the full step-by-step, the **measured cost**, and "Try it yourself".
