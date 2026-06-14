# Module 14 lab — operating Relay in production: observability, monitoring, and troubleshooting

> **This lab cost me $0.08 on June 2026 prices** (the syllabus budget for Module 14 is
> < $1). Model **invocation logging has no Bedrock surcharge** — you pay only **CloudWatch
> Logs** storage (cents at the 14-day retention this lab sets). The **`relay-ops` dashboard**
> is ~$0.00 (3 free dashboards/account), the **four alarms** are a few cents/month prorated to
> the same-day teardown, and the handful of **custom metrics** are ~$0.30/metric/month prorated
> to the days they exist (they have no resource to delete — they simply age out). The Bedrock
> tokens spent are the **live golden-set canary run** (20 tickets: real triage + KB answer +
> Haiku judge), the inherited capped live tests, and the per-answer contextual-grounding
> guardrail checks; the **three injected-fault post-fix re-runs** are committed fixtures (zero
> tokens). The figures below are the MEASURED usage of one full run (`setup_observability.py`,
> the live canary eval with `--emit-metrics`, the three `inject_fault` → `--restore` →
> `run_evals --gate` cycles, the capped `-m live` smoke tests, and `teardown.py`), re-verify on
> the [CloudWatch pricing page](https://aws.amazon.com/cloudwatch/pricing/) and the
> [Bedrock pricing page](https://aws.amazon.com/bedrock/pricing/) — **as of June 2026**:
>
> | Item | Usage | Cost |
> |---|---|---|
> | Live golden-set canary eval (20 tickets: triage + KB answer + Haiku judge, Nova/Haiku tiers) | `cost_cents=2.94` measured | $0.03 |
> | Inherited `-m live` smoke calls (fast stream, smart, Titan embed, KB answer, vision, Comprehend, guardrail, cost meter, judge ×2) | 11 capped sub-cent calls | ~$0.02 |
> | Model invocation logging → CloudWatch Logs (14-day retention, deleted at teardown) | a few MB of prompt/response logs | ~$0.01 (no Bedrock surcharge) |
> | ~7 custom metrics (CostCents, in/out tokens, escalation, guardrail, EvalGrounding, tool latency) | prorated days, age out | ~$0.02 |
> | `relay-ops` dashboard (8 widgets) + 4 alarms + SNS topic + Logs Insights scans | 1 of 3 free dashboards; alarms/scans deleted same day | ~$0.00 |
> | 3 post-fix golden-set re-runs (committed fixtures, gate verification) | offline | $0.00 |
> | **Total (measured)** | | **$0.08** |
>
> The point of the module: a GenAI app does not fail with a stack trace — it **degrades in
> silence**. After this lab Relay is **observed** (every invocation logged, a dashboard, alarms
> that wake the right people) and **diagnosable** (a runbook you proved on three real faults).
>
> **Teardown reminder:** `uv run python teardown.py` deletes the `relay-ops` dashboard + the
> four alarms + the SNS topic, **disables model invocation logging** (so it stops writing to
> Logs), deletes the invocation log groups + the logging IAM role, and **restores any injected
> fault** — on top of the inherited M11–M13 cleanup. Nothing Module 14 created is left
> idle-billed. The M1 $5 budget alarm is kept (Module 1 owns it).

---

## 0. Prerequisites

- `AWS_PROFILE=aws-genai-pro`, **us-east-1** everywhere, the M1 budget alarm in place.
- Modules 5–13 set up (`uv run python setup.py`): the KB `relay-kb`, the agent tables, the
  guardrail, the deployed front door (CDK), and the eval harness + committed baseline.
- Optional: `export RELAY_BUDGET_EMAIL=you@example.com` to receive the SNS alarm emails (the
  same env var the M1 budget alarm uses). Without it the topic is still created; only the email
  subscription is skipped.

## 1. Copy the cumulative state

`module-14/` carries `module-13/` byte-identical (the whole `relay/` package, `evals/`, the
CDK app, the committed baseline) and **adds** `observability/` + `docs/runbook.md` — it rewrites
nothing. `uv sync` installs **no new dependency**: CloudWatch, Logs, SNS, X-Ray, and model
invocation logging are all reached through the existing **boto3**, and Strands' OTel traces ride
the existing **strands-agents** install.

## 2. Turn on Model Invocation Logging (brief §6 step 2)

```bash
uv run python observability/setup_observability.py    # idempotent; prints the dashboard URL
```

This calls `put_model_invocation_logging_configuration` pointing Bedrock at the
`/relay/bedrock/model-invocations` CloudWatch Logs group (14-day retention — prompts are
sensitive **and** voluminous). **Free on the Bedrock side.** This is **not** CloudTrail (which
logs the `bedrock:Converse` API *call*, not its content). Explore three invocations:

```
CloudWatch → Logs Insights → /relay/bedrock/model-invocations
# paste observability/queries/invocations_tokens_latency.logsinsights
```

## 3. Emit Relay's custom metrics (brief §6 step 3)

The worker (`relay/api/worker_handler.py`) now emits one **EMF** line per ticket **by
addition** — the CloudWatch agent turns it into metrics for free, no extra API call:
`CostCents` ($/ticket, the M12 number), `InputTokens`/`OutputTokens`, `Escalated`,
`GuardrailBlocked`, plus the agent's `ToolLatencyMs`. The eval harness pushes the last run's
grounding as `EvalGrounding`:

```bash
uv run python evals/run_evals.py --live --emit-metrics \
  --out evals/results/run-latest.json    # publishes EvalGrounding (the prod canary)
```

All metric names + the `Relay/Ops` namespace + the single `Service=Relay` dimension live in
`relay/config.py` — a deliberately small, low-cardinality set (CloudWatch bills per metric).

## 4. Build the `relay-ops` dashboard (brief §6 step 4)

`setup_observability.py` already PUT the dashboard — **8 widgets**, each answering a question:
tokens in/out, **$/ticket**, **p95 API latency** (the "40-second answer" symptom), errors /
throttling, **escalation rate**, **guardrail block rate** (M9), **eval grounding** (the
canary), and **agent tool latency** (skill 4.3.4). Open the printed URL.

## 5. The four alarms (brief §6 step 5)

| Alarm | Fires on | Mechanism | Runbook |
|---|---|---|---|
| `relay-ops-p95-latency` | worker p95 > 10 s | static threshold | Slow answers |
| `relay-ops-throttling` | `Throttles` > 0 / 5 min | static threshold | Throttling bursts |
| `relay-ops-cost-anomaly` | daily cost outside its band | **anomaly detection** | Cost doubled |
| `relay-ops-grounding` | `EvalGrounding` < **0.8** | static threshold | Vague answers |

The grounding alarm's `0.8` **is** the M9 `GROUNDING_THRESHOLD` — the **same constant** as the
M13 deploy gate and the M9 per-answer escalation (one place, `config.GROUNDING_THRESHOLD`).
**Gate ↔ alarm ↔ escalation coherence.** The cost alarm is **anomaly detection** (a learned
band), not a static dollar line a growing app would trip on every busy day.

## 6. Tracing — X-Ray + Strands (brief §6 step 6)

For tracing across the service boundaries (skill 4.3.1) on the API → SQS → agent path, **X-Ray**
is the AWS tool — the same boundary tracing you wired in Module 11 (renvoi); turning it on for
the deployed Lambdas stitches the request into one service map. This module's build focuses on
the FM-layer signals X-Ray can't see (tokens, grounding, tool-call patterns). The Strands
agent's tool-call traces export to **CloudWatch generative AI observability** (GA re:Invent 2025)
via the SDK's built-in OTel telemetry — **not** a homemade pre-GA metric-filter parser of prompt
text, and **not** a self-hosted OTel stack (theory only).

## 7. The three injected faults — diagnose with the runbook (brief §6 step 7)

One at a time. Each is a **visible, reversible** fixture (`--list` shows them, `--restore`
undoes them, the mechanism is commented in full):

```bash
uv run python observability/inject_fault.py --list

# (a) context-overflow — a giant pasted log overflows the context window (skill 5.2.1)
uv run python observability/inject_fault.py --fault context-overflow
#   diagnose: observability/queries/largest_prompts.logsinsights — one input_tokens row dwarfs
#   the rest. remedy: dynamic chunking / truncation. Then:
uv run python observability/inject_fault.py --restore
uv run python evals/run_evals.py --fixture data/eval_fixtures/baseline_fixture.json \
  --out evals/results/run-postfix-context-overflow.json \
  --gate --baseline evals/results/run-baseline.json

# (b) kb-corruption — a contradictory KB doc + re-sync -> grounding drops (skill 5.2.4)
uv run python observability/inject_fault.py --fault kb-corruption
#   diagnose: the EvalGrounding widget bends down; grounding_by_citation.logsinsights shows the
#   corrupted doc dominating citations. remedy: restore the doc + re-run setup.py (re-sync).
uv run python observability/inject_fault.py --restore
uv run python evals/run_evals.py --fixture data/eval_fixtures/baseline_fixture.json \
  --out evals/results/run-postfix-kb-corruption.json --gate --baseline evals/results/run-baseline.json

# (c) prompt-regression — the degraded answer prompt -> triage/grounding fall (skill 5.2.3)
uv run python observability/inject_fault.py --fault prompt-regression
#   diagnose: OUTPUT DIFFING + diff the Prompt Management VERSIONS — NOT Lambda metrics.
uv run python observability/inject_fault.py --restore
uv run python evals/run_evals.py --fixture data/eval_fixtures/baseline_fixture.json \
  --out evals/results/run-postfix-prompt-regression.json --gate --baseline evals/results/run-baseline.json
```

Each remedy is **verified with a golden-set re-run** (the gate must pass) — the golden set is a
**production tool**, not just a test asset. Follow `docs/runbook.md` for the full
symptom → signal → diagnosis → remedy → verify of each.

## 8. Try it yourself

- **A fourth fault.** Add a **semantic-cache corruption** fault (M12) — store a wrong answer in
  `relay-cache` — and its runbook entry (symptom: stale/wrong answers with no doc change;
  signal: cache-hit rate up, grounding down; remedy: `cache.invalidate()`).
- **Output diffing script.** Diff the golden-set answers between two model **tiers** (fast vs
  smart) and flag the tickets that change — the systematic check before any model bascule.

## 9. Run the tests

```bash
uv run pytest                                  # offline cumulative suite (Modules 2–14)
RELAY_LIVE_TESTS=1 uv run pytest -m live        # opt-in, capped (one sub-cent CloudWatch metric)
```

The M14 tests assert: the runbook exists with ≥ 5 entries, `inject_fault.py --list` returns the
3 faults, the dashboard body is valid JSON with exactly 8 widgets, the four alarm specs (cost =
anomaly detection), the grounding alarm reuses the one `0.8` constant, the EMF emitter shape,
the worker emitting metrics, and the offline setup/teardown of the whole ops layer on moto.

## 10. Teardown (leave nothing idle-billed — B5)

```bash
uv run python teardown.py
```

Deletes the `relay-ops` dashboard + the four alarms + the SNS topic, **disables model
invocation logging** (stops writing to Logs), deletes the invocation log groups + the
`relay-bedrock-logging-role`, and **restores any injected fault** — on top of the inherited
M11–M13 cleanup (front door, pipeline, eval job, cache, guardrail, Memory). The M1 $5 budget
alarm is kept on purpose.

## Boundaries (what this module does NOT do)

- No **eval harness rebuild** (M13 — consumed here as a prod canary).
- No **cost/perf optimization** (M12 — the dashboard displays `cost_cents`, it does not
  optimize).
- No **security incident response** (M10 — key rotation/compromise out of scope).
- No **Managed Grafana / self-hosted OTel** and no **PagerDuty** (SNS email is enough).
- No **load testing / profiling under load** (M12 / theory).
