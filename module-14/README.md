# Module 14 — Operating GenAI in Production: Observability, Monitoring, and Troubleshooting

**What:** Relay is deployed (Module 11), cost-instrumented (Module 12), and evaluated
(Module 13) — but in production it is **blind**. If the Knowledge Base serves a bad answer at
3 a.m., nobody knows until a furious customer ticket lands. There are no centralized
invocation logs, no dashboard, no alarms, no procedure when something breaks. *A GenAI app
without observability doesn't fail loudly — it degrades in silence.*

Module 14 builds Relay's **ops layer**: **Model Invocation Logs** (Bedrock → CloudWatch Logs,
free on the Bedrock side), Relay's custom **CloudWatch metrics** ($/ticket, escalation rate,
guardrail block rate, eval grounding, tokens, tool latency), the **`relay-ops` dashboard** (8
widgets), **four alarms** (p95 latency, throttling, **cost anomaly detection**, grounding < 0.8
reusing the M9/M13 gate constant), and a **troubleshooting runbook** you battle-test against
**three injected faults**. (For cross-service tracing on the API → SQS → agent path, **X-Ray**
is the right tool — the same boundary tracing you wired in Module 11; this module focuses the
build on the FM-layer signals X-Ray can't see.)

```bash
uv run python observability/setup_observability.py   # invocation logging + relay-ops + 4 alarms; prints the URL
uv run python observability/inject_fault.py --list   # the 3 reversible faults the runbook covers
uv run python observability/inject_fault.py --fault context-overflow
# diagnose with the dashboard + Logs Insights (docs/runbook.md), remedy, then:
uv run python observability/inject_fault.py --restore
uv run python evals/run_evals.py \
  --fixture data/eval_fixtures/baseline_fixture.json \
  --out evals/results/run-postfix-context-overflow.json \
  --gate --baseline evals/results/run-baseline.json   # back to baseline -> GATE PASSED
```

The worst GenAI failures are **200 OK**: ungrounded answers, **retrieval drift**,
hallucinations. "No errors in CloudWatch" does **not** mean healthy — that is why the dashboard
carries **quality** signals (the golden set re-run in prod as a canary), not just error codes,
and why every remedy is verified with a golden-set re-run, not a green HTTP code.

## What this module builds (on top of Module 13)

- **`observability/setup_observability.py` (NEW)** — turns on **Model Invocation Logs**
  (`put_model_invocation_logging_configuration` → CloudWatch Logs, **free on Bedrock**, you pay
  only Logs storage), builds the **`relay-ops`** dashboard (8 widgets), and creates **four
  alarms** wired to an SNS email topic. Pure builder functions (the dashboard body, the alarm
  specs) so the smoke test asserts them offline.
- **`observability/inject_fault.py` (NEW)** — three **visible, reversible** faults
  (`context-overflow`, `kb-corruption`, `prompt-regression`) + `--restore` + `--list`. Every
  fault is a documented fixture, never opaque sabotage.
- **`observability/metrics.py` (NEW)** — the **EMF / PutMetricData** emitter. The worker logs
  one EMF line per ticket (metrics for free, no extra API call); `run_evals.py` pushes the last
  run's grounding as the **prod-canary** metric the grounding alarm watches.
- **`observability/queries/` (NEW)** — the **Logs Insights** queries the runbook cites by
  name (largest prompts, throttling errors, grounding-by-citation, cost-per-ticket).
- **`docs/runbook.md` (NEW)** — the **symptom → signal → diagnosis → remedy → verify** runbook,
  **5 entries** (the 3 faults + throttling + cost anomaly), each linked to an alarm.
- **`relay/api/worker_handler.py` (MODIFIED, by addition)** — emits the ticket's custom metrics
  (EMF) from facts it already has; best-effort (observability never fails a shipped ticket).
- **`evals/run_evals.py` (MODIFIED, by addition)** — `--emit-metrics` publishes the run's
  aggregate grounding as `EvalGrounding` (the golden set as a production canary).
- **`relay/config.py` (MODIFIED, by addition)** — the metric namespace + names, the dashboard
  name, the four alarm names + thresholds (the grounding alarm **= the one M9/M13 0.8
  constant**), the SNS topic + invocation-log group names, and the injected-fault marker.
- **`setup.py` / `teardown.py` (MODIFIED)** — `setup.py` delegates to
  `setup_observability.py`; `teardown.py` deletes the dashboard + alarms, **disables invocation
  logging**, deletes the log groups + SNS topic + logging role, and **restores any injected
  fault** (B5).

## Frozen contracts (bible §3)

**Zero Pydantic change** — Module 14 adds no field; it observes the existing schemas.
`converse(messages, *, tier="auto", stream=False, **params)` is **byte-identical M3→M15**.
The **Évals contract** is **consumed identically** (the `run_evals.py` CLI + output, the
`0.8`/`>5pt` gate) — the post-fix verification runs use the exact same harness. The grounding
**alarm** threshold **is** the M9 `GROUNDING_THRESHOLD` (= the M13 `EVAL_GROUNDING_FLOOR`) —
one `0.8` constant: **gate ↔ alarm ↔ escalation coherence**. The dashboard is the canonical
`relay-ops` (8 widgets, bible §3.3).

## Model Invocation Logs ≠ CloudTrail (the exam's distinction)

| | Logs | Carries | Turn on? |
|---|---|---|---|
| **Model Invocation Logs** | CloudWatch Logs (this module) | the Bedrock **request + response** (prompt, tokens) | yes — `put_model_invocation_logging_configuration` |
| **CloudTrail** (M10) | CloudTrail | the **management API call** (`bedrock:Converse`), not its content | already on |

Invocation logs are **free on the Bedrock side** (you pay only Logs storage, cents). They
carry prompts/responses — **sensitive** (link M10 PII) — so the log group gets a **short
14-day retention**.

## Run it

```bash
export AWS_PROFILE=aws-genai-pro              # us-east-1 everywhere; no keys in code/.env
export RELAY_BUDGET_EMAIL=you@example.com     # optional — the SNS alarm email (M1 convention)
uv sync                                       # M14 adds NO new runtime dep (boto3 + strands cover it)
uv run python setup.py                        # upstream (M5–M13) + invocation logging + relay-ops + alarms
uv run python observability/setup_observability.py   # (idempotent) re-asserts the ops layer; prints the URL
uv run python observability/inject_fault.py --fault kb-corruption   # break Relay on purpose
# follow docs/runbook.md: dashboard -> Logs Insights -> hypothesis -> remedy
uv run python observability/inject_fault.py --restore
uv run python evals/run_evals.py --fixture data/eval_fixtures/baseline_fixture.json \
  --out evals/results/run-postfix-kb-corruption.json \
  --gate --baseline evals/results/run-baseline.json    # verify back to baseline
uv run pytest                                 # offline cumulative suite (Modules 2–14)
RELAY_LIVE_TESTS=1 uv run pytest -m live       # opt-in, capped (one sub-cent CloudWatch metric)
uv run python teardown.py                     # deletes dashboard/alarms, disables logging, purges log groups (+ M11 front door)
```

For cross-service tracing on the API → SQS → agent path, **X-Ray** is the AWS tool — the same
boundary tracing covered in Module 11 (renvoi); enabling it on the deployed stack stitches the
request into one service map. Strands agent traces export to **CloudWatch generative AI
observability** (GA re:Invent 2025) via the SDK's built-in OTel telemetry — no self-hosted OTel
stack (theory only).

## Boundaries (what this module does NOT do)

- No **eval harness rebuild** — Module 13 owns it; here the golden set is a **prod canary**.
- No **cost/perf optimization** — Module 12 owns it; the dashboard **displays** `cost_cents`.
- No **security incident response** — Module 10 (key rotation, compromise) is out of scope.
- No **Managed Grafana / self-hosted OTel** and no **PagerDuty** — SNS email is the on-call
  signal (theory mentions only).
- No **load testing / profiling under load** (Module 12 / theory).

See `lab.md` for the full step-by-step, the measured cost, and "Try it yourself".
