# Module 13 — Evaluating GenAI Applications: Bedrock Evaluations, LLM-as-a-Judge, and RAG Metrics

**What:** Relay is deployed (Module 11) and optimized (Module 12) — and at Module 12 we
compressed prompts and moved triage to Nova Micro to save ~40%. A customer just complained
about an off-the-mark refund answer. **Regression, or noise?** Nobody can say: there is no
**golden dataset**, so we reread five tickets by hand, mutter "looks fine", and cross our
fingers. *A GenAI system without evaluation isn't debugged — it's guessed at.*

Module 13 builds Relay's **eval harness**: a **20-ticket golden set**, a **Bedrock RAG
evaluation** job on the Knowledge Base, a custom **LLM-as-a-judge**, a **fairness** rubric, a
**user-feedback** loop, and a **regression gate** wired into the Module 11 pipeline — so a
prompt change that degrades grounding is **blocked before it deploys**, automatically.

```bash
uv run python evals/run_evals.py \
  --fixture data/eval_fixtures/baseline_fixture.json \
  --out evals/results/run-baseline.json            # the per-ticket table + the committed baseline

uv run python evals/run_evals.py \
  --fixture data/eval_fixtures/degraded_fixture.json \
  --out evals/results/run-degraded.json \
  --gate --baseline evals/results/run-baseline.json   # a bad prompt -> GATE FAILED (exit 1)
```

```
=== Relay eval run: baseline ===
ticket                                   kind        triage  ground  cover   cite
--------------------------------------------------------------------------------
gold-01-duplicate-charge                 nominal     ok      1.000   1.000   yes
gold-07-package-lost                     nominal     ok      0.750   0.750   yes
gold-16-edge-multi-question              edge        ok      0.750   0.750   yes
...
--------------------------------------------------------------------------------
AGGREGATE  triage_accuracy=1.000  grounding=0.963  coverage=0.963  citation_rate=1.000
COST       cost_cents=0.3600 ($0.0036)
```

A degraded run drops `grounding` to **0.400** — below the **0.8** floor *and* > 5 pts under
baseline — and `run_evals.py` exits non-zero. In the pipeline, that **blocks the deploy**.

## What this module builds (on top of Module 12)

- **`evals/golden_set.json` + `evals/golden_set.py` (NEW)** — the **golden dataset**: 20
  CloudCart tickets (12 nominal, 4 edge, 2 adversarial from Module 9, 2 multimodal from
  Module 6) in the **frozen Évals contract** `{id, ticket, expected_intent, expected_points[],
  must_cite}`. A **versioned asset**, not a test file: it grows from the feedback loop.
- **`evals/judge.py` (NEW)** — the **LLM-as-a-judge**. A rubric-driven scorer (triage match,
  expected-point **coverage**, **grounding**, **citations**, **tool usage**, **task
  completion**) returning **Pydantic-validated** JSON with **one retry** on a schema miss
  (no silent `try/except`). The judge is **Anthropic Claude Haiku 4.5** on the **Flex tier
  (−50%)** — a **different model family** from every Relay candidate (Amazon Nova
  fast/smart/vision), so **self-preference bias is designed out**. It also carries the
  **fairness** rubric (skill 3.4.2) over twin-ticket pairs.
- **`evals/run_evals.py` (NEW)** — the orchestrator + the **regression gate**. Runs the
  candidate over the golden set, scores each ticket with the judge, prints the table, writes
  the frozen results JSON, and with `--gate` **fails** on `aggregate.grounding < 0.8` or a
  **> 5-pt** drop vs the committed `evals/results/run-baseline.json` **(NEW)**.
- **`relay/models.py` (MODIFIED, by addition)** — `TicketRecord.feedback_rating: int | None
  = None` — the user-feedback signal (skill 5.1.3).
- **`relay/api/feedback_handler.py` (NEW)** + **`relay/api/__init__.py` (MODIFIED)** —
  `POST /tickets/{id}/feedback {feedback_rating: int}` writes the rating onto the record. It
  **wraps** the store; no model call, no model ID.
- **`relay/config.py` (MODIFIED, by addition)** — the **judge** tier (Claude Haiku 4.5) +
  `judge_profile()` (which **enforces** judge ≠ candidate), the judge price + Flex pin, the
  eval IAM role / S3 prefixes / RAG-metric names, and `EVAL_GROUNDING_FLOOR` — **the same
  `0.8` `GROUNDING_THRESHOLD` the M9 escalation + M14 alarm use** (one constant).
- **`cdk/` (MODIFIED)** — the **eval-gate** stage is now **wired** into the CodePipeline
  (after Smoke, before promotion), and the fifth API route `POST /…/feedback` is added.
- **`setup.py` / `teardown.py` (MODIFIED)** — `setup.py` creates the `relay-eval-role` and
  the Bedrock **RAG-evaluation** job on `relay-kb` (no job surcharge — tokens only);
  `teardown.py` stops the job, deletes the role, and purges the `evals/` S3 artifacts (B5).

## Frozen contracts (bible §3.1 / §3.4)

`TicketRecord.feedback_rating` is added **by addition** (default `None`) — every earlier
record still validates; no field is renamed/retyped/removed. `converse(messages, *,
tier="auto", stream=False, **params)` is **byte-identical M3→M15**: the judge runs through it
on the appended `judge` tier, never a parallel client. The Évals contract (golden-set shape,
the `run_evals.py` CLI + output, the `0.8`/`>5pt` gate) is **frozen here** and reproduced
field-for-field. **Judge ≠ candidate** is a hard invariant, enforced in `config.judge_profile()`.

## The two grounding numbers (and why neither is enough alone)

| Eval | Scores | On | When |
|---|---|---|---|
| **Bedrock RAG evaluation** | context relevance/coverage, correctness, faithfulness | the **system** (retrieval + generation) on **your** KB | the managed, audited view of grounding |
| **LLM-as-a-judge** | grounding, coverage, citations, tool usage, task completion | the **full answer** against expected points | the fast, customizable, gating view |
| **model evaluation** | generic benchmarks | a **bare model** | *not* what tells you Relay is good on your tickets |

A public benchmark score (MMLU…) does **not** predict quality on **your** tickets. RAG
evaluation ≠ model evaluation: the first scores the **system on your data**, the second a
model on benchmarks.

## Run it

```bash
export AWS_PROFILE=aws-genai-pro             # us-east-1 everywhere; no keys in code/.env
uv sync                                      # M13 adds NO new runtime dep (boto3 + pydantic cover it)
uv run python setup.py                       # upstream (M5–M12) + the eval role + the RAG-eval job (uploaded)
uv run python -m evals.golden_set            # validate + summarize the 20-ticket golden set
uv run python evals/run_evals.py \
  --fixture data/eval_fixtures/baseline_fixture.json \
  --out evals/results/run-baseline.json      # the eval table + the committed baseline (offline)
uv run python evals/run_evals.py --live --gate \
  --out evals/results/run-latest.json        # a REAL run (triage + KB answer + Haiku judge) + the gate
uv run python evals/run_evals.py --fairness \
  --fairness-fixture data/eval_fixtures/fairness_fixture.json   # the fairness rubric (offline)
uv run pytest                                # offline cumulative suite (Modules 2–13)
RELAY_LIVE_TESTS=1 uv run pytest -m live     # opt-in, capped (a few sub-cent calls)
uv run python teardown.py                    # stops the eval job + deletes role + purges artifacts (+ M11 front door)
```

`setup.py --submit-eval` actually submits the (minutes-long) Bedrock RAG-evaluation job; by
default it only uploads the dataset and prints the exact `create_evaluation_job` call.

## Boundaries (what this module does NOT do)

- No **dashboards / alarms** on the scores — Module 14 (M13 measures; M14 watches them drift).
- No **troubleshooting / fault diagnosis** — Module 14 (the other half of Domain 5).
- No **A/B or canary infra** — theory only (Module 11 laid the deployment infra).
- No **eval-driven fine-tuning** (out of the AIP-C01 role); no large-scale **human eval**.
- Reporting is **partial** (the `run_evals.py` table) — the dashboard widget is Module 14.

See `lab.md` for the full step-by-step, the measured cost, and "Try it yourself".
