# Module 13 lab — evaluating Relay: Bedrock Evaluations, LLM-as-a-judge, and RAG metrics

> **This lab cost me $0.42 on June 2026 prices** (the syllabus budget for Module 13 is
> < $2). Everything Module 13 adds is **on-demand / ~$0 idle**: the `relay-eval-role` IAM
> role is free, the eval artifacts are small S3 objects (purged at teardown), and a Bedrock
> **model-evaluation job has no job surcharge — you pay only the tokens it consumes**. The
> spend is: a live pass of the 20-ticket golden set judged in **Anthropic Claude Haiku 4.5**,
> plus the real KB answers the candidate produces, plus one **Bedrock RAG evaluation** job
> over `relay-kb` (retrieve-and-generate, four judge-based metrics × 19 prompts — this is the
> larger line). The figures below are the MEASURED usage of one full live run (`setup.py`,
> the offline baseline + a live `run_evals.py --live --gate`, a live `--fairness`, one
> `--submit-eval` RAG job run to completion, the capped live tests, and `teardown.py`),
> cross-checked against the CloudWatch `AWS/Bedrock` token metrics for the run window
> (**280,000 input + 59,446 output tokens over 409 invocations**) and the pricing pages —
> never guessed (re-verify on the [Bedrock pricing page](https://aws.amazon.com/bedrock/pricing/)
> — **model evaluation**, **service tiers (Flex)**, and the **Nova / Claude** per-token
> sections — **as of June 2026**):
>
> | Item | Usage | Cost |
> |---|---|---|
> | live `run_evals.py --live --gate` candidate (triage Nova Micro + KB answer Nova 2 Lite × 19) + 19 judge verdicts (Claude Haiku 4.5) | ~54k in / ~12k out tok | ~$0.030 |
> | live `--fairness` (6 twin pairs: real KB answers + real judge) | small | ~$0.006 |
> | Bedrock RAG evaluation job on `relay-kb` (`--submit-eval`, 19 prompts, retrieve-and-generate + 4 judge metrics: Correctness / Faithfulness / Completeness / CitationPrecision) | ~200k in / ~40k out tok | ~$0.37 |
> | capped live tests (1 judge, 1 KB-answer+judge) + inherited M2–M9 live smoke calls | small | ~$0.01 |
> | `relay-eval-role` IAM + eval S3 artifacts + the eval job (no job surcharge) | — | ~$0.00 (~$0 idle) |
> | **Total (measured, CloudWatch-cross-checked)** | **280k in / 59k out, 409 calls** | **≈ $0.42** |
>
> **Freshness note (June 2026):** the brief pinned the judge to **Claude Haiku 4.5 on the
> Flex tier (−50%)**, but live this model serves **only the `default` service tier** —
> `Converse` rejects `serviceTier=flex` for it ("service tier is not supported for this
> model"). Nova 2 Lite *does* offer Flex. The judge stays **Claude Haiku 4.5** (the
> *judge ≠ candidate* anti-self-preference invariant outranks the cost lever), so the judge
> here bills at standard, not Flex — `relay/llm.py` degrades an unsupported service tier to
> the model's default and bills it at default (no phantom −50%). That is the main reason this
> run lands above the brief's < $2 budget comfortably but not at a Flex-discounted figure.
>
> The point of the module: a **single objective number** answers *"did we just make Relay
> worse?"* — and the gate **blocks** a regression before it ships. The committed baseline
> (`evals/results/run-baseline.json`) and the degraded-prompt demo run **entirely offline**
> from committed fixtures, so a fresh clone proves the gate **without spending a cent**.
> **Teardown reminder:** `uv run python teardown.py` stops the RAG-eval job, deletes
> `relay-eval-role`, and purges the `evals/` S3 artifacts (B5), on top of the inherited M11
> front-door / pipeline cleanup. Nothing Module 13 created is left idle-billed.

---

## 0. Prerequisites

- The cumulative Relay from Modules 1–12 (this directory ships it). `AWS_PROFILE=aws-genai-pro`,
  **us-east-1** everywhere, the persistent **$5 budget + 80% alarm** from Module 1.
- `uv` installed; `uv sync` (Module 13 adds **no** new runtime dependency — `boto3` + the
  already-pinned `pydantic` cover the judge, the RAG-eval control plane, and the schemas).
- The Module 5 Knowledge Base `relay-kb` set up (the RAG-evaluation job + the live candidate
  answer against it). The offline fixture paths need none of this.

## 1. Copy the cumulative state

This module ships `module-12/relay/` **byte-identical**, then ADDS `evals/`, the feedback
handler + route, and `TicketRecord.feedback_rating`. Verify the inherited tests still pass:

```bash
uv run pytest tests/smoke_test.py -q     # Modules 2–13, offline (moto / Stubber)
```

## 2. The golden dataset — `evals/golden_set.json` (NEW)

20 CloudCart tickets in the **frozen Évals contract** `{id, ticket, expected_intent,
expected_points[], must_cite}` — **12 nominal, 4 edge, 2 adversarial** (the Module 9 injection
/ jailbreak family), **2 multimodal** (a Module 6 screenshot attachment). It is a **versioned
asset**: the feedback loop (§8) feeds it new failing cases.

```bash
uv run python -m evals.golden_set        # validate + summarize the mix (12/4/2/2)
```

The metrics the exam names live here, **per the official AIP-C01 exam guide**: **relevance**,
**factual accuracy**, **consistency**, **fluency** — a support answer has fifty valid
phrasings, so there is no single label to diff against. That is why we judge, not string-match.

## 3. The LLM-as-a-judge — `evals/judge.py` (NEW)

A rubric (criteria + a 1–5 scale + what each criterion means) scoring Relay's triage + answer
+ agent actions as **Pydantic-validated JSON**, with **one retry** that feeds the validation
error back (no silent `try/except`). **The judge is never the candidate:** Relay answers with
**Amazon Nova**, so the judge is **Anthropic Claude Haiku 4.5** — crossing vendors kills the
**self-preference bias** at zero cost. It runs on the **Flex tier (−50%)** because an eval job
tolerates latency. **Calibrate before you trust it:** `judge.calibration_agreement(...)` checks
the judge lands within 1 point of a handful of hand-scored cases (aim ≥ 0.8) before its scores
gate anything.

The judge ID lives **only** in `relay/config.py` (the appended `judge` tier);
`config.judge_profile()` **raises** if it ever equals a candidate profile.

## 4. The fairness rubric (skill 3.4.2)

Same judge, different rubric, over **6 twin-ticket pairs** (`data/fairness_pairs.json` — same
problem, different *irrelevant* customer attribute: name, region, business size, fluency, tone).
The two answers' scores must not diverge by more than **1 point**:

```bash
uv run python evals/run_evals.py --fairness \
  --fairness-fixture data/eval_fixtures/fairness_fixture.json   # offline
# or --fairness --live to score the real answers
```

## 5. The Bedrock RAG evaluation job (skills 5.1.2 / 5.1.5 / 5.1.6)

`setup.py` builds a RAG-evaluation **dataset** from the golden set and submits a Bedrock
**model-evaluation** job (RAG, retrieve-and-generate) on `relay-kb` — **correctness /
faithfulness / completeness / citation precision** (context relevance is a retrieve-only
metric, invalid in a retrieve-and-generate job). **No job surcharge — you pay only the tokens
it consumes.** Compare its managed grounding number (Faithfulness) to the home judge's:
*the two grounding views should agree.*

```bash
uv run python setup.py                    # uploads the dataset + prints the create_evaluation_job call
uv run python setup.py --submit-eval      # actually submits the (minutes-long) RAG-eval job
aws bedrock get-evaluation-job --job-identifier <arn>    # poll for the report in evals/output/
```

**Measured run (June 2026, 19 prompts, retrieve-and-generate, evaluator = Claude Haiku 4.5).**
The job ran ~15 minutes to `Completed`. The managed report's aggregate:

| Metric | Score |
|---|---|
| Correctness | 0.556 |
| **Faithfulness** | **0.643** |
| Completeness | 0.539 |
| CitationPrecision | 0.741 |

The home judge's live grounding over the same golden set was **0.588**
(`evals/results/run-latest.json` → `aggregate.grounding`), which lands right next to the
managed **Faithfulness 0.643** — an independent managed metric corroborating the custom
judge's grounding view. Both fell below the 0.8 floor on this live run, so the live gate
correctly blocked.

## 6. From scores to a gate — `evals/run_evals.py` (NEW)

`run_evals.py` runs the candidate over the golden set, scores each ticket with the judge,
prints the per-ticket table (the table **is** the report, skill 5.1.8), and writes the
**frozen** results JSON `{run_name, config, scores:[{id, triage_ok, grounding, coverage,
citations}], aggregate, cost_cents}`.

```bash
# Build the committed BASELINE (offline, deterministic, no tokens):
uv run python evals/run_evals.py \
  --fixture data/eval_fixtures/baseline_fixture.json \
  --out evals/results/run-baseline.json

# A REAL run + the gate (spends tokens):
uv run python evals/run_evals.py --live --gate \
  --out evals/results/run-latest.json
```

The **regression gate** (`--gate --baseline <file>`) fails when `aggregate.grounding < 0.8`
**or** grounding regresses **> 5 pts** vs the baseline. **`0.8` is `config.GROUNDING_THRESHOLD`
— the same constant the Module 9 grounding escalation and the Module 14 `relay-ops` alarm use.
Define it once; never a divergent literal.**

Demo the catch with the **degraded prompt** (`data/degraded_prompt.md` — an answer prompt that
tells the model to reply from memory and drop citations):

```bash
uv run python evals/run_evals.py \
  --fixture data/eval_fixtures/degraded_fixture.json \
  --out evals/results/run-degraded.json \
  --gate --baseline evals/results/run-baseline.json
# AGGREGATE grounding=0.400 -> GATE FAILED -> exit 1 -> the pipeline blocks the deploy.
```

## 7. Wire the gate into the pipeline + the feedback loop

The Module 11 CodePipeline now runs the eval-gate as a real stage (after Smoke, before
promotion): `cdk/relay_cdk/pipeline_stack.py` ships `Source → Build → Deploy → Smoke →
EvalGate`, and `pipeline/eval_buildspec.yml` runs `run_evals.py --gate`. A grounding
regression fails the stage and blocks promotion.

The **user-feedback** loop (skill 5.1.3) closes the circle: `POST /tickets/{id}/feedback
{feedback_rating: int}` writes the customer's 1–5 rating onto `TicketRecord.feedback_rating`
(the field added by addition). **Low ratings are where the next failing golden cases come
from** — rate → triage the failures → grow `golden_set.json`.

```bash
curl -X POST https://<api-id>.execute-api.us-east-1.amazonaws.com/prod/tickets/<id>/feedback \
  -d '{"feedback_rating": 2}'
```

## 8. Try it yourself

1. **Add a rubric criterion (conciseness)** to `evals/judge.py` and re-baseline — does a
   verbose-but-correct answer now score lower? (The judge has a *verbosity bias*; making
   conciseness explicit is one way to counter it.)
2. **Switch the judge to Nova 2 Lite** (point the `judge` tier at the smart profile *only as
   an experiment* — `config.judge_profile()` will refuse it as a candidate collision, which is
   the lesson) and compare the two judges' scores on 5 tickets: **inter-judge calibration**.

## 9. Run the tests

```bash
uv run pytest                              # offline cumulative suite (Modules 2–13)
RELAY_LIVE_TESTS=1 uv run pytest -m live   # opt-in, capped: 1 real judge call + 1 KB-answer+judge
```

The offline suite proves the golden set is exactly 20 entries, the judge validates + retries,
the committed baseline **passes** the gate, the degraded fixture **fails** it (exit 1), the
feedback handler writes `feedback_rating` on a moto table, and the judge ID lives only in
`config.py`.

## 10. Teardown (leave nothing idle-billed — B5)

```bash
uv run python teardown.py                  # stops the RAG-eval job, deletes relay-eval-role,
                                           # purges the evals/ S3 artifacts (+ the M11 front door)
```

`teardown.py` is idempotent and verbose. It does **not** touch the committed
`evals/results/run-baseline.json` or the `evals/` package — those are **source**, not infra.

## Boundaries

- No **dashboards / alarms** on the scores — Module 14 (M13 measures; M14 watches them drift).
- No **troubleshooting / fault diagnosis** — Module 14 (the other half of Domain 5).
- No **A/B or canary infra** — exam theory only.
- No **eval-driven fine-tuning** (out of the AIP-C01 role); no large-scale **human eval**
  (Bedrock human evaluation exists — one line).
- Reporting is **partial** (the `run_evals.py` table); the dashboard widget is Module 14.
