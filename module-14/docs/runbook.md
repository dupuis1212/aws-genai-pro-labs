# Relay Operations Runbook (Module 14)

> An alarm without a runbook entry is just scheduled panic. Each `relay-ops` alarm links to
> exactly one entry below; each entry names the **precise** signal to look at (a CloudWatch
> Logs Insights query under `observability/queries/`, a dashboard widget, a metric) — never
> "look at the logs". The diagnosis always runs in this order: **symptom → signal →
> diagnosis → remedy → verify**.

The worst GenAI failures are **200 OK**: ungrounded answers, retrieval drift, hallucinated
policy. "No errors in CloudWatch" does **not** mean healthy — that is why three of the five
entries below are quality failures, not error-code failures, and why the verification step is
almost always a golden-set re-run (`evals/run_evals.py`), not a green HTTP code.

**Severity scale.** SEV1 = customer-facing wrong/blocked answers at scale or runaway cost;
SEV2 = degraded quality / partial outage; SEV3 = single-ticket or contained.

**Escalation.** SNS topic `relay-ops-alarms` → on-call email. SEV1 pages the GenAI on-call and
loops in the CloudCart CSM lead; SEV2 is handled by the on-call engineer; SEV3 is logged and
batched. (No PagerDuty in the lab — the SNS email is the on-call signal.)

The three injected faults (`observability/inject_fault.py --fault …`) map to the first three
entries; the last two cover the remaining `relay-ops` alarms.

---

## 1. Truncated answers / context-window overflow

- **Symptom.** Relay errors or returns a truncated/empty answer **only on long inputs** (a
  customer who pasted a huge log, a giant order export). Short tickets are fine.
- **Severity.** SEV3 (contained to oversized tickets) — SEV2 if a whole intake channel pastes
  large payloads.
- **Alarm / signal.** No infra alarm fires (the Lambda often returns 200). The first signal is
  a **size validation error** in the answer, or — proactively — an `inputTokenCount` far above
  the median in the invocation logs.
- **Diagnosis.**
  1. Run `observability/queries/largest_prompts.logsinsights` against
     `/relay/bedrock/model-invocations`. One row's `input_tokens` dwarfs the rest — that is the
     overflowing call (skill 5.2.1). Confirm with `errorCode` (a size/validation error).
  2. This is **content**, not infra: nothing in Lambda/API Gateway changed; the prompt got
     too big.
- **Remedy.** Trim the input before the FM call: **dynamic chunking / truncation** of the
  oversized field (cap the pasted-log portion, summarize-then-answer, or chunk + retrieve).
  The fix lives in the intake/answer path, not in any AWS resource.
- **Verify.** Re-run the offending ticket; the answer returns. Then
  `uv run python evals/run_evals.py --fixture data/eval_fixtures/baseline_fixture.json --out
  evals/results/run-postfix-context-overflow.json --gate
  --baseline evals/results/run-baseline.json` — grounding back to baseline, gate passes.
- **Reproduce (lab).** `uv run python observability/inject_fault.py --fault context-overflow`.

---

## 2. Vague answers / grounding drop (retrieval drift)

- **Symptom.** Answers are **fluent but wrong or vague**, citing the right *document* but the
  wrong *fact* (e.g. "refunds are instant"). Zero API errors — a textbook 200-OK failure.
- **Severity.** SEV1 (Relay is confidently telling customers something false).
- **Alarm / signal.** `relay-ops-grounding` (the `EvalGrounding` metric < **0.8**, the one
  M9/M13/M14 constant) — the golden set used as a **production canary**. The dashboard's *Eval
  grounding* widget bends down.
- **Diagnosis.**
  1. Run `observability/queries/grounding_by_citation.logsinsights` — the citations cluster on
     one source doc. Open that doc: it now contains a falsehood (a doc edit / bad re-sync
     introduced **retrieval drift**, skill 5.2.4).
  2. Confirm with **output diffing**: diff the golden-set answers before/after the last KB
     re-sync — the regressed tickets all touch that doc.
- **Remedy.** Restore the correct doc in `data/docs/`, **re-sync the Knowledge Base**
  (`uv run python setup.py` re-runs the ingestion job). Embedding diagnostics: confirm the
  re-embedded doc retrieves correctly.
- **Verify.** Re-run the golden set (as above, `--out run-postfix-kb-corruption.json`):
  grounding returns above 0.8, the grounding alarm clears.
- **Reproduce (lab).** `uv run python observability/inject_fault.py --fault kb-corruption`.

---

## 3. Triage JSON suddenly wrong / answers degrade (prompt regression)

- **Symptom.** Triage starts mis-classifying, or answers stop citing sources — right after a
  prompt change. **Nothing changed in infra.**
- **Severity.** SEV2 (quality regression across many tickets; SEV1 if it ships to all traffic).
- **Alarm / signal.** `relay-ops-grounding` may trip; the *Eval grounding* and (for triage) the
  golden-set `triage_ok` rate fall. The tell is **timing**: the drop coincides with a prompt
  deploy, not an infra change.
- **Diagnosis.**
  1. **Do NOT chase Lambda metrics** — if infra is unchanged, the cause is the prompt. Diff the
     **Prompt Management versions** (the new vs the prior revision) and do **output diffing** on
     the golden set across the two versions (skill 5.2.3). The degraded version answers "from
     memory", uncited.
  2. The invocation logs corroborate: the system prompt changed; citations vanished.
- **Remedy.** Revert the Prompt Management prompt to its prior version; re-deploy. Add the
  regressed tickets to the golden set so the **regression gate** (M13) blocks this class of
  change next time.
- **Verify.** Golden-set re-run (`--out run-postfix-prompt-regression.json`): `triage_ok` and
  grounding recover; the gate passes.
- **Reproduce (lab).** `uv run python observability/inject_fault.py --fault prompt-regression`.

---

## 4. Throttling bursts (FM API-integration errors)

- **Symptom.** Intermittent failures / slowdowns; tickets occasionally fail and redeliver.
- **Severity.** SEV2 (degraded throughput) — SEV1 if sustained and the DLQ fills.
- **Alarm / signal.** `relay-ops-throttling` (`Throttles` > 0 over 5 min). The dashboard's
  *Errors / throttling* widget spikes.
- **Diagnosis.** Run `observability/queries/throttling_errors.logsinsights` — a burst of
  `ThrottlingException` rows means you are hitting a **model quota** (skill 5.2.2), not a code
  bug. Check whether traffic spiked or a batch/eval run is contending for the same model.
- **Remedy.** `relay/llm.py` already does **exponential backoff + jitter** (M3) — confirm it is
  engaging (retries in the logs). If sustained, request a **quota increase** or move the
  latency-tolerant eval/backfill traffic to the **Flex** tier (M12) so it stops contending with
  interactive tickets. Never silently swallow the error.
- **Verify.** Throttle count returns to 0 over a 5-minute window; the alarm clears; the DLQ
  drains.

---

## 5. Cost doubled, no extra traffic (agent tool loop / cost anomaly)

- **Symptom.** The Bedrock bill (or `$/ticket`) **jumps with flat ticket volume**. The
  weekend's spend doubled; nobody deployed.
- **Severity.** SEV1 (runaway cost).
- **Alarm / signal.** `relay-ops-cost-anomaly` — **anomaly detection** on daily `CostCents`
  (a learned band, not a static line, so a normal busy day does not trip it; skill 4.3.2). The
  *$/ticket* and *FM tokens in/out* widgets rise without the volume rising.
- **Diagnosis.** Run `observability/queries/cost_per_ticket.logsinsights`: ticket *count* is
  flat but *tokens per ticket* climbed — more `converse()` calls per ticket. That is an **agent
  tool loop** (the agent re-calling a tool in a cycle; skill 4.3.4). The agent's *tool latency*
  widget and the X-Ray traces (API → SQS → agent) show the repeated tool span.
- **Remedy.** Cap the agent's tool-iteration / step budget (the M7 stop conditions); fix the
  tool whose result keeps re-triggering the loop. This is **agent observability**, not raw
  Lambda logs — the loop is invisible in `Duration` but obvious in the tool-call pattern.
- **Verify.** `$/ticket` and tokens/ticket return to their band; the anomaly alarm clears.

---

### Why these reference queries, not "the logs"

Every diagnosis above names a file in `observability/queries/` or a specific metric/widget. A
runbook that says "check the logs" fails at 3 a.m.; a runbook that says "run
`largest_prompts.logsinsights` and read the top row" works. The golden set is a **production
tool**, not just a test asset — it is the verification step for three of the five entries, run
as a canary and after every remedy.
