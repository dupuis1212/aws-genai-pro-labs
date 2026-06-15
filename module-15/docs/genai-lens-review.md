# Relay v1.0 — AWS Well-Architected Generative AI Lens review

> **Scope.** A pillar-by-pillar review of Relay against the **AWS Well-Architected Generative AI
> Lens** (the official whitepaper, docs.aws.amazon.com/wellarchitected/latest/generative-ai-lens,
> 2026 revision — re-verify the version on review day). This is the skill **1.1.3** deliverable:
> not paperwork, a **checklist a reviewer can tick** before a go-live. For each pillar it names
> the CONCRETE Relay control that satisfies it (and the module that built it) and the **gap** that
> remains — the gaps are exactly the "In production" items (article T7), not pretend-done rows.
>
> The Lens organizes the **six Well-Architected pillars** across **six generative-AI lifecycle
> phases** (Scoping → Model selection → Customization → Integration → Deployment → Iteration).
> Relay sits in *Integration → Deployment → Iteration*; this review reads each pillar against
> Relay's deployed shape.
>
> Status legend: ✅ = a built control covers it · ⚠️ = partially covered, a gap noted (→ T7).

---

## 1. Operational Excellence

| Best practice (Lens) | Status | Relay control (module) | Gap → "In production" |
|---|---|---|---|
| Observe the workload end to end | ✅ | Model invocation logs → CloudWatch + the **`relay-ops`** dashboard (8 widgets) + X-Ray on API→SQS→agent (M14) | — |
| Alarm on the signals that matter | ✅ | 4 alarms: p95 latency, throttling, cost anomaly, grounding<0.8 (M14) | A real on-call rotation + paging (SNS email only) |
| Operate with a runbook | ✅ | `docs/runbook.md` — symptom→signal→diagnosis→remedy→verify, 5 entries proven on 3 injected faults (M14) | Extend the runbook as new faults are seen |
| Use the golden set as a prod canary | ✅ | `evals/run_evals.py --emit-metrics` publishes `EvalGrounding`; the grounding alarm watches it (M13/M14) | Scheduled canary cadence (here it is on-demand) |
| Quality is a release gate, not a vibe | ✅ | The CodePipeline eval-gate (grounding<0.8 / >5 pts) blocks promote (M13) | Blue/green or canary **model** rollout (theory) |

## 2. Security

| Best practice (Lens) | Status | Relay control (module) | Gap → "In production" |
|---|---|---|---|
| Treat customer content as untrusted | ✅ | **`relay-guardrail`** (content/topics/words/prompt-attack) — standalone ApplyGuardrail + the `converse()` hook (M9); the agent's ReAct path leans on the system prompt + IAM tool boundary | Wire the guardrail onto the agent's request path (M9 next step) |
| Defend against prompt injection / exfiltration | ✅ | Prompt-attack filter (M9) + the IAM tool boundary on `lookup_order`/`create_ticket` (M7) — a slipped injection still cannot read another customer's data | A red-team pass beyond the 12-attack suite |
| Keep PII out of the model and the logs | ✅ | Comprehend `DetectPiiEntities` redaction at intake **before** any FM call, `Ticket.pii_redacted` (M10); guardrail PII mask; events carry id+status only (M11) | KMS CMK on the data bucket / table (default SSE today) |
| Least-privilege IAM per component | ✅ | `iam/policies/*.json` — explicit Action/Resource ARNs, zero wildcards (M10) | A periodic access review / Access Analyzer in CI |
| Audit the management plane | ✅ | CloudTrail management events + `audit_report.py` (M10) | Centralized SIEM / data-residency controls (theory) |

## 3. Reliability

| Best practice (Lens) | Status | Relay control (module) | Gap → "In production" |
|---|---|---|---|
| Retry transient model failures | ✅ | `relay/llm.converse()` exponential backoff + jitter on throttling/5xx; documented cross-Region inference fallback (M3) | — |
| Idempotent intake (no duplicate side-effects) | ✅ | **Idempotency** key on the front-door `received` write — conditional `attribute_not_exists(ticket_id)` PutItem; a duplicate webhook is an idempotent no-op, never a second pipeline or refund (M15) | Idempotency tokens spanning the whole CloudCart→Relay edge |
| Bound the agent (no runaway) | ✅ | **Timeouts** + stop conditions: `AGENT_TIMEOUT_S=60`, `MAX_ITERATIONS=6`, bedrock-runtime read/connect timeouts (M7, applied M15) | — |
| Handle partial failures cleanly | ✅ | A failing tool returns a model-readable error (not a crash); a stuck/over-budget run ends `failed`; the SQS redrive policy DLQs a poison job after `RELAY_QUEUE_MAX_RECEIVE=3` (M7/M11) | — |
| Decouple async work | ✅ | API returns 202 + SQS buffer + worker; `relay-events` bus routes escalations/approvals loosely (M11) | Multi-Region active/active (single-Region by design — theory) |

## 4. Performance Efficiency

| Best practice (Lens) | Status | Relay control (module) | Gap → "In production" |
|---|---|---|---|
| Right-size the model per task | ✅ | The tier router: `fast` (Nova Micro) for triage, `smart` (Nova 2 Lite) for answers/agent, `auto` complexity routing (M3) | — |
| Cache to cut repeated work | ✅ | Prompt caching on the system prefix + the **semantic cache** (Titan V2 + DynamoDB, threshold+TTL) (M12) | Cache-hit-rate SLO + tuning under real traffic |
| Measure p95, not just averages | ✅ | The `relay-ops` p95-latency widget + alarm; the capstone recap prints p95 (M14/M15) | Load testing under realistic concurrency (theory) |
| Scale the integration layer | ⚠️ | Lambda reserved/provisioned concurrency + API GW throttling as the FM auto-scaling lever (M11) | Tuned concurrency floors from real peak data |

## 5. Cost Optimization

| Best practice (Lens) | Status | Relay control (module) | Gap → "In production" |
|---|---|---|---|
| Meter cost per unit of work | ✅ | `TicketRecord.cost_cents` summed from token usage × the per-tier price map; the capstone recap prints total $/ticket (M12/M15) | — |
| Pick the cheap-not-free building blocks | ✅ | **S3 Vectors** over OpenSearch Serverless (~$0 idle vs ~$174/mo); on-demand DynamoDB; idle-free AgentCore Runtime (M4/M5/M7/M8) | — |
| Discount the non-interactive paths | ✅ | Flex / batch tier (-50%) for eval/backfill only, never interactive (M12) | Reserved/committed-use as volume grows |
| Watch for cost regressions | ✅ | The cost-anomaly alarm (learned band, not a static line) (M14) | A continuous FinOps review of $/ticket |
| **Leave nothing idle-billed** | ✅ | `teardown.py` is exhaustive — incl. the **AgentCore long-term Memory** purge (the only monthly idle item, ~$0.75/1K records); verified ~$0/month (M15, B5) | — |

## 6. Sustainability

| Best practice (Lens) | Status | Relay control (module) | Gap → "In production" |
|---|---|---|---|
| Use the smallest sufficient model | ✅ | Nova Micro/Lite tiers (not a frontier model on every ticket); the router avoids over-provisioning compute per token (M3) | A measured tokens-per-ticket budget |
| Avoid wasted inference | ✅ | Semantic/prompt caching skips repeated generation; the agent's turn cap avoids runaway loops (M12/M7) | — |
| Decommission idle resources | ✅ | The verified teardown returns the account to ~$0/month — no idle accelerators, no provisioned vector store (M15) | — |

---

## Generative-AI-specific considerations (the Lens's cross-cutting AI concerns)

| Concern | Status | Relay control (module) | Gap |
|---|---|---|---|
| **Model selection** fits the task/constraints | ✅ | Inference profiles only (never bare regional IDs); tiers chosen by complexity/cost (M1/M3) | Periodic re-benchmark as new Nova/Claude versions ship |
| **RAG grounding** is real, not assumed | ✅ | Bedrock Knowledge Base `relay-kb` on a dedicated S3 Vectors index; contextual-grounding check recomputes `Answer.grounded` and escalates ungrounded answers (M5/M9) | Hybrid search (unsupported on S3 Vectors — reranker is the precision lever) |
| **Responsible AI** (safety, fairness, transparency) | ⚠️ | Guardrail + PII redaction + a `docs/model-card.md` + a fairness rubric in evals (M9/M10/M13) | A formal third-party responsible-AI / bias audit |
| **Human-in-the-loop** on sensitive actions | ✅ | Refunds park in `awaiting_approval`; nothing executes without `POST /tickets/{id}/approve` (M8/M11) | — |
| **Evaluation** before and after release | ✅ | 20-ticket golden set + LLM-as-a-judge (judge ≠ candidate) + the regression gate (M13) | A larger, continuously-grown golden set from low feedback ratings |

---

## Verdict

Relay v1.0 **passes** every pillar with a built control. The remaining ⚠️ rows are the
deliberate "In production" frontier (article T7): a paging on-call, multi-Region active/active,
tuned auto-scaling from real peak data, KMS CMKs, a third-party security/responsible-AI audit,
and continuous FinOps. None is a blocker for a single-Region production launch with the controls
above; each is the **next** thing a reviewer would ask for — which is exactly what a Lens review
is for: surfacing the gaps *before* the go-live, not stamping the project done.

**Try it yourself:** add one row to a pillar above for the gap you would close FIRST in production,
and say why (the cost of being wrong, or the blast radius, should drive the order).
