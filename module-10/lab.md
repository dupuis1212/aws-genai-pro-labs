# Module 10 lab — security, privacy, and governance: IAM, PII, and responsible AI

> **This lab cost me $0.02 on June 2026 prices** (the syllabus budget for Module 10 is
> < $1). The Module 10 increment is **almost free**: **Amazon Comprehend**
> `DetectPiiEntities` bills per unit of text (pennies on the fixtures), **CloudTrail**
> management events are **free**, and **IAM** roles/policies are **free** — there is **no
> idle-billed resource** in this module. Every figure below is read from the API, never
> guessed (re-verify on the
> [Comprehend pricing page](https://aws.amazon.com/comprehend/pricing/), **as of June
> 2026**):
>
> | Item | Usage | Cost |
> |---|---|---|
> | `relay.intake` PII redaction + entity pass (Comprehend `DetectPiiEntities`/`DetectEntities`) | a few units of text | ~$0.002 |
> | four least-privilege IAM roles + inline policies (`setup.py`) | IAM is free | $0.000 |
> | `audit_report.py` CloudTrail `LookupEvents` (management events) | free | $0.000 |
> | M10 live smoke (2 Comprehend calls: 1 masked, 1 clean) | 2 units of text | <$0.001 |
> | inherited live smoke (fast/smart stream, 2× Titan embed, KB answer, vision, 2× guardrail) | small, maxTokens≤64 | ~$0.0015 |
> | two doc-only agent runs (real `search_kb` + smart-tier ReAct, decision-log evidence) | 12 turns ~2.5k tok | ~$0.011 |
> | **Total (measured)** | | **≈ $0.02** |
>
> Comprehend PII detection is ~$0.0001 per unit (100 characters), **as of June 2026** —
> re-verify on the pricing page and always cite the date. The two agent runs are the
> optional end-to-end evidence path; the core lab (intake redaction + IAM + audit +
> live smoke) lands at ~$0.004.
>
> **No idle-billed resource is created.** IAM is free; CloudTrail management events are
> free; Comprehend bills only on use. `teardown.py` deletes the four IAM roles and the
> decision log anyway (course rule B5: leave nothing behind that you created). The
> inherited guardrail / tables / KB / S3 Vectors are ~$0 idle; AgentCore long-term Memory
> (the one idle-billed item) is purged by `teardown.py` as before.
>
> **Teardown reminder:** run `uv run python teardown.py` when you're done — it **deletes
> the four IAM component roles + the decision log** first, then the guardrail, purges
> AgentCore Memory, and removes the MCP Lambda, **keeping** the on-demand tables and the
> Knowledge Base (~$0 idle). The M1 $5 budget stays.

**Goal:** mask **PII** in Relay's intake **before any foundation-model call**, give each
component a **least-privilege** IAM role, make the agent's decisions auditable, and ship
Relay's **model card**.

Region for the whole course: **us-east-1**. Profile: `AWS_PROFILE=aws-genai-pro`. No AWS
key in code or `.env`.

---

## Step 1 — Carry the cumulative state forward

Module 10 starts from Module 9's `relay/` package byte-for-byte (`models.py`, `config.py`,
`llm.py`, `triage.py`, `kb.py`, `intake.py`, `tools.py`, `agent.py`, `specialists.py`,
`approve.py`, `run.py`, `safety.py`) plus the inherited `ingest/` pipeline, the
`mcp_server/` package, the DynamoDB tables, the AgentCore deployment, the **Module 5
Knowledge Base `relay-kb`**, and the **Module 9 guardrail `relay-guardrail`**.

Module 10 adds, **by addition only**:

- `relay/pii.py` (NEW) — PII detection + masking with Comprehend, by offset.
- `relay/intake.py` (MODIFIED) — redact before the entity pass and before the vision call.
- `relay/models.py` (MODIFIED) — `Ticket.pii_redacted: bool = False` (the **only** change).
- `relay/agent.py` (MODIFIED) — the structured, redacted decision log.
- `iam/policies/*.json` (NEW) — one least-privilege policy per component.
- `docs/model-card.md`, `audit_report.py`, `data/tickets/pii_ticket.json` (NEW).
- `setup.py` / `teardown.py` (MODIFIED) — create/delete the IAM component roles.

```bash
uv sync                 # installs the pinned deps (no new dependency this module)
uv run pytest -q        # the cumulative offline suite (Modules 2–10) goes green
```

---

## Step 2 — `relay/pii.py`: detect and mask PII with Comprehend, by offset

`relay/pii.py` runs Amazon Comprehend `DetectPiiEntities` over a piece of text and masks
each detected entity **by character offset** — never with a home-grown regex. Comprehend
returns `{Type, BeginOffset, EndOffset, Score}` (offsets, not the substring), so we slice
the original text at those offsets and replace each span with its typed placeholder
(`[NAME]`, `[EMAIL]`, `[PHONE]`, …). A confidence floor (`config.PII_MIN_CONFIDENCE`)
drops borderline guesses; the entity allowlist (`config.PII_ENTITY_TYPES`) decides what we
mask — and deliberately **keeps order numbers** (a business key, not PII) and
**`DATE_TIME`** (operational signal the agent needs).

A failed Comprehend call **raises** `PiiError` — it is **never** treated as "no PII found"
(that would leak raw data). No silent `try/except`.

```bash
uv run python -m relay.pii "Hi, I'm Dana Lee (dana.lee@example.com, 555-0100), order #1042."
# -> Hi, I'm [NAME] ([EMAIL], [PHONE]), order #1042.
# [pii] masked 1 EMAIL, 1 NAME, 1 PHONE
```

The `[pii]` summary carries **counts only** — no raw value — so it is safe to log.

---

## Step 3 — Redact at the edge in `relay/intake.py` (before any FM call)

The principle: **redact at the edge, and everything downstream inherits the protection.**
`relay/intake.py` now runs PII redaction on the normalized text **before** the Comprehend
entity pass and **before** the Nova Lite vision read — so the foundation model, the
`[Entities]` / `[Attachment summary]` enrichment, the agent's decision log, the persisted
`TicketRecord`, and AgentCore Memory **all** see `[NAME]`/`[EMAIL]`/`[PHONE]`, never the
customer's real values. The `Ticket.pii_redacted` flag records that it happened.

```bash
uv run python -m relay.intake data/tickets/pii_ticket.json
# prints the normalized Ticket with name/email/phone masked and "pii_redacted": true
```

`pii_redacted` is the **only** field Module 10 adds to `Ticket` — by addition, default
`False`, so every earlier fixture still validates (06 §2 / bible §3.1).

---

## Step 4 — Least-privilege IAM: one role per component (`iam/policies/`)

A single broad lab role would fail an enterprise security review. Module 10 gives **each
component its own role** with **explicit actions and resource ARNs** and **zero
wildcards** — the policies live in `iam/policies/*.json`:

| Role | Can do (only) |
|---|---|
| `relay-intake-role` | Comprehend detect; `PutObject` under `attachments/`; the Nova Lite vision Converse call |
| `relay-agent-role` | reason on the smart/fast profiles; `ApplyGuardrail` on `relay-guardrail`; **read** `relay-orders`; **write** `relay-tickets`; `Retrieve` from `relay-kb` |
| `relay-kb-reader-role` | `Retrieve`/`RetrieveAndGenerate` on `relay-kb`; the reranker/embedder; **read** `docs/` |
| `relay-api-role` | **write/read** `relay-tickets`; its own logs (the deployment module adds the rest) |

**Why no `*`?** The intake role **cannot** read the order book; the api role **cannot**
call a model. A bug or a compromised component is bounded by IAM, not just by convention.
`setup.py` substitutes `${ACCOUNT_ID}` / `${REGION}` (the account id is resolved from STS,
never hard-coded) and attaches each JSON as the role's inline policy.

```bash
uv run python setup.py            # creates relay-guardrail (M9) + the four IAM roles (M10)
grep -rE '"Action":\s*"\*"|"Resource":\s*"\*"' iam/    # -> zero results (the gate)
```

---

## Step 5 — Audit trail: the decision log + CloudTrail

Two trails the exam keeps distinct:

- **CloudTrail** management events (free): **who** called **which** AWS API, and **when** —
  the call, **not** the prompt content.
- the **decision log** (`relay/agent.py` → `decision_log.jsonl`): what **Relay decided**
  and why — the tool calls, their **redacted** inputs, the result, the outcome. Inputs are
  **re-redacted** before writing, so **no raw email/phone** ever lands in the log.

`audit_report.py` crosses the two:

```bash
uv run python audit_report.py --last 1h          # decision log + CloudTrail, last hour
uv run python audit_report.py --last 1h --no-cloudtrail   # decision log only (offline)
```

---

## Step 6 — `docs/model-card.md`: document the system

A **model card** documents intended use, the models used, the data touched, and the
**known limitations** — it **documents**, it does **not** enforce (the guardrail and IAM
do that). Relay's card (`docs/model-card.md`) honestly lists the M9 attacks that **still
slip past** the guardrail, the single-Region/English-first scope, and that fairness is a
documented principle whose tooling arrives later. It lists **only ACTIVE inference-profile
IDs** — no bare or legacy model IDs.

---

## Try it yourself

1. **Consistent anonymization** (instead of masking): replace each detected NAME with a
   **stable pseudonym** (e.g. hash the value to `Customer-7F3A`) so the same customer reads
   the same way across tickets, without exposing the real name. Hook it in `relay/pii.py`
   alongside `mask_spans` (a different replacement, same offsets).
2. **S3 Lifecycle expiry** on attachments: add a Lifecycle rule that expires objects under
   `s3://relay-<account_id>/attachments/` after N days — the retention control the auditor
   asks for. (One `put_bucket_lifecycle_configuration` call; ~$0.)

---

## What this lab does NOT do (taught as theory in the article)

No **VPC endpoints**/NAT, no **Macie** job, no **Lake Formation**, no **Kendra** — all
billed-by-the-hour or out of scope, so they are exam-corner theory, **not** provisioned.
No fairness judge (Module 13), no dashboards (Module 14), no public-API auth/Cognito
(Module 11), no guardrail change (Module 9 owns it).

---

## Teardown (tested, idempotent)

```bash
uv run python teardown.py
# deletes: the four IAM component roles + their inline policies, decision_log.jsonl,
#          relay-guardrail (all versions), purges AgentCore Memory, removes the MCP Lambda
# keeps:   relay-orders + relay-tickets (on-demand, ~$0 idle), relay-kb (~$0 idle)
# the M1 $5 budget stays (Module 1 owns it).
```

Verified: after teardown, **no idle-billed resource remains** (IAM is free and removed
anyway; CloudTrail/Comprehend bill only on use).
