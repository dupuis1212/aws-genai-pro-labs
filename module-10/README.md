# Module 10 — Security, Privacy, and Governance: IAM, PII, and Responsible AI

**What:** Relay resists attacks now (Module 9), but it would still fail an enterprise
security review. A SOC 2 auditor asks: *"Your support bot sends customer data to a
foundation model. Show me what data it saw last Tuesday, who could access it, how long you
keep it, and your documented limitations."* Today the tickets — names, emails, phone
numbers — go **raw** to Bedrock; the lab role can do almost anything; there is no audit
trail and no document.

Module 10 closes Domain 3: Relay **masks PII before any model call**, each component has a
**least-privilege** IAM role, its decisions are **auditable**, and it ships a **model
card**.

```bash
uv run python -m relay.intake data/tickets/pii_ticket.json
# prints the normalized Ticket with name/email/phone masked and "pii_redacted": true:
#   "customer_message": "Hi support team, My name is [NAME] and my order #1042 ...
#                        email me at [EMAIL]? You can also reach me at [PHONE]."
#   "pii_redacted": true
# [pii] masked 2 NAME, 1 EMAIL, 1 PHONE

uv run python audit_report.py --last 1h
# DECISION LOG (what Relay decided)  +  CLOUDTRAIL (who called which AWS API)
```

## What this module builds (on top of Module 9)

- **`relay/pii.py` (NEW)** — mask **PII** with **Amazon Comprehend** `DetectPiiEntities`,
  **by offset** (`[NAME]`/`[EMAIL]`/`[PHONE]`). A confidence floor + entity allowlist live
  in `relay/config.py`; order numbers and dates are deliberately **kept** (business
  signal). Comprehend is a managed NLP service — **not** a Bedrock FM, **not** a guardrail.
  A failed scan **raises** (never "no PII found"). Holds **no model ID**, makes **no**
  Bedrock call.
- **`relay/intake.py` (MODIFIED, by addition)** — redact on the normalized text **before**
  the Comprehend entity pass **and before** the Nova Lite vision call. Redact at the edge,
  everything downstream inherits the protection.
- **`relay/models.py` (MODIFIED, by addition)** — `Ticket.pii_redacted: bool = False` —
  the **only** schema change in the module (default load-bearing for backward compat).
- **`relay/agent.py` (MODIFIED, by addition)** — a structured, **redacted** decision log
  (`decision_log.jsonl`): the application "why" trail, the complement to CloudTrail.
- **`iam/policies/*.json` (NEW)** — one **least-privilege** policy per component
  (`intake`, `agent`, `kb-reader`, `api`), explicit action/resource ARNs, **zero
  wildcards**.
- **`docs/model-card.md`, `audit_report.py`, `data/tickets/pii_ticket.json` (NEW)** — the
  governance artifact, the audit-trail report, and the demo fixture.
- **`setup.py` / `teardown.py` (MODIFIED)** — setup creates the four IAM roles from the
  JSON files; teardown deletes them + the decision log.

## Frozen contracts (one field added — bible §3.1)

Module 10 adds **exactly one field**, `Ticket.pii_redacted`, **by addition**. Every other
schema is byte-identical to Module 9:

```python
class Ticket(BaseModel):                      # M2 (4) + M6 (attachments) + M10 (pii_redacted)
    ticket_id: str
    channel: Literal["email", "chat"]
    customer_message: str
    attachments: list[Attachment] = []        # ADDED M6
    pii_redacted: bool = False                # ADDED M10 — the only change this module
    created_at: str
```

Resource names in the IAM ARNs reuse the **canonical** frozen names (`relay-orders`,
`relay-tickets`, `relay-guardrail`, `relay-kb`, `relay-<account_id>`).

## The right PII tool in the right place (the exam tests this)

| Tool | Where it acts | When |
|---|---|---|
| **Amazon Comprehend** `DetectPiiEntities` | text **in flight** | what `relay.intake` runs, **before** the FM call — entity + offset |
| **Amazon Macie** | S3 **at rest** | scheduled discovery jobs over a bucket (theory — no job provisioned) |
| **Bedrock Guardrails** PII filter | at the **model-call** moment | mask/block (the Module 9 layer) |

Macie scans S3 at rest — it does **not** redact text in flight (that is Comprehend); the
exam loves to swap them. CloudTrail records the **API call** (who/what/when), **not** the
prompt **content** — content lives in the (redacted) decision log.

## Run it

```bash
export AWS_PROFILE=aws-genai-pro          # us-east-1 everywhere; no keys in code/.env
uv sync                                   # no new dependency — Comprehend/IAM are in boto3
uv run python setup.py                    # creates the guardrail (M9) + 4 IAM roles (M10)
uv run python -m relay.pii "I'm Dana Lee, dana.lee@example.com, 555-0100."
uv run python -m relay.intake data/tickets/pii_ticket.json   # masked ticket, flag true
uv run python audit_report.py --last 1h   # decision log + CloudTrail
uv run pytest                             # offline cumulative suite (Modules 2–10)
RELAY_LIVE_TESTS=1 uv run pytest -m live  # opt-in, capped (~$0.03, see lab.md)
uv run python teardown.py                 # deletes the 4 IAM roles + decision log + guardrail
```

## Boundaries (what this module does NOT do)

- No **VPC endpoints**/NAT, no **Macie** job, no **Lake Formation**, no **Kendra** — taught
  as exam-corner theory (billed by the hour or out of scope), **not** provisioned.
- No **fairness** judge / golden set — Module 13 (defs + A/B only here).
- No **dashboards / alarms** — Module 14.
- No **public-API auth** (federation, RBAC, Cognito) — Module 11. Here the **internal**
  IAM of the components, not consumer authentication.
- No **guardrail** change — Module 9 owns it (the PII filter is only the comparison row).

See `lab.md` for the full step-by-step, the measured cost, and "Try it yourself".
