# Relay — Model Card

> A **model card** documents a system: its intended use, the models it relies on, the
> data it touches, and its known limitations. It **documents — it does not enforce**;
> enforcement is the job of the guardrail (Module 9) and the least-privilege IAM roles
> (Module 10). This card is the governance artifact an auditor asks for when they want
> "the documented limitations of the system." Reproduced from the Module 10 template.
>
> *Last reviewed: 2026-06 — re-review on any model, guardrail, or scope change.*

## 1. Overview

| Field | Value |
|---|---|
| **System name** | Relay |
| **Owner** | CloudCart — Customer Support Engineering |
| **Purpose** | An agentic AI support assistant that triages CloudCart support tickets, answers from the help-center Knowledge Base with citations, looks up real order status, and records outcomes — handing financial actions to a human. |
| **Status** | Internal lab build (AWS GenAI Pro Mastery, Modules 1–10). Not yet exposed to the public internet (that is Module 11). |
| **Region** | us-east-1 (single-Region by design). |

## 2. Intended use

- **In scope.** Classifying a ticket (intent / priority / sentiment), answering
  how-to and policy questions from CloudCart's documentation **with source
  citations**, retrieving the live status of a specific order, and recording a
  `TicketRecord` of what was done.
- **Human-in-the-loop.** A **refund** is **proposed, never executed** by the agent.
  The ticket is parked in `awaiting_approval` and a human approves or rejects it
  (`relay.approve`). Relay moves no money on its own.
- **Out of scope / not intended.** Legal, medical, or financial advice; endorsing or
  disparaging competitors (denied topics in the guardrail); any action outside the
  three tools (`search_kb`, `lookup_order`, `create_ticket`) and the proposed `refund`;
  decisions about a customer's account standing. Relay is an assistant, not an
  authority.

## 3. Models used

All generation goes through the single `converse()` call site; model IDs live only in
`relay/config.py`. Relay uses **Amazon Bedrock inference profiles** (cross-Region
inference) — never a bare regional model ID, never a legacy model.

| Role | Inference profile (as of June 2026) | Where |
|---|---|---|
| Triage / fast classification | `us.amazon.nova-micro-v1:0` | `relay.triage`, the router floor |
| Complex answers / agent reasoning | `us.amazon.nova-2-lite-v1:0` | `relay.kb.answer`, `relay.agent` |
| Multimodal (screenshot) read | `us.amazon.nova-lite-v1:0` | `relay.intake` vision step |
| Embeddings (retrieval) | `amazon.titan-embed-text-v2:0` (1024-dim) | `ingest/embed.py`, the KB index |
| Reranker (retrieval precision) | `cohere.rerank-v3-5:0` | `relay.kb` |

Amazon **Comprehend** (`DetectPiiEntities`, `DetectEntities`) is used for PII redaction
and entity extraction at intake — it is a managed NLP service, **not** a foundation
model.

## 4. Data

| Data | Where it lives | Protection |
|---|---|---|
| Help-center docs (CloudCart corpus) | `s3://relay-<account_id>/docs/` → KB index `relay-kb-docs` | Encrypted at rest (S3 default / **KMS**); read by the KB only. |
| Customer ticket text | In memory at intake → masked → `relay-tickets` | **PII masked at the intake edge** (Comprehend, before any model call). |
| Screenshots | `s3://relay-<account_id>/attachments/` | Encrypted at rest; **S3 Lifecycle** expiry is the documented retention control (see "Try it yourself"). |
| Order book | DynamoDB `relay-orders` (read-only to the agent) | Read-only IAM grant; fictional seed data in the lab. |
| Outcomes | DynamoDB `relay-tickets` | `TicketRecord` with a redacted message; write-scoped IAM grant. |
| Decision log | `decision_log.jsonl` | Application "why" trail; inputs **re-redacted** — no raw PII written. |
| Cross-session memory | AgentCore Memory (long-term) | Distils **non-PII** facts only; bounded retention. |

**PII handling.** Customer name / email / phone / address and similar entities are
detected with Amazon Comprehend and **masked by character offset** to typed
placeholders (`[NAME]`, `[EMAIL]`, `[PHONE]`) **before any foundation-model call,
log write, or memory write** — so the model provider, the invocation logs, the decision
log, the persisted record, and AgentCore Memory never see raw PII. **Amazon Bedrock does
not use your prompts/completions to train its models and does not share them with model
providers** (see *AWS — Bedrock data protection*); the real PII risk is in *your* own
stores, which is why redaction happens at the edge and retention is bounded.

## 5. Known limitations

Honest limitations the team has measured — an auditor wants these stated, not hidden:

- **The guardrail is probabilistic, not a guarantee.** The Module 9 adversarial suite
  shows that some prompt-injection variants **still slip past** `relay-guardrail`.
  Defense in depth (intake validation, PII redaction, IAM tool boundaries, the
  guardrail, the grounding check) is why no single miss is catastrophic — but Relay is
  **not** "safe", it is *defended*.
- **Grounding is checked, not perfect.** Answers below the grounding threshold (0.8)
  are escalated rather than shipped, but a confidently-wrong answer that *cites* a real
  doc can still pass the check. Citations let a human verify.
- **PII redaction can miss or over-mask.** Comprehend is a classifier with a confidence
  floor; a borderline value may be left in (miss) or a non-PII token may be masked
  (over-mask). Order numbers are deliberately **not** masked (business keys).
- **Single-Region, English-first.** us-east-1 only; Comprehend runs with
  `LanguageCode="en"`. Multi-jurisdiction redaction is out of scope here.
- **No fairness evaluation yet.** Fairness is **documented** as a principle and tested
  by A/B prompts, but the LLM-as-a-judge fairness harness is **Module 13** — this card
  does not claim a measured fairness result.

## 6. Governance & oversight

- **Least-privilege IAM**: one role per component (`relay-intake-role`,
  `relay-agent-role`, `relay-kb-reader-role`, `relay-api-role`) with explicit
  action/resource ARNs and **zero wildcards** (`iam/policies/*.json`).
- **Auditability**: **CloudTrail** management events record who called which AWS API;
  the **decision log** records what Relay decided and why. `audit_report.py` reads both.
- **Human oversight**: the HITL refund gate; escalation on ungrounded answers.
- **Review cadence**: this card and the IAM roles are reviewed on any model, guardrail,
  scope, or data-flow change (and, in production, on a fixed schedule via IAM Access
  Analyzer — Module 11 "in production").

## 7. Contact

CloudCart Customer Support Engineering — `support-eng@cloudcart.example` (fictional).
