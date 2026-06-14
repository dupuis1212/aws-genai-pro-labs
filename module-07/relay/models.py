"""relay/models.py — the frozen Pydantic v2 schemas for Relay.

Module 2 introduces — and FREEZES — the first two schemas of the cumulative
Relay data model: Ticket and Triage. They are reproduced field-for-field from
the Relay spec (06-FIL-ROUGE-SPEC.md §2) and the PRODUCTION-BIBLE §3.1, and must
stay identical everywhere they appear (article, code, quiz, downstream modules).

These schemas grow ONLY BY ADDITION in later modules:
  - Ticket gains two more fields later (an attachment list in Module 6 and a
    redaction flag in Module 10). Neither exists yet — adding them early would
    break the "exactly 4 fields" contract Module 2 freezes.
  - Triage is COMPLETE as defined here; it is never extended.
  - Module 5 ADDS two new schemas, Citation and Answer (below) — the output
    contract of Relay's managed Knowledge Base. They are frozen with NO score /
    confidence field, ever.

No field is ever renamed, retyped, or removed. There is no `refund` intent and
no `severity` rename — the literals below are LAW.

Module 6 ADDS the `Attachment` schema and extends `Ticket` with one field —
`attachments: list[Attachment] = []` — BY ADDITION ONLY. The default empty list is
load-bearing: every M2–M5 ticket fixture (no `attachments` key) still validates.
There is still NO `pii_redacted` field — that is a Module 10 addition; adding it
here would break the module boundary.

Module 7 ADDS the last two schemas of the agent's data model — `AgentAction` (the
journal entry for one tool call) and `TicketRecord` (the persisted record of a
handled ticket, with its `actions[]` log) — BY ADDITION ONLY. They are frozen
field-for-field (06 §2 / bible §3.1). Two boundary facts the bible pins:
  - `AgentAction.approved` is EFFECTIVE only at Module 8 (human-in-the-loop). At
    Module 7 it is ALWAYS `None` (proposed, never approved/rejected) — no approval
    flow exists yet.
  - `TicketRecord.status` carries its FULL 7-value enum from the moment it is frozen
    here, even though Module 7 only ever writes four of them
    (`received|triaged|answered|failed`). Module 8 turns on `awaiting_approval`;
    Module 11 reaches `escalated|closed`. `cost_cents` is present as a `0.0`
    placeholder now and is really populated at Module 12. `feedback_rating` is NOT
    part of the M7 definition — it is added by addition at Module 13 only.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class Attachment(BaseModel):
    """One file attached to a ticket — exactly 3 fields. Frozen at Module 6.

    Reproduced field-for-field from 06 §2 / bible §3.1. Relay's intake pipeline
    (relay.intake) validates an attachment's type, uploads it to the data bucket's
    attachments/ prefix, and records:

      - filename:   the original file name (e.g. "payment_error.png");
      - media_type: the validated MIME type (e.g. "image/png"). An image whose
                    type is not in the admitted set is REJECTED at the gate,
                    before any upload or FM call;
      - s3_uri:     where the upload landed —
                    s3://relay-<account_id>/attachments/<...>.

    No bytes, no size, no checksum here — those are intake internals, not part of
    the Ticket contract. This schema is COMPLETE and frozen; it is never extended.
    """

    filename: str
    media_type: str
    s3_uri: str


class Ticket(BaseModel):
    """A raw CloudCart support ticket.

    Module 2 froze the first four fields. Module 6 adds — BY ADDITION ONLY — the
    `attachments` list (default empty) so a ticket can carry the screenshots/files
    a customer sent. Module 10 will add one more field (a redaction flag) the same
    way; it does NOT exist yet. No field here is ever renamed, retyped, or removed.
    """

    ticket_id: str
    channel: Literal["email", "chat"]
    customer_message: str
    attachments: list[Attachment] = []        # ADDED M6 (by addition; default [])
    created_at: str


class Triage(BaseModel):
    """Relay's structured classification of a ticket — complete and frozen at M2.

    The three enums are exact literals from the spec:
      - intent:    5 values
      - priority:  4 values
      - sentiment: 3 values

    This is the object `relay.triage.triage()` must produce, fully validated,
    for every ticket — the whole point of the module.
    """

    intent: Literal["billing", "technical", "account", "shipping", "other"]
    priority: Literal["low", "normal", "high", "urgent"]
    sentiment: Literal["negative", "neutral", "positive"]


# =============================================================================
# Module 5 ADDITIONS — the Knowledge Base answer contract (Citation, Answer).
# =============================================================================
# Frozen at Module 5 (06 §2 / bible §3.1), reproduced field-for-field. These are
# the output of relay.kb.answer(): a grounded answer with the sources it cited.
#
# HARD invariant: NO score / confidence field is ever added to either schema.
# A "score" is a retrieval/rerank internal; what leaves Relay is the answer text,
# the human-readable citations, and one boolean for whether it is grounded.


class Citation(BaseModel):
    """One source Relay cited when it answered — exactly 2 fields. Frozen at M5.

    `source_uri` is the s3:// URI of the doc the cited chunk came from;
    `snippet` is the retrieved passage text, for a human to verify the claim.
    No score, no confidence, no rank — those are retrieval internals, not part of
    the answer contract.
    """

    source_uri: str
    snippet: str


class Answer(BaseModel):
    """Relay's grounded answer to a question — text + citations + grounded. Frozen M5.

    `grounded` is a BOOLEAN. At Module 5 it is the heuristic `bool(citations)`:
    an answer that cited at least one retrieved source is treated as grounded.
    Module 9 keeps this exact field but recomputes it from a real contextual
    grounding check (a guardrail), escalating ungrounded answers — SAME field name
    and type, different computation. No field is added or renamed between M5 and M9.
    """

    text: str
    citations: list[Citation]
    grounded: bool


# =============================================================================
# Module 7 ADDITIONS — the agent's action journal + the persisted TicketRecord.
# =============================================================================
# Frozen at Module 7 (06 §2 / bible §3.1), reproduced field-for-field. These are
# the contract for an AGENT that acts: every tool the Strands agent calls is logged
# as an AgentAction, and the whole handled ticket is persisted as a TicketRecord in
# the `relay-tickets` DynamoDB table. Both grow only BY ADDITION downstream — no
# field is ever renamed, retyped, or removed.


class AgentAction(BaseModel):
    """One action the agent took — exactly 4 fields. Frozen at Module 7.

    The agent's ReAct loop decides to call a tool (search_kb / lookup_order /
    create_ticket); each call is journaled here so the TicketRecord carries a full,
    auditable trail of what the agent DID, not just what it said.

      - tool:       the canonical tool name (06 §5.4) — "search_kb", "lookup_order",
                    or "create_ticket". No synonyms.
      - tool_input: the arguments the model passed the tool (a plain dict).
      - result:     the tool's textual result (or a clean error string the tool
                    returned to the model — a failing tool reports, it does not crash
                    the loop).
      - approved:   `None` at Module 7 and ALWAYS `None` here — the field is frozen
                    now but EFFECTIVE only at Module 8 (human-in-the-loop):
                    None = proposed, True = approved, False = rejected. Module 7
                    exercises no approval flow; do not assign it anything but None.
    """

    tool: str
    tool_input: dict
    result: str
    approved: bool | None = None


class TicketRecord(BaseModel):
    """The persisted record of a ticket the agent handled. Frozen at Module 7.

    This is what lands in the `relay-tickets` DynamoDB table: the ticket's id, its
    status, the structured triage (M2) and grounded answer (M5) when present, the
    `actions[]` journal of every tool call, and bookkeeping fields. Reproduced
    field-for-field from 06 §2 / bible §3.1.

    The `status` enum carries ALL SEVEN values from the moment it is frozen here,
    even though Module 7 only ever WRITES four of them:
      - received   : the record was created for an incoming ticket;
      - triaged    : triage classified it (intent/priority/sentiment);
      - answered   : the agent produced a final answer / took its actions;
      - failed     : the run hit a stop condition or an error before answering.
    The other three are reached downstream and are present now ON PURPOSE so the
    contract never changes shape later:
      - awaiting_approval : Module 8 (the HITL refund gate);
      - escalated, closed : Module 11 (end-to-end status lifecycle).

    Other fields:
      - triage / answer : optional (a ticket may be recorded before it is triaged or
                          answered) — the frozen Triage / Answer schemas, or None.
      - actions         : the AgentAction journal (≥1 once the agent has acted).
      - escalated       : whether the ticket was handed to a human (False at M7 —
                          escalation/HITL is Module 8/11).
      - cost_cents      : a `float`. A `0.0` PLACEHOLDER at Module 7; really populated
                          (summed token usage × the M3 per-tier price map) at Module 12.
                          It is the SAME field, never "new" and never re-typed.
      - updated_at      : ISO-8601 timestamp of the last write.

    `feedback_rating` is deliberately ABSENT here — it is added by addition at Module
    13 only. Adding it now would break the module boundary the bible pins.
    """

    ticket_id: str
    status: Literal[
        "received", "triaged", "awaiting_approval",
        "answered", "escalated", "closed", "failed",
    ]
    triage: Triage | None
    answer: Answer | None
    actions: list[AgentAction]
    escalated: bool
    cost_cents: float            # placeholder 0.0 at M7 -> really populated at M12
    updated_at: str
