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
