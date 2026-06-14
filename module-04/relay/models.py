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

No field is ever renamed, retyped, or removed. There is no `refund` intent and
no `severity` rename — the literals below are LAW.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class Ticket(BaseModel):
    """A raw CloudCart support ticket — exactly 4 fields in Module 2.

    (Module 6 and Module 10 each add one more field, by addition only.)
    """

    ticket_id: str
    channel: Literal["email", "chat"]
    customer_message: str
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
