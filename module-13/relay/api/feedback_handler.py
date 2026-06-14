"""relay/api/feedback_handler.py — POST /tickets/{ticket_id}/feedback (Module 13).

The user-feedback loop (skill 5.1.3). Module 11 gave Relay a door; Module 13 lets the
customer rate the answer that came back through it — the thumb on the response. That rating
lands on the frozen TicketRecord as `feedback_rating` (the field Module 13 added by
addition), so a stream of low-rated answers becomes the source of new failing cases for
`evals/golden_set.json`: rating -> triage the failures -> grow the golden set (the article's
feedback->golden-set loop).

Contract (06 §2 / bible §3.3 — reproduced field-for-field):
    POST /tickets/{ticket_id}/feedback   body {"feedback_rating": int}
  - 200 + the updated TicketRecord on success.
  - 400 on a missing / non-integer / out-of-range `feedback_rating`.
  - 404 when the ticket does not exist.

This handler WRAPS the unchanged store, exactly like the other three handlers: it READS the
persisted TicketRecord through the single relay-tickets read path (mcp_server.store.get_ticket),
sets the one new field on the FROZEN schema, and re-persists the record's JSON document under
the SAME item shape mcp_server.store.create_ticket writes — preserving every other field
(status, triage, answer, actions, escalated, cost_cents, updated_at). It never re-implements
the agent and never re-runs a model: leaving feedback is a single DynamoDB round-trip, NO
foundation-model call, NO model ID. The table name comes from relay.config.

Why a 1-5 scale: the rating is meant to be human-meaningful and to feed the judge's
calibration (the lab's "rate, then re-baseline" loop). The handler accepts an integer in
[FEEDBACK_MIN, FEEDBACK_MAX]; a thumbs up/down UI maps to 5/1 on the client. The bounds live
in this module (a request-shape concern, not a Relay-wide config constant) so the validation
message is specific and the smoke test asserts them.
"""

from __future__ import annotations

import datetime as dt
import json

import boto3

from relay import config
from relay.api import common

# The accepted rating range. A small, human-meaningful 1-5 scale: 1 = bad answer, 5 = great
# answer; a thumbs-down/up UI maps to 1/5. Inclusive bounds. Kept in this handler because it
# is a REQUEST-VALIDATION concern (the shape of the public body), not a model/resource
# constant — the other handlers keep their body rules local too (approve_handler's boolean).
FEEDBACK_MIN = 1
FEEDBACK_MAX = 5


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_feedback(body: dict) -> int:
    """Extract the required integer `feedback_rating` in [FEEDBACK_MIN, FEEDBACK_MAX].

    Strict, like approve_handler's boolean: `feedback_rating` MUST be a JSON integer (not a
    string, not a float, not a bool). A bool is a Python int subclass, so it is rejected
    explicitly — `true` is not a 1-5 rating. Out-of-range is a BadRequest, never silently
    clamped (a clamped 0 or 7 would corrupt the aggregate the eval loop reads).
    """
    if "feedback_rating" not in body:
        raise common.BadRequest(
            "Body must include an integer 'feedback_rating' "
            f"({FEEDBACK_MIN}-{FEEDBACK_MAX})."
        )
    rating = body["feedback_rating"]
    # bool is a subclass of int in Python — reject it before the isinstance(int) check so
    # `true`/`false` cannot pass as 1/0.
    if isinstance(rating, bool) or not isinstance(rating, int):
        raise common.BadRequest(
            "'feedback_rating' must be a JSON integer "
            f"({FEEDBACK_MIN}-{FEEDBACK_MAX})."
        )
    if not (FEEDBACK_MIN <= rating <= FEEDBACK_MAX):
        raise common.BadRequest(
            f"'feedback_rating' must be between {FEEDBACK_MIN} and {FEEDBACK_MAX} "
            f"(got {rating})."
        )
    return rating


def _tickets_table(resource=None):
    """The relay-tickets table resource (a moto table in tests, the real one in Lambda)."""
    ddb = resource or boto3.resource("dynamodb", region_name=config.REGION)
    return ddb.Table(config.RELAY_TICKETS_TABLE)


def apply_feedback(ticket_id: str, rating: int, *, load=None, save=None, resource=None):
    """Set feedback_rating on the persisted TicketRecord and re-persist it. Returns it.

    `load` reads the record (defaults to the single relay-tickets reader,
    mcp_server.store.get_ticket); `save` writes the updated item (defaults to a direct
    put_item under the SAME item shape the store writes — wrapping the store, not rewriting
    it). Tests inject both to stay offline. Returns None when the ticket does not exist (the
    handler maps that to a 404).
    """
    if load is None:
        from mcp_server import store

        def load(tid):  # the single relay-tickets reader
            return store.get_ticket(tid, resource=resource)

    record = load(ticket_id)
    if record is None:
        return None

    # Set the ONE new field on the frozen schema. Everything else (status, triage, answer,
    # actions, escalated, cost_cents) is preserved exactly as it was persisted — feedback
    # never re-runs the agent or recomputes a cost.
    record.feedback_rating = rating
    record.updated_at = _now_iso()

    if save is None:
        def save(rec):
            # Re-persist under the SAME item shape mcp_server.store.create_ticket writes, so
            # the table has one row shape and the record round-trips through the frozen
            # schema. We wrap the store's write contract; we do not modify the store.
            table = _tickets_table(resource)
            table.put_item(Item={
                config.TICKETS_KEY: rec.ticket_id,
                "status": rec.status,
                "escalated": rec.escalated,
                "updated_at": rec.updated_at,
                "record": rec.model_dump_json(),
            })
            return rec

    return save(record)


def handle(event: dict, *, load=None, save=None, resource=None) -> dict:
    """POST /tickets/{ticket_id}/feedback. Validate -> set feedback_rating -> 200 record.

    Args mirror the testable seams: `load` (the relay-tickets reader), `save` (the writer),
    `resource` (a moto DynamoDB resource in tests). In the deployed Lambda they resolve from
    the boto3 session. Returns the API Gateway proxy response dict.
    """
    try:
        ticket_id = common.path_param(event, "ticket_id")
        body = common.parse_json_body(event)
        rating = parse_feedback(body)
    except common.BadRequest as err:
        return common.error(400, str(err))

    try:
        record = apply_feedback(ticket_id, rating, load=load, save=save, resource=resource)
    except Exception as err:  # noqa: BLE001 — log the real cause, return a clean 500.
        print(f"[feedback_handler] feedback failed for {ticket_id!r}: "
              f"{type(err).__name__}: {err}")
        return common.error(500, "Could not record the feedback.")

    if record is None:
        return common.error(
            404, f"No ticket {ticket_id!r} found. Check the id returned by POST /tickets."
        )

    return common.response(200, record.model_dump(mode="json"))


def lambda_handler(event, context=None):  # noqa: ANN001 - Lambda signature
    """The AWS Lambda handler API Gateway invokes for POST /tickets/{id}/feedback."""
    return handle(event)
