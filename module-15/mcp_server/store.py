"""mcp_server/store.py — the DynamoDB data access for the CloudCart MCP tools.

Module 7. This is the SOLE place the two business tools read/write their tables, so
the table names (frozen in relay.config, 06 §2) and the TicketRecord/AgentAction
schemas (frozen in relay.models) live in exactly one path. The MCP server (server.py)
wraps these functions as MCP @tools; the offline smoke test drives them directly on a
moto DynamoDB backend.

Two design rules the brief pins (skills 2.1.6 + 2.1.3):

  - VALIDATE PARAMETERS, RETURN CLEAN ERRORS. A tool given a bad/missing order id, or
    an unknown order, raises a structured `ToolInputError` / `OrderNotFound` whose
    MESSAGE is meant to go back to the model so it can recover (ask the user, try the
    KB, give up gracefully) — not a stack trace, and never a silent empty result. The
    server turns these into a clean tool-result string the model reads.
  - LEAST-PRIVILEGE I/O. This layer touches ONLY relay-orders (read) and relay-tickets
    (write). The Lambda's IAM role (setup.py) is bounded to exactly those two table
    ARNs (skill 2.1.3); a write anywhere else is denied by IAM, not just by convention.

No foundation-model call, no inference profile, no model-invoke path here — pure
business I/O over DynamoDB. The account/Region come from the boto3 default session; the
table NAMES come from relay.config (never typed here).
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass

import boto3
from botocore.exceptions import ClientError

from relay import config
from relay.models import AgentAction, Answer, TicketRecord, Triage

REGION = config.REGION


# =============================================================================
# Errors — structured, meant to be READ BY THE MODEL (skill 2.1.6).
# =============================================================================
class StoreError(RuntimeError):
    """Base for a CloudCart store failure. The message is model-facing."""


class ToolInputError(StoreError):
    """A tool was called with an invalid/missing parameter (validation failure).

    The message tells the model exactly what was wrong so it can fix the call or ask
    the customer — e.g. "order_id is required". Raised, never swallowed.
    """


class OrderNotFound(StoreError):
    """`lookup_order` was given an order id that is not in the order book.

    This is a normal business outcome, not a crash: the model should tell the
    customer the order was not found (and maybe ask them to re-check the number),
    so the message is phrased for exactly that.
    """


# =============================================================================
# Clients — built lazily from the boto3 default session (AWS_PROFILE / Region).
# =============================================================================
def _dynamodb():
    """A DynamoDB RESOURCE (the high-level Table API). Cheap to build; no state."""
    return boto3.resource("dynamodb", region_name=REGION)


def _orders_table(resource=None):
    return (resource or _dynamodb()).Table(config.RELAY_ORDERS_TABLE)


def _tickets_table(resource=None):
    return (resource or _dynamodb()).Table(config.RELAY_TICKETS_TABLE)


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _normalize_order_id(order_id) -> str:
    """Validate + canonicalize an order id. Accepts '1042', '#1042', 1042 -> '1042'.

    Customers (and the model reading their text) write order ids as "#1042", "1042",
    or even an int. We strip a leading '#' and surrounding whitespace and require a
    non-empty digit-or-alphanumeric token. A missing/blank id is a ToolInputError the
    model can recover from.
    """
    if order_id is None:
        raise ToolInputError("order_id is required (e.g. '1042'). None was given.")
    text = str(order_id).strip().lstrip("#").strip()
    if not text:
        raise ToolInputError("order_id is empty. Pass a CloudCart order id, e.g. '1042'.")
    return text


# =============================================================================
# lookup_order — READ relay-orders.
# =============================================================================
def lookup_order(order_id, *, resource=None) -> dict:
    """Look up one CloudCart order by id. Returns the order item as a plain dict.

    Validates the id, reads the `relay-orders` table, and either returns the order
    (a dict with status / dates / total / items) or raises OrderNotFound with a
    model-facing message. A DynamoDB error surfaces as StoreError (the real cause),
    never a silent empty result.
    """
    oid = _normalize_order_id(order_id)
    table = _orders_table(resource)
    try:
        response = table.get_item(Key={config.ORDERS_KEY: oid})
    except ClientError as err:
        raise StoreError(
            f"Failed to read {config.RELAY_ORDERS_TABLE} for order {oid!r}: "
            f"{err.response['Error']['Code']} — {err.response['Error']['Message']}"
        ) from err

    item = response.get("Item")
    if item is None:
        raise OrderNotFound(
            f"No order {oid!r} found in the CloudCart order book. Double-check the "
            "order number with the customer; it may be mistyped."
        )
    return _decimals_to_native(item)


def _decimals_to_native(obj):
    """DynamoDB returns numbers as Decimal; make them JSON-friendly (int/float).

    The order items carry a total and quantities; converting Decimal -> int/float
    here keeps the tool result clean JSON the model reads, with no Decimal noise.
    """
    from decimal import Decimal

    if isinstance(obj, list):
        return [_decimals_to_native(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _decimals_to_native(v) for k, v in obj.items()}
    if isinstance(obj, Decimal):
        return int(obj) if obj == obj.to_integral_value() else float(obj)
    return obj


# =============================================================================
# create_ticket — WRITE relay-tickets (persist a TicketRecord).
# =============================================================================
def create_ticket(
    ticket_id: str,
    *,
    status: str = "answered",
    summary: str | None = None,
    triage: dict | Triage | None = None,
    answer: dict | Answer | None = None,
    actions: list | None = None,
    escalated: bool = False,
    resource=None,
) -> dict:
    """Persist a TicketRecord in `relay-tickets`. Returns the stored record as a dict.

    The agent calls this to record what it did for a ticket. We BUILD a frozen
    TicketRecord (06 §2) — validating the shape — then write it to DynamoDB. The
    write is IDEMPOTENT on ticket_id (PutItem overwrites the same key), so an agent
    retry does not create a duplicate row (the brief's idempotence note, T7).

    Args:
        ticket_id: the ticket's id (required; the table's primary key).
        status: one of the frozen TicketRecord statuses (validated by the schema).
            Module 7 only writes received/triaged/answered/failed.
        summary: an optional short free-text note the agent wants stored (kept as a
            top-level attribute alongside the record — handy for a human scanning the
            table; it is NOT a TicketRecord schema field, so the schema stays frozen).
        triage / answer: the structured Triage / Answer for this ticket, if any.
        actions: the AgentAction journal (list of dicts or AgentAction).
        escalated: whether it was handed to a human (False at M7).

    Raises ToolInputError on a bad/missing id or an invalid status/shape (the message
    goes back to the model), and StoreError on a DynamoDB failure.
    """
    if not ticket_id or not str(ticket_id).strip():
        raise ToolInputError("ticket_id is required to create a ticket.")

    record = _build_record(
        ticket_id=str(ticket_id).strip(),
        status=status,
        triage=triage,
        answer=answer,
        actions=actions,
        escalated=escalated,
    )

    # Store the validated record as a JSON document under one attribute, plus the
    # key + a couple of scannable top-level attributes. JSON keeps the nested
    # triage/answer/actions intact without fighting DynamoDB's type system, and the
    # record round-trips through TicketRecord.model_validate_json on read.
    item: dict = {
        config.TICKETS_KEY: record.ticket_id,
        "status": record.status,
        "escalated": record.escalated,
        "updated_at": record.updated_at,
        "record": record.model_dump_json(),
    }
    if summary:
        item["summary"] = str(summary)

    table = _tickets_table(resource)
    try:
        table.put_item(Item=item)
    except ClientError as err:
        raise StoreError(
            f"Failed to write {config.RELAY_TICKETS_TABLE} for ticket "
            f"{record.ticket_id!r}: {err.response['Error']['Code']} — "
            f"{err.response['Error']['Message']}"
        ) from err

    return record.model_dump(mode="json")


# =============================================================================
# Module 15 ADDITION — idempotent first-write on ticket_id (BY ADDITION ONLY).
# =============================================================================
# The capstone HARDENS the front door against the real-world failure mode the
# brief calls out: a CloudCart webhook delivers the SAME ticket twice (they all
# do). create_ticket already upserts (PutItem on ticket_id), so a redelivery never
# makes TWO rows. But "the row is overwritten" is not the same guarantee as "the
# FIRST write wins and a duplicate is rejected" — and an action (a refund) must be
# proposed/recorded EXACTLY ONCE, never a second time on a redelivery.
#
# This adds a CONDITIONAL first-write helper (skill 1.1.1 hardening, T5.2): the
# very first record for a ticket_id is written with a DynamoDB condition
# `attribute_not_exists(ticket_id)`. A second create_ticket_first_seen() for the
# SAME id raises IdempotentReplay INSTEAD of silently clobbering the
# first record — the caller (the POST handler / the worker) treats the replay as a
# no-op and returns the existing record. The agent's own status advances
# (received -> triaged -> answered) still use the plain create_ticket upsert; only
# the FRONT-DOOR `received` write is conditional, so a duplicated webhook cannot
# start a second pipeline (and therefore cannot trigger a second refund action).
#
# BY ADDITION: a NEW function + a NEW exception. create_ticket's signature, the
# frozen TicketRecord schema, and the table name are all untouched.

class IdempotentReplay(StoreError):
    """A ticket_id that already has a record was re-submitted as a first write.

    Raised by create_ticket_first_seen() when a CloudCart webhook (or any caller)
    delivers the same ticket twice. The message is operator-facing; the API/worker
    catch it and treat the replay as a no-op — one ticket, one pipeline, one action.
    """


def create_ticket_first_seen(
    ticket_id: str,
    *,
    status: str = "received",
    summary: str | None = None,
    triage: dict | Triage | None = None,
    answer: dict | Answer | None = None,
    actions: list | None = None,
    escalated: bool = False,
    resource=None,
) -> dict:
    """Conditionally write the FIRST record for a ticket (idempotency gate). Module 15.

    Identical to create_ticket EXCEPT the PutItem carries the DynamoDB condition
    `attribute_not_exists(<ticket_id key>)`. The first delivery of a ticket succeeds
    and returns the stored record; a SECOND delivery of the same ticket_id raises
    IdempotentReplay (a ConditionalCheckFailedException underneath) rather than
    overwriting the first row or starting a second run. This is the brief's
    idempotency key (T5.2 "webhook delivered twice -> dedup by ticket_id, conditional
    DynamoDB write"): one webhook redelivery cannot create a duplicate TicketRecord
    nor a second refund action.

    The agent's later status writes still use create_ticket (an UPSERT) to ADVANCE
    the same row through its lifecycle — only this front-door `received` write is
    conditional. Raises ToolInputError on a bad id/shape (same as create_ticket).
    """
    if not ticket_id or not str(ticket_id).strip():
        raise ToolInputError("ticket_id is required to create a ticket.")

    record = _build_record(
        ticket_id=str(ticket_id).strip(),
        status=status,
        triage=triage,
        answer=answer,
        actions=actions,
        escalated=escalated,
    )
    item: dict = {
        config.TICKETS_KEY: record.ticket_id,
        "status": record.status,
        "escalated": record.escalated,
        "updated_at": record.updated_at,
        "record": record.model_dump_json(),
    }
    if summary:
        item["summary"] = str(summary)

    table = _tickets_table(resource)
    try:
        table.put_item(
            Item=item,
            # The idempotency gate: only write when no row for this ticket_id exists.
            ConditionExpression=f"attribute_not_exists({config.TICKETS_KEY})",
        )
    except ClientError as err:
        code = err.response["Error"]["Code"]
        if code == "ConditionalCheckFailedException":
            raise IdempotentReplay(
                f"Ticket {record.ticket_id!r} already exists — this is a duplicate "
                "delivery (idempotent replay). The first record stands; no second "
                "pipeline or action is started."
            ) from err
        raise StoreError(
            f"Failed to write {config.RELAY_TICKETS_TABLE} for ticket "
            f"{record.ticket_id!r}: {code} — {err.response['Error']['Message']}"
        ) from err

    return record.model_dump(mode="json")


def _build_record(
    *,
    ticket_id: str,
    status: str,
    triage,
    answer,
    actions,
    escalated: bool,
) -> TicketRecord:
    """Coerce loose inputs into a validated, frozen TicketRecord (06 §2).

    Each piece is validated through its frozen schema, so a malformed triage/answer/
    action is caught HERE (and reported to the model as a ToolInputError) rather than
    written as junk. `approved` on every action stays None at Module 7 — the agent
    proposes actions; nothing approves them yet (Module 8).
    """
    try:
        triage_obj = _as_triage(triage)
        answer_obj = _as_answer(answer)
        action_objs = _as_actions(actions)
        return TicketRecord(
            ticket_id=ticket_id,
            status=status,                 # Literal-validated by the schema
            triage=triage_obj,
            answer=answer_obj,
            actions=action_objs,
            escalated=bool(escalated),
            cost_cents=0.0,                # PLACEHOLDER at M7 — really populated at M12
            updated_at=_now_iso(),
        )
    except (ValueError, TypeError) as err:
        raise ToolInputError(
            f"create_ticket was given a value that does not match the TicketRecord "
            f"contract: {err}"
        ) from err


def _as_triage(value) -> Triage | None:
    if value is None or isinstance(value, Triage):
        return value
    if isinstance(value, str):
        return Triage.model_validate_json(value)
    return Triage.model_validate(value)


def _as_answer(value) -> Answer | None:
    if value is None or isinstance(value, Answer):
        return value
    if isinstance(value, str):
        return Answer.model_validate_json(value)
    return Answer.model_validate(value)


def _as_actions(value) -> list[AgentAction]:
    if not value:
        return []
    out: list[AgentAction] = []
    for item in value:
        if isinstance(item, AgentAction):
            out.append(item)
        elif isinstance(item, str):
            out.append(AgentAction.model_validate_json(item))
        else:
            out.append(AgentAction.model_validate(item))
    return out


def get_ticket(ticket_id: str, *, resource=None) -> TicketRecord | None:
    """Read back a persisted TicketRecord (used by the demo + tests). Returns None if
    the ticket is not in the table. Reconstructs the frozen TicketRecord from the
    stored JSON document."""
    table = _tickets_table(resource)
    try:
        response = table.get_item(Key={config.TICKETS_KEY: str(ticket_id).strip()})
    except ClientError as err:
        raise StoreError(
            f"Failed to read {config.RELAY_TICKETS_TABLE} for ticket "
            f"{ticket_id!r}: {err.response['Error']['Code']} — "
            f"{err.response['Error']['Message']}"
        ) from err
    item = response.get("Item")
    if item is None:
        return None
    return TicketRecord.model_validate_json(item["record"])


# =============================================================================
# Seeding — load data/orders.json into relay-orders (idempotent; used by setup.py).
# =============================================================================
def seed_orders(items: list[dict], *, resource=None) -> int:
    """Batch-write the 25 seed orders into relay-orders. Idempotent (PutItem upserts).

    Each item must carry the order_id key. Numbers are written as-is (boto3 maps
    Python numbers to DynamoDB N). Returns the count written.
    """
    from decimal import Decimal

    table = _orders_table(resource)
    written = 0
    with table.batch_writer() as batch:
        for raw in items:
            if config.ORDERS_KEY not in raw:
                raise ToolInputError(
                    f"Seed order is missing the {config.ORDERS_KEY!r} key: {raw!r}"
                )
            # DynamoDB rejects float; round-trip through JSON with Decimal for any
            # numeric fields (totals, quantities) so seeding never raises on a float.
            item = json.loads(json.dumps(raw), parse_float=Decimal, parse_int=Decimal)
            batch.put_item(Item=item)
            written += 1
    return written
