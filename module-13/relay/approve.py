"""relay/approve.py — the human-in-the-loop decision point for a gated refund.

Module 8 of AWS GenAI Pro Mastery (skill 2.1.5). The agent's HITL gate (relay.agent)
PROPOSES a refund: it records an AgentAction(approved=None) and parks the ticket in
`awaiting_approval` — it never executes the refund itself. THIS module is where a human
makes the call:

    uv run python -m relay.approve <ticket_id> --approve   # execute the refund
    uv run python -m relay.approve <ticket_id> --reject    # abandon + escalate

  approve(ticket_id, decision):
    1. reads the `awaiting_approval` TicketRecord from relay-tickets;
    2. finds the pending refund AgentAction (approved is None, tool == refund);
    3. sets AgentAction.approved = True (approve) or False (reject);
    4. on approve: EXECUTES the refund (journals the execution), status -> `answered`;
       on reject: escalated = True, status -> `escalated`;
    5. re-persists the updated TicketRecord (idempotent on ticket_id).

This is the LOCAL / PROGRAMMATIC approval the lab exercises. The public approval HTTP
endpoint and the approval-required event bus are Module 11 — this module deliberately
does NOT create them (no API, no bus). The human decision and the resulting status
transitions are the same; Module 11 only wraps them in an API and an event.

`AgentAction.approved` is the frozen-since-M7 field made EFFECTIVE here: None (proposed),
True (approved/executed), False (rejected). No new schema — the field and the
`answered`/`escalated`/`awaiting_approval` statuses were frozen at M7 (06 §2); M8 only
USES them.

No foundation-model call here — approving a refund is a business decision + a DynamoDB
write, not a generation. The refund "execution" is a logged, idempotent state change in
CloudCart's systems (the order book), not a payment-gateway integration (out of scope).
"""

from __future__ import annotations

import datetime as dt
import sys

from relay import config
from relay.agent import find_pending_refund
from relay.models import AgentAction, TicketRecord


class ApprovalError(RuntimeError):
    """A refund could not be approved/rejected (no such ticket, not pending, ...).

    The message is meant for the human running the command (or, in M11, the API
    caller) — actionable, never a bare stack trace.
    """


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _execute_refund(action: AgentAction, *, resource=None) -> str:
    """EXECUTE an approved refund: a logged, idempotent state change in CloudCart.

    The lab's "refund" is a recorded business event on the order book (mark the order
    refunded), not a payment-gateway call — that integration is out of scope. We record
    the execution so the audit trail shows the refund actually happened after approval,
    distinct from the proposal. Idempotent: re-running approve on an already-approved
    ticket does not double-refund (the caller guards on `approved`).
    """
    order_id = (action.tool_input or {}).get("order_id", "(unknown)")
    amount = (action.tool_input or {}).get("amount_cents", 0)
    # The order-book write is intentionally minimal and defensive: if the orders table
    # is reachable we annotate the order as refunded; if not (offline/dev), the refund
    # is still recorded on the ticket's action result below. No model call, no payment.
    try:
        from mcp_server import store  # local import keeps this module import-light
        import boto3

        ddb = resource or boto3.resource("dynamodb", region_name=config.REGION)
        table = ddb.Table(config.RELAY_ORDERS_TABLE)
        table.update_item(
            Key={config.ORDERS_KEY: str(order_id)},
            UpdateExpression="SET refunded = :r, refund_amount_cents = :a, "
                             "refunded_at = :t",
            ExpressionAttributeValues={
                ":r": True, ":a": int(amount), ":t": _now_iso(),
            },
        )
        _ = store  # store is imported to keep the single-writer path discoverable
        where = f"order {order_id} marked refunded in {config.RELAY_ORDERS_TABLE}"
    except Exception as err:  # noqa: BLE001 — record it, do not crash the approval.
        where = (
            f"order {order_id} refund recorded on the ticket only "
            f"(order book not updated: {type(err).__name__})"
        )
    return f"refund EXECUTED after human approval: {amount} cents — {where}"


def approve(
    ticket_id: str,
    decision: bool,
    *,
    load=None,
    persist=None,
    resource=None,
) -> TicketRecord:
    """Approve (True) or reject (False) the gated refund on a ticket. Returns the record.

    Args:
        ticket_id: the ticket whose refund is awaiting approval.
        decision: True approves (execute the refund, status -> answered); False rejects
            (abandon + escalate, status -> escalated).
        load: callable(ticket_id) -> TicketRecord | None. Defaults to
            mcp_server.store.get_ticket; tests inject a fake to stay offline.
        persist: callable(ticket_id, *, status, summary, triage, answer, actions,
            escalated) -> stored record. Defaults to mcp_server.store.create_ticket
            (the single relay-tickets writer); tests inject a fake.
        resource: optional DynamoDB resource (moto in tests) for the order-book write.

    Raises ApprovalError if the ticket does not exist, is not awaiting approval, or has
    no pending refund action (e.g. it was already decided — idempotent guard).
    """
    if load is None:
        from mcp_server import store
        load = lambda tid: store.get_ticket(tid, resource=resource)  # noqa: E731
    if persist is None:
        from mcp_server import store
        def persist(tid, **kw):  # the single relay-tickets writer
            return store.create_ticket(tid, resource=resource, **kw)

    record = load(ticket_id)
    if record is None:
        raise ApprovalError(
            f"No ticket {ticket_id!r} in {config.RELAY_TICKETS_TABLE}. Check the id "
            "(the run printed it), or run setup.py if the table is missing."
        )
    if record.status != "awaiting_approval":
        raise ApprovalError(
            f"Ticket {ticket_id!r} is {record.status!r}, not 'awaiting_approval'. "
            "There is no pending refund to approve — it may already be decided."
        )

    idx = find_pending_refund(record.actions)
    if idx is None:
        raise ApprovalError(
            f"Ticket {ticket_id!r} has no pending refund action (approved is None). "
            "Nothing to approve — it may already be decided."
        )

    action = record.actions[idx]
    action.approved = bool(decision)

    if decision:
        # APPROVE: execute the refund and journal the execution, then close the ticket.
        execution = _execute_refund(action, resource=resource)
        record.actions.append(
            AgentAction(
                tool=config.REFUND_TOOL_NAME,
                tool_input=action.tool_input,
                result=execution,
                approved=True,
            )
        )
        new_status = "answered"
        escalated = record.escalated
        summary = f"Refund APPROVED and executed by a human. {execution}"
    else:
        # REJECT: abandon the refund and escalate to a human owner.
        new_status = "escalated"
        escalated = True
        summary = (
            "Refund REJECTED by a human; ticket escalated for follow-up "
            "(no money moved)."
        )

    stored = persist(
        ticket_id,
        status=new_status,
        summary=summary,
        triage=record.triage,
        answer=record.answer,
        actions=[a.model_dump() for a in record.actions],
        escalated=escalated,
    )
    return TicketRecord.model_validate(stored) if isinstance(stored, dict) else stored


# =============================================================================
# CLI — `uv run python -m relay.approve <ticket_id> --approve|--reject`
# =============================================================================
def _usage() -> str:
    return (
        "Usage: uv run python -m relay.approve <ticket_id> (--approve | --reject)\n"
        "  --approve : execute the proposed refund (status -> answered)\n"
        "  --reject  : abandon + escalate the ticket (status -> escalated)"
    )


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2 or argv[1] not in ("--approve", "--reject"):
        print(_usage(), file=sys.stderr)
        return 1

    ticket_id, flag = argv[0], argv[1]
    decision = flag == "--approve"
    try:
        record = approve(ticket_id, decision)
    except ApprovalError as err:
        print(f"[approve] {err}", file=sys.stderr)
        return 1
    except Exception as err:  # noqa: BLE001
        print(f"[approve] failed: {type(err).__name__}: {err}", file=sys.stderr)
        return 1

    verb = "APPROVED — refund executed" if decision else "REJECTED — ticket escalated"
    print(f"Ticket {ticket_id}: {verb}.")
    print(f"  status   : {record.status}")
    print(f"  escalated: {record.escalated}")
    print("\n--- updated TicketRecord ---")
    print(record.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
