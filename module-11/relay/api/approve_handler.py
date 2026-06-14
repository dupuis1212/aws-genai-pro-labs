"""relay/api/approve_handler.py — POST /tickets/{ticket_id}/approve (Module 11).

This endpoint REALIZES the Module 8 human-in-the-loop refund gate over HTTP. In Module 8
the gate parked a refund in `awaiting_approval` and a human ran a LOCAL command
(`uv run python -m relay.approve <id> --approve`). The agent prompt in T2 said it best:
the gate was waiting for a `POST /tickets/{id}/approve` that did not exist. Now it exists.

Contract (06 §2 — reproduced field-for-field):
    POST /tickets/{ticket_id}/approve   body {"approved": bool}
  - approved=true  -> execute the refund (relay.approve), status -> answered.
  - approved=false -> abandon + escalate, status -> escalated.

The handler is a THIN wrapper: it parses {approved: bool}, then calls the SAME
relay.approve.approve(ticket_id, decision) the CLI used. The business logic, the
idempotent guard (a ticket already decided returns a clean 409), and the relay-tickets
writes all live in relay.approve — Module 11 only opens the door; it does not re-implement
the gate. NO foundation-model call (approving a refund is a business decision + a DynamoDB
write, never a generation), no model ID.

  - 200 + the updated TicketRecord on a successful approve/reject.
  - 400 on a missing/non-boolean `approved`.
  - 404 / 409 when the ticket does not exist or is not awaiting approval (ApprovalError).
"""

from __future__ import annotations

from relay.api import common


def parse_decision(body: dict) -> bool:
    """Extract the required boolean `approved` from the body, or raise BadRequest (->400).

    Strict: `approved` MUST be a JSON boolean. We do NOT coerce "true"/1 — an ambiguous
    truthy value on a financial action is a request error the caller should fix, not a
    silent guess (a refund is money moving)."""
    if "approved" not in body:
        raise common.BadRequest("Body must include a boolean 'approved' field.")
    decision = body["approved"]
    if not isinstance(decision, bool):
        raise common.BadRequest("'approved' must be a JSON boolean (true or false).")
    return decision


def handle(event: dict, *, approve=None, load=None, persist=None, resource=None) -> dict:
    """POST /tickets/{ticket_id}/approve. Wrap relay.approve.approve over HTTP.

    `approve` (the relay.approve.approve callable), `load`/`persist`/`resource` (the
    relay-tickets read/write + order-book resource) are injectable for offline tests; in
    the deployed Lambda they default to relay.approve on the boto3 session.
    """
    try:
        ticket_id = common.path_param(event, "ticket_id")
        body = common.parse_json_body(event)
        decision = parse_decision(body)
    except common.BadRequest as err:
        return common.error(400, str(err))

    if approve is None:
        from relay.approve import approve as approve_fn

        approve = approve_fn

    # relay.approve raises ApprovalError when the ticket does not exist or is not
    # awaiting approval (already decided / wrong status). Both are CLIENT-correctable, so
    # they map to 404 / 409 with the module's own actionable message.
    try:
        from relay.approve import ApprovalError
    except Exception:  # pragma: no cover - relay.approve always imports
        ApprovalError = RuntimeError  # noqa: N806

    try:
        record = approve(ticket_id, decision, load=load, persist=persist,
                         resource=resource)
    except ApprovalError as err:
        message = str(err)
        # "No ticket ..." -> 404; anything else (not awaiting / already decided) -> 409
        # Conflict (the ticket exists but is not in an approvable state).
        status = 404 if message.lower().startswith("no ticket") else 409
        return common.error(status, message)
    except Exception as err:  # noqa: BLE001 — log the real cause, return a clean 500.
        print(f"[approve_handler] approve failed for {ticket_id!r}: "
              f"{type(err).__name__}: {err}")
        return common.error(500, "Could not process the approval.")

    return common.response(200, record.model_dump(mode="json"))


def lambda_handler(event, context=None):  # noqa: ANN001 - Lambda signature
    """The AWS Lambda handler API Gateway invokes for POST /tickets/{id}/approve."""
    return handle(event)
