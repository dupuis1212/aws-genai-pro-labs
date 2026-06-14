"""relay/api/get_handler.py — GET /tickets/{ticket_id} -> the TicketRecord (Module 11).

The other half of the async pattern: the client POSTed a ticket and got a `ticket_id`
back (202); now it POLLS this endpoint until the status is terminal. The handler reads the
persisted TicketRecord from relay-tickets and returns it as JSON — the FULL frozen status
lifecycle Module 11 exercises (received -> triaged -> awaiting_approval -> answered /
escalated / closed / failed).

  - 200 + the TicketRecord JSON when the ticket exists.
  - 404 (clean message) when there is no such ticket — never an empty 200 that a client
    would mistake for "still processing".

NO foundation-model call, no model ID — a single relay-tickets read through the same store
path the agent/worker write. The table name comes from relay.config.
"""

from __future__ import annotations

from relay.api import common


def load_record(ticket_id: str, *, load=None):
    """Read the persisted TicketRecord (or None). Uses the single relay-tickets read path
    (mcp_server.store.get_ticket) so the record round-trips through the frozen schema.
    Tests inject `load` to stay offline."""
    if load is None:
        from mcp_server import store

        load = store.get_ticket
    return load(ticket_id)


def handle(event: dict, *, load=None) -> dict:
    """GET /tickets/{ticket_id}. Returns 200 + the TicketRecord, or 404, or 400/500.

    `load` is the relay-tickets reader (a fake on a moto table in tests). In the deployed
    Lambda it defaults to mcp_server.store.get_ticket on the boto3 session.
    """
    try:
        ticket_id = common.path_param(event, "ticket_id")
    except common.BadRequest as err:
        return common.error(400, str(err))

    try:
        record = load_record(ticket_id, load=load)
    except Exception as err:  # noqa: BLE001 — log the real cause, return a clean 500.
        print(f"[get_handler] read failed for {ticket_id!r}: "
              f"{type(err).__name__}: {err}")
        return common.error(500, "Could not read the ticket.")

    if record is None:
        return common.error(
            404, f"No ticket {ticket_id!r} found. Check the id returned by POST /tickets."
        )

    # The frozen TicketRecord serialized to JSON (mode='json' so nested Triage/Answer/
    # AgentAction and the status enum are plain JSON). This is the 06 §2 response body.
    return common.response(200, record.model_dump(mode="json"))


def lambda_handler(event, context=None):  # noqa: ANN001 - Lambda signature
    """The AWS Lambda handler API Gateway invokes for GET /tickets/{ticket_id}."""
    return handle(event)
