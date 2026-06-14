"""relay/api/post_handler.py — POST /tickets -> 202 {ticket_id} (Module 11).

The front door. A CloudCart ticket arrives over HTTP; an agent run takes several seconds
(triage -> retrieve -> tool calls -> answer), far longer than a client should hold an HTTP
connection (skill 2.4.1: sync vs async). So this handler does the FAST part synchronously
and pushes the SLOW part onto a queue:

  1. PARSE + VALIDATE the body into the frozen Ticket schema (06 §2). API Gateway already
     ran a JSON-Schema request model (the cheap structural gate in the CDK stack); this is
     the second, business-contract layer. Bad input -> a clean 400, never a stack trace.
  2. GENERATE a ticket_id (when the caller did not supply one).
  3. PERSIST TicketRecord{status:"received"} to relay-tickets so an immediate
     GET /tickets/{id} already returns a real record (status `received`) — the client can
     poll from the very first millisecond.
  4. ENQUEUE the job on the SQS work queue (the ticket_id + the customer message + the
     optional triage intent / customer id). The worker Lambda consumes it.
  5. RETURN 202 Accepted with {ticket_id}. 202, not 200: the work is ACCEPTED, not done.

This is the canonical async-API pattern the exam tests (D2): respond immediately, process
in the background, let the client poll GET for the final status. NO foundation-model call
here — generation happens in the worker, through the unchanged agent. No model ID; the
queue URL comes from relay.config (env var the CDK stack injects, or GetQueueUrl).
"""

from __future__ import annotations

import datetime as dt
import json
import uuid

import boto3

from relay import config
from relay.api import common
from relay.models import Ticket


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sqs():
    """An SQS client from the boto3 default session (AWS_PROFILE / Region). Built lazily
    so importing the handler stays offline for tests."""
    return boto3.client("sqs", region_name=config.REGION)


def validate_ticket(body: dict) -> Ticket:
    """Validate the POST body into a frozen Ticket, generating fields the caller may omit.

    The API contract (06 §2) is intentionally small: a caller MUST send a non-empty
    `customer_message`; `channel` defaults to "email" and is checked against the frozen
    Literal; `ticket_id` / `created_at` are generated when absent. `attachments` and
    `pii_redacted` are NOT part of the public request (attachments are an intake-pipeline
    concern, pii_redacted is a server-set state flag) — a caller that sends them is
    ignored for those, keeping the door simple. Raises common.BadRequest (-> 400) on any
    contract violation, with a caller-facing message.
    """
    message = body.get("customer_message")
    if not message or not str(message).strip():
        raise common.BadRequest("'customer_message' is required and must be non-empty.")

    channel = body.get("channel", "email")
    if channel not in ("email", "chat"):
        raise common.BadRequest("'channel' must be 'email' or 'chat'.")

    ticket_id = body.get("ticket_id") or f"ticket-{uuid.uuid4().hex[:8]}"
    try:
        return Ticket(
            ticket_id=str(ticket_id),
            channel=channel,
            customer_message=str(message),
            created_at=body.get("created_at") or _now_iso(),
        )
    except ValueError as err:  # a stray Literal/type violation -> a clean 400.
        raise common.BadRequest(f"Ticket does not match the contract: {err}") from err


def enqueue(ticket: Ticket, *, body: dict, sqs_client=None, queue_url=None) -> str:
    """Push the ticket job onto the SQS work queue. Returns the SQS message id.

    The message body is the minimal job envelope the worker needs: the ticket_id, the
    customer message, and the optional triage intent / customer id / session id (passed
    straight through to the frozen run_relay payload — relay.run owns those keys). The
    queue URL resolves from the arg / RELAY_QUEUE_URL env var / GetQueueUrl (config).
    """
    client = sqs_client or _sqs()
    url = config.resolve_queue_url(queue_url, sqs_client=client)
    job = {
        "ticket_id": ticket.ticket_id,
        "customer_message": ticket.customer_message,
        "channel": ticket.channel,
        # Pass the optional routing hints through to the agent's frozen payload. None of
        # these is required; the worker tolerates their absence.
        "triage_intent": body.get("triage_intent"),
        "customer_id": body.get("customer_id"),
        "session_id": body.get("session_id"),
    }
    sent = client.send_message(QueueUrl=url, MessageBody=json.dumps(job))
    return sent["MessageId"]


def write_received(ticket: Ticket, *, persist=None) -> dict:
    """Persist TicketRecord{status:"received"} so an immediate GET returns a real record.

    Uses the SAME single relay-tickets writer the agent uses (mcp_server.store.create_ticket)
    so the table has exactly one writer path and the record round-trips through the frozen
    TicketRecord schema. `received` is the first of the full status lifecycle Module 11
    exercises end-to-end (received -> the worker advances it). Tests inject `persist`.
    """
    if persist is None:
        from mcp_server import store

        persist = store.create_ticket
    return persist(
        ticket.ticket_id,
        status="received",
        summary="Ticket accepted at the API; queued for processing.",
        triage=None,
        answer=None,
        actions=[],
        escalated=False,
    )


def handle(event: dict, *, sqs_client=None, persist=None, queue_url=None) -> dict:
    """POST /tickets. Validate -> persist `received` -> enqueue -> 202 {ticket_id}.

    Args mirror the testable seams: `sqs_client` (moto/stub), `persist` (the relay-tickets
    writer, a fake on a moto table in tests), `queue_url` (explicit in tests). In the
    deployed Lambda all three resolve from the boto3 session + the CDK-injected env var.
    Returns the API Gateway proxy response dict.
    """
    try:
        body = common.parse_json_body(event)
        ticket = validate_ticket(body)
    except common.BadRequest as err:
        return common.error(400, str(err))

    # Persist `received` FIRST so a GET right after the POST already finds the record, then
    # enqueue. If the enqueue fails we surface a 500 (the record stays `received`; the
    # client can retry the POST — idempotent on a supplied ticket_id, the worker upserts).
    try:
        write_received(ticket, persist=persist)
        enqueue(ticket, body=body, sqs_client=sqs_client, queue_url=queue_url)
    except ValueError as err:
        # resolve_queue_url raises ValueError with an actionable message when the queue is
        # not configured (stack not deployed) — a server-side misconfiguration -> 500.
        print(f"[post_handler] enqueue failed: {err}")
        return common.error(500, "Could not accept the ticket (queue unavailable).")
    except Exception as err:  # noqa: BLE001 — log the real cause, return a clean 500.
        print(f"[post_handler] unexpected failure: {type(err).__name__}: {err}")
        return common.error(500, "Could not accept the ticket.")

    # 202 Accepted: the ticket is queued, not yet processed. The client polls GET.
    return common.response(202, {"ticket_id": ticket.ticket_id, "status": "received"})


# AWS Lambda entrypoint name the CDK stack points the function at.
def lambda_handler(event, context=None):  # noqa: ANN001 - Lambda signature
    """The AWS Lambda handler API Gateway invokes for POST /tickets."""
    return handle(event)
