"""relay/api/worker_handler.py — the SQS consumer that runs Relay async (Module 11).

This is where the slow work happens. POST /tickets enqueued a job and returned 202; this
Lambda is triggered by Amazon SQS, dequeues the job, and does the real processing:

  1. PARSE the SQS message into the frozen run_relay payload (the M8 invoke contract:
     customer_message + the optional ticket_id / triage_intent / customer_id / session_id).
  2. INVOKE THE DEPLOYED AGENT through relay.run.run_relay() — the FROZEN seam from
     Module 8 (bible §4 M11). The worker NEVER re-implements the agent; run_relay wires
     the handoff, the HITL gate, the guardrail, and PERSISTS the final TicketRecord to
     relay-tickets (advancing it from `received` to its terminal status). The worker reads
     the outcome, it does not re-write the record.
  3. PUBLISH events to the `relay-events` EventBridge bus by OUTCOME (loose coupling —
     skill 2.3.1/2.3.2):
       - status `escalated`        -> a `relay.escalation` event;
       - status `awaiting_approval`-> a `relay.approval_required` event.
     Relay does not know CloudCart's human queue or approval inbox; it publishes and moves
     on. EventBridge RULES (in the CDK stack) route the event to whatever targets CloudCart
     wires (a human-escalation SQS queue, an approval inbox). A `relay.escalation` consumer
     is a new rule, not a code change in Relay.
  4. IDEMPOTENCE: run_relay persists with PutItem on ticket_id (the single relay-tickets
     writer upserts), so a REDELIVERED SQS message overwrites the same row — never two
     records for one ticket (the "In production" idempotence note, built in).

A message the worker cannot process (a genuinely broken job, or a ticket that fails over
and over) is RE-RAISED so SQS redelivers it; after RELAY_QUEUE_MAX_RECEIVE attempts the
redrive policy parks it in the DLQ (config + the CDK stack) instead of looping forever.

NO model ID and no bare model-invoke path here — all generation stays inside the agent,
through relay.llm.converse(). cost_cents stays the M7 `0.0` placeholder (Module 12
populates it from the M3 price map; this worker is the hook). The queue/bus names come
from relay.config; clients resolve from the boto3 session.
"""

from __future__ import annotations

import json

import boto3

from relay import config


def _events_client():
    """An EventBridge client from the boto3 default session. Built lazily so importing the
    worker stays offline for tests."""
    return boto3.client("events", region_name=config.REGION)


def parse_job(record: dict) -> dict:
    """Turn one SQS record into the frozen run_relay payload (M8 invoke contract).

    The SQS record's `body` is the JSON job post_handler enqueued. We map it onto the
    EXACT run_relay keys (relay.run owns them): customer_message (required) + the optional
    ticket_id / triage_intent / customer_id / session_id. A record with no parseable body
    or no customer_message raises ValueError so the worker re-raises and SQS redelivers
    (then DLQs) — a broken job must not be silently dropped.
    """
    raw = record.get("body")
    if raw is None:
        raise ValueError("SQS record has no body.")
    try:
        job = raw if isinstance(raw, dict) else json.loads(raw)
    except json.JSONDecodeError as err:
        raise ValueError(f"SQS record body is not valid JSON: {err.msg}.") from err
    if not job.get("customer_message"):
        raise ValueError("SQS job is missing 'customer_message'.")
    # Pass through exactly the frozen run_relay payload keys (drop anything else).
    return {
        "customer_message": job["customer_message"],
        "ticket_id": job.get("ticket_id"),
        "triage_intent": job.get("triage_intent"),
        "customer_id": job.get("customer_id"),
        "session_id": job.get("session_id"),
    }


def publish_outcome_event(response: dict, *, events_client=None, bus_name=None) -> str | None:
    """Publish a `relay-events` event for an escalation / approval-required outcome.

    Reads the run_relay RESPONSE (the frozen dict: status / gated / record / ...) and emits
    the matching detail-type. Returns the emitted detail-type, or None when the outcome
    needs no event (a plain `answered` ticket — the client just polls GET, no human needed).
    Loose coupling: ONE PutEvents, then the worker moves on; routing is the bus's job.
    """
    status = response.get("status")
    if status == "escalated":
        detail_type = config.RELAY_DETAIL_ESCALATION
    elif status == "awaiting_approval":
        detail_type = config.RELAY_DETAIL_APPROVAL_REQUIRED
    else:
        return None  # answered / failed: no human-routing event to publish.

    client = events_client or _events_client()
    bus = config.resolve_event_bus_name(bus_name)
    # The event detail is a SMALL envelope — the ticket id, the status, and whether a
    # refund is gated — NOT the whole record (a consumer fetches the record via GET if it
    # needs it). Keep PII out of the event by construction: only the id + status travel.
    detail = {
        "ticket_id": response.get("ticket_id"),
        "status": status,
        "gated": bool(response.get("gated")),
    }
    client.put_events(Entries=[{
        "Source": config.RELAY_EVENT_SOURCE,
        "DetailType": detail_type,
        "Detail": json.dumps(detail),
        "EventBusName": bus,
    }])
    return detail_type


def process_record(record: dict, *, run=None, events_client=None, bus_name=None) -> dict:
    """Process ONE SQS record: invoke the agent, then publish the outcome event.

    `run` is the relay.run.run_relay callable (the FROZEN M8 seam) — injectable so tests
    drive a scripted agent offline. Returns the run_relay response dict (with an added
    `event` key recording which detail-type was published, or None). Raises on a broken
    job / agent failure so SQS redelivers (then DLQs) — the worker never silently drops.
    """
    if run is None:
        from relay.run import run_relay

        run = run_relay

    payload = parse_job(record)
    # Invoke the deployed agent through the FROZEN contract. run_relay persists the final
    # TicketRecord (received -> terminal status) — the worker does not re-write it.
    response = run(payload)
    detail_type = publish_outcome_event(
        response, events_client=events_client, bus_name=bus_name
    )
    return {**response, "event": detail_type}


def handle(event: dict, *, run=None, events_client=None, bus_name=None) -> dict:
    """SQS event handler: process each record in the batch.

    Returns a small summary ({processed, results}) for logging/tests. A record that raises
    is RE-RAISED after the batch attempt so SQS redelivers the WHOLE batch (lab simplicity:
    the batch size is 1 in the CDK stack, so one bad job never blocks others). After
    RELAY_QUEUE_MAX_RECEIVE attempts the redrive policy parks it in the DLQ.
    """
    records = event.get("Records", [])
    results = []
    for record in records:
        # Let a failure propagate so SQS retries -> DLQ (no silent swallow). With batch
        # size 1 (CDK stack) this is exactly "redeliver this one ticket".
        outcome = process_record(record, run=run, events_client=events_client,
                                 bus_name=bus_name)
        results.append({"ticket_id": outcome.get("ticket_id"),
                        "status": outcome.get("status"),
                        "event": outcome.get("event")})
    return {"processed": len(results), "results": results}


def lambda_handler(event, context=None):  # noqa: ANN001 - Lambda signature
    """The AWS Lambda handler SQS invokes with a batch of ticket jobs."""
    return handle(event)
