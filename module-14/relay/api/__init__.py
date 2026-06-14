"""relay.api — Relay's serverless front door (Module 11).

Module 11 of AWS GenAI Pro Mastery gives Relay a DOOR. Through Module 10 the agent
ran on a laptop (`uv run python -m relay.run "..."`): no API, no reproducible
deployment, no pipeline. The M8 HITL refund gate waited for a `POST /tickets/{id}/
approve` endpoint that did not exist. This package is that door — four AWS Lambda
handlers behind Amazon API Gateway, wired with Amazon SQS for async processing and
Amazon EventBridge for loose coupling. It WRAPS the unchanged `relay/` package; it
never rewrites the agent (M7/M8), the guardrail (M9), the intake/PII (M6/M10), or the
`converse()` layer (M3).

The four handlers (06 §2 frozen API contract — paths/bodies reproduced identically):

  - post_handler    : `POST /tickets`  -> 202 {ticket_id}. Validates the Ticket,
                      generates a ticket_id, writes TicketRecord{status:"received"} to
                      relay-tickets, ENQUEUES the job on SQS, and returns immediately.
                      A ticket takes several seconds (an agent loop); the client must
                      not block, so the work goes async.
  - worker_handler  : the SQS consumer. Dequeues a job, invokes the DEPLOYED agent
                      through the FROZEN relay.run.run_relay contract (M8), updates the
                      TicketRecord to its final status (answered / escalated /
                      awaiting_approval / failed), and PUBLISHES escalation /
                      approval-required events to the `relay-events` bus.
  - get_handler     : `GET /tickets/{ticket_id}` -> the persisted TicketRecord (with the
                      full status lifecycle the worker writes). The client POSTs, then
                      polls this.
  - approve_handler : `POST /tickets/{ticket_id}/approve` body {approved: bool} ->
                      REALIZES the M8 HITL refund gate over HTTP. It calls the SAME
                      relay.approve.approve() the local CLI used; the API only wraps it.

Module 13 ADDS — BY ADDITION — a FIFTH handler, the user-feedback loop (skill 5.1.3):

  - feedback_handler : `POST /tickets/{ticket_id}/feedback` body {feedback_rating: int}
                      -> sets the Module-13 `TicketRecord.feedback_rating` (a 1-5 customer
                      rating of Relay's answer) and re-persists the record. It WRAPS the
                      single relay-tickets read/write path (it never modifies the store),
                      makes NO foundation-model call, and holds NO model ID — leaving
                      feedback is one DynamoDB round-trip. Low ratings are where the next
                      failing cases for evals/golden_set.json come from (the feedback ->
                      golden-set loop the Module 13 article describes).

Why this layering holds the contracts (bible §4 M11):
  - The worker calls run_relay() — it NEVER re-implements the agent.
  - No model ID appears here: generation stays in relay.llm.converse() via relay.run.
  - Resource NAMES (the SQS queue, the `relay-events` bus, relay-tickets) come from
    relay.config; the handlers resolve clients from the boto3 default session.
  - At Module 11, `TicketRecord` gained NO field — M11 only EXERCISES the full frozen
    status enum end-to-end (received -> triaged/awaiting_approval/answered/escalated/
    closed/failed). Module 13 adds the FIFTH route `POST /tickets/{id}/feedback` and the
    `feedback_rating` field (by addition); the M11 feedback_handler note above is now live.

Streaming over this API (API Gateway WebSocket / Lambda response streaming) sits ABOVE
the M3 converse() layer and is NOT re-taught or wired here (renvoi M3). The provisioned-
throughput / SageMaker-endpoint deployment modes are taught in the article as theory;
the lab is on-demand only (B5 — nothing idle-billed beyond a torn-down pipeline).
"""

from __future__ import annotations

__all__ = [
    "common",
    "post_handler",
    "get_handler",
    "approve_handler",
    "worker_handler",
    "feedback_handler",   # ADDED M13 (by addition) — POST /tickets/{id}/feedback
]
