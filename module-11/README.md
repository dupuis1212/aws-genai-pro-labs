# Module 11 — Shipping Relay: Serverless Deployment, Enterprise Integration, and CI/CD

**What:** Relay is safe and governed now (Modules 9–10), but it has **no front door**. It
runs in local scripts — `uv run python -m relay.run "..."` on a laptop — with no API, no
reproducible deployment, no pipeline. The M8 HITL refund gate is still waiting for a
`POST /tickets/{id}/approve` endpoint that does not exist. CloudCart can't plug anything in.

Module 11 **ships** Relay: an API (`POST /tickets` → `{ticket_id}`) returns immediately and
pushes the work onto **Amazon SQS**; a worker Lambda processes the ticket and writes the
`TicketRecord`; escalations go out on **Amazon EventBridge** `relay-events`; the whole
front door is described in **AWS CDK**; and a **CodePipeline** re-deploys Relay on every
commit, with smoke tests as the guardrail.

```bash
uv run python setup.py                 # upstream: tables/KB/guardrail/MCP/IAM (M5–M10)
uv run cdk deploy RelayApiStack        # the front door: API Gateway + 4 Lambda + SQS + bus

curl -X POST https://<api-id>.execute-api.us-east-1.amazonaws.com/prod/tickets \
  -d @data/tickets/sample.json
# -> 202 {"ticket_id": "ticket-XXXXXXXX", "status": "received"}
curl https://<api-id>.execute-api.us-east-1.amazonaws.com/prod/tickets/ticket-XXXXXXXX
# -> the TicketRecord (status answered | escalated | awaiting_approval)
```

## What this module builds (on top of Module 10)

- **`relay/api/` (NEW)** — four AWS Lambda handlers behind **Amazon API Gateway**:
  - `post_handler` — `POST /tickets`: validate the `Ticket`, write
    `TicketRecord{status:"received"}`, **enqueue** on SQS, return **202** `{ticket_id}`.
  - `worker_handler` — the **SQS** consumer: invoke the **deployed agent** through the
    frozen `relay.run.run_relay` contract (M8), update the `TicketRecord` to its final
    status, and **publish** `relay.escalation` / `relay.approval_required` to `relay-events`.
  - `get_handler` — `GET /tickets/{ticket_id}`: read the `TicketRecord`.
  - `approve_handler` — `POST /tickets/{ticket_id}/approve` body `{approved: bool}`:
    realize the **M8 HITL refund gate** over HTTP (calls `relay.approve.approve`).
- **`cdk/` (NEW)** — the AWS CDK app: `RelayApiStack` (API Gateway + 4 Lambda + SQS + DLQ +
  the `relay-events` bus + a demo escalation rule) and `RelayPipelineStack` (the
  CodePipeline). It **references** upstream tables/KB/guardrail by canonical name — never
  recreates them (decision **B6**: M11+ infra → CDK).
- **`pipeline/` (NEW)** — the CI/CD buildspecs: `build` (offline tests + a security scan),
  `smoke` (a real `POST`→poll-`GET` against the deployed API), and `rollback` on failure.
  The **eval-gate** stage is commented for Module 13.
- **`setup.py` (MODIFIED)** — kept for the **upstream** resources only (B6); the M11 infra
  is CDK. **`teardown.py` (MODIFIED)** — `cdk destroy` + **delete the CodePipeline** (the
  only idle-billed M11 resource) + sweep SQS/EventBridge.

The whole `relay/` package (the agent, guardrail, intake, PII, `converse()` layer) is
**WRAPPED, not rewritten**. No model ID appears in `relay/api/` — generation stays in
`relay.llm.converse()` via `relay.run`.

## Frozen contracts (no schema change — bible §3.1)

Module 11 adds **no Pydantic field**. It first **freezes the API + the bus** (06 §2) and
**exercises the full `TicketRecord` status lifecycle** end-to-end:

```
POST /tickets                    -> {ticket_id}                 (HTTP 202)
GET  /tickets/{ticket_id}        -> TicketRecord
POST /tickets/{ticket_id}/approve  body {approved: bool}        (the M8 HITL gate)
```

```
EventBridge bus relay-events
  detail-type relay.escalation          (the agent handed the ticket to a human)
  detail-type relay.approval_required   (a refund is awaiting human approval)
```

`TicketRecord.status` ∈ `received | triaged | awaiting_approval | answered | escalated |
closed | failed` — the full enum frozen at M7, now reached end-to-end. `POST /feedback` +
`feedback_rating` are **Module 13**, not here.

## Sync vs async — why `POST` returns 202 (the exam tests this)

A ticket runs an agent loop (several seconds). A synchronous API would time out the client.
So `POST /tickets` does the **fast** part (validate + enqueue) and returns **202 Accepted**;
the **slow** part (the agent) runs in the SQS worker; the client **polls** `GET`. This is
the canonical async-API pattern for long GenAI work (skill 2.4.1).

## Internal IAM (M10) vs consumer auth (M11)

| | What it controls | Module |
|---|---|---|
| **Internal IAM** | what each Relay component can **touch** (the worker reads `relay-orders`, writes `relay-tickets`) | M10 |
| **Consumer auth** | **who** may **call** the API from outside (Cognito / RBAC / federation) | M11 (the "Securing the front door" article + a Try-it-yourself) |

## Run it

```bash
export AWS_PROFILE=aws-genai-pro          # us-east-1 everywhere; no keys in code/.env
uv sync                                   # adds aws-cdk-lib v2 (the cdk CLI is npm-installed)
uv run python setup.py                    # upstream: tables/KB/guardrail/MCP/IAM
uv run cdk deploy RelayApiStack           # the front door
uv run python pipeline/smoke_test_live.py https://<api-id>.execute-api.us-east-1.amazonaws.com/prod
uv run cdk deploy RelayPipelineStack -c repo_owner=<you> -c connection_arn=<arn>   # CI/CD
uv run pytest                             # offline cumulative suite (Modules 2–11)
RELAY_LIVE_TESTS=1 RELAY_API_URL=<url> uv run pytest -m live   # opt-in, capped
uv run python teardown.py                 # cdk destroy + delete the pipeline + M7–M10 cleanup
```

## Boundaries (what this module does NOT do)

- No **provisioned throughput**, no **SageMaker AI endpoint** — taught as theory/exam
  scenario (idle-billed), **never** provisioned. No **NAT gateway**, no paid VPC.
- No **eval-gate** wired into the pipeline — Module 13 (M11 builds the pipeline + smoke
  tests; the eval-gate stage is commented).
- No **dashboard / invocation logs** — Module 14.
- No **agent rebuild** — M7/M8 own the agent; M11 **wraps** it.
- No **caching / $-per-ticket math** — Module 12.
- No **Amplify** front-end deploy, no **WebSocket streaming** wired (renvoi M3).

See `lab.md` for the full step-by-step, the measured cost, and "Try it yourself".
