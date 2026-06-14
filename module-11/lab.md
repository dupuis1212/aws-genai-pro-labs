# Module 11 lab — shipping Relay: serverless deployment, enterprise integration, CI/CD

> **This lab cost me $0.10 on June 2026 prices** (the syllabus budget for Module 11 is
> < $2). Almost everything in the front door is **on-demand / free-tier**; the only
> resource with a real *idle* cost is the **CodePipeline** (~$1/active pipeline/month) —
> **destroyed at teardown**. The figures below are the MEASURED usage of one full live
> run (`cdk deploy` of `RelayApiStack`, real `POST`→poll-`GET` round-trips, the live
> smoke suite, and `teardown.py`), read from CloudWatch Bedrock token metrics and the
> pricing pages — never guessed (re-verify on the
> [API Gateway](https://aws.amazon.com/api-gateway/pricing/),
> [Lambda](https://aws.amazon.com/lambda/pricing/), [SQS](https://aws.amazon.com/sqs/pricing/),
> [EventBridge](https://aws.amazon.com/eventbridge/pricing/),
> [Bedrock](https://aws.amazon.com/bedrock/pricing/), and
> [CodePipeline](https://aws.amazon.com/codepipeline/pricing/) pricing pages, **as of June
> 2026**):
>
> | Item | Usage | Cost |
> |---|---|---|
> | API Gateway REST (`POST`/`GET`/`approve`) | ~60 requests | ~$0.0001 (free tier) |
> | AWS Lambda (4 functions, post/get/approve/worker) | ~60 invokes, <512 MB-s each | ~$0.00 (free tier) |
> | Amazon SQS (work queue + DLQ) | ~30 messages | ~$0.00 (free tier) |
> | Amazon EventBridge (`relay-events`, custom bus) | ~10 events | ~$0.00 ($1/M events) |
> | agent runs (smart-tier ReAct, on-demand Bedrock) | 242 invocations, ~271k in / ~9k out tok | ~$0.10 |
> | CloudFormation / CDK deploy + destroy (+ one-time bootstrap S3 staging) | — | ~$0.00 |
> | inherited live smoke (fast/smart/embed/KB/vision/guardrail) | small | ~$0.0015 |
> | CodePipeline (synth-validated; NOT deployed — needs a GitHub connection) | 0 runs | $0.00 |
> | **Total (measured)** | | **≈ $0.10** (essentially all Bedrock agent runs) |
>
> The agent-run figure is read straight from the **CloudWatch `AWS/Bedrock`
> InputTokenCount / OutputTokenCount** for the session (~271k input + ~9k output tokens
> across 242 model calls — the multi-turn ReAct loops dominate) and priced at the
> smart-tier (`us.amazon.nova-2-lite-v1:0`) on-demand rate as an upper bound. The
> **CodePipeline** was synth-validated (`cdk synth RelayPipelineStack` produces a valid
> 1-pipeline / 3-CodeBuild-project template) but not deployed live — it needs a GitHub
> CodeStar connection, and it is the one idle-billed resource the brief says to destroy —
> so it added **$0** here; its build/smoke gates were exercised by hand
> (`pipeline/smoke_test_live.py` against the deployed API).
>
> **No provisioned throughput, no SageMaker endpoint, no NAT gateway** — all on-demand /
> pay-per-use (B5). The CodePipeline is the **one** idle-billed M11 resource; `teardown.py`
> runs `cdk destroy` for both stacks **and deletes the pipeline** so nothing is idle-billed.
>
> **Teardown reminder:** run `uv run python teardown.py` when you're done — it
> **`cdk destroy`s `RelayApiStack` + `RelayPipelineStack` and deletes the CodePipeline**
> first (the only idle-billed M11 resource), then does the M7–M10 cleanup (MCP Lambda,
> guardrail, AgentCore Memory), **keeping** the on-demand tables and the Knowledge Base
> (~$0 idle). The M1 $5 budget stays.

This lab gives Relay a **front door**. We'll expose it behind `POST /tickets` (API Gateway
+ Lambda, async via SQS) and `GET /tickets/{id}`, publish escalations on EventBridge
`relay-events`, describe the whole stack in **AWS CDK**, and deploy it via a **CodePipeline**
with smoke tests and rollback.

---

## Step 1 — Carry the cumulative state forward

`module-11/relay/` is the **byte-identical** Module 10 `relay/` package, plus the new
`relay/api/` subpackage. Confirm the agent still answers locally before wrapping it:

```bash
export AWS_PROFILE=aws-genai-pro          # us-east-1 everywhere; no keys in code/.env
uv sync                                   # adds aws-cdk-lib v2 (+ constructs)
uv run python setup.py                    # upstream: tables/KB/guardrail/MCP/IAM (M5–M10)
uv run python -m relay.run "How do refunds work?"   # the agent still runs locally
```

`setup.py` is now **upstream-only** (decision **B6**): it creates the resources the front
door *references* (the DynamoDB tables, the Knowledge Base, the guardrail, the MCP Lambda,
the IAM roles), but **not** the API/queue/bus — those move to AWS CDK.

## Step 2 — `relay/api/`: the four Lambda handlers

The new subpackage wraps the unchanged agent:

- `post_handler` validates the `Ticket`, generates a `ticket_id`, writes
  `TicketRecord{status:"received"}` to `relay-tickets`, **enqueues** the job on SQS, and
  returns **202** `{ticket_id}`.
- `get_handler` reads the `TicketRecord` back (`GET /tickets/{id}`).
- `approve_handler` realizes the **M8 HITL gate** over HTTP (`POST /tickets/{id}/approve`
  body `{approved: bool}`) by calling `relay.approve.approve`.
- `worker_handler` consumes SQS, invokes the deployed agent through the **frozen**
  `relay.run.run_relay` contract, updates the `TicketRecord` to its final status, and
  publishes escalation / approval-required events.

No model ID lives here; generation stays in `relay.llm.converse()` via `relay.run`.

## Step 3 — Request validation

`POST /tickets` is guarded twice (skill 2.4.1): API Gateway runs a **JSON-Schema request
model** (`POST_TICKETS_REQUEST_SCHEMA` in the CDK stack) that rejects a malformed payload
**before** a Lambda cold-starts, and `post_handler.validate_ticket` enforces the business
contract (a non-empty `customer_message`, a valid `channel`) and returns a **clean 400** on
bad input — never a stack trace.

## Step 4 — Escalations on EventBridge `relay-events`

When the agent **escalates**, the worker publishes a `relay.escalation` event; when a refund
is **awaiting approval**, it publishes `relay.approval_required`. An EventBridge **rule**
(in the CDK stack) routes `relay.escalation` to a demo "human queue" SQS sink. That is
**loose coupling**: Relay does not know CloudCart's human queue — it publishes and moves on;
a new consumer is a new **rule**, not a code change.

## Step 5 — `cdk/`: the stack in AWS CDK

`cdk/relay_cdk/api_stack.py` describes API Gateway + the four Lambda + the SQS queue (+ DLQ)
+ the `relay-events` bus + the IAM grants (reusing the M10 canonical ARNs). The DynamoDB
tables and the KB stay owned by `setup.py` — the stack **references** them, never recreates
them. The `cdk` CLI is installed outside the Python deps (`npm install -g aws-cdk`).

```bash
uv run cdk deploy RelayApiStack           # prints ApiUrl + QueueUrl outputs
API=$(aws cloudformation describe-stacks --stack-name RelayApiStack \
  --query "Stacks[0].Outputs[?OutputKey=='ApiUrl'].OutputValue" --output text)

curl -X POST "${API}tickets" -d @data/tickets/sample.json
# -> 202 {"ticket_id": "ticket-XXXXXXXX", "status": "received"}
sleep 8
curl "${API}tickets/ticket-XXXXXXXX"
# -> the TicketRecord with status answered | escalated | awaiting_approval
```

The deployed API **is** an OpenAPI contract (skill 2.5.2) — accessible interfaces are not a
separate build, they fall out of the stack. Export the live REST API as an OpenAPI 3.0
(`oas30`) document and commit it as `cdk/openapi.json`, so any **Amplify** / OpenAPI client
generates a typed SDK against the real contract:

```bash
ID=$(aws cloudformation describe-stacks --stack-name RelayApiStack \
  --query "Stacks[0].Outputs[?OutputKey=='ApiId'].OutputValue" --output text)
aws apigateway get-export --rest-api-id "$ID" --stage-name prod \
  --export-type oas30 --accepts application/json cdk/openapi.json
```

The committed `cdk/openapi.json` is **this exported file**, not a hand-written parallel doc.
At Module 11 it carries exactly the three frozen routes (`POST /tickets`,
`GET /tickets/{ticket_id}`, `POST /tickets/{ticket_id}/approve`); `POST /tickets/{id}/feedback`
is added in Module 13. The smoke suite asserts it is valid `oas30` and matches the CDK route
spec field-for-field, so a drift between the deployed API and the committed contract fails CI.

## Step 6 — `pipeline/`: the CodePipeline (CI/CD)

`cdk/relay_cdk/pipeline_stack.py` describes a CodePipeline:
`source → build (uv sync + offline tests + a pip-audit security scan) → deploy
(cdk deploy) → smoke (curl POST/GET against the deployed API) → rollback on failure`. A
failed smoke stage stops the pipeline before promotion; CloudFormation rolls back a failed
deploy. The **eval-gate** stage is **commented** — it's added in Module 13.

```bash
uv run cdk deploy RelayPipelineStack \
  -c repo_owner=<you> -c repo_name=aws-genai-pro-labs \
  -c connection_arn=arn:aws:codestar-connections:us-east-1:<acct>:connection/<id>
git push        # -> the pipeline runs; a green smoke stage promotes the deploy
```

Run the smoke check by hand any time:

```bash
uv run python pipeline/smoke_test_live.py "$API"
# POSTs sample.json, polls GET until a terminal status, exits non-zero on failure.
```

## Step 7 — Try it yourself

1. **Outbound webhook on escalation.** Add an EventBridge rule that routes
   `relay.escalation` to a small Lambda which `POST`s to an external URL (a CloudCart
   webhook). Loose coupling: the worker doesn't change — you add a rule + a target.
2. **Cognito authorizer on `POST /tickets`.** Add an Amazon Cognito user pool + a JWT
   authorizer on the route, then call the API with a bearer token. This is **consumer auth**
   (M11) — distinct from the **internal** IAM of the components (M10).

## Run the tests

```bash
uv run pytest                              # offline cumulative suite (Modules 2–11)
RELAY_LIVE_TESTS=1 RELAY_API_URL="$API" uv run pytest -m live   # opt-in, capped (see budget)
```

The offline suite drives the four handlers with a `moto` DynamoDB/SQS backend and a
scripted agent (no Bedrock, no network): `post_handler` writes `received` + enqueues,
`worker_handler` produces a valid `TicketRecord` and publishes the right event,
`get_handler` reads it back, and `approve_handler` realizes the HITL gate. The CDK wiring is
asserted from dependency-light spec constants, so the suite runs even without `aws-cdk-lib`
installed. The `live` marker runs **one** real `POST`→poll-`GET` round-trip against a
deployed API (the worker runs one agent loop, < $0.02), skipping cleanly when `RELAY_API_URL`
is unset.

## Tear it down

```bash
uv run python teardown.py                  # cdk destroy both stacks + delete the pipeline
                                           #   + the M7–M10 cleanup; KEEP tables + KB
uv run python teardown.py --delete-tables  # ALSO drop relay-orders + relay-tickets
uv run python teardown.py --keep-stacks    # SKIP the CDK destroy (infra already gone)
```

`teardown.py` is idempotent and **tested** (B5): it `cdk destroy`s `RelayApiStack` +
`RelayPipelineStack` and **deletes the CodePipeline** (with a boto3 sweep fallback when the
`cdk` CLI is absent — it deletes the pipeline, the SQS queue + DLQ, and the `relay-events`
bus directly), so **nothing idle-billed survives**.
