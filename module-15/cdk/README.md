# `cdk/` — Relay's front-door infrastructure in AWS CDK (Module 11)

From Module 11 on, Relay's **new** infrastructure is described in **AWS CDK** (`aws-cdk-lib`
v2) instead of imperative `boto3` scripts (decision **B6**). The upstream resources (the
DynamoDB tables, the Knowledge Base, the guardrail) stay managed by `setup.py`; this stack
**references** them by their canonical names (`06 §2`) and never recreates them.

## Two stacks

| Stack | What it creates |
|---|---|
| `RelayApiStack` | API Gateway REST API (`POST /tickets`, `GET /tickets/{ticket_id}`, `POST /tickets/{ticket_id}/approve`, **`POST /tickets/{ticket_id}/feedback`** *(Module 13)*), the five Lambda handlers (`post`/`get`/`approve`/`worker`/**`feedback`**), the SQS work queue + DLQ, and the `relay-events` EventBridge bus with a demo escalation rule. |
| `RelayPipelineStack` | The CodePipeline (`source → build → deploy → smoke → eval-gate`). Module 13 **wired the eval-gate** stage (golden-set regression: block on `aggregate.grounding < 0.8` or a >5-pt drop vs the committed baseline) after smoke, before promotion. |

## Deploy

```bash
# from module-11/ (so uv picks up the project + the lock):
uv run cdk deploy RelayApiStack            # the API + queue + bus
uv run cdk deploy RelayPipelineStack \      # the CI/CD pipeline (needs a source connection)
    -c repo_owner=<you> -c repo_name=aws-genai-pro-labs \
    -c connection_arn=arn:aws:codestar-connections:us-east-1:<acct>:connection/<id>
uv run cdk deploy --all
```

`RelayApiStack` needs no context — region is pinned to `us-east-1` (B8), the account is
resolved from the active `AWS_PROFILE`. Run `setup.py` first so the upstream tables / KB /
guardrail / MCP Lambda exist (the agent the worker invokes needs them).

The stack prints the API URL and the queue URL as CloudFormation outputs:

```bash
curl -X POST https://<api-id>.execute-api.us-east-1.amazonaws.com/prod/tickets \
  -d @data/tickets/sample.json
# -> 202 {"ticket_id": "ticket-XXXXXXXX", "status": "received"}
curl https://<api-id>.execute-api.us-east-1.amazonaws.com/prod/tickets/ticket-XXXXXXXX
# -> the TicketRecord (status answered | escalated | awaiting_approval)
```

## What it does NOT do (B5 / brief §9)

No **provisioned throughput**, no **SageMaker AI endpoint**, no **NAT gateway**, no VPC —
everything is on-demand / pay-per-use, so the only idle-billed M11 resource is the
CodePipeline (~$1/active pipeline/month), which `teardown.py` destroys.

## Tear it down

```bash
uv run python teardown.py     # cdk destroy both stacks + the upstream M7-M10 cleanup
# or just the infra:
uv run cdk destroy --all
```
