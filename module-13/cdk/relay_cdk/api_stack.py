"""cdk/relay_cdk/api_stack.py — Relay's serverless front door, in AWS CDK (Module 11).

RelayApiStack describes the WHOLE front door declaratively (decision B6):

  API Gateway REST API
    POST /tickets                       -> post Lambda     (202 + enqueue)
    GET  /tickets/{ticket_id}           -> get  Lambda     (read TicketRecord)
    POST /tickets/{ticket_id}/approve   -> approve Lambda  (HITL refund gate)
    POST /tickets/{ticket_id}/feedback  -> feedback Lambda (Module 13 — feedback_rating)
  SQS work queue `relay-tickets-queue` (+ DLQ, redrive) -> worker Lambda (the agent)
  EventBridge bus `relay-events`        + a demo escalation RULE -> a human-escalation
                                          SQS queue (the loose-coupling target)

Frozen contracts reproduced field-for-field (06 §2 / bible §3.3): the API paths/bodies,
the queue + bus NAMES (from relay.config), and the upstream resource names this stack only
REFERENCES — relay-orders, relay-tickets, relay-kb, relay-guardrail, the data bucket. The
stack NEVER recreates those (they stay owned by setup.py / the upstream modules); it grants
each Lambda least-privilege access to exactly the ones it needs, reusing the M10
iam/policies ARNs by canonical name.

A request VALIDATION model on POST /tickets rejects a malformed payload at the edge
(skill 2.4.1) before a Lambda cold-starts. Lambda RESERVED concurrency + API Gateway
throttling are set as the FM auto-scaling lever (skill 4.2.5). The whole stack is
on-demand / pay-per-use — NO provisioned throughput, NO SageMaker endpoint, NO NAT
gateway, NO VPC (B5: nothing idle-billed; the agent reaches Bedrock/DynamoDB over the
public AWS endpoints, like every other module).

This module imports `aws_cdk` (aws-cdk-lib v2) lazily inside the class so the smoke test —
which asserts the stack's WIRING via the lightweight, dependency-light helpers below — runs
offline even when aws-cdk-lib is not installed. The deployable Stack is built only when
the CDK app actually synthesizes it.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `relay` importable when this file is run by the `cdk` CLI from the cdk/ dir.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from relay import config  # noqa: E402


# =============================================================================
# Dependency-light WIRING SPEC — the single source of truth the Stack reads AND
# the smoke test asserts against (no aws-cdk-lib import needed to check the wiring).
# =============================================================================
# The Lambda handlers: (logical id, handler dotted path, the HTTP/queue trigger).
# The handler path is `relay.api.<module>.lambda_handler` — the deployed code is the
# unchanged relay/ package (the agent is WRAPPED, never re-implemented). Module 13 ADDS the
# fifth handler, `feedback` (POST /tickets/{id}/feedback), by addition.
LAMBDA_HANDLERS: dict[str, str] = {
    "post": "relay.api.post_handler.lambda_handler",
    "get": "relay.api.get_handler.lambda_handler",
    "approve": "relay.api.approve_handler.lambda_handler",
    "worker": "relay.api.worker_handler.lambda_handler",
    "feedback": "relay.api.feedback_handler.lambda_handler",   # ADDED M13
}

# The API routes (method, path) -> which Lambda serves them. Reproduced FIELD-FOR-FIELD
# from 06 §2 — the smoke test diffs this against the frozen contract. Module 13 ADDS the
# fifth route, POST /tickets/{ticket_id}/feedback, by addition.
API_ROUTES: tuple[tuple[str, str, str], ...] = (
    ("POST", "/tickets", "post"),
    ("GET", "/tickets/{ticket_id}", "get"),
    ("POST", "/tickets/{ticket_id}/approve", "approve"),
    ("POST", "/tickets/{ticket_id}/feedback", "feedback"),   # ADDED M13
)

# The JSON-Schema request model API Gateway validates POST /tickets against (the cheap
# structural gate before a Lambda runs — skill 2.4.1). customer_message is required and
# non-empty; channel, when present, must be one of the frozen Literals. additionalProperties
# is allowed (the optional triage_intent/customer_id/session_id routing hints flow through).
POST_TICKETS_REQUEST_SCHEMA: dict = {
    "type": "object",
    "required": ["customer_message"],
    "properties": {
        "customer_message": {"type": "string", "minLength": 1},
        "channel": {"type": "string", "enum": ["email", "chat"]},
        "ticket_id": {"type": "string"},
        "triage_intent": {"type": "string"},
        "customer_id": {"type": "string"},
        "session_id": {"type": "string"},
    },
}

# The demo target the relay.escalation rule routes to: a "human queue" SQS queue (the
# loose-coupling sink CloudCart would replace with its real ticketing inbox).
HUMAN_ESCALATION_QUEUE_NAME = "relay-human-escalation"

# API Gateway throttling + Lambda reserved concurrency — the FM auto-scaling lever
# (skill 4.2.5). Modest caps so a traffic burst is shaped, not unbounded (and the lab stays
# cheap). Tunable in the "Try it yourself".
API_THROTTLE_RATE_LIMIT = 20      # steady-state requests/sec across the stage
API_THROTTLE_BURST_LIMIT = 40     # burst bucket
WORKER_RESERVED_CONCURRENCY = 5   # cap concurrent agent runs (cost + downstream pressure)


def upstream_table_arns(account: str, region: str = config.REGION) -> dict[str, str]:
    """The ARNs of the UPSTREAM DynamoDB tables this stack references (never creates).

    Built from the canonical names (06 §2) — relay-orders / relay-tickets — so the CDK
    grants reference EXACTLY the same resources the M10 iam/policies/*.json name. Returned
    for the smoke test to assert zero drift between the CDK grants and the M10 ARNs.
    """
    return {
        config.RELAY_ORDERS_TABLE:
            f"arn:aws:dynamodb:{region}:{account}:table/{config.RELAY_ORDERS_TABLE}",
        config.RELAY_TICKETS_TABLE:
            f"arn:aws:dynamodb:{region}:{account}:table/{config.RELAY_TICKETS_TABLE}",
    }


# =============================================================================
# The deployable CDK Stack — built only when the app synthesizes (lazy CDK import).
# =============================================================================
def build_api_stack(scope, construct_id: str = config.RELAY_STACK_NAME, **kwargs):
    """Construct RelayApiStack. Imports aws-cdk-lib LAZILY so the smoke test stays offline.

    Returns the synthesized Stack. The CDK app (cdk/app.py) calls this; the smoke test does
    NOT (it asserts the WIRING SPEC constants above, which need no CDK install).
    """
    import aws_cdk as cdk
    from aws_cdk import (
        Duration,
        RemovalPolicy,
        Stack,
        aws_apigateway as apigw,
        aws_events as events,
        aws_events_targets as targets,
        aws_iam as iam,
        aws_lambda as lambda_,
        aws_lambda_event_sources as event_sources,
        aws_sqs as sqs,
    )
    from constructs import Construct

    class RelayApiStack(Stack):
        """Relay's serverless front door (API Gateway + 4 Lambda + SQS + relay-events)."""

        def __init__(self, scope: Construct, cid: str, **kw) -> None:
            super().__init__(scope, cid, **kw)
            account, region = self.account, self.region

            # --- The async work queue + its dead-letter queue (redrive) --------------
            dlq = sqs.Queue(
                self, "WorkDLQ", queue_name=config.RELAY_DLQ_NAME,
                retention_period=Duration.days(14),
            )
            queue = sqs.Queue(
                self, "WorkQueue", queue_name=config.RELAY_QUEUE_NAME,
                visibility_timeout=Duration.seconds(
                    config.RELAY_QUEUE_VISIBILITY_TIMEOUT_S),
                dead_letter_queue=sqs.DeadLetterQueue(
                    max_receive_count=config.RELAY_QUEUE_MAX_RECEIVE, queue=dlq),
            )

            # --- The relay-events bus + a demo escalation target ---------------------
            bus = events.EventBus(self, "RelayEvents",
                                  event_bus_name=config.RELAY_EVENT_BUS_NAME)
            human_queue = sqs.Queue(self, "HumanEscalationQueue",
                                    queue_name=HUMAN_ESCALATION_QUEUE_NAME)
            # A RULE routing relay.escalation events to the human queue — the loose-
            # coupling sink. CloudCart swaps this target for its real ticketing inbox; the
            # publisher (the worker) does not change. relay.approval_required is published
            # too; an approval-inbox rule is a Try-it-yourself (a second rule, not a code
            # change).
            events.Rule(
                self, "EscalationRule", event_bus=bus,
                rule_name="relay-escalation-to-human",
                event_pattern=events.EventPattern(
                    source=[config.RELAY_EVENT_SOURCE],
                    detail_type=[config.RELAY_DETAIL_ESCALATION],
                ),
                targets=[targets.SqsQueue(human_queue)],
            )

            # --- The deployment code asset (the unchanged relay/ + mcp_server/) ------
            # The deployed handlers ARE the unchanged relay package. The asset bundles
            # the repo SOURCE (relay/, mcp_server/) and the runtime deps (pydantic,
            # strands-agents, mcp, bedrock-agentcore — boto3 is already in the Lambda
            # runtime, aws-cdk-lib is synth-only) into /asset-output, via the official
            # Lambda Python 3.12 build image. The `exclude` keeps the local .venv /
            # cdk.out / caches OUT of the bundle (otherwise CDK recurses into cdk.out and
            # the multi-hundred-MB .venv — the lab-machine asset blow-up).
            _RUNTIME_DEPS = (
                "pydantic~=2.0 strands-agents~=1.43 mcp~=1.27 bedrock-agentcore~=1.14"
            )
            code = lambda_.Code.from_asset(
                str(_REPO_ROOT),
                exclude=[
                    ".venv", "cdk.out", "cdk/cdk.out", "**/__pycache__", "**/*.pyc",
                    ".git", ".pytest_cache", "node_modules", "tests", "data/raw",
                    "*.zip", "assets",
                    # The deployed worker must NOT carry the laptop-written .mcp_url: in
                    # THIS account an org policy blocks public Lambda Function URLs (AuthType
                    # NONE), so a bundled URL would make MCP init 403 and crash every run.
                    # Excluded, the worker resolves the MCP URL from an env var if set, else
                    # degrades to a tool-light run (the lab's designed fallback — a doc/refund
                    # ticket still completes). The guardrail / memory markers STAY so the
                    # worker keeps the M9 guardrail + M8 memory. decision_log.jsonl is
                    # read-only in /var/task, so drop it (best-effort write skips cleanly).
                    ".mcp_url", "decision_log.jsonl", "*.bak",
                ],
                bundling=cdk.BundlingOptions(
                    image=lambda_.Runtime.PYTHON_3_12.bundling_image,
                    # Pin the wheels to the Lambda's x86_64/manylinux2014 platform so the
                    # native extension packages (pydantic_core) match the function arch —
                    # otherwise an Apple-Silicon build machine bundles arm64 wheels and the
                    # x86_64 Lambda fails with "No module named pydantic_core._pydantic_core".
                    command=[
                        "bash", "-c",
                        "pip install --no-cache-dir -t /asset-output "
                        "--platform manylinux2014_x86_64 --implementation cp "
                        "--python-version 3.12 --only-binary=:all: --upgrade "
                        + _RUNTIME_DEPS
                        + " && cp -r relay mcp_server /asset-output/",
                    ],
                ),
            )

            def make_fn(name: str, *, timeout_s: int, reserved=None) -> "lambda_.Function":
                fn = lambda_.Function(
                    self, f"{name.capitalize()}Fn",
                    function_name=f"relay-api-{name}",
                    runtime=lambda_.Runtime.PYTHON_3_12,
                    handler=LAMBDA_HANDLERS[name],
                    code=code,
                    timeout=Duration.seconds(timeout_s),
                    memory_size=512,
                    environment={
                        config.RELAY_QUEUE_URL_ENV: queue.queue_url,
                        config.RELAY_EVENT_BUS_ENV: config.RELAY_EVENT_BUS_NAME,
                    },
                    reserved_concurrent_executions=reserved,
                )
                return fn

            post_fn = make_fn("post", timeout_s=15)
            get_fn = make_fn("get", timeout_s=15)
            approve_fn = make_fn("approve", timeout_s=30)
            worker_fn = make_fn("worker", timeout_s=config.RELAY_WORKER_TIMEOUT_S,
                                reserved=WORKER_RESERVED_CONCURRENCY)
            feedback_fn = make_fn("feedback", timeout_s=15)   # ADDED M13

            # --- Least-privilege grants — reference upstream by canonical name --------
            # relay-tickets table (Table.from_table_name -> an L2 ref, no recreate).
            from aws_cdk import aws_dynamodb as dynamodb

            tickets = dynamodb.Table.from_table_name(
                self, "TicketsTable", config.RELAY_TICKETS_TABLE)
            orders = dynamodb.Table.from_table_name(
                self, "OrdersTable", config.RELAY_ORDERS_TABLE)

            # post/get/approve read+write tickets; the worker (via the agent) reads orders
            # and writes tickets. Each grant is exactly what that Lambda needs — no '*'.
            tickets.grant_read_write_data(post_fn)
            tickets.grant_read_data(get_fn)
            tickets.grant_read_write_data(approve_fn)
            orders.grant_read_write_data(approve_fn)   # approve executes the refund write
            tickets.grant_read_write_data(worker_fn)
            orders.grant_read_data(worker_fn)
            # feedback reads the record + writes back feedback_rating (no model, no orders).
            tickets.grant_read_write_data(feedback_fn)   # ADDED M13

            # post enqueues; the worker consumes; the worker publishes to the bus.
            queue.grant_send_messages(post_fn)
            queue.grant_consume_messages(worker_fn)
            bus.grant_put_events_to(worker_fn)

            # The worker runs the AGENT, which calls Bedrock (Converse) + the guardrail.
            # Reference the guardrail by canonical name; scope Converse to the inference
            # profiles + foundation models (the model IDs come from relay.config, not here).
            worker_fn.add_to_role_policy(iam.PolicyStatement(
                sid="InvokeConverseAndGuardrail",
                actions=["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream",
                         "bedrock:Converse", "bedrock:ConverseStream",
                         "bedrock:ApplyGuardrail", "bedrock:Retrieve",
                         "bedrock:RetrieveAndGenerate"],
                resources=[
                    f"arn:aws:bedrock:{region}:{account}:inference-profile/*",
                    # A `us.` cross-Region inference profile FANS OUT the actual model
                    # invocation to the region members (us-east-1 / us-east-2 / us-west-2),
                    # so the foundation-model ARN must allow ALL regions, not just the
                    # calling one — otherwise ConverseStream is AccessDenied on the
                    # us-east-2 (or us-west-2) member (the real M3 inference-profile rule).
                    "arn:aws:bedrock:*::foundation-model/*",
                    f"arn:aws:bedrock:{region}:{account}:guardrail/*",
                    f"arn:aws:bedrock:{region}:{account}:knowledge-base/*",
                ],
            ))

            # --- The API Gateway REST API + a request-validation model ---------------
            api = apigw.RestApi(
                self, "RelayApi", rest_api_name="relay-api",
                deploy_options=apigw.StageOptions(
                    stage_name=config.RELAY_API_STAGE,
                    throttling_rate_limit=API_THROTTLE_RATE_LIMIT,
                    throttling_burst_limit=API_THROTTLE_BURST_LIMIT,
                ),
            )
            post_model = api.add_model(
                "PostTicketModel",
                content_type="application/json",
                model_name="PostTicket",
                schema=apigw.JsonSchema(**_to_cdk_schema(POST_TICKETS_REQUEST_SCHEMA)),
            )
            body_validator = api.add_request_validator(
                "BodyValidator", validate_request_body=True)

            tickets_res = api.root.add_resource("tickets")
            tickets_res.add_method(
                "POST", apigw.LambdaIntegration(post_fn),
                request_models={"application/json": post_model},
                request_validator=body_validator,
            )
            ticket_id_res = tickets_res.add_resource("{ticket_id}")
            ticket_id_res.add_method("GET", apigw.LambdaIntegration(get_fn))
            approve_res = ticket_id_res.add_resource("approve")
            approve_res.add_method("POST", apigw.LambdaIntegration(approve_fn))
            # Module 13: POST /tickets/{ticket_id}/feedback -> the feedback Lambda.
            feedback_res = ticket_id_res.add_resource("feedback")
            feedback_res.add_method("POST", apigw.LambdaIntegration(feedback_fn))

            # --- Wire the worker Lambda to the SQS queue (batch size 1 -> per-ticket) -
            worker_fn.add_event_source(event_sources.SqsEventSource(
                queue, batch_size=1))

            # The human-escalation queue is a demo sink; allow CloudFormation to clean it
            # up on stack destroy (it carries no durable state worth keeping).
            human_queue.apply_removal_policy(RemovalPolicy.DESTROY)

            # NOTE: the eval-gate stage is added to the PIPELINE in Module 13 — see
            # pipeline_stack.py (commented stage). Nothing eval-related is wired here.

            self.api_url = api.url
            self.queue_url = queue.queue_url
            self.event_bus_name = bus.event_bus_name

            cdk.CfnOutput(self, "ApiUrl", value=api.url)
            cdk.CfnOutput(self, "QueueUrl", value=queue.queue_url)

    app_scope = scope
    return RelayApiStack(app_scope, construct_id, **kwargs)


def _to_cdk_schema(schema: dict) -> dict:
    """Translate a plain JSON-Schema dict into apigateway.JsonSchema kwargs.

    aws-cdk-lib expects JsonSchema(type=JsonSchemaType.OBJECT, properties={...}, ...). We
    map our small dialect (object/string/enum/required/minLength) onto it. Imported lazily
    so this helper is only exercised when CDK is installed (inside build_api_stack)."""
    from aws_cdk import aws_apigateway as apigw

    type_map = {
        "object": apigw.JsonSchemaType.OBJECT,
        "string": apigw.JsonSchemaType.STRING,
        "integer": apigw.JsonSchemaType.INTEGER,
        "boolean": apigw.JsonSchemaType.BOOLEAN,
    }
    out: dict = {"type": type_map[schema["type"]]}
    if "required" in schema:
        out["required"] = list(schema["required"])
    if "enum" in schema:
        out["enum"] = list(schema["enum"])
    if "minLength" in schema:
        out["min_length"] = schema["minLength"]
    if "properties" in schema:
        out["properties"] = {
            k: apigw.JsonSchema(**_to_cdk_schema(v))
            for k, v in schema["properties"].items()
        }
    return out
