"""relay_cdk — the AWS CDK app that describes Relay's serverless front door (Module 11).

Module 11 transitions Relay's infrastructure from imperative boto3 scripts (setup.py,
M1-M10) to DECLARATIVE AWS CDK (decision B6). Two stacks:

  - api_stack.RelayApiStack      : the front door — Amazon API Gateway REST API
                                   (POST /tickets, GET /tickets/{id},
                                   POST /tickets/{id}/approve), the four AWS Lambda
                                   handlers (post / get / approve / worker), the Amazon
                                   SQS work queue + DLQ, and the `relay-events` Amazon
                                   EventBridge bus with a demo escalation rule. It
                                   REFERENCES the upstream tables / KB / guardrail by
                                   their canonical names (06 §2) — it never recreates them.
  - pipeline_stack.RelayPipelineStack : the CodePipeline (source -> build -> deploy ->
                                   smoke -> rollback) that re-deploys RelayApiStack on
                                   every commit. The eval-gate stage is left commented for
                                   Module 13.

Why CDK and not more boto3: declarative infra gives you a diff (`cdk diff`), an automatic
rollback on a failed deploy, and a single source of truth — the trade-off the article's
T5.5 makes. The upstream resources (the DynamoDB tables, the KB, the guardrail) stay
managed by setup.py for now; the M11 INFRA is what moves to CDK (B6).
"""

from __future__ import annotations

__all__ = ["api_stack", "pipeline_stack"]
