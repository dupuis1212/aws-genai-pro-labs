#!/usr/bin/env python3
"""cdk/app.py — the AWS CDK app entrypoint for Relay's front door (Module 11).

`cdk deploy` (run from the cdk/ directory, or with `--app "python cdk/app.py"`) synthesizes
two stacks:

    uv run cdk deploy RelayApiStack        # the API + 4 Lambda + SQS + relay-events
    uv run cdk deploy RelayPipelineStack   # the CodePipeline (build->smoke->deploy)
    uv run cdk deploy --all                # both

The pipeline source defaults to placeholders; pass your fork + a CodeStar connection ARN
via context when you deploy the pipeline:

    uv run cdk deploy RelayPipelineStack \\
        -c repo_owner=<you> -c repo_name=aws-genai-pro-labs \\
        -c connection_arn=arn:aws:codestar-connections:...

Everything else (the API stack) needs no context — it references the upstream tables / KB /
guardrail by their canonical names (06 §2). Region is pinned to us-east-1 (B8) and the
account is resolved from the CDK environment (the active AWS_PROFILE), never hard-coded.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Make `relay` and `relay_cdk` importable when run by the cdk CLI from any cwd.
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
for p in (_REPO_ROOT, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import aws_cdk as cdk  # noqa: E402

from relay import config  # noqa: E402
from relay_cdk import api_stack, pipeline_stack  # noqa: E402


def main() -> None:
    app = cdk.App()
    # The deploy environment: us-east-1 (B8), account from the active session/profile.
    env = cdk.Environment(
        account=os.environ.get("CDK_DEFAULT_ACCOUNT"),
        region=config.REGION,
    )

    api_stack.build_api_stack(app, config.RELAY_STACK_NAME, env=env)

    # The pipeline source comes from CDK context (-c repo_owner=... etc.), with placeholders
    # so `cdk synth RelayApiStack` works with no extra flags. Deploy the pipeline only when
    # you have wired a CodeStar connection.
    pipeline_stack.build_pipeline_stack(
        app, config.RELAY_PIPELINE_STACK_NAME, env=env,
        repo_owner=app.node.try_get_context("repo_owner") or "OWNER",
        repo_name=app.node.try_get_context("repo_name") or "aws-genai-pro-labs",
        branch=app.node.try_get_context("branch") or "main",
        connection_arn=app.node.try_get_context("connection_arn"),
    )

    app.synth()


if __name__ == "__main__":
    main()
