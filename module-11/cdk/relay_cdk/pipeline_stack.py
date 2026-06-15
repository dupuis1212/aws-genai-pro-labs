"""cdk/relay_cdk/pipeline_stack.py — the CI/CD pipeline for Relay, in AWS CDK (Module 11).

RelayPipelineStack describes a CodePipeline that re-deploys Relay on every commit (skill
2.3.5):

    Source (the repo) ->
    Build  (CodeBuild: uv sync + the OFFLINE smoke tests + a security scan) ->
    Deploy (cdk deploy RelayApiStack) ->
    Smoke  (CodeBuild: curl POST /tickets + poll GET against the DEPLOYED API) ->
    [eval-gate]  # ADDED IN MODULE 13 — left commented below.

ROLLBACK is automatic: a CloudFormation deploy that fails rolls the stack back to the last
good state, and a FAILED smoke stage stops the pipeline BEFORE any promotion — so a commit
that breaks the API never reaches a healthy "deployed" state without the smoke tests
passing first. Security scans run IN the build (the brief's 2.3.5 "security scans in the
build" line).

A running CodePipeline is the ONE M11 resource with a real idle cost (~$1/active
pipeline/month, as of June 2026), so teardown.py DELETES it (B5) — `cdk destroy
RelayPipelineStack`. The buildspec / smoke commands live in pipeline/ (buildspec.yml,
smoke_buildspec.yml) so CodeBuild and the lab run the SAME steps.

The wiring is exposed as dependency-light SPEC constants (PIPELINE_STAGES, the buildspec
paths) that the smoke test asserts, so it verifies the source->build->deploy->smoke order
and the commented eval-gate WITHOUT an aws-cdk-lib install. The deployable Stack is built
only when the CDK app synthesizes it.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from relay import config  # noqa: E402

# The ORDER of pipeline stages (source -> build -> deploy -> smoke -> rollback): you
# smoke-test the DEPLOYED API, so Smoke runs AFTER Deploy, and a failed smoke stage stops
# promotion (CloudFormation rolls the failed deploy back). The eval-gate is NOT in this
# list — it is the Module 13 addition (a stage
# inserted after Deploy / before promotion). The smoke test asserts this exact order AND
# that "eval-gate" is absent at M11.
PIPELINE_STAGES: tuple[str, ...] = ("Source", "Build", "Deploy", "Smoke")

# The stage Module 13 will INSERT (named here only so the smoke test can assert it is NOT
# wired yet — the forward-dependency boundary). M11 builds the pipeline + smoke tests; M13
# branches the eval-gate onto it.
EVAL_GATE_STAGE_M13 = "EvalGate"

# The buildspec files CodeBuild runs (kept in pipeline/, so the pipeline and a local run
# share the SAME steps). build = offline tests + security scan; smoke = curl the deployed API.
BUILD_BUILDSPEC = "pipeline/buildspec.yml"
SMOKE_BUILDSPEC = "pipeline/smoke_buildspec.yml"


def build_pipeline_stack(scope, construct_id: str = config.RELAY_PIPELINE_STACK_NAME,
                         *, repo_owner: str = "OWNER", repo_name: str = "aws-genai-pro-labs",
                         branch: str = "main", connection_arn: str | None = None,
                         **kwargs):
    """Construct RelayPipelineStack. Imports aws-cdk-lib LAZILY (offline-safe smoke test).

    `repo_owner`/`repo_name`/`branch`/`connection_arn` configure the source. In the lab you
    pass your fork + a CodeStar (GitHub) connection ARN; the defaults are placeholders the
    `cdk deploy` overrides via context (-c repo_owner=...). The smoke test never calls this.
    """
    import aws_cdk as cdk
    from aws_cdk import (
        Stack,
        aws_codebuild as codebuild,
        aws_codepipeline as codepipeline,
        aws_codepipeline_actions as cpactions,
    )
    from constructs import Construct

    class RelayPipelineStack(Stack):
        """The CodePipeline that re-deploys Relay on every commit (Source->Build->Deploy->Smoke)."""

        def __init__(self, scope: Construct, cid: str, **kw) -> None:
            super().__init__(scope, cid, **kw)

            source_output = codepipeline.Artifact("SourceOut")
            build_output = codepipeline.Artifact("BuildOut")

            # SOURCE — the repo (a CodeStar/GitHub connection). On commit, the pipeline runs.
            source_action = cpactions.CodeStarConnectionsSourceAction(
                action_name="Source",
                owner=repo_owner, repo=repo_name, branch=branch,
                connection_arn=connection_arn or "REPLACE_WITH_CONNECTION_ARN",
                output=source_output,
            )

            # BUILD — uv sync + the OFFLINE smoke tests + a security scan (buildspec).
            build_project = codebuild.PipelineProject(
                self, "BuildProject", project_name="relay-build",
                build_spec=codebuild.BuildSpec.from_source_filename(BUILD_BUILDSPEC),
                environment=codebuild.BuildEnvironment(
                    build_image=codebuild.LinuxBuildImage.STANDARD_7_0),
            )
            build_action = cpactions.CodeBuildAction(
                action_name="Build", project=build_project,
                input=source_output, outputs=[build_output],
            )

            # DEPLOY — cdk deploy RelayApiStack. CloudFormation rolls back automatically on
            # a failed deploy (the article's rollback story). Run as a CodeBuild step that
            # invokes the CDK CLI (kept simple for the lab; a real pipeline uses a CDK
            # Pipelines self-mutating stage).
            deploy_project = codebuild.PipelineProject(
                self, "DeployProject", project_name="relay-deploy",
                build_spec=codebuild.BuildSpec.from_object({
                    "version": "0.2",
                    "phases": {
                        "install": {"commands": [
                            "npm install -g aws-cdk", "pip install uv", "uv sync"]},
                        "build": {"commands": [
                            f"uv run cdk deploy {config.RELAY_STACK_NAME} "
                            "--require-approval never"]},
                    },
                }),
                environment=codebuild.BuildEnvironment(
                    build_image=codebuild.LinuxBuildImage.STANDARD_7_0),
            )
            deploy_action = cpactions.CodeBuildAction(
                action_name="Deploy", project=deploy_project, input=source_output)

            # SMOKE — curl POST /tickets + poll GET against the DEPLOYED API. A FAILED smoke
            # stage stops the pipeline before any further promotion (rollback gate).
            smoke_project = codebuild.PipelineProject(
                self, "SmokeProject", project_name="relay-smoke",
                build_spec=codebuild.BuildSpec.from_source_filename(SMOKE_BUILDSPEC),
                environment=codebuild.BuildEnvironment(
                    build_image=codebuild.LinuxBuildImage.STANDARD_7_0),
            )
            smoke_action = cpactions.CodeBuildAction(
                action_name="Smoke", project=smoke_project, input=source_output)

            pipeline = codepipeline.Pipeline(
                self, "Pipeline", pipeline_name=config.RELAY_PIPELINE_NAME,
                stages=[
                    codepipeline.StageProps(stage_name="Source", actions=[source_action]),
                    codepipeline.StageProps(stage_name="Build", actions=[build_action]),
                    codepipeline.StageProps(stage_name="Deploy", actions=[deploy_action]),
                    codepipeline.StageProps(stage_name="Smoke", actions=[smoke_action]),
                    # ----------------------------------------------------------------
                    # ADDED IN MODULE 13 — the eval-gate stage. After Deploy + Smoke, run
                    # the golden-set evals against the deployed API and BLOCK promotion if
                    # grounding regresses (aggregate.grounding < 0.8 or >5 pts vs baseline,
                    # the same 0.8 constant relay.config.GROUNDING_THRESHOLD defines once).
                    # Module 11 builds the pipeline + smoke tests; Module 13 branches the
                    # eval-gate onto it. Left commented here on purpose (no forward dep).
                    #
                    # codepipeline.StageProps(stage_name="EvalGate", actions=[eval_action]),
                    # ----------------------------------------------------------------
                ],
            )

            self.pipeline_name = pipeline.pipeline_name
            cdk.CfnOutput(self, "PipelineName", value=pipeline.pipeline_name)

    return RelayPipelineStack(scope, construct_id, **kwargs)
