"""cdk/relay_cdk/pipeline_stack.py — the CI/CD pipeline for Relay, in AWS CDK (Module 11 +
Module 13 eval-gate).

RelayPipelineStack describes a CodePipeline that re-deploys Relay on every commit (skill
2.3.5), with the Module 13 EVAL-GATE wired in after Smoke:

    Source (the repo) ->
    Build  (CodeBuild: uv sync + the OFFLINE smoke tests + a security scan) ->
    Deploy (cdk deploy RelayApiStack) ->
    Smoke  (CodeBuild: curl POST /tickets + poll GET against the DEPLOYED API) ->
    EvalGate (CodeBuild: run the golden-set evals + the regression gate against the
              DEPLOYED API; a grounding regression BLOCKS promotion)   # ADDED M13

ROLLBACK is automatic: a CloudFormation deploy that fails rolls the stack back to the last
good state, and a FAILED smoke OR eval-gate stage stops the pipeline BEFORE any promotion —
so a commit that breaks the API (smoke) OR degrades answer quality (eval-gate) never reaches
customers. The eval-gate fails when aggregate grounding drops below 0.8
(config.EVAL_GROUNDING_FLOOR — the SAME 0.8 the M9 escalation and the M14 alarm use) or
regresses more than 5 pts vs the committed baseline. Security scans run IN the build (the
brief's 2.3.5 "security scans in the build" line).

A running CodePipeline is the ONE M11 resource with a real idle cost (~$1/active
pipeline/month, as of June 2026), so teardown.py DELETES it (B5) — `cdk destroy
RelayPipelineStack`. The buildspec / smoke / eval commands live in pipeline/ (buildspec.yml,
smoke_buildspec.yml, eval_buildspec.yml) so CodeBuild and the lab run the SAME steps.

The wiring is exposed as dependency-light SPEC constants (PIPELINE_STAGES, the buildspec
paths) that the smoke test asserts, so it verifies the source->build->deploy->smoke->eval-gate
order WITHOUT an aws-cdk-lib install. The deployable Stack is built only when the CDK app
synthesizes it.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from relay import config  # noqa: E402

# The ORDER of pipeline stages (source -> build -> deploy -> smoke -> EVAL-GATE): you
# smoke-test then EVAL the DEPLOYED API, so Smoke and EvalGate run AFTER Deploy, and a failed
# smoke OR eval-gate stage stops promotion (CloudFormation rolls the failed deploy back).
# Module 13 ADDED "EvalGate" after "Smoke" (the golden-set regression gate). The smoke test
# asserts this exact order.
PIPELINE_STAGES: tuple[str, ...] = ("Source", "Build", "Deploy", "Smoke", "EvalGate")

# The eval-gate stage name (now WIRED, Module 13). M11 built the pipeline + smoke tests; M13
# branches the eval-gate onto it — a stage inserted after Deploy + Smoke, before promotion.
EVAL_GATE_STAGE = "EvalGate"

# The buildspec files CodeBuild runs (kept in pipeline/, so the pipeline and a local run
# share the SAME steps). build = offline tests + security scan; smoke = curl the deployed API;
# eval = the golden-set regression gate against the deployed API (Module 13).
BUILD_BUILDSPEC = "pipeline/buildspec.yml"
SMOKE_BUILDSPEC = "pipeline/smoke_buildspec.yml"
EVAL_BUILDSPEC = "pipeline/eval_buildspec.yml"


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

            # EVAL-GATE (Module 13) — run the golden-set evals + the regression gate against
            # the DEPLOYED API. A grounding regression (aggregate grounding < 0.8 =
            # config.EVAL_GROUNDING_FLOOR, the SAME constant the M9 escalation + M14 alarm
            # use, OR a >5-pt drop vs the committed baseline) makes run_evals.py exit
            # non-zero, FAILS this stage, and BLOCKS promotion — so a quality regression never
            # reaches a customer, even when smoke (the API works) passes. M11 built the
            # pipeline + smoke tests; M13 branches the eval-gate onto it.
            eval_project = codebuild.PipelineProject(
                self, "EvalGateProject", project_name="relay-eval-gate",
                build_spec=codebuild.BuildSpec.from_source_filename(EVAL_BUILDSPEC),
                environment=codebuild.BuildEnvironment(
                    build_image=codebuild.LinuxBuildImage.STANDARD_7_0),
            )
            eval_action = cpactions.CodeBuildAction(
                action_name="EvalGate", project=eval_project, input=source_output)

            pipeline = codepipeline.Pipeline(
                self, "Pipeline", pipeline_name=config.RELAY_PIPELINE_NAME,
                stages=[
                    codepipeline.StageProps(stage_name="Source", actions=[source_action]),
                    codepipeline.StageProps(stage_name="Build", actions=[build_action]),
                    codepipeline.StageProps(stage_name="Deploy", actions=[deploy_action]),
                    codepipeline.StageProps(stage_name="Smoke", actions=[smoke_action]),
                    # ADDED IN MODULE 13 — the eval-gate stage (after Deploy + Smoke).
                    codepipeline.StageProps(stage_name="EvalGate", actions=[eval_action]),
                ],
            )

            self.pipeline_name = pipeline.pipeline_name
            cdk.CfnOutput(self, "PipelineName", value=pipeline.pipeline_name)

    return RelayPipelineStack(scope, construct_id, **kwargs)
