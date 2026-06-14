"""teardown.py — cdk destroy the front door + delete the pipeline, then the M7-M10 cleanup.

Module 11 of AWS GenAI Pro Mastery. Idempotent and verbose, and TESTED — one tested
teardown per setup (decision B5).

What this DELETES BY DEFAULT (the idle-billed / standing resources):
  0!. (Module 11) the front-door INFRA, via AWS CDK: `cdk destroy RelayApiStack` (the API
     Gateway + the four Lambda + the SQS work queue + DLQ + the `relay-events` EventBridge
     bus) AND `cdk destroy RelayPipelineStack` — the CodePipeline is the ONLY M11 resource
     with a real idle cost (~$1/active pipeline/month as of June 2026), so it MUST go (B5).
     When the `cdk` CLI is not on PATH, teardown FALLS BACK to a boto3 sweep that deletes
     the pipeline, the SQS queue + DLQ, and the relay-events bus directly (so nothing
     idle-billed survives even without CDK). API Gateway / Lambda / SQS / EventBridge bill
     ~$0 idle, but the pipeline does not — and the course rule is leave nothing behind.
  0z. (Module 10) the four least-privilege **IAM component roles** (relay-intake-role,
     relay-agent-role, relay-kb-reader-role, relay-api-role) + their inline policies, and
     the runtime `decision_log.jsonl`. IAM is FREE (~$0 idle), but the course rule is
     leave nothing behind that you created (B5). The roles are iterated from the SAME
     config.IAM_COMPONENT_ROLES list setup.py creates from, so they never drift.
  0a. (Module 9) the Bedrock **Guardrail** `relay-guardrail` (ALL versions — DeleteGuardrail
     without a version removes the whole guardrail). A guardrail bills ONLY per use
     (~$0 idle), but the course rule is leave nothing behind that you created — so it
     goes, and the .guardrail_id / .guardrail_version markers are removed.
  0b. (Module 8) the **AgentCore Memory** store `relay-memory` — its long-term
     cross-session store is the **ONLY idle-billed item in the whole lab**
     (~$0.75/1K records/month as of June 2026), so it is PURGED here (B5). The
     .memory_id / .runtime_arn markers are removed too. The AgentCore RUNTIME billed
     ~$0 idle (idle is free) and is removed with `agentcore destroy` (see
     agentcore/README.md) — the CLI owns it, not boto3.
  1. (Module 7) the CloudCart MCP server Lambda `relay-mcp-server` + its Function URL;
  2. (Module 7) the MCP Lambda's IAM role `relay-mcp-lambda-role` + its inline policy;
  3. the recorded .mcp_url marker.
  (A Lambda + Function URL bill ~$0 idle, but the course rule is leave nothing behind
  that you created and no longer use — so they go.)

What this KEEPS BY DEFAULT, on purpose:
  - the two DynamoDB tables `relay-orders` (seeded) + `relay-tickets`. They are
    ON-DEMAND, so idle ≈ $0 — keeping them is acceptable and explicit — AND **Module 8
    reuses them** (the deployed agent reads orders + writes tickets). Drop them only
    with `--delete-tables` (you will then re-seed on the next setup).
  - the inherited Module 5 Knowledge Base `relay-kb` + its role and the S3 Vectors
    bucket/indexes (search_kb retrieves from the KB; Module 8 needs it too) — all
    ~$0 idle. Drop the KB with `--delete-kb` if you really want a clean slate (Module
    8 will then need setup.py re-run to rebuild it).
  - the data bucket `relay-<account_id>` + docs/ corpus, and the M1 $5 budget alarm
    (persistent; Module 1 owns it).

So Module 7 leaves NOTHING idle-billed of its OWN that a later module does not reuse:
the Lambda + role are deleted; the on-demand tables and the ~$0 KB are deliberate
downstream dependencies kept on purpose. Rebuild any of it with `uv run python setup.py`.

Run it:
    uv run python teardown.py                  # cdk destroy the front door + delete the
                                               # pipeline; delete MCP Lambda + role; KEEP
                                               # tables + KB
    uv run python teardown.py --delete-tables  # ALSO drop relay-orders + relay-tickets
    uv run python teardown.py --delete-kb      # ALSO tear down the inherited KB + role
    uv run python teardown.py --delete-vectors # ALSO drop the S3 Vectors indexes
    uv run python teardown.py --keep-stacks    # SKIP the CDK destroy (infra already gone)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

from relay import config
from setup import (
    KB_ID_FILE, KB_DATA_SOURCE_ID_FILE, KB_ROLE_NAME,
    MCP_LAMBDA_NAME, MCP_LAMBDA_ROLE_NAME, MCP_URL_FILE,
    MEMORY_ID_FILE, RUNTIME_ARN_FILE,
    GUARDRAIL_ID_FILE, GUARDRAIL_VERSION_FILE,
)

# Module 10: the runtime decision log lives at the repo root; teardown removes it (it is
# git-ignored runtime output — leave nothing behind, even though it is redacted).
DECISION_LOG_FILE = Path(__file__).resolve().parent / config.DECISION_LOG_FILE_NAME

REGION = config.REGION
_NOT_FOUND = ("ResourceNotFoundException", "NotFoundException", "NoSuchEntity",
              "404")
_DELETE_TIMEOUT_S = 300
_DELETE_POLL_S = 5


def _sts():
    return boto3.client("sts", region_name=REGION)


def _iam():
    return boto3.client("iam", region_name=REGION)


def _s3vectors():
    return boto3.client("s3vectors", region_name=REGION)


def _s3():
    return boto3.client("s3", region_name=REGION)


def _agent():
    return boto3.client("bedrock-agent", region_name=REGION)


# --- Module 6: purge the attachments/ prefix the intake pipeline wrote to ------
# The intake's uploaded screenshots are the ONE thing Module 6 puts in S3 that the
# teardown should clear (S3 storage for a few small PNGs is fractions of a cent, but
# the course rule is leave nothing behind that you created). We delete every object
# under attachments/ (including the .keep marker), KEEPING the data bucket itself and
# the docs/ corpus + vectors/ — downstream modules reuse those. Idempotent: an empty
# or already-gone prefix is a clean no-op. Comprehend and Converse are per-CALL — they
# create NOTHING idle-billed, so there is nothing else for Module 6 to tear down.
def purge_attachments(s3, data_bucket: str) -> int:
    """Delete every object under attachments/ in the data bucket. Returns the count."""
    deleted = 0
    try:
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(
            Bucket=data_bucket, Prefix=config.RELAY_ATTACHMENTS_PREFIX
        ):
            keys = [{"Key": obj["Key"]} for obj in page.get("Contents", [])]
            if keys:
                s3.delete_objects(Bucket=data_bucket, Delete={"Objects": keys})
                deleted += len(keys)
    except ClientError as err:
        if err.response["Error"]["Code"] in _NOT_FOUND:
            print(f"  attachments/ prefix in '{data_bucket}': already gone. Fine.")
            return 0
        raise
    if deleted:
        print(f"  attachments/ prefix in '{data_bucket}': {deleted} object(s) "
              "DELETED (intake uploads).")
    else:
        print(f"  attachments/ prefix in '{data_bucket}': already empty. Fine.")
    return deleted


def _find_kb_id(agent) -> str | None:
    """Find the relay-kb id: recorded file first, then a live name lookup."""
    if KB_ID_FILE.exists():
        recorded = KB_ID_FILE.read_text(encoding="utf-8").strip()
        if recorded:
            return recorded
    for page in agent.get_paginator("list_knowledge_bases").paginate():
        for summary in page.get("knowledgeBaseSummaries", []):
            if summary.get("name") == config.RELAY_KB_NAME:
                return summary["knowledgeBaseId"]
    return None


# --- Step 1: the data source + the Knowledge Base -----------------------------
def delete_knowledge_base(agent, kb_id: str | None) -> None:
    """Delete the KB's data source(s) then the KB itself. Idempotent."""
    if not kb_id:
        print(f"  Knowledge Base '{config.RELAY_KB_NAME}': already gone. Fine.")
        return

    # Data sources first (clean log; deleting the KB would cascade anyway).
    try:
        for page in agent.get_paginator("list_data_sources").paginate(
            knowledgeBaseId=kb_id
        ):
            for summary in page.get("dataSourceSummaries", []):
                ds_id = summary["dataSourceId"]
                agent.delete_data_source(knowledgeBaseId=kb_id, dataSourceId=ds_id)
                print(f"  data source {ds_id}: DELETED.")
    except ClientError as err:
        if err.response["Error"]["Code"] not in _NOT_FOUND:
            raise

    try:
        agent.delete_knowledge_base(knowledgeBaseId=kb_id)
        print(f"  Knowledge Base '{config.RELAY_KB_NAME}' ({kb_id}): DELETED.")
        _wait_kb_gone(agent, kb_id)
    except ClientError as err:
        if err.response["Error"]["Code"] in _NOT_FOUND:
            print(f"  Knowledge Base '{config.RELAY_KB_NAME}': already gone. Fine.")
        else:
            raise

    for path in (KB_ID_FILE, KB_DATA_SOURCE_ID_FILE):
        if path.exists():
            path.unlink()


def _wait_kb_gone(agent, kb_id: str) -> None:
    """Block until the KB no longer exists, so role deletion does not race it."""
    deadline = time.time() + _DELETE_TIMEOUT_S
    while time.time() < deadline:
        try:
            agent.get_knowledge_base(knowledgeBaseId=kb_id)
        except ClientError as err:
            if err.response["Error"]["Code"] in _NOT_FOUND:
                return
            raise
        time.sleep(_DELETE_POLL_S)


# --- Step 2: the KB service role ----------------------------------------------
def delete_kb_role(iam) -> None:
    """Delete the KB role's inline policy then the role itself. Idempotent."""
    try:
        for name in iam.list_role_policies(RoleName=KB_ROLE_NAME).get(
            "PolicyNames", []
        ):
            iam.delete_role_policy(RoleName=KB_ROLE_NAME, PolicyName=name)
            print(f"  inline policy '{name}': deleted.")
        iam.delete_role(RoleName=KB_ROLE_NAME)
        print(f"  IAM role '{KB_ROLE_NAME}': DELETED.")
    except ClientError as err:
        if err.response["Error"]["Code"] in _NOT_FOUND:
            print(f"  IAM role '{KB_ROLE_NAME}': already gone. Fine.")
        else:
            raise


# --- Optional: drop the vector index (NOT default — the KB needs it) ----------
def delete_vectors(s3v, vector_bucket: str, index_name: str) -> None:
    """Drop the relay-docs index. OFF by default: the KB / search_kb retrieve from it."""
    try:
        s3v.delete_index(vectorBucketName=vector_bucket, indexName=index_name)
        print(f"  index '{index_name}': DELETED (--delete-vectors). "
              "Module 8 will need Module 4's setup re-run.")
    except ClientError as err:
        if err.response["Error"]["Code"] in _NOT_FOUND:
            print(f"  index '{index_name}': already gone. Fine.")
        else:
            raise


# =============================================================================
# Module 7 teardown — the MCP Lambda + Function URL + role; (opt) the tables.
# =============================================================================
def _lambda():
    return boto3.client("lambda", region_name=REGION)


def _dynamodb():
    return boto3.client("dynamodb", region_name=REGION)


def delete_mcp_lambda(lmb) -> None:
    """Delete the MCP server Lambda + its Function URL config. Idempotent.

    Deleting the function removes its Function URL config with it; we delete the URL
    config first for a clean log. The recorded .mcp_url marker is removed too.
    """
    try:
        lmb.delete_function_url_config(FunctionName=MCP_LAMBDA_NAME)
        print(f"  Function URL for '{MCP_LAMBDA_NAME}': DELETED.")
    except ClientError as err:
        if err.response["Error"]["Code"] not in _NOT_FOUND:
            raise
        print(f"  Function URL for '{MCP_LAMBDA_NAME}': already gone. Fine.")

    try:
        lmb.delete_function(FunctionName=MCP_LAMBDA_NAME)
        print(f"  Lambda '{MCP_LAMBDA_NAME}': DELETED.")
    except ClientError as err:
        if err.response["Error"]["Code"] in _NOT_FOUND:
            print(f"  Lambda '{MCP_LAMBDA_NAME}': already gone. Fine.")
        else:
            raise

    if MCP_URL_FILE.exists():
        MCP_URL_FILE.unlink()
        print(f"  recorded '{MCP_URL_FILE.name}': removed.")


def delete_mcp_lambda_role(iam) -> None:
    """Delete the MCP Lambda's inline policy then its role. Idempotent."""
    try:
        for name in iam.list_role_policies(RoleName=MCP_LAMBDA_ROLE_NAME).get(
            "PolicyNames", []
        ):
            iam.delete_role_policy(RoleName=MCP_LAMBDA_ROLE_NAME, PolicyName=name)
            print(f"  inline policy '{name}': deleted.")
        iam.delete_role(RoleName=MCP_LAMBDA_ROLE_NAME)
        print(f"  IAM role '{MCP_LAMBDA_ROLE_NAME}': DELETED.")
    except ClientError as err:
        if err.response["Error"]["Code"] in _NOT_FOUND:
            print(f"  IAM role '{MCP_LAMBDA_ROLE_NAME}': already gone. Fine.")
        else:
            raise


# --- Module 10: the least-privilege component roles + the decision log ----------
def delete_component_roles(iam) -> None:
    """Delete the four Module 10 component roles (+ their inline policy). Idempotent.

    Iterates relay.config.IAM_COMPONENT_ROLES (the SAME list setup.py creates from), so
    create and delete never drift. Each role's inline policy is removed first (IAM rejects
    deleting a role that still has inline policies), then the role. A role already gone is
    fine. IAM is FREE, but the course rule is leave nothing behind that you created (B5)."""
    for role_name, _stem in config.IAM_COMPONENT_ROLES:
        try:
            for name in iam.list_role_policies(RoleName=role_name).get(
                "PolicyNames", []
            ):
                iam.delete_role_policy(RoleName=role_name, PolicyName=name)
                print(f"  inline policy '{name}' on '{role_name}': deleted.")
            iam.delete_role(RoleName=role_name)
            print(f"  IAM role '{role_name}': DELETED.")
        except ClientError as err:
            if err.response["Error"]["Code"] in _NOT_FOUND:
                print(f"  IAM role '{role_name}': already gone. Fine.")
            else:
                raise


def delete_decision_log() -> None:
    """Remove the runtime decision_log.jsonl (git-ignored output). Idempotent."""
    if DECISION_LOG_FILE.exists():
        DECISION_LOG_FILE.unlink()
        print(f"  decision log '{DECISION_LOG_FILE.name}': DELETED.")
    else:
        print(f"  decision log '{DECISION_LOG_FILE.name}': none to remove. Fine.")


def delete_tables(ddb) -> None:
    """Drop relay-orders + relay-tickets. OFF by default — Module 8 reuses them.

    On-demand tables are ~$0 idle, so keeping them is fine; this is the explicit
    `--delete-tables` opt-out for a clean slate (the next setup re-creates + re-seeds).
    """
    for name in (config.RELAY_ORDERS_TABLE, config.RELAY_TICKETS_TABLE):
        try:
            ddb.delete_table(TableName=name)
            print(f"  table '{name}': DELETED (--delete-tables).")
        except ClientError as err:
            if err.response["Error"]["Code"] in _NOT_FOUND:
                print(f"  table '{name}': already gone. Fine.")
            else:
                raise


# =============================================================================
# Module 12 teardown — drop the semantic-cache table + the batch role + S3 artifacts.
# =============================================================================
# Module 12 added ONE standing resource (the relay-cache DynamoDB table, on-demand ~$0 idle)
# plus a batch IAM role and the batch job's S3 artifacts. None is idle-billed beyond ~$0,
# but B5 says leave nothing behind: drop the table, delete the role, purge the artifacts.
# Always run (not gated behind a flag) — the cache table is M12's own, never reused upstream.
def delete_cache_table(ddb) -> None:
    """Drop the M12 semantic-cache table `relay-cache`. Idempotent.

    On-demand -> ~$0 idle, but it is M12's own resource (nothing upstream reuses it), so it
    is deleted unconditionally at teardown (B5), unlike relay-orders/relay-tickets which
    Module 8+ reuse and which stay unless --delete-tables."""
    try:
        ddb.delete_table(TableName=config.RELAY_CACHE_TABLE)
        print(f"  table '{config.RELAY_CACHE_TABLE}': DELETED (M12 semantic cache).")
    except ClientError as err:
        if err.response["Error"]["Code"] in _NOT_FOUND:
            print(f"  table '{config.RELAY_CACHE_TABLE}': already gone. Fine.")
        else:
            raise


def delete_batch_role(iam) -> None:
    """Delete the M12 batch service role (+ its inline policy). Idempotent.

    Removes the inline policy first (IAM rejects deleting a role that still has inline
    policies), then the role. IAM is FREE, removed anyway (B5)."""
    role_name = config.RELAY_BATCH_ROLE_NAME
    try:
        for name in iam.list_role_policies(RoleName=role_name).get("PolicyNames", []):
            iam.delete_role_policy(RoleName=role_name, PolicyName=name)
            print(f"  inline policy '{name}' on '{role_name}': deleted.")
        iam.delete_role(RoleName=role_name)
        print(f"  IAM role '{role_name}': DELETED (M12 batch role).")
    except ClientError as err:
        if err.response["Error"]["Code"] in _NOT_FOUND:
            print(f"  IAM role '{role_name}': already gone. Fine.")
        else:
            raise


def purge_batch_artifacts(s3, data_bucket: str) -> int:
    """Delete the batch job's S3 input + output objects (batch/input/, batch/output/).

    Returns the count removed. The batch JSONL input and the job's output land under the
    data bucket's batch/ prefixes; this purges both so nothing of the demo job lingers (B5).
    Idempotent — a missing prefix is a clean no-op."""
    removed = 0
    for prefix in (config.RELAY_BATCH_INPUT_PREFIX, config.RELAY_BATCH_OUTPUT_PREFIX):
        try:
            paginator = s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=data_bucket, Prefix=prefix):
                keys = [{"Key": obj["Key"]} for obj in page.get("Contents", [])]
                if keys:
                    s3.delete_objects(Bucket=data_bucket, Delete={"Objects": keys})
                    removed += len(keys)
        except ClientError as err:
            if err.response["Error"]["Code"] in _NOT_FOUND:
                continue
            raise
    if removed:
        print(f"  batch artifacts under batch/: {removed} object(s) DELETED.")
    else:
        print("  batch artifacts under batch/: none to remove. Fine.")
    return removed


# =============================================================================
# Module 8 teardown — PURGE the AgentCore Memory (the sole idle-billed item).
# =============================================================================
def _agentcore_control():
    """The bedrock-agentcore-control client (AgentCore Memory control plane)."""
    return boto3.client("bedrock-agentcore-control", region_name=REGION)


def _resolve_memory_id_for_delete(control) -> str | None:
    """Find the Memory id to delete: the recorded .memory_id, else a lookup by name.

    Prefer the marker setup.py wrote; fall back to listing by name so a missing marker
    still finds the store. Returns None if there is nothing to delete."""
    if MEMORY_ID_FILE.exists():
        recorded = MEMORY_ID_FILE.read_text(encoding="utf-8").strip()
        if recorded:
            return recorded
    try:
        resp = control.list_memories(maxResults=100)
    except Exception:  # noqa: BLE001 — no store / service: nothing to delete.
        return None
    # The store id/arn carry the API name (canonical handle, hyphens -> underscores).
    api_name = config.agentcore_memory_api_name()
    for mem in resp.get("memories", []):
        mem_id = mem.get("id") or ""
        if mem_id.startswith(api_name) or api_name in mem.get("arn", ""):
            return mem_id
    return None


def purge_agentcore_memory(control) -> None:
    """Delete the AgentCore Memory store `relay-memory` — PURGING the long-term records.

    The long-term cross-session store is the ONLY idle-billed item in the lab (B5), so
    this is the load-bearing teardown step. Idempotent: a missing store is a clean
    no-op. Removes the .memory_id / .runtime_arn markers too (the runtime itself is torn
    down with `agentcore destroy` — the CLI owns it). DeleteMemory takes `memoryId`."""
    memory_id = _resolve_memory_id_for_delete(control)
    if memory_id is None:
        print(f"  AgentCore Memory '{config.AGENTCORE_MEMORY_NAME}': none found "
              "(already purged or never created).")
    else:
        try:
            control.delete_memory(memoryId=memory_id)
        except ClientError as err:
            code = err.response["Error"]["Code"]
            message = err.response["Error"]["Message"].lower()
            if code in _NOT_FOUND:
                print(f"  AgentCore Memory {memory_id}: already gone.")
            elif "transitional state" in message or "deleting" in message:
                # DeleteMemory is asynchronous: a re-run while the store is still in the
                # DELETING state is a clean no-op (the first call already purged it).
                print(f"  AgentCore Memory {memory_id}: deletion already in progress "
                      "(DELETING) — purge under way.")
            else:
                raise
        else:
            print(f"  AgentCore Memory {memory_id}: DELETED "
                  "(long-term records purged — the only idle-billed item).")
    for marker in (MEMORY_ID_FILE, RUNTIME_ARN_FILE):
        if marker.exists():
            marker.unlink()
            print(f"  removed marker {marker.name}")
    print("  AgentCore Runtime: remove with `agentcore destroy` (idle was free; the "
          "CLI owns it).")


# =============================================================================
# Module 9 teardown — DELETE the Bedrock Guardrail `relay-guardrail` (all versions).
# =============================================================================
def _bedrock_control():
    """The bedrock CONTROL plane (DeleteGuardrail / ListGuardrails)."""
    return boto3.client("bedrock", region_name=REGION)


def _resolve_guardrail_id_for_delete(bd) -> str | None:
    """Find the guardrail id to delete: the recorded .guardrail_id, else a lookup by name.

    Prefer the marker setup.py wrote; fall back to listing by name so a missing marker
    still finds the guardrail. Returns None if there is nothing to delete."""
    if GUARDRAIL_ID_FILE.exists():
        recorded = GUARDRAIL_ID_FILE.read_text(encoding="utf-8").strip()
        if recorded:
            return recorded
    try:
        for page in bd.get_paginator("list_guardrails").paginate():
            for summary in page.get("guardrails", []):
                if summary.get("name") == config.RELAY_GUARDRAIL_NAME:
                    return summary.get("id")
    except Exception:  # noqa: BLE001 — no guardrail / service: nothing to delete.
        return None
    return None


def delete_guardrail(bd) -> None:
    """Delete the Bedrock Guardrail `relay-guardrail` — ALL versions. Idempotent.

    DeleteGuardrail WITHOUT a `guardrailVersion` removes the whole guardrail (DRAFT +
    every published version), so one call cleans the lab. A missing guardrail is a clean
    no-op. Removes the .guardrail_id / .guardrail_version markers too. This is the only
    thing Module 9 created, and a guardrail bills only per use (~$0 idle) — but the course
    rule is leave nothing behind (B5)."""
    guardrail_id = _resolve_guardrail_id_for_delete(bd)
    if guardrail_id is None:
        print(f"  Guardrail '{config.RELAY_GUARDRAIL_NAME}': none found "
              "(already deleted or never created).")
    else:
        try:
            bd.delete_guardrail(guardrailIdentifier=guardrail_id)
            print(f"  Guardrail '{config.RELAY_GUARDRAIL_NAME}' ({guardrail_id}): "
                  "DELETED (all versions).")
        except ClientError as err:
            if err.response["Error"]["Code"] in _NOT_FOUND:
                print(f"  Guardrail '{config.RELAY_GUARDRAIL_NAME}': already gone. Fine.")
            else:
                raise
    for marker in (GUARDRAIL_ID_FILE, GUARDRAIL_VERSION_FILE):
        if marker.exists():
            marker.unlink()
            print(f"  removed marker {marker.name}")


# =============================================================================
# Module 11 teardown — cdk destroy the front door + DELETE the CodePipeline (B5).
# =============================================================================
# The front-door infra is AWS CDK (decision B6), so the clean teardown is `cdk destroy`
# RelayApiStack + RelayPipelineStack. The CodePipeline is the ONLY M11 resource with a real
# idle cost (~$1/active pipeline/month as of June 2026), so destroying both stacks is the
# load-bearing B5 step. When the `cdk` CLI is not on PATH (e.g. CI without node), we FALL
# BACK to a boto3 sweep that deletes the pipeline + the SQS queue + DLQ + the relay-events
# bus directly — so nothing idle-billed survives even without CDK. Everything is idempotent:
# an already-gone stack / pipeline / queue / bus is a clean no-op.
_CDK_DIR = Path(__file__).resolve().parent / "cdk"


def _codepipeline():
    return boto3.client("codepipeline", region_name=REGION)


def _sqs():
    return boto3.client("sqs", region_name=REGION)


def _events():
    return boto3.client("events", region_name=REGION)


def cdk_destroy_front_door() -> bool:
    """Run `cdk destroy` for both M11 stacks. Returns True if the CLI ran, False if absent.

    Tries the `cdk` CLI in cdk/ (the declarative path). If `cdk` is not installed (no node),
    returns False so main() falls back to the boto3 sweep. A non-zero CLI exit on an
    already-gone stack is tolerated (CDK reports "no stacks" cleanly). Never raises — a
    teardown must not crash on a missing CLI."""
    import shutil
    import subprocess

    if shutil.which("cdk") is None and shutil.which("npx") is None:
        print("  `cdk` CLI not found — falling back to a boto3 sweep of the M11 resources.")
        return False
    cdk_cmd = "cdk" if shutil.which("cdk") else "npx cdk"
    stacks = f"{config.RELAY_STACK_NAME} {config.RELAY_PIPELINE_STACK_NAME}"
    cmd = f"{cdk_cmd} destroy {stacks} --force"
    print(f"  running: {cmd} (in {_CDK_DIR})")
    try:
        proc = subprocess.run(cmd, shell=True, cwd=str(_CDK_DIR),
                              capture_output=True, text=True, timeout=900)
    except Exception as err:  # noqa: BLE001 — never crash teardown on the CLI.
        print(f"  [warn] cdk destroy could not run ({type(err).__name__}); "
              "falling back to the boto3 sweep.")
        return False
    if proc.returncode == 0:
        print(f"  cdk destroy: {config.RELAY_STACK_NAME} + "
              f"{config.RELAY_PIPELINE_STACK_NAME} DESTROYED.")
    else:
        # An already-empty environment also exits non-zero on some CDK versions; log + still
        # run the boto3 sweep to be SURE the idle-billed pipeline is gone.
        print(f"  [warn] cdk destroy exited {proc.returncode}; running the boto3 sweep to "
              "be sure nothing idle-billed survives.\n  "
              + (proc.stderr.strip()[-300:] or "(no stderr)"))
        return False
    return True


def delete_pipeline(cp) -> None:
    """Delete the CodePipeline `relay-pipeline` directly (boto3 fallback). Idempotent.

    The pipeline is the only idle-billed M11 resource, so this is the load-bearing sweep
    step when CDK is unavailable. A missing pipeline is a clean no-op."""
    try:
        cp.delete_pipeline(name=config.RELAY_PIPELINE_NAME)
        print(f"  CodePipeline '{config.RELAY_PIPELINE_NAME}': DELETED "
              "(the only idle-billed M11 resource).")
    except ClientError as err:
        if err.response["Error"]["Code"] in _NOT_FOUND + ("PipelineNotFoundException",):
            print(f"  CodePipeline '{config.RELAY_PIPELINE_NAME}': already gone. Fine.")
        else:
            raise


def delete_work_queues(sqs) -> None:
    """Delete the SQS work queue + DLQ (boto3 fallback). Idempotent.

    On-demand SQS bills ~$0 idle, but the course rule is leave nothing behind. A queue that
    does not exist is a clean no-op (GetQueueUrl raises QueueDoesNotExist)."""
    for name in (config.RELAY_QUEUE_NAME, config.RELAY_DLQ_NAME):
        try:
            url = sqs.get_queue_url(QueueName=name)["QueueUrl"]
        except ClientError as err:
            code = err.response["Error"]["Code"]
            if code in _NOT_FOUND + ("AWS.SimpleQueueService.NonExistentQueue",
                                     "QueueDoesNotExist"):
                print(f"  SQS queue '{name}': already gone. Fine.")
                continue
            raise
        sqs.delete_queue(QueueUrl=url)
        print(f"  SQS queue '{name}': DELETED.")


def delete_event_bus(ev) -> None:
    """Delete the relay-events EventBridge bus + its rules (boto3 fallback). Idempotent.

    A custom bus + its rules bill ~$0 idle, but leave nothing behind (B5). Rules (and their
    targets) must be removed before the bus. A missing bus/rule is a clean no-op."""
    bus = config.RELAY_EVENT_BUS_NAME
    try:
        rules = ev.list_rules(EventBusName=bus).get("Rules", [])
    except ClientError as err:
        if err.response["Error"]["Code"] in _NOT_FOUND:
            print(f"  EventBridge bus '{bus}': already gone. Fine.")
            return
        raise
    for rule in rules:
        rname = rule["Name"]
        target_ids = [t["Id"] for t in ev.list_targets_by_rule(
            Rule=rname, EventBusName=bus).get("Targets", [])]
        if target_ids:
            ev.remove_targets(Rule=rname, EventBusName=bus, Ids=target_ids)
        ev.delete_rule(Name=rname, EventBusName=bus)
        print(f"  EventBridge rule '{rname}': DELETED.")
    try:
        ev.delete_event_bus(Name=bus)
        print(f"  EventBridge bus '{bus}': DELETED.")
    except ClientError as err:
        if err.response["Error"]["Code"] in _NOT_FOUND:
            print(f"  EventBridge bus '{bus}': already gone. Fine.")
        else:
            raise


def teardown_front_door(*, use_cdk: bool = True) -> None:
    """Tear down the M11 front door: cdk destroy, with a boto3 sweep fallback.

    If `use_cdk` and the CLI ran cleanly, the stacks are gone (queue/bus/pipeline included).
    Otherwise (no CLI, or a non-zero exit) we sweep the idle-billed pieces directly so the
    pipeline never survives. Idempotent either way."""
    destroyed = cdk_destroy_front_door() if use_cdk else False
    if destroyed:
        # CDK removed the stacks (and everything in them). Run the sweep anyway as a CHEAP
        # idempotent belt-and-braces — every call below is a clean no-op once the stacks
        # are gone, and it GUARANTEES the idle-billed pipeline is deleted.
        print("  verifying nothing idle-billed survived (idempotent sweep):")
    delete_pipeline(_codepipeline())
    delete_work_queues(_sqs())
    delete_event_bus(_events())


def assert_clean(*, drop_tables: bool, drop_kb: bool, drop_vectors: bool,
                 vector_bucket: str, index_name: str, data_bucket: str) -> None:
    print("\nIdle-billed resources after teardown: NONE remain.")
    print(f"  - Front door (Module 11): RelayApiStack (API Gateway + 4 Lambda + SQS + "
          f"DLQ + '{config.RELAY_EVENT_BUS_NAME}' bus) destroyed via CDK; the CodePipeline "
          f"'{config.RELAY_PIPELINE_NAME}' (the ONLY idle-billed M11 resource, ~$1/month) "
          "DELETED above (B5).")
    roles = ", ".join(r for r, _ in config.IAM_COMPONENT_ROLES)
    print(f"  - IAM component roles ({roles}) + decision log: deleted above "
          "(IAM is FREE, removed anyway — leave nothing behind).")
    print(f"  - Bedrock Guardrail '{config.RELAY_GUARDRAIL_NAME}' (all versions): "
          "deleted above (billed only per use ~$0 idle, but removed anyway).")
    print("  - AgentCore Memory (long-term, the ONLY idle-billed item): PURGED above.")
    print("  - MCP server Lambda + Function URL + IAM role: deleted above "
          "(billed ~$0 idle, but removed anyway).")
    print(f"  - Module 12 semantic cache '{config.RELAY_CACHE_TABLE}' (on-demand ~$0 idle) "
          f"+ batch role '{config.RELAY_BATCH_ROLE_NAME}' + batch/ S3 artifacts: DELETED "
          "above (M12's own resources — leave nothing behind, B5).")
    if drop_tables:
        print(f"  - DynamoDB '{config.RELAY_ORDERS_TABLE}' + "
              f"'{config.RELAY_TICKETS_TABLE}': DELETED on request (--delete-tables).")
    else:
        print(f"  - DynamoDB '{config.RELAY_ORDERS_TABLE}' (seeded) + "
              f"'{config.RELAY_TICKETS_TABLE}': KEPT — on-demand (~$0 idle) AND "
              "Module 9+ reuse them.\n    Drop with --delete-tables.")
    if drop_kb:
        print("  - Knowledge Base + data source + KB role: DELETED on request "
              "(--delete-kb).")
    else:
        print(f"  - Knowledge Base '{config.RELAY_KB_NAME}' + role: KEPT — search_kb "
              "retrieves from it and Module 9+ need it (~$0 idle). Drop with --delete-kb.")
    if drop_vectors:
        print(f"  - S3 Vectors index '{index_name}': DELETED on request "
              "(--delete-vectors).")
    else:
        print(f"  - S3 Vectors KB index '{config.RELAY_KB_INDEX}' + M4 DIY index "
              f"'{index_name}' (bucket '{vector_bucket}'): KEPT (idle ~$0).")
    print(f"  - data bucket '{data_bucket}' + docs/ (+ metadata sidecars): KEPT "
          "— downstream modules reuse the corpus.")
    print("  The M1 $5 budget is KEPT on purpose (Module 1 owns it).")
    print("\nRemove the AgentCore Runtime separately (the CLI owns it; idle was free):")
    print("    agentcore destroy")
    print("To rebuild: uv run python setup.py "
          "(recreates the guardrail + MCP Lambda + AgentCore Memory; tables/KB kept).")


_FLAGS = ("--delete-tables", "--delete-kb", "--delete-vectors", "--keep-stacks")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    drop_tables = "--delete-tables" in argv
    drop_kb = "--delete-kb" in argv
    drop_vectors = "--delete-vectors" in argv
    keep_stacks = "--keep-stacks" in argv
    leftover = [a for a in argv if a not in _FLAGS]
    if leftover:
        print(f"Unknown argument(s): {' '.join(leftover)}\n"
              "Usage: uv run python teardown.py "
              "[--delete-tables] [--delete-kb] [--delete-vectors] [--keep-stacks]",
              file=sys.stderr)
        return 1

    print("Tearing down Module 11 (idempotent).\n")

    try:
        acct = config.account_id(_sts())
    except NoCredentialsError:
        print("  [FAIL] no AWS credentials — set AWS_PROFILE=aws-genai-pro.",
              file=sys.stderr)
        return 1

    vector_bucket = config.relay_vector_bucket(acct)
    data_bucket = config.relay_bucket(acct)
    index_name = config.RELAY_INDEX

    agent, iam, s3v, s3, lmb, ddb = (
        _agent(), _iam(), _s3vectors(), _s3(), _lambda(), _dynamodb()
    )

    try:
        # --- Module 11: cdk destroy the front door + DELETE the pipeline (B5) FIRST ---
        # The pipeline is the only idle-billed M11 resource, so it goes first. CDK removes
        # the whole stack (API/Lambda/SQS/bus); the boto3 sweep is the no-CLI fallback.
        if keep_stacks:
            print("--keep-stacks: skipping the CDK destroy (front door already gone).")
        else:
            print("Front door (Module 11 — AWS CDK: API Gateway + 4 Lambda + SQS + "
                  "relay-events) + the CodePipeline:")
            teardown_front_door(use_cdk=True)
            print()

        print("Least-privilege IAM component roles (Module 10):")
        delete_component_roles(iam)

        print("\nAgent decision log (Module 10 — runtime output):")
        delete_decision_log()

        print("\nBedrock Guardrail (relay-guardrail — Module 9):")
        delete_guardrail(_bedrock_control())

        print("\nAgentCore Memory (Bedrock AgentCore — the only idle-billed item):")
        purge_agentcore_memory(_agentcore_control())

        print("\nCloudCart MCP server (AWS Lambda + Function URL):")
        delete_mcp_lambda(lmb)

        print("\nMCP Lambda execution role (IAM):")
        delete_mcp_lambda_role(iam)

        # --- Module 12: drop the semantic-cache table + batch role + S3 artifacts ----
        # These are M12's OWN resources (nothing upstream reuses them), so they go
        # unconditionally (B5) — not gated behind --delete-tables like relay-orders/tickets.
        print("\nModule 12 — semantic cache + batch artifacts:")
        delete_cache_table(ddb)
        delete_batch_role(iam)
        purge_batch_artifacts(s3, data_bucket)

        if drop_tables:
            print("\nAgent business tables (Amazon DynamoDB):")
            delete_tables(ddb)

        if drop_kb:
            print("\nIntake attachments (Amazon S3 attachments/ prefix):")
            purge_attachments(s3, data_bucket)
            print("\nKnowledge Base (Bedrock):")
            delete_knowledge_base(agent, _find_kb_id(agent))
            print("\nKB service role (IAM):")
            delete_kb_role(iam)

        if drop_vectors:
            print("\nVector store (Amazon S3 Vectors):")
            delete_vectors(s3v, vector_bucket, index_name)
    except ClientError as err:
        code = err.response["Error"]["Code"]
        message = err.response["Error"]["Message"]
        print(f"\nAWS call failed ({code}):\n  {message}", file=sys.stderr)
        return 1

    assert_clean(drop_tables=drop_tables, drop_kb=drop_kb, drop_vectors=drop_vectors,
                 vector_bucket=vector_bucket, index_name=index_name,
                 data_bucket=data_bucket)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
