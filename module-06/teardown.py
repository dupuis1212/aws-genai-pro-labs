"""teardown.py — remove Module 6's intake uploads + the KB, keep what M7 needs.

Module 6 of AWS GenAI Pro Mastery. Idempotent and verbose, and TESTED — one tested
teardown per setup (decision B5).

What this DELETES (Module 6's + Module 5's own resources):
  0. (Module 6) every object the intake pipeline uploaded under the data bucket's
     attachments/ prefix (the screenshots). Comprehend and Converse vision are
     per-CALL services — they create nothing idle-billed, so there is nothing else
     for Module 6 to delete.
  1. the Knowledge Base `relay-kb` and its S3 data source (deleting the KB removes
     the data source with it; we delete the data source first for a clean log);
  2. the KB service role `relay-kb-role` and its inline policy.

What this KEEPS, on purpose:
  - the S3 Vectors bucket `relay-vectors-<account_id>` and BOTH its indexes (idle
    ~$0): the KB-owned index `relay-kb-docs` (Module 7's agent KB-search tool
    retrieves from the KB built on it) and Module 4's DIY index `relay-docs` —
    both survive Module 5 teardown (bible §3.3);
  - the data bucket `relay-<account_id>` and the docs/ corpus (+ metadata
    sidecars) — same reason;
  - the M1 $5 budget alarm (persistent; Module 1 owns it).

So Module 5 leaves NOTHING idle-billed of its OWN: the KB and role bill ~$0 idle
anyway, and the kept resources (S3 Vectors + S3) are deliberate downstream
dependencies, also ~$0 idle. If you tore the KB down between modules, recreate it
with `uv run python setup.py` (it rebuilds the KB over the still-present index).

Note: deleting the KB does NOT delete the vectors in `relay-kb-docs` — those
vectors live in the S3 Vectors index, which the vector bucket owns and Module 7
reuses. That is the point: the managed KB is a thin control layer over a vector
store you keep. (`setup.py` re-ingests cleanly on rebuild.)

Run it:
    uv run python teardown.py
    uv run python teardown.py --delete-vectors   # ALSO drop relay-docs (M7 will
                                                 # then need M4's setup re-run)
"""

from __future__ import annotations

import sys
import time

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

from relay import config
from setup import KB_ID_FILE, KB_DATA_SOURCE_ID_FILE, KB_ROLE_NAME

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


# --- Optional: drop the vector index (NOT default — M7 needs it) --------------
def delete_vectors(s3v, vector_bucket: str, index_name: str) -> None:
    """Drop the relay-docs index. OFF by default: Module 7 retrieves from it."""
    try:
        s3v.delete_index(vectorBucketName=vector_bucket, indexName=index_name)
        print(f"  index '{index_name}': DELETED (--delete-vectors). "
              "Module 7 will need Module 4's setup re-run.")
    except ClientError as err:
        if err.response["Error"]["Code"] in _NOT_FOUND:
            print(f"  index '{index_name}': already gone. Fine.")
        else:
            raise


def assert_clean(delete_vectors_flag: bool, vector_bucket: str,
                 index_name: str, data_bucket: str) -> None:
    print("\nIdle-billed resources from Module 6: NONE.")
    print("  - intake attachments/ uploads: purged above (S3 storage was cents).")
    print("  - Comprehend + Converse (Nova Lite vision): per-CALL services — they "
          "create\n    nothing idle-billed, so there is nothing to delete.")
    print("  - Knowledge Base + data source + IAM role: deleted above "
          "(idle was ~$0).")
    if delete_vectors_flag:
        print(f"  - S3 Vectors index '{index_name}': DELETED on request "
              "(--delete-vectors).")
    else:
        print(f"  - S3 Vectors KB index '{config.RELAY_KB_INDEX}' + M4 DIY index "
              f"'{index_name}' (bucket '{vector_bucket}'): KEPT — Module 7's "
              "KB-search tool reuses the KB built on these (idle ~$0).")
    print(f"  - data bucket '{data_bucket}' + docs/ (+ metadata sidecars): KEPT "
          "— downstream modules reuse the corpus.")
    print("  The M1 $5 budget is KEPT on purpose (Module 1 owns it).")
    print("\nTo rebuild for a later module: uv run python setup.py "
          "(recreates the KB over the kept index).")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    drop_vectors = "--delete-vectors" in argv
    leftover = [a for a in argv if a != "--delete-vectors"]
    if leftover:
        print(f"Unknown argument(s): {' '.join(leftover)}\n"
              "Usage: uv run python teardown.py [--delete-vectors]",
              file=sys.stderr)
        return 1

    print("Tearing down Module 6 (idempotent).\n")

    try:
        acct = config.account_id(_sts())
    except NoCredentialsError:
        print("  [FAIL] no AWS credentials — set AWS_PROFILE=aws-genai-pro.",
              file=sys.stderr)
        return 1

    vector_bucket = config.relay_vector_bucket(acct)
    data_bucket = config.relay_bucket(acct)
    index_name = config.RELAY_INDEX

    agent, iam, s3v, s3 = _agent(), _iam(), _s3vectors(), _s3()

    try:
        print("Intake attachments (Amazon S3 attachments/ prefix):")
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

    assert_clean(drop_vectors, vector_bucket, index_name, data_bucket)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
