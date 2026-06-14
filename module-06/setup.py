"""setup.py — stand up Relay's intake attachments prefix + the KB (Module 6).

Module 6 of AWS GenAI Pro Mastery. Idempotent and verbose: safe to run twice, and
it tells you exactly what it creates, what it reuses, and what it costs.

Module 6 adds an INTAKE pipeline (relay.intake) upstream of everything. Its only
standing AWS resource is the data bucket's `attachments/` prefix, where it uploads
the screenshots customers send — so this setup ensures that prefix exists (a tiny
marker), on top of the inherited Module 5 Knowledge Base setup below. The intake's
other calls — Amazon Comprehend entity extraction and the Amazon Nova Lite vision
read via Converse — are per-CALL services that need no standing resource.

Module 4 built the storage layer by hand — the data bucket `relay-<account_id>`
with the docs/ corpus, and the S3 Vectors index `relay-docs` (1024-dim Titan V2).
Module 5 hands ingestion to a **Bedrock Knowledge Base** that creates and owns its
OWN dedicated S3 Vectors index `relay-kb-docs` in Module 4's vector bucket — a
Bedrock KB writes its own vector-metadata schema and cannot read Module 4's raw
`relay-docs` vectors, so the KB gets a clean index and `relay-docs` stays the DIY
benchmark baseline (see config.RELAY_KB_INDEX for the full rationale). setup.py,
in order:

  1. PRECHECK the Module 4 prerequisites exist (data bucket + docs/ + the
     relay-docs index). It only READS relay-docs as a prerequisite — it never
     writes it. If they are missing, it tells you to run Module 4's setup first;
     it does NOT silently rebuild someone else's resource.
  2. Create the KB SERVICE ROLE (IAM) the Knowledge Base assumes: trust to
     bedrock.amazonaws.com, and least-privilege access to invoke Titan embeddings,
     read the docs/ prefix, and read/write the S3 Vectors index. Idempotent.
  3. Create the KB's OWN S3 Vectors index `relay-kb-docs`, then the Knowledge Base
     `relay-kb` with S3 Vectors storage pointed at that KB-owned index (NOT a
     Quick-Create always-on serverless search collection — that bills ~$174/month
     idle; S3 Vectors bills ~$0 idle).
  4. Attach the S3 DATA SOURCE over s3://relay-<account_id>/docs/.
  5. Start the FIRST ingestion job and wait for COMPLETE (parse -> chunk -> embed
     with Titan -> write vectors into relay-kb-docs).
  6. Record the KB id and data-source id in .kb_id / .kb_data_source_id so
     relay/kb.py, compare_retrieval.py, and freshness_test.py find them.

It does NOT create any always-on serverless search cluster, Aurora, or Kendra.
Every model ID, resource name, and the reranker come from relay/config.py — none
is typed here. The account id is resolved from STS at run time, never hard-coded.

What it costs (as of June 2026 — re-verify on the Bedrock pricing page):
  - one ingestion job over a handful of small docs: Titan embeddings, a few cents.
  - the KB itself + S3 Vectors idle: ~$0/month. No always-on cluster.

Run it:
    uv run python setup.py
    uv run python setup.py --no-wait    # start ingestion, do not block on COMPLETE
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

from relay import config

REGION = config.REGION

_ROOT = Path(__file__).resolve().parent
DOCS_DIR = _ROOT / "data" / "docs"
KB_ID_FILE = _ROOT / ".kb_id"
KB_DATA_SOURCE_ID_FILE = _ROOT / ".kb_data_source_id"

# The IAM role the Knowledge Base assumes to do its work.
KB_ROLE_NAME = "relay-kb-role"

# How long to wait for the first ingestion job to reach COMPLETE before giving up
# (the job is small; this is a generous ceiling).
_INGESTION_TIMEOUT_S = 600
_INGESTION_POLL_S = 10


def _s3():
    return boto3.client("s3", region_name=REGION)


def _s3vectors():
    return boto3.client("s3vectors", region_name=REGION)


def _sts():
    return boto3.client("sts", region_name=REGION)


def _iam():
    return boto3.client("iam", region_name=REGION)


def _agent():
    """bedrock-agent: the Knowledge Base CONTROL plane (create KB / data source)."""
    return boto3.client("bedrock-agent", region_name=REGION)


# --- Step 1: the Module 4 prerequisites must already exist --------------------
def precheck_prerequisites(s3, s3v, data_bucket: str, vector_bucket: str,
                           index_name: str) -> None:
    """Confirm the M4 storage layer is present. Raise a clear instruction if not.

    Module 5 REUSES Module 4's resources; it never recreates them. If they are
    missing, the fix is to run Module 4's setup, not to silently provision here.
    """
    problems: list[str] = []
    try:
        s3.head_bucket(Bucket=data_bucket)
        listed = s3.list_objects_v2(Bucket=data_bucket, Prefix="docs/", MaxKeys=1)
        if listed.get("KeyCount", 0) == 0:
            problems.append(f"data bucket '{data_bucket}' has no objects under docs/")
    except ClientError:
        problems.append(f"data bucket '{data_bucket}' is missing")

    try:
        s3v.get_index(vectorBucketName=vector_bucket, indexName=index_name)
    except ClientError:
        problems.append(
            f"S3 Vectors index '{index_name}' in '{vector_bucket}' is missing"
        )

    if problems:
        raise SystemExit(
            "Module 5 reuses Module 4's storage layer, which is not ready:\n  - "
            + "\n  - ".join(problems)
            + "\n\nRun Module 4's setup + ingestion first (from module-04/, or copy\n"
            "its data/docs/ here and run its pipeline), then re-run this setup.\n"
            "Module 5 never recreates Module 4's bucket or index."
        )
    print(f"  prerequisites OK: data bucket '{data_bucket}' (docs/ populated) and "
          f"index '{index_name}' present.")


# --- Filterable metadata: per-doc sidecars so `category` is a filter key --------
_FRONT_MATTER_CATEGORY = re.compile(r"^category:\s*(\S+)", re.MULTILINE)


def _doc_category(text: str) -> str:
    """Read the `category:` value from a doc's YAML front matter (defaults safe)."""
    m = _FRONT_MATTER_CATEGORY.search(text)
    return m.group(1) if m else "uncategorized"


def ensure_metadata_sidecars(s3, data_bucket: str,
                             docs_dir: Path = DOCS_DIR) -> int:
    """Upload a `<doc>.md.metadata.json` sidecar per doc so the KB indexes
    `category` as a FILTERABLE metadata attribute. Idempotent (overwrites).

    A Bedrock S3 data source reads `<object-key>.metadata.json` next to each object
    and turns its `metadataAttributes` into filterable fields. Without this, the
    KB has no `category` filter, so retrieve(category=...) (the multi-tenant /
    scoped-retrieval lever, and the lab's 'Try it yourself' #1) returns nothing.
    We derive the category from the doc's own front matter — single source of truth.
    """
    docs = sorted(docs_dir.glob("*.md"))
    written = 0
    for doc in docs:
        category = _doc_category(doc.read_text(encoding="utf-8"))
        sidecar = {
            "metadataAttributes": {
                "category": {
                    "value": {"type": "STRING", "stringValue": category},
                    # Keep the tag OUT of the embedded text — it is a filter, not
                    # content to retrieve on.
                    "includeForEmbedding": False,
                }
            }
        }
        key = f"{config.RELAY_KB_INCLUSION_PREFIX}{doc.name}.metadata.json"
        s3.put_object(
            Bucket=data_bucket, Key=key,
            Body=json.dumps(sidecar).encode("utf-8"),
            ContentType="application/json",
        )
        written += 1
    print(f"  metadata sidecars: {written} uploaded "
          "(category -> filterable KB metadata).")
    return written


# --- Step 2: the KB service role ---------------------------------------------
def _trust_policy() -> str:
    return json.dumps({
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "bedrock.amazonaws.com"},
            "Action": "sts:AssumeRole",
        }],
    })


def _kb_permissions_policy(account: str, data_bucket: str,
                           vector_bucket: str) -> str:
    """Least-privilege inline policy for the KB role (explicit ARNs, no wildcards
    on resources). The KB needs: invoke the Titan embedder, read docs/, and
    read/write the S3 Vectors store.
    """
    return json.dumps({
        "Version": "2012-10-17",
        "Statement": [
            {
                # The KB role invokes the Titan embedder (ingestion + query
                # embedding) by explicit ARN.
                "Sid": "InvokeTitanEmbeddings",
                "Effect": "Allow",
                "Action": ["bedrock:InvokeModel"],
                "Resource": [
                    f"arn:aws:bedrock:{REGION}::foundation-model/"
                    f"{config.EMBED_MODEL_ID}"
                ],
            },
            {
                # Retrieve with rerank=True assumes THIS role to invoke the Bedrock
                # reranker model — scoped to the reranker's foundation-model ARN.
                "Sid": "InvokeReranker",
                "Effect": "Allow",
                "Action": ["bedrock:InvokeModel"],
                "Resource": [config.rerank_model_arn()],
            },
            {
                # The bedrock:Rerank ACTION itself does NOT support resource-level
                # scoping — AWS requires Resource "*" for it (see the Bedrock
                # "Permissions for reranking" doc). The actual model is still pinned
                # by the InvokeReranker statement above, so this is least-privilege
                # in practice: Rerank is an API-level capability, not a per-model
                # grant. This is the ONLY statement allowed to use "*" (the smoke
                # test's least-privilege gate carves out exactly bedrock:Rerank).
                "Sid": "RerankAction",
                "Effect": "Allow",
                "Action": ["bedrock:Rerank"],
                "Resource": ["*"],
            },
            {
                "Sid": "ReadDocsPrefix",
                "Effect": "Allow",
                "Action": ["s3:GetObject", "s3:ListBucket"],
                "Resource": [
                    f"arn:aws:s3:::{data_bucket}",
                    f"arn:aws:s3:::{data_bucket}/docs/*",
                ],
            },
            {
                "Sid": "UseS3VectorsIndex",
                "Effect": "Allow",
                "Action": [
                    "s3vectors:GetVectorBucket",
                    "s3vectors:GetIndex",
                    "s3vectors:PutVectors",
                    "s3vectors:GetVectors",
                    "s3vectors:QueryVectors",
                    "s3vectors:ListVectors",
                    "s3vectors:DeleteVectors",
                ],
                "Resource": [
                    f"arn:aws:s3vectors:{REGION}:{account}:bucket/{vector_bucket}",
                    # The KB owns and writes its dedicated index relay-kb-docs.
                    f"arn:aws:s3vectors:{REGION}:{account}:bucket/{vector_bucket}/"
                    f"index/{config.RELAY_KB_INDEX}",
                ],
            },
        ],
    })


def ensure_kb_role(iam, account: str, data_bucket: str, vector_bucket: str) -> str:
    """Create (or reuse) the KB service role and its inline policy. Returns the ARN."""
    try:
        role = iam.get_role(RoleName=KB_ROLE_NAME)
        print(f"  IAM role '{KB_ROLE_NAME}': already exists. Reusing.")
    except ClientError as err:
        if err.response["Error"]["Code"] != "NoSuchEntity":
            raise
        role = iam.create_role(
            RoleName=KB_ROLE_NAME,
            AssumeRolePolicyDocument=_trust_policy(),
            Description="Relay Knowledge Base service role (Module 5).",
        )
        print(f"  IAM role '{KB_ROLE_NAME}': CREATED.")

    iam.put_role_policy(
        RoleName=KB_ROLE_NAME,
        PolicyName="relay-kb-permissions",
        PolicyDocument=_kb_permissions_policy(account, data_bucket, vector_bucket),
    )
    print("    inline policy 'relay-kb-permissions': put (least-privilege).")
    return role["Role"]["Arn"]


# --- Step 3 + 4: the Knowledge Base + its S3 data source ----------------------
def _find_kb_by_name(agent, name: str) -> str | None:
    """Return the id of a KB with this name, or None. Lets setup be idempotent."""
    paginator = agent.get_paginator("list_knowledge_bases")
    for page in paginator.paginate():
        for summary in page.get("knowledgeBaseSummaries", []):
            if summary.get("name") == name:
                return summary["knowledgeBaseId"]
    return None


def ensure_kb_index(s3v, vector_bucket: str) -> None:
    """Create the KB's DEDICATED S3 Vectors index (1024 dims, cosine). Idempotent.

    The Knowledge Base owns this index (`relay-kb-docs`) and is its sole writer, so
    its retrieval is consistent — see config.RELAY_KB_INDEX for WHY it is separate
    from Module 4's `relay-docs` DIY index (a Bedrock KB cannot read Module 4's
    raw-vector metadata schema, so mixing the two populations breaks retrieval and
    the reranker). Same vector bucket, same Titan V2 / 1024-dim contract, idle ~$0.
    """
    try:
        s3v.get_index(vectorBucketName=vector_bucket, indexName=config.RELAY_KB_INDEX)
        print(f"  KB index '{config.RELAY_KB_INDEX}': already exists "
              f"({config.EMBED_DIMENSIONS} dims, {config.EMBED_DISTANCE_METRIC}). "
              "Reusing.")
        return
    except ClientError as err:
        if err.response["Error"]["Code"] not in (
            "NotFoundException", "ResourceNotFoundException"
        ):
            raise
    s3v.create_index(
        vectorBucketName=vector_bucket,
        indexName=config.RELAY_KB_INDEX,
        dataType="float32",
        dimension=config.EMBED_DIMENSIONS,
        distanceMetric=config.EMBED_DISTANCE_METRIC,
        # S3 Vectors caps FILTERABLE metadata at 2048 bytes per vector. A Bedrock
        # KB stores the full chunk text under AMAZON_BEDROCK_TEXT and a JSON blob
        # under AMAZON_BEDROCK_METADATA — both routinely exceed 2048 bytes, so they
        # MUST be declared non-filterable or ingestion fails with "Filterable
        # metadata must have at most 2048 bytes". `category` (from our sidecars)
        # stays filterable — that is the multi-tenant lever we want to filter on.
        metadataConfiguration={
            "nonFilterableMetadataKeys": [
                "AMAZON_BEDROCK_TEXT",
                "AMAZON_BEDROCK_METADATA",
            ]
        },
    )
    print(f"  KB index '{config.RELAY_KB_INDEX}': CREATED "
          f"({config.EMBED_DIMENSIONS} dims, {config.EMBED_DISTANCE_METRIC}, "
          "Titan Text Embeddings V2) — KB-owned, separate from Module 4's "
          f"'{config.RELAY_INDEX}'.")


# A freshly created IAM role is not instantly assumable: IAM is eventually
# consistent, so create_knowledge_base can fail with a ValidationException
# ("Bedrock Knowledge Base was unable to assume the given role") for a few seconds
# after ensure_kb_role() created the role. This is a TRANSIENT propagation race, not
# a permissions bug — the role's trust + inline policy are correct. We retry the
# create with a short bounded backoff so a FIRST-EVER setup run (role created this
# same run) succeeds without a manual re-run. Idempotency is preserved: a real
# permission error (different message) still surfaces immediately.
_KB_ROLE_ASSUME_RETRIES = 6
_KB_ROLE_ASSUME_BACKOFF_S = 10


def _is_role_not_assumable(err: ClientError) -> bool:
    """True if the error is the transient 'KB cannot assume the role yet' race."""
    if err.response["Error"]["Code"] != "ValidationException":
        return False
    return "unable to assume" in err.response["Error"]["Message"].lower()


def _create_kb_with_role_retry(agent, *, kwargs: dict):
    """create_knowledge_base, retrying ONLY the transient role-propagation race.

    On the very first setup run the KB service role was created moments earlier and
    IAM has not finished propagating it, so Bedrock cannot assume it yet. We retry
    with a bounded backoff; any other error (or running out of retries) is raised."""
    last_err: ClientError | None = None
    for attempt in range(_KB_ROLE_ASSUME_RETRIES):
        try:
            return agent.create_knowledge_base(**kwargs)
        except ClientError as err:
            if not _is_role_not_assumable(err):
                raise
            last_err = err
            if attempt < _KB_ROLE_ASSUME_RETRIES - 1:
                print(f"    KB role not assumable yet (IAM still propagating) — "
                      f"retry {attempt + 1}/{_KB_ROLE_ASSUME_RETRIES - 1} in "
                      f"{_KB_ROLE_ASSUME_BACKOFF_S}s.")
                time.sleep(_KB_ROLE_ASSUME_BACKOFF_S)
    raise SystemExit(
        "Knowledge Base creation kept failing because Bedrock could not assume "
        f"'{KB_ROLE_NAME}' after {_KB_ROLE_ASSUME_RETRIES} tries. The role's trust "
        "policy allows bedrock.amazonaws.com; if this persists, the role was likely "
        f"deleted mid-run. Last error: {last_err}"
    )


def ensure_knowledge_base(agent, s3v, role_arn: str, vector_bucket: str,
                          account: str) -> str:
    """Create the KB `relay-kb` over its DEDICATED S3 Vectors index. Idempotent.

    The storage config points at the KB-owned `relay-kb-docs` index (by ARN), with
    the Titan V2 embedder and the pinned 1024 dimensions — never a new always-on
    serverless search collection. Returns the KB id.
    """
    existing = _find_kb_by_name(agent, config.RELAY_KB_NAME)
    if existing:
        print(f"  Knowledge Base '{config.RELAY_KB_NAME}': already exists "
              f"(id {existing}). Reusing.")
        return existing

    # Resolve the ARNs of the vector bucket + the KB's own index (never hard-coded).
    vb = s3v.get_vector_bucket(vectorBucketName=vector_bucket)
    vector_bucket_arn = vb["vectorBucket"]["vectorBucketArn"]
    idx = s3v.get_index(
        vectorBucketName=vector_bucket, indexName=config.RELAY_KB_INDEX
    )
    index_arn = idx["index"]["indexArn"]

    created = _create_kb_with_role_retry(
        agent,
        kwargs=dict(
            name=config.RELAY_KB_NAME,
            description="Relay's CloudCart help-center Knowledge Base (Module 5).",
            roleArn=role_arn,
            knowledgeBaseConfiguration={
                "type": "VECTOR",
                "vectorKnowledgeBaseConfiguration": {
                    "embeddingModelArn": (
                        f"arn:aws:bedrock:{REGION}::foundation-model/"
                        f"{config.EMBED_MODEL_ID}"
                    ),
                    "embeddingModelConfiguration": {
                        "bedrockEmbeddingModelConfiguration": {
                            "dimensions": config.EMBED_DIMENSIONS,
                            "embeddingDataType": "FLOAT32",
                        }
                    },
                },
            },
            storageConfiguration={
                "type": "S3_VECTORS",
                "s3VectorsConfiguration": {
                    # The index ARN fully identifies the index. The current Bedrock
                    # API rejects passing indexName ALONGSIDE the ARN ("Vector index
                    # name should not be present with namespace arn") — the ARN is
                    # the namespace, so we address the index by ARN only. (Pass
                    # indexName INSTEAD of the ARN only when you let the KB create it.)
                    "vectorBucketArn": vector_bucket_arn,
                    "indexArn": index_arn,
                },
            },
        ),
    )
    kb_id = created["knowledgeBase"]["knowledgeBaseId"]
    print(f"  Knowledge Base '{config.RELAY_KB_NAME}': CREATED (id {kb_id}, "
          f"S3 Vectors index '{config.RELAY_KB_INDEX}', Titan V2 / "
          f"{config.EMBED_DIMENSIONS} dims).")
    _wait_kb_active(agent, kb_id)
    return kb_id


def _wait_kb_active(agent, kb_id: str) -> None:
    """Block until the KB leaves CREATING (so the first ingestion job can start).

    A freshly created KB is briefly in status CREATING; StartIngestionJob raises
    ConflictException until it is ACTIVE. We poll a short, bounded loop — never a
    silent sleep-and-hope.
    """
    deadline = time.time() + 180
    while time.time() < deadline:
        status = agent.get_knowledge_base(
            knowledgeBaseId=kb_id
        )["knowledgeBase"]["status"]
        if status == "ACTIVE":
            return
        if status in ("FAILED", "DELETING"):
            raise SystemExit(f"Knowledge Base {kb_id} entered status {status}.")
        time.sleep(_INGESTION_POLL_S)
    raise SystemExit(f"Knowledge Base {kb_id} did not become ACTIVE within 180s.")


def _find_data_source_by_name(agent, kb_id: str, name: str) -> str | None:
    for page in agent.get_paginator("list_data_sources").paginate(
        knowledgeBaseId=kb_id
    ):
        for summary in page.get("dataSourceSummaries", []):
            if summary.get("name") == name:
                return summary["dataSourceId"]
    return None


def ensure_data_source(agent, kb_id: str, data_bucket: str, account: str) -> str:
    """Attach the S3 data source over docs/. Idempotent. Returns the data-source id."""
    existing = _find_data_source_by_name(
        agent, kb_id, config.RELAY_KB_DATA_SOURCE_NAME
    )
    if existing:
        print(f"  data source '{config.RELAY_KB_DATA_SOURCE_NAME}': already exists "
              f"(id {existing}). Reusing.")
        return existing

    created = agent.create_data_source(
        knowledgeBaseId=kb_id,
        name=config.RELAY_KB_DATA_SOURCE_NAME,
        dataSourceConfiguration={
            "type": "S3",
            "s3Configuration": {
                "bucketArn": f"arn:aws:s3:::{data_bucket}",
                "inclusionPrefixes": [config.RELAY_KB_INCLUSION_PREFIX],
            },
        },
    )
    ds_id = created["dataSource"]["dataSourceId"]
    print(f"  data source '{config.RELAY_KB_DATA_SOURCE_NAME}': CREATED "
          f"(id {ds_id}, s3://{data_bucket}/{config.RELAY_KB_INCLUSION_PREFIX}).")
    return ds_id


# --- Step 5: the first ingestion job -----------------------------------------
def start_ingestion(agent, kb_id: str, ds_id: str, *, wait: bool) -> str:
    """Start an ingestion job and (optionally) wait for COMPLETE. Returns its status."""
    job = agent.start_ingestion_job(knowledgeBaseId=kb_id, dataSourceId=ds_id)
    job_id = job["ingestionJob"]["ingestionJobId"]
    status = job["ingestionJob"]["status"]
    print(f"  ingestion job {job_id}: started (status {status}).")
    if not wait:
        print("    --no-wait: not blocking on COMPLETE. Check with "
              "list_ingestion_jobs.")
        return status

    deadline = time.time() + _INGESTION_TIMEOUT_S
    while status not in ("COMPLETE", "FAILED", "STOPPED"):
        if time.time() > deadline:
            raise SystemExit(
                f"Ingestion job {job_id} did not finish within "
                f"{_INGESTION_TIMEOUT_S}s (last status {status})."
            )
        time.sleep(_INGESTION_POLL_S)
        got = agent.get_ingestion_job(
            knowledgeBaseId=kb_id, dataSourceId=ds_id, ingestionJobId=job_id
        )
        status = got["ingestionJob"]["status"]
        print(f"    ingestion job {job_id}: {status}")

    if status != "COMPLETE":
        raise SystemExit(f"Ingestion job {job_id} ended in {status}, not COMPLETE.")
    print(f"  ingestion job {job_id}: COMPLETE — docs/ embedded into "
          f"'{config.RELAY_KB_INDEX}'.")
    return status


def _record_ids(kb_id: str, ds_id: str) -> None:
    KB_ID_FILE.write_text(kb_id + "\n", encoding="utf-8")
    KB_DATA_SOURCE_ID_FILE.write_text(ds_id + "\n", encoding="utf-8")
    print(f"  recorded KB id -> {KB_ID_FILE.name}, "
          f"data-source id -> {KB_DATA_SOURCE_ID_FILE.name}.")


# =============================================================================
# Module 6 ADDITION — the attachments/ prefix the intake pipeline writes to.
# =============================================================================
# Module 4 created the data bucket with three prefixes (docs/ attachments/
# vectors/) but only filled docs/. Module 6's intake pipeline is the FIRST writer
# of attachments/: every accepted screenshot is uploaded there. S3 has no real
# "folders", so a prefix only "exists" once an object lives under it. We write a
# tiny zero-byte marker key so the prefix is visible in the console and the
# intake's PutObject lands in a place the teardown knows to purge. Idempotent.
#
# Permissions: this writes with the SAME course credentials the intake runs under
# (boto3 default session / AWS_PROFILE). No new IAM role is created here — the
# intake's S3 PutObject and Comprehend detect_entities run on the caller's
# least-privilege course role (the model card in Module 10 formalizes per-component
# roles). We only ensure the prefix marker exists.
def ensure_attachments_prefix(s3, data_bucket: str) -> None:
    """Make the attachments/ prefix exist (zero-byte marker). Idempotent (overwrites).

    Module 6's intake uploads screenshots to s3://<data_bucket>/attachments/. This
    marker just makes the prefix visible and gives teardown a known place to purge.
    """
    key = f"{config.RELAY_ATTACHMENTS_PREFIX}.keep"
    s3.put_object(
        Bucket=data_bucket, Key=key, Body=b"",
        ContentType="application/octet-stream",
    )
    print(f"  attachments prefix: s3://{data_bucket}/"
          f"{config.RELAY_ATTACHMENTS_PREFIX} ready (intake uploads land here).")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    wait = "--no-wait" not in argv
    leftover = [a for a in argv if a != "--no-wait"]
    if leftover:
        print(f"Unknown argument(s): {' '.join(leftover)}\n"
              "Usage: uv run python setup.py [--no-wait]", file=sys.stderr)
        return 1

    print("Setting up Module 6 — Relay's intake pipeline + the managed Knowledge "
          "Base 'relay-kb'.")
    print("Adds the attachments/ prefix the intake pipeline writes screenshots to;")
    print("creates the KB's own S3 Vectors index 'relay-kb-docs' in Module 4's "
          "vector bucket;")
    print("only PRECHECKS (never writes) Module 4's 'relay-docs' DIY index "
          "(no always-on search cluster).")
    print("Expected cost: a few cents (one Titan ingestion job); KB idle ~$0. The "
          "intake's\nNova Lite vision + Comprehend calls bill per-call (cents) — "
          "no idle billing.\n")

    try:
        acct = config.account_id(_sts())
    except NoCredentialsError:
        print("  [FAIL] no AWS credentials — set AWS_PROFILE=aws-genai-pro.",
              file=sys.stderr)
        return 1

    data_bucket = config.relay_bucket(acct)
    vector_bucket = config.relay_vector_bucket(acct)
    index_name = config.RELAY_INDEX

    s3, s3v, iam, agent = _s3(), _s3vectors(), _iam(), _agent()

    try:
        print("Prerequisites (Module 4 storage layer):")
        precheck_prerequisites(s3, s3v, data_bucket, vector_bucket, index_name)

        print("\nIntake attachments prefix (Module 6):")
        ensure_attachments_prefix(s3, data_bucket)

        print("\nKnowledge Base service role (IAM, least-privilege):")
        role_arn = ensure_kb_role(iam, acct, data_bucket, vector_bucket)

        print("\nKnowledge Base index (Amazon S3 Vectors, KB-owned):")
        ensure_kb_index(s3v, vector_bucket)

        print("\nFilterable metadata (category sidecars in S3):")
        ensure_metadata_sidecars(s3, data_bucket)

        print("\nKnowledge Base (Bedrock, S3 Vectors storage):")
        kb_id = ensure_knowledge_base(agent, s3v, role_arn, vector_bucket, acct)
        ds_id = ensure_data_source(agent, kb_id, data_bucket, acct)

        print("\nFirst ingestion job (parse -> chunk -> embed -> upsert):")
        start_ingestion(agent, kb_id, ds_id, wait=wait)

        _record_ids(kb_id, ds_id)
    except ClientError as err:
        code = err.response["Error"]["Code"]
        message = err.response["Error"]["Message"]
        print(f"\nAWS call failed ({code}):\n  {message}\n\n"
              "If this is AccessDenied, your course IAM role needs bedrock:* on the\n"
              "Knowledge Base, iam:CreateRole/PutRolePolicy/PassRole on relay-kb-role,\n"
              "and the S3/S3 Vectors access from Module 4. See lab.md.",
              file=sys.stderr)
        return 1

    print("\nDone. Try it:")
    print("  uv run python -m relay.intake data/raw/email_billing_error.txt \\")
    print("      --attachment data/raw/payment_error.png")
    print('  uv run python -m relay.kb "How do I change my CloudCart subscription plan?"')
    print("  uv run python compare_retrieval.py")
    print("  uv run python freshness_test.py")
    print("\nResources created (frozen names, 06 §2):")
    print(f"  attachments    : s3://{data_bucket}/{config.RELAY_ATTACHMENTS_PREFIX} "
          "(intake uploads, Module 6)")
    print(f"  knowledge base : {config.RELAY_KB_NAME}")
    print(f"  data source    : {config.RELAY_KB_DATA_SOURCE_NAME} "
          f"(s3://{data_bucket}/docs/)")
    print(f"  service role   : {KB_ROLE_NAME}")
    print(f"  KB vector index: {config.RELAY_KB_INDEX} (KB-owned, S3 Vectors, "
          f"in bucket {vector_bucket})")
    print(f"  M4 DIY index   : {index_name} (Module 4's, unchanged — the "
          "compare_retrieval baseline)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
