"""setup.py — stand up the agent's tables + MCP Lambda, on top of the KB (Module 7).

Module 7 of AWS GenAI Pro Mastery. Idempotent and verbose: safe to run twice, and
it tells you exactly what it creates, what it reuses, and what it costs.

MODULE 7 ADDS (the agent's standing resources — see module_07_setup() below):
  A. two DynamoDB ON-DEMAND tables — `relay-orders` (SEEDED with the 25 orders in
     data/orders.json, so lookup_order returns a real status) and `relay-tickets`
     (where the agent persists a TicketRecord). On-demand -> ~$0 idle.
  B. an IAM-BOUNDED execution role for the MCP Lambda (skill 2.1.3): it can read ONLY
     relay-orders and write ONLY relay-tickets — nothing else. This is the IAM
     resource boundary the article and the lab demo (a write outside relay-tickets is
     denied by IAM, not just by convention).
  C. the stateless CloudCart **MCP server** (mcp_server/) packaged + deployed to an
     AWS Lambda, fronted by a Lambda **Function URL**, whose URL is recorded in
     .mcp_url so relay/tools.py can build an MCP client. The agent (relay.agent) is the
     MCP client; lookup_order / create_ticket are served by this Lambda.

It is ADDITIVE on top of the inherited Module 6 + Module 5 setup, which it KEEPS
running unchanged (the agent's search_kb tool retrieves from the same KB):

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
    uv run python setup.py --skip-kb    # only the Module 7 tables + MCP Lambda (KB
                                        # already set up) — handy for re-deploys
"""

from __future__ import annotations

import io
import json
import re
import sys
import time
import zipfile
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

# --- Module 7 names (the agent's resources) ----------------------------------
ORDERS_SEED_FILE = _ROOT / "data" / "orders.json"
MCP_URL_FILE = _ROOT / config.MCP_URL_FILE_NAME

# The IAM execution role the MCP Lambda runs under — bounded to relay-orders (read)
# and relay-tickets (write) ONLY (skill 2.1.3). Frozen name so teardown finds it.
MCP_LAMBDA_ROLE_NAME = "relay-mcp-lambda-role"

# The MCP server Lambda function name + its handler (Function-URL adapter in app.py).
MCP_LAMBDA_NAME = "relay-mcp-server"
MCP_LAMBDA_HANDLER = "mcp_server.app.handler"
# Generous Lambda timeout: an MCP request is one quick DynamoDB call, but cold start +
# import of strands/mcp wants headroom. Kept well under API/agent timeouts.
MCP_LAMBDA_TIMEOUT_S = 30
MCP_LAMBDA_MEMORY_MB = 512
# The Python runtime the Lambda runs on (matches the course pin, Python 3.12).
MCP_LAMBDA_RUNTIME = "python3.12"


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


# =============================================================================
# Module 7 ADDITION — the agent's tables, the IAM-bounded MCP Lambda + Function URL.
# =============================================================================
# Everything below is the Module 7 increment, additive on top of the inherited M5/M6
# KB setup above. Same containment law: every name comes from relay.config, the account
# id is resolved from STS at run time, never hard-coded.
def _dynamodb():
    return boto3.client("dynamodb", region_name=REGION)


def _dynamodb_resource():
    return boto3.resource("dynamodb", region_name=REGION)


def _lambda():
    return boto3.client("lambda", region_name=REGION)


# --- A: the two on-demand DynamoDB tables ------------------------------------
def ensure_table(ddb, *, name: str, key_attr: str) -> None:
    """Create an ON-DEMAND DynamoDB table with a single string hash key. Idempotent.

    PAY_PER_REQUEST -> no provisioned capacity, ~$0 idle, nothing to scale down at
    teardown. We wait until ACTIVE so seeding (orders) can write immediately.
    """
    try:
        ddb.describe_table(TableName=name)
        print(f"  table '{name}': already exists. Reusing.")
        return
    except ClientError as err:
        if err.response["Error"]["Code"] != "ResourceNotFoundException":
            raise
    ddb.create_table(
        TableName=name,
        AttributeDefinitions=[{"AttributeName": key_attr, "AttributeType": "S"}],
        KeySchema=[{"AttributeName": key_attr, "KeyType": "HASH"}],
        BillingMode="PAY_PER_REQUEST",  # on-demand: ~$0 idle
    )
    ddb.get_waiter("table_exists").wait(TableName=name)
    print(f"  table '{name}': CREATED (on-demand, hash key '{key_attr}', ~$0 idle).")


def seed_orders_table(resource) -> int:
    """Seed relay-orders from data/orders.json (idempotent upsert). Returns the count."""
    from mcp_server import store

    items = json.loads(ORDERS_SEED_FILE.read_text(encoding="utf-8"))
    written = store.seed_orders(items, resource=resource)
    print(f"  relay-orders: SEEDED {written} CloudCart orders from "
          f"{ORDERS_SEED_FILE.name} (idempotent upsert).")
    return written


# --- B: the IAM-bounded MCP Lambda execution role (skill 2.1.3) --------------
def _lambda_trust_policy() -> str:
    return json.dumps({
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "lambda.amazonaws.com"},
            "Action": "sts:AssumeRole",
        }],
    })


def _mcp_lambda_policy(account: str) -> str:
    """The LEAST-PRIVILEGE inline policy for the MCP Lambda (skill 2.1.3).

    The IAM RESOURCE BOUNDARY the article and the lab demo: the Lambda can
      - READ relay-orders (GetItem/Query) and
      - WRITE relay-tickets (PutItem/GetItem/UpdateItem) —
    and NOTHING else. No '*' on resources; the table ARNs are explicit. A write to any
    other table is denied by IAM at call time, not just by convention. Plus the basic
    CloudWatch Logs grant any Lambda needs (scoped to its own log group).
    """
    orders_arn = f"arn:aws:dynamodb:{REGION}:{account}:table/{config.RELAY_ORDERS_TABLE}"
    tickets_arn = f"arn:aws:dynamodb:{REGION}:{account}:table/{config.RELAY_TICKETS_TABLE}"
    log_arn = (f"arn:aws:logs:{REGION}:{account}:log-group:"
               f"/aws/lambda/{MCP_LAMBDA_NAME}:*")
    return json.dumps({
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "ReadOrdersOnly",
                "Effect": "Allow",
                "Action": ["dynamodb:GetItem", "dynamodb:Query",
                           "dynamodb:BatchGetItem"],
                "Resource": [orders_arn],
            },
            {
                "Sid": "WriteTicketsOnly",
                "Effect": "Allow",
                "Action": ["dynamodb:PutItem", "dynamodb:GetItem",
                           "dynamodb:UpdateItem"],
                "Resource": [tickets_arn],
            },
            {
                "Sid": "OwnLogsOnly",
                "Effect": "Allow",
                "Action": ["logs:CreateLogGroup", "logs:CreateLogStream",
                           "logs:PutLogEvents"],
                "Resource": [log_arn],
            },
        ],
    })


def ensure_mcp_lambda_role(iam, account: str) -> str:
    """Create (or reuse) the IAM-bounded MCP Lambda role + its inline policy. Returns ARN."""
    try:
        role = iam.get_role(RoleName=MCP_LAMBDA_ROLE_NAME)
        print(f"  IAM role '{MCP_LAMBDA_ROLE_NAME}': already exists. Reusing.")
    except ClientError as err:
        if err.response["Error"]["Code"] != "NoSuchEntity":
            raise
        role = iam.create_role(
            RoleName=MCP_LAMBDA_ROLE_NAME,
            AssumeRolePolicyDocument=_lambda_trust_policy(),
            Description="Relay CloudCart MCP server Lambda role (Module 7) - bounded "
                        "to relay-orders (read) + relay-tickets (write).",
        )
        print(f"  IAM role '{MCP_LAMBDA_ROLE_NAME}': CREATED.")
    iam.put_role_policy(
        RoleName=MCP_LAMBDA_ROLE_NAME,
        PolicyName="relay-mcp-permissions",
        PolicyDocument=_mcp_lambda_policy(account),
    )
    print("    inline policy 'relay-mcp-permissions': put (read relay-orders, "
          "write relay-tickets, own logs — nothing else).")
    return role["Role"]["Arn"]


# --- C: package + deploy the MCP server Lambda, expose a Function URL ----------
# The Lambda runs on Linux x86_64 / Python 3.12 (MCP_LAMBDA_RUNTIME). One runtime
# dep — pydantic_core — ships a COMPILED (Rust) extension, so the wheel is
# platform-specific: a macOS/arm64 build will not import on Lambda
# ("No module named 'pydantic_core._pydantic_core'"). So we RESOLVE the third-party
# deps for the Lambda TARGET (manylinux x86_64) rather than copy the host venv's
# site-packages. `uv pip install --python-platform x86_64-manylinux2014
# --python-version 3.12 --target <dir>` downloads the right manylinux wheels (a no-op
# extra resolve when the host already matches). boto3/botocore are provided by the
# Lambda runtime, so they are excluded to keep the zip small.
LAMBDA_TARGET_PLATFORM = "x86_64-manylinux2014"
LAMBDA_TARGET_PY = "3.12"
# The handler's third-party runtime deps (mcp + pydantic pull the rest transitively;
# starlette/uvicorn/sse-starlette/python-multipart back FastMCP's streamable-HTTP
# transport). Pinned to the project's locked versions so the deployed code matches.
LAMBDA_RUNTIME_DEPS = (
    "mcp~=1.27",
    "pydantic~=2.0",
    "starlette",
    "sse-starlette",
    "uvicorn",
    "python-multipart",
)


def _build_lambda_zip() -> bytes:
    """Build the Lambda deployment zip: mcp_server/ + relay/ + Linux-target deps.

    The handler imports mcp_server.app -> mcp_server.server (FastMCP) and
    relay.config/relay.models, plus the `mcp` / `pydantic` / `starlette` packages.
    We zip the two first-party packages (pure Python, copied from the repo) and the
    third-party deps RESOLVED FOR THE LAMBDA TARGET (manylinux x86_64 / Py 3.12) so the
    compiled pydantic_core wheel matches the runtime — not the host's macOS/arm64 build.
    (In a real pipeline this is a container image or a layer; for the lab a targeted
    `uv pip install --target` zip is the simplest reproducible build.)
    """
    import subprocess
    import tempfile

    buf = io.BytesIO()
    with tempfile.TemporaryDirectory() as tmp:
        deps_dir = Path(tmp) / "deps"
        deps_dir.mkdir()
        # Resolve the runtime deps FOR THE LAMBDA TARGET into deps_dir. This pulls the
        # manylinux x86_64 pydantic_core wheel, so the extension imports on Lambda.
        cmd = [
            "uv", "pip", "install",
            "--python-platform", LAMBDA_TARGET_PLATFORM,
            "--python-version", LAMBDA_TARGET_PY,
            "--target", str(deps_dir),
            *LAMBDA_RUNTIME_DEPS,
        ]
        print(f"    resolving Lambda deps for {LAMBDA_TARGET_PLATFORM} "
              f"(py{LAMBDA_TARGET_PY})...")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise SystemExit(
                "Failed to resolve the Lambda's deps for the Linux target with uv:\n"
                f"  cmd: {' '.join(cmd)}\n{proc.stderr.strip()}\n"
                "Install uv (https://docs.astral.sh/uv/) or build the zip on a Linux "
                "x86_64 host."
            )

        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            # First-party packages (pure Python) from the repo.
            for pkg in ("mcp_server", "relay"):
                _add_tree(zf, _ROOT / pkg, arc_root=pkg)
            # Third-party deps, resolved for the Lambda target. Everything uv put in
            # deps_dir (each top-level dir/module) goes in at the zip root, skipping
            # dist-info/caches we do not need at runtime.
            for path in sorted(deps_dir.rglob("*")):
                if "__pycache__" in path.parts or path.suffix in (".pyc", ".pyo"):
                    continue
                if path.is_file():
                    zf.write(path, path.relative_to(deps_dir).as_posix())
    return buf.getvalue()


def _add_tree(zf: zipfile.ZipFile, root: Path, *, arc_root: str) -> None:
    """Add a directory tree to the zip under arc_root/, skipping caches."""
    for path in sorted(root.rglob("*")):
        if "__pycache__" in path.parts or path.suffix in (".pyc", ".pyo"):
            continue
        if path.is_file():
            zf.write(path, f"{arc_root}/{path.relative_to(root).as_posix()}")


def ensure_mcp_lambda(lmb, role_arn: str, zip_bytes: bytes) -> str:
    """Create or update the MCP server Lambda. Returns its ARN. Idempotent.

    A freshly created role is not instantly assumable (IAM is eventually consistent),
    so create_function can fail transiently with InvalidParameterValueException ("The
    role ... cannot be assumed") — we retry that ONE transient with a short backoff.
    """
    try:
        existing = lmb.get_function(FunctionName=MCP_LAMBDA_NAME)
        print(f"  Lambda '{MCP_LAMBDA_NAME}': exists — updating code + config.")
        lmb.update_function_code(FunctionName=MCP_LAMBDA_NAME, ZipFile=zip_bytes)
        lmb.get_waiter("function_updated").wait(FunctionName=MCP_LAMBDA_NAME)
        lmb.update_function_configuration(
            FunctionName=MCP_LAMBDA_NAME, Role=role_arn, Handler=MCP_LAMBDA_HANDLER,
            Runtime=MCP_LAMBDA_RUNTIME, Timeout=MCP_LAMBDA_TIMEOUT_S,
            MemorySize=MCP_LAMBDA_MEMORY_MB,
        )
        lmb.get_waiter("function_updated").wait(FunctionName=MCP_LAMBDA_NAME)
        return existing["Configuration"]["FunctionArn"]
    except ClientError as err:
        if err.response["Error"]["Code"] != "ResourceNotFoundException":
            raise

    arn = _create_function_with_retry(lmb, role_arn, zip_bytes)
    lmb.get_waiter("function_active_v2").wait(FunctionName=MCP_LAMBDA_NAME)
    print(f"  Lambda '{MCP_LAMBDA_NAME}': CREATED ({MCP_LAMBDA_RUNTIME}, "
          f"{MCP_LAMBDA_MEMORY_MB} MB, {MCP_LAMBDA_TIMEOUT_S}s timeout).")
    return arn


_ROLE_ASSUME_RETRIES = 6
_ROLE_ASSUME_BACKOFF_S = 8


def _create_function_with_retry(lmb, role_arn: str, zip_bytes: bytes) -> str:
    last: ClientError | None = None
    for attempt in range(_ROLE_ASSUME_RETRIES):
        try:
            created = lmb.create_function(
                FunctionName=MCP_LAMBDA_NAME,
                Runtime=MCP_LAMBDA_RUNTIME,
                Role=role_arn,
                Handler=MCP_LAMBDA_HANDLER,
                Code={"ZipFile": zip_bytes},
                Timeout=MCP_LAMBDA_TIMEOUT_S,
                MemorySize=MCP_LAMBDA_MEMORY_MB,
                Description="Relay's stateless CloudCart MCP server (Module 7).",
            )
            return created["FunctionArn"]
        except ClientError as err:
            code = err.response["Error"]["Code"]
            msg = err.response["Error"]["Message"].lower()
            if code == "InvalidParameterValueException" and "cannot be assumed" in msg:
                last = err
                if attempt < _ROLE_ASSUME_RETRIES - 1:
                    print(f"    role not assumable yet (IAM propagating) — retry "
                          f"{attempt + 1}/{_ROLE_ASSUME_RETRIES - 1} in "
                          f"{_ROLE_ASSUME_BACKOFF_S}s.")
                    time.sleep(_ROLE_ASSUME_BACKOFF_S)
                    continue
            raise
    raise SystemExit(
        f"Lambda creation kept failing because the role could not be assumed after "
        f"{_ROLE_ASSUME_RETRIES} tries. Last error: {last}"
    )


def ensure_function_url(lmb, account: str) -> str:
    """Create (or reuse) a public Lambda Function URL for the MCP server. Returns it.

    The Function URL is the agent's MCP endpoint. AuthType NONE keeps the lab simple
    (the article's 'In production' box flags the managed gateway/identity story for real
    endpoint auth, Module 8) — the IAM boundary on what the Lambda can TOUCH is the
    security control this module teaches, not endpoint auth. We add the public invoke
    permission the Function URL needs. Idempotent.
    """
    try:
        existing = lmb.get_function_url_config(FunctionName=MCP_LAMBDA_NAME)
        url = existing["FunctionUrl"]
        print(f"  Function URL: already exists. Reusing.")
    except ClientError as err:
        if err.response["Error"]["Code"] != "ResourceNotFoundException":
            raise
        created = lmb.create_function_url_config(
            FunctionName=MCP_LAMBDA_NAME, AuthType="NONE",
        )
        url = created["FunctionUrl"]
        print(f"  Function URL: CREATED (AuthType NONE — lab simplicity).")

    # The public-invoke permission the Function URL needs (idempotent: ignore exists).
    try:
        lmb.add_permission(
            FunctionName=MCP_LAMBDA_NAME,
            StatementId="relay-mcp-function-url",
            Action="lambda:InvokeFunctionUrl",
            Principal="*",
            FunctionUrlAuthType="NONE",
        )
    except ClientError as err:
        if err.response["Error"]["Code"] != "ResourceConflictException":
            raise

    # Record the MCP endpoint (URL + the mount path) so relay/tools.py finds it.
    mcp_url = url.rstrip("/") + config.MCP_SERVER_PATH
    MCP_URL_FILE.write_text(mcp_url + "\n", encoding="utf-8")
    print(f"  recorded MCP endpoint -> {MCP_URL_FILE.name}: {mcp_url}")
    return mcp_url


def module_07_setup(*, account: str) -> None:
    """Stand up the Module 7 agent resources: tables (seeded), IAM role, MCP Lambda.

    Idempotent and verbose. Called by main() after the inherited KB setup.
    """
    ddb = _dynamodb()
    ddb_resource = _dynamodb_resource()
    iam = _iam()
    lmb = _lambda()

    print("\nAgent business tables (Amazon DynamoDB, on-demand):")
    ensure_table(ddb, name=config.RELAY_ORDERS_TABLE, key_attr=config.ORDERS_KEY)
    ensure_table(ddb, name=config.RELAY_TICKETS_TABLE, key_attr=config.TICKETS_KEY)
    seed_orders_table(ddb_resource)

    print("\nMCP Lambda execution role (IAM, least-privilege resource boundary):")
    role_arn = ensure_mcp_lambda_role(iam, account)

    print("\nCloudCart MCP server (AWS Lambda, stateless):")
    print("  packaging mcp_server/ + relay/ + deps into a deployment zip...")
    zip_bytes = _build_lambda_zip()
    print(f"  zip built: {len(zip_bytes) // 1024} KiB.")
    ensure_mcp_lambda(lmb, role_arn, zip_bytes)
    ensure_function_url(lmb, account)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    wait = "--no-wait" not in argv
    skip_kb = "--skip-kb" in argv
    leftover = [a for a in argv if a not in ("--no-wait", "--skip-kb")]
    if leftover:
        print(f"Unknown argument(s): {' '.join(leftover)}\n"
              "Usage: uv run python setup.py [--no-wait] [--skip-kb]", file=sys.stderr)
        return 1

    print("Setting up Module 7 — Relay as a Strands agent: the agent's DynamoDB tables "
          "+ the\nstateless CloudCart MCP server on AWS Lambda, on top of the inherited "
          "Knowledge Base.")
    print("Adds: relay-orders (seeded 25) + relay-tickets (on-demand, ~$0 idle); an "
          "IAM-bounded\nMCP Lambda role (read relay-orders, write relay-tickets ONLY) + "
          "the MCP Lambda + URL.")
    if not skip_kb:
        print("Keeps the inherited Module 6 intake attachments/ prefix and the Module 5 "
              "Knowledge\nBase 'relay-kb' (the agent's search_kb tool retrieves from it).")
    print("Expected cost: tables/Lambda/Function URL idle ~$0; one small Titan KB "
          "ingestion is\na few cents. The agent's smart-tier runs bill per-call (cents) "
          "— no idle billing.\n")

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
        if not skip_kb:
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
        else:
            print("--skip-kb: leaving the inherited Module 5/6 KB setup untouched.")

        # --- Module 7: the agent's tables + the MCP Lambda + Function URL ----------
        module_07_setup(account=acct)
    except ClientError as err:
        code = err.response["Error"]["Code"]
        message = err.response["Error"]["Message"]
        print(f"\nAWS call failed ({code}):\n  {message}\n\n"
              "If this is AccessDenied, your course IAM role needs: bedrock:* on the\n"
              "Knowledge Base; iam:CreateRole/PutRolePolicy/PassRole on relay-kb-role +\n"
              "relay-mcp-lambda-role; dynamodb:* on relay-orders/relay-tickets; and\n"
              "lambda:CreateFunction/CreateFunctionUrlConfig/AddPermission. See lab.md.",
              file=sys.stderr)
        return 1

    print("\nDone. Try it:")
    print('  uv run python -m relay.agent "Where is order 1042? It was supposed to '
          'arrive Monday."')
    print('  uv run python -m relay.agent "How do refunds work?"   '
          '# -> the agent chooses search_kb')
    print("\nResources created (frozen names, 06 §2):")
    print(f"  orders table   : {config.RELAY_ORDERS_TABLE} (DynamoDB on-demand, seeded 25)")
    print(f"  tickets table  : {config.RELAY_TICKETS_TABLE} (DynamoDB on-demand)")
    print(f"  MCP Lambda role: {MCP_LAMBDA_ROLE_NAME} (bounded: read orders, write tickets)")
    print(f"  MCP server     : {MCP_LAMBDA_NAME} (AWS Lambda, stateless; URL in "
          f"{MCP_URL_FILE.name})")
    if not skip_kb:
        print(f"  knowledge base : {config.RELAY_KB_NAME} (inherited; search_kb backend)")
        print(f"  KB service role: {KB_ROLE_NAME}")
        print(f"  KB vector index: {config.RELAY_KB_INDEX} (KB-owned, S3 Vectors, "
              f"in bucket {vector_bucket})")
        print(f"  M4 DIY index   : {index_name} (Module 4's, unchanged — the "
              "compare_retrieval baseline)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
