"""setup.py — provision Relay's data bucket and S3 Vectors store (Module 4).

Module 4 of AWS GenAI Pro Mastery. Idempotent and verbose: safe to run twice, and
it tells you exactly what it creates and what it costs.

This is the module that INTRODUCES Relay's storage layer — the bucket and vector
store the managed Knowledge Base (M5) and the agent (M7) build on. setup.py:

  1. Creates the DATA BUCKET  relay-<account_id>  with the three canonical
     prefixes (docs/, attachments/, vectors/). attachments/ is created empty now
     and filled by the multimodal intake at Module 6.
  2. Uploads every doc in data/docs/ under the docs/ prefix (the corpus Relay
     learns CloudCart from).
  3. Creates the S3 VECTORS bucket  relay-vectors-<account_id>  and the index
     relay-docs  (1024 dimensions, cosine distance — the Titan V2 contract).

It does NOT create any always-on search cluster, Aurora, or a managed Knowledge
Base. S3 Vectors is the course's choice (GA December 2025): it bills ~$0 idle,
where a provisioned serverless search cluster would bill ~$174/month around the
clock (the article makes the full vector-store comparison).

What it costs:
  - S3 storage for a few dozen small docs: a fraction of a cent per month.
  - S3 Vectors bucket + empty index: $0 to create; storage is $0.06/GB-month and
    queries $2.50/M (as of June 2026 — re-verify on the S3 pricing page). Idle ~$0.

Every resource name and the embedder ID come from relay/config.py — none is typed
here. The account ID is resolved from STS at run time, never hard-coded.

Run it:
    uv run python setup.py
    uv run python setup.py --skip-upload   # create buckets/index, skip doc upload
"""

from __future__ import annotations

import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

from relay import config

REGION = config.REGION

_ROOT = Path(__file__).resolve().parent
DOCS_DIR = _ROOT / "data" / "docs"


def _s3():
    return boto3.client("s3", region_name=REGION)


def _s3vectors():
    return boto3.client("s3vectors", region_name=REGION)


def _sts():
    return boto3.client("sts", region_name=REGION)


# --- Step 1: the data bucket + its three prefixes -----------------------------
def ensure_data_bucket(s3, bucket: str) -> None:
    """Create the data bucket if missing, then seed the three prefixes. Idempotent.

    S3 has no real folders; a "prefix" is just a key. We create a zero-byte marker
    at each prefix so the structure is visible in the console and tools, and so
    attachments/ and vectors/ exist before M6/M12 need them.
    """
    if _bucket_exists(s3, bucket):
        print(f"  data bucket '{bucket}': already exists. Reusing.")
    else:
        _create_bucket(s3, bucket)
        print(f"  data bucket '{bucket}': CREATED.")

    for prefix in config.RELAY_BUCKET_PREFIXES:
        # Put an empty marker only if the prefix has no objects yet (idempotent).
        existing = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
        if existing.get("KeyCount", 0) == 0:
            s3.put_object(Bucket=bucket, Key=prefix)
            print(f"    prefix '{prefix}': created (empty marker).")
        else:
            print(f"    prefix '{prefix}': already populated.")


def _bucket_exists(s3, bucket: str) -> bool:
    try:
        s3.head_bucket(Bucket=bucket)
        return True
    except ClientError as err:
        if err.response["Error"]["Code"] in ("404", "NoSuchBucket", "NotFound"):
            return False
        if err.response["Error"]["Code"] == "403":
            # Exists but owned by someone else — surface it, do not pretend it's ours.
            raise
        return False


def _create_bucket(s3, bucket: str) -> None:
    """Create a bucket in REGION (us-east-1 needs no LocationConstraint)."""
    if REGION == "us-east-1":
        s3.create_bucket(Bucket=bucket)
    else:
        s3.create_bucket(
            Bucket=bucket,
            CreateBucketConfiguration={"LocationConstraint": REGION},
        )


# --- Step 2: upload the CloudCart docs under docs/ ----------------------------
def upload_docs(s3, bucket: str, docs_dir: Path = DOCS_DIR) -> int:
    """Upload every *.md doc under the docs/ prefix. Returns the count uploaded."""
    docs = sorted(docs_dir.glob("*.md"))
    if not docs:
        raise SystemExit(f"No docs found in {docs_dir}.")
    for doc in docs:
        key = f"docs/{doc.name}"
        s3.upload_file(str(doc), bucket, key,
                       ExtraArgs={"ContentType": "text/markdown"})
        print(f"    uploaded docs/{doc.name}")
    return len(docs)


# --- Step 3: the S3 Vectors bucket + the relay-docs index ---------------------
def ensure_vector_bucket(s3v, vector_bucket: str) -> None:
    """Create the S3 Vectors bucket if missing. Idempotent."""
    try:
        s3v.get_vector_bucket(vectorBucketName=vector_bucket)
        print(f"  vector bucket '{vector_bucket}': already exists. Reusing.")
    except ClientError as err:
        if err.response["Error"]["Code"] in ("NotFoundException", "ResourceNotFoundException"):
            s3v.create_vector_bucket(vectorBucketName=vector_bucket)
            print(f"  vector bucket '{vector_bucket}': CREATED.")
        else:
            raise


def ensure_index(s3v, vector_bucket: str, index_name: str) -> None:
    """Create the relay-docs index if missing (1024 dims, cosine). Idempotent.

    `nonFilterableMetadataKeys` excludes the human-readable `snippet` from the
    filter structures (it is only for inspection in compare_chunking.py). The
    keys we DO filter on — category, source_uri, chunk_index, strategy, heading —
    stay filterable.
    """
    try:
        s3v.get_index(vectorBucketName=vector_bucket, indexName=index_name)
        print(f"  index '{index_name}': already exists "
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
        indexName=index_name,
        dataType="float32",
        dimension=config.EMBED_DIMENSIONS,
        distanceMetric=config.EMBED_DISTANCE_METRIC,
        metadataConfiguration={"nonFilterableMetadataKeys": ["snippet"]},
    )
    print(f"  index '{index_name}': CREATED "
          f"({config.EMBED_DIMENSIONS} dims, {config.EMBED_DISTANCE_METRIC}, "
          "Titan Text Embeddings V2).")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    skip_upload = "--skip-upload" in argv
    leftover = [a for a in argv if a != "--skip-upload"]
    if leftover:
        print(f"Unknown argument(s): {' '.join(leftover)}\n"
              "Usage: uv run python setup.py [--skip-upload]", file=sys.stderr)
        return 1

    print("Setting up Module 4 — Relay's data bucket and S3 Vectors store.")
    print("Expected cost: a fraction of a cent (S3 storage); S3 Vectors idle ~$0.")
    print("NO always-on search cluster is created (that would bill ~$174/month).\n")

    try:
        acct = config.account_id(_sts())
    except NoCredentialsError:
        print("  [FAIL] no AWS credentials — set AWS_PROFILE=aws-genai-pro.",
              file=sys.stderr)
        return 1

    data_bucket = config.relay_bucket(acct)
    vector_bucket = config.relay_vector_bucket(acct)
    index_name = config.RELAY_INDEX

    s3 = _s3()
    s3v = _s3vectors()

    try:
        print("Data bucket (S3):")
        ensure_data_bucket(s3, data_bucket)

        if not skip_upload:
            print("\nUploading CloudCart docs under docs/:")
            count = upload_docs(s3, data_bucket)
            print(f"  {count} docs uploaded.")
        else:
            print("\nDoc upload: SKIPPED (--skip-upload).")

        print("\nVector store (Amazon S3 Vectors):")
        ensure_vector_bucket(s3v, vector_bucket)
        ensure_index(s3v, vector_bucket, index_name)
    except ClientError as err:
        code = err.response["Error"]["Code"]
        message = err.response["Error"]["Message"]
        print(f"\nAWS call failed ({code}):\n  {message}\n\n"
              "If this is AccessDenied, your course IAM role needs s3:* on the two\n"
              "relay-* buckets and s3vectors:* on the vector bucket. See lab.md.",
              file=sys.stderr)
        return 1

    print("\nDone. Next — ingest the docs under each chunking strategy:")
    print("  uv run python -m ingest.run --strategy fixed")
    print("  uv run python -m ingest.run --strategy hierarchical")
    print("  uv run python -m ingest.run --strategy semantic")
    print("  uv run python compare_chunking.py")
    print("\nResources created (frozen names, 06 §2):")
    print(f"  data bucket   : {data_bucket}  (docs/ attachments/ vectors/)")
    print(f"  vector bucket : {vector_bucket}")
    print(f"  index         : {index_name}  "
          f"({config.EMBED_DIMENSIONS} dims, {config.EMBED_DISTANCE_METRIC})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
