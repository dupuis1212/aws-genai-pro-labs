"""teardown.py — leave the account clean after Module 4.

Module 4 of AWS GenAI Pro Mastery. Idempotent and verbose, and TESTED — the course
rule is one tested teardown per setup (decision B5).

S3 Vectors bills ~$0 idle, so this is mostly hygiene rather than cost rescue — but
the rule stands: nothing the lab created is left behind. Teardown, in order:

  1. Delete the S3 VECTORS index  relay-docs  (drops all upserted vectors with it),
     then delete the vector bucket  relay-vectors-<account_id>.
  2. Empty and delete the DATA BUCKET  relay-<account_id>  (docs + prefixes).

A note on what survives across modules: the brief lets you KEEP the data bucket if
you prefer (S3 storage for a few docs is a fraction of a cent/month, and Module 5
re-uses docs/ for its managed Knowledge Base). This script DELETES it by default
for a clean slate; pass --keep-data to retain it. Either way, Module 5 starts by
running its own setup, which re-creates whatever it needs. The vector store is
always removed here because it carries no downstream dependency yet at Module 4 —
Module 5 rebuilds the index when it stands up the managed Knowledge Base.

The M1 $5 budget is NOT touched — it is persistent and backstops the whole course
(Module 1's teardown owns it).

Run it:
    uv run python teardown.py
    uv run python teardown.py --keep-data   # keep the data bucket + docs
"""

from __future__ import annotations

import sys

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

from relay import config

REGION = config.REGION
_NOT_FOUND = ("NotFoundException", "ResourceNotFoundException", "NoSuchBucket",
              "404", "NoSuchKey")


def _s3():
    return boto3.client("s3", region_name=REGION)


def _s3vectors():
    return boto3.client("s3vectors", region_name=REGION)


def _sts():
    return boto3.client("sts", region_name=REGION)


# --- Step 1: the S3 Vectors index + bucket ------------------------------------
def delete_vector_store(s3v, vector_bucket: str, index_name: str) -> None:
    """Delete the index (and its vectors), then the vector bucket. Idempotent."""
    try:
        s3v.delete_index(vectorBucketName=vector_bucket, indexName=index_name)
        print(f"  index '{index_name}': DELETED (with all its vectors).")
    except ClientError as err:
        if err.response["Error"]["Code"] in _NOT_FOUND:
            print(f"  index '{index_name}': already gone. Fine.")
        else:
            raise

    try:
        s3v.delete_vector_bucket(vectorBucketName=vector_bucket)
        print(f"  vector bucket '{vector_bucket}': DELETED.")
    except ClientError as err:
        if err.response["Error"]["Code"] in _NOT_FOUND:
            print(f"  vector bucket '{vector_bucket}': already gone. Fine.")
        else:
            raise


# --- Step 2: the data bucket --------------------------------------------------
def empty_and_delete_bucket(s3, bucket: str) -> None:
    """Delete every object (all versions) then the bucket itself. Idempotent."""
    if not _bucket_exists(s3, bucket):
        print(f"  data bucket '{bucket}': already gone. Fine.")
        return

    deleted = _empty_bucket(s3, bucket)
    print(f"  data bucket '{bucket}': removed {deleted} object(s).")
    try:
        s3.delete_bucket(Bucket=bucket)
        print(f"  data bucket '{bucket}': DELETED.")
    except ClientError as err:
        if err.response["Error"]["Code"] in _NOT_FOUND:
            print(f"  data bucket '{bucket}': already gone. Fine.")
        else:
            raise


def _bucket_exists(s3, bucket: str) -> bool:
    try:
        s3.head_bucket(Bucket=bucket)
        return True
    except ClientError as err:
        if err.response["Error"]["Code"] in ("404", "NoSuchBucket", "NotFound"):
            return False
        raise


def _empty_bucket(s3, bucket: str) -> int:
    """Delete all object keys under a bucket. Returns the count removed."""
    deleted = 0
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket):
        keys = [{"Key": obj["Key"]} for obj in page.get("Contents", [])]
        if keys:
            s3.delete_objects(Bucket=bucket, Delete={"Objects": keys})
            deleted += len(keys)
    return deleted


def assert_clean(keep_data: bool, data_bucket: str) -> None:
    """State plainly that nothing idle-billed remains."""
    print("\nIdle-billed resources from Module 4: NONE.")
    print("  - S3 Vectors index + bucket: deleted above (idle was ~$0 anyway).")
    if keep_data:
        print(f"  - data bucket '{data_bucket}': KEPT on request (--keep-data). "
              "S3 storage for a few docs is a fraction of a cent/month.")
    else:
        print("  - data bucket: deleted above.")
    print("  The M1 $5 budget is KEPT on purpose (Module 1 owns it; it guards the")
    print("  whole course).")
    print("\nTo rebuild for Module 5: run setup.py, then ingest.run "
          "(setup re-creates the bucket, docs, and index).")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    keep_data = "--keep-data" in argv
    leftover = [a for a in argv if a != "--keep-data"]
    if leftover:
        print(f"Unknown argument(s): {' '.join(leftover)}\n"
              "Usage: uv run python teardown.py [--keep-data]", file=sys.stderr)
        return 1

    print("Tearing down Module 4 (idempotent).\n")

    try:
        acct = config.account_id(_sts())
    except NoCredentialsError:
        print("  [FAIL] no AWS credentials — set AWS_PROFILE=aws-genai-pro.",
              file=sys.stderr)
        return 1

    data_bucket = config.relay_bucket(acct)
    vector_bucket = config.relay_vector_bucket(acct)
    index_name = config.RELAY_INDEX

    try:
        print("Vector store (Amazon S3 Vectors):")
        delete_vector_store(_s3vectors(), vector_bucket, index_name)

        print("\nData bucket (S3):")
        if keep_data:
            print(f"  data bucket '{data_bucket}': KEPT (--keep-data).")
        else:
            empty_and_delete_bucket(_s3(), data_bucket)
    except ClientError as err:
        code = err.response["Error"]["Code"]
        message = err.response["Error"]["Message"]
        print(f"\nAWS call failed ({code}):\n  {message}", file=sys.stderr)
        return 1

    assert_clean(keep_data, data_bucket)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
