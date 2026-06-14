"""freshness_test.py — prove the Knowledge Base stays fresh after a doc changes.

The observable result of Module 5's data-maintenance half (skill 1.4.5). A managed
Knowledge Base does not magically know the docs changed — you re-run an INGESTION
JOB, and the KB does an INCREMENTAL sync: it detects which objects changed (by
S3 ETag / last-modified) and re-embeds only those, not the whole corpus. This
script proves it end to end:

  1. ASK the KB the price of the Growth plan, BEFORE. Record the answer + citation.
  2. EDIT data/docs/billing-plans.md — change the Growth price (e.g. $79 -> $99) —
     and re-upload it to s3://relay-<account_id>/docs/.
  3. RE-SYNC: StartIngestionJob and wait for COMPLETE. The job re-embeds only the
     one changed doc (incremental change detection), not all of them.
  4. ASK the same question AFTER, and show the new price in the answer.
  5. RESTORE the doc to its original price and re-sync, so the lab is repeatable
     (unless you pass --no-restore).

This is the test that turns "the KB syncs automatically" from a claim into a proof.
In production you would not edit a doc by hand — an EventBridge schedule or an S3
event triggers StartIngestionJob (the event-driven version is Module 11); the
mechanism the job uses is the same incremental sync you watch here.

Run it after setup.py (the KB must exist and be synced):
    uv run python freshness_test.py
    uv run python freshness_test.py --new-price '$129'   # use a different price
    uv run python freshness_test.py --no-restore         # leave the edit in place
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

from relay import config, kb

REGION = config.REGION

_ROOT = Path(__file__).resolve().parent
PLANS_DOC = _ROOT / "data" / "docs" / "billing-plans.md"
PLANS_KEY = "docs/billing-plans.md"

QUESTION = "How much does the Growth plan cost per month?"
ORIGINAL_PRICE = "$79"
DEFAULT_NEW_PRICE = "$99"

_INGESTION_TIMEOUT_S = 600
_INGESTION_POLL_S = 10


def _sts():
    return boto3.client("sts", region_name=REGION)


def _s3():
    return boto3.client("s3", region_name=REGION)


def _agent():
    return boto3.client("bedrock-agent", region_name=REGION)


def _set_growth_price(doc_text: str, new_price: str) -> str:
    """Replace the Growth plan's monthly price in the doc. Returns the new text.

    The doc line reads: '- **Growth** — $79 per month.' We rewrite the dollar
    amount immediately after the bolded plan name, leaving the rest untouched.
    """
    pattern = re.compile(r"(\*\*Growth\*\*\s+—\s+)\$\d+(\s+per month)")
    new_text, n = pattern.subn(rf"\g<1>{new_price}\g<2>", doc_text)
    if n == 0:
        raise SystemExit(
            "Could not find the Growth price line in billing-plans.md — did the "
            "doc change shape? Expected '- **Growth** — $NN per month.'"
        )
    return new_text


def _upload_doc(s3, bucket: str, text: str) -> None:
    s3.put_object(
        Bucket=bucket, Key=PLANS_KEY,
        Body=text.encode("utf-8"), ContentType="text/markdown",
    )


def _resync(agent, kb_id: str, ds_id: str) -> None:
    """StartIngestionJob and wait for COMPLETE (incremental — only changed docs)."""
    job = agent.start_ingestion_job(knowledgeBaseId=kb_id, dataSourceId=ds_id)
    ingestion_job = job["ingestionJob"]
    job_id = ingestion_job["ingestionJobId"]
    status = ingestion_job["status"]
    print(f"    ingestion job {job_id}: started ({status}) — incremental sync.")
    deadline = time.time() + _INGESTION_TIMEOUT_S
    while status not in ("COMPLETE", "FAILED", "STOPPED"):
        if time.time() > deadline:
            raise SystemExit(f"Ingestion job {job_id} timed out ({status}).")
        time.sleep(_INGESTION_POLL_S)
        ingestion_job = agent.get_ingestion_job(
            knowledgeBaseId=kb_id, dataSourceId=ds_id, ingestionJobId=job_id
        )["ingestionJob"]
        status = ingestion_job["status"]
    if status != "COMPLETE":
        raise SystemExit(f"Ingestion job {job_id} ended in {status}.")
    stats = ingestion_job.get("statistics", {})
    print(f"    ingestion job {job_id}: COMPLETE. {stats}")


def _resolve_data_source_id(agent, kb_id: str) -> str:
    from setup import KB_DATA_SOURCE_ID_FILE

    if KB_DATA_SOURCE_ID_FILE.exists():
        recorded = KB_DATA_SOURCE_ID_FILE.read_text(encoding="utf-8").strip()
        if recorded:
            return recorded
    for page in agent.get_paginator("list_data_sources").paginate(
        knowledgeBaseId=kb_id
    ):
        for summary in page.get("dataSourceSummaries", []):
            if summary.get("name") == config.RELAY_KB_DATA_SOURCE_NAME:
                return summary["dataSourceId"]
    raise SystemExit("No data source found on the KB. Run setup.py first.")


def _ask_and_show(label: str) -> str:
    """Ask the freshness question and print the answer + first citation."""
    result = kb.answer(QUESTION)
    print(f"\n  [{label}] {QUESTION}")
    print(f"    answer: {result.text.strip()[:300]}")
    if result.citations:
        print(f"    cited : {result.citations[0].source_uri}")
    print(f"    grounded: {result.grounded}")
    return result.text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prove the KB re-syncs after a doc edit (freshness test)."
    )
    parser.add_argument("--new-price", default=DEFAULT_NEW_PRICE,
                        help=f"the new Growth price to write (default "
                             f"{DEFAULT_NEW_PRICE}).")
    parser.add_argument("--no-restore", action="store_true",
                        help="leave the edited price in place (do not restore).")
    args = parser.parse_args(argv)

    try:
        acct = config.account_id(_sts())
    except NoCredentialsError:
        print("  [FAIL] no AWS credentials — set AWS_PROFILE=aws-genai-pro.",
              file=sys.stderr)
        return 1

    bucket = config.relay_bucket(acct)
    agent, s3 = _agent(), _s3()

    try:
        kb_id = kb.resolve_kb_id()
        ds_id = _resolve_data_source_id(agent, kb_id)
    except (kb.KBError, SystemExit) as err:
        print(f"  {err}", file=sys.stderr)
        return 1

    original_text = PLANS_DOC.read_text(encoding="utf-8")

    print("Freshness test — edit a doc, re-sync, prove the answer changed.\n")
    print("1. Answer BEFORE the change:")
    try:
        _ask_and_show("before")

        print("\n2. Editing billing-plans.md (Growth price -> "
              f"{args.new_price}) and re-uploading:")
        edited = _set_growth_price(original_text, args.new_price)
        PLANS_DOC.write_text(edited, encoding="utf-8")
        _upload_doc(s3, bucket, edited)
        print(f"    uploaded s3://{bucket}/{PLANS_KEY}")

        print("\n3. Re-syncing the Knowledge Base (incremental):")
        _resync(agent, kb_id, ds_id)

        print("\n4. Answer AFTER the change:")
        after = _ask_and_show("after")
        if args.new_price.lstrip("$") in after:
            print(f"\n  PROVEN: the new price {args.new_price} appears in the "
                  "answer — the KB re-synced.")
        else:
            print(f"\n  NOTE: the new price {args.new_price} was not echoed "
                  "verbatim; read the answer above to confirm the update.")

        if not args.no_restore:
            print("\n5. Restoring the original price and re-syncing "
                  "(repeatable lab):")
            PLANS_DOC.write_text(original_text, encoding="utf-8")
            _upload_doc(s3, bucket, original_text)
            _resync(agent, kb_id, ds_id)
            print(f"    restored Growth -> {ORIGINAL_PRICE}.")
        else:
            print("\n5. --no-restore: leaving the edited price in place.")
    except (ClientError, kb.KBError) as err:
        # Always restore the local doc on failure so the repo is never left dirty.
        PLANS_DOC.write_text(original_text, encoding="utf-8")
        print(f"\nFreshness test failed: {err}", file=sys.stderr)
        print("(local billing-plans.md restored).", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
