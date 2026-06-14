"""teardown.py — remove Module 2's Bedrock Prompt Management prompt.

Module 2 of AWS GenAI Pro Mastery. Idempotent and verbose.

Module 2 creates exactly one cloud resource: the `relay-triage` prompt (with its
version 1) in Amazon Bedrock Prompt Management. Deleting the prompt removes the
prompt and all its versions in one call. Prompt Management does not bill for
stored prompts, so this is hygiene rather than cost control — but the course rule
is that every setup.py has a matching, tested teardown.py that leaves the account
clean.

The M1 $5 budget is NOT touched here — it is persistent and backstops the whole
course (Module 1's teardown owns it).

Run it:
    uv run python teardown.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

REGION = "us-east-1"
PROMPT_NAME = "relay-triage"  # must match setup.py

_ROOT = Path(__file__).resolve().parent
PROMPT_ID_FILE = _ROOT / "prompts" / ".prompt_id"


def _client():
    return boto3.client("bedrock-agent", region_name=REGION)


def _find_prompt_id(client) -> str | None:
    """Return the ID of the `relay-triage` prompt, or None if it does not exist."""
    paginator_args = {}
    while True:
        resp = client.list_prompts(maxResults=100, **paginator_args)
        for summary in resp.get("promptSummaries", []):
            if summary["name"] == PROMPT_NAME:
                return summary["id"]
        token = resp.get("nextToken")
        if not token:
            return None
        paginator_args = {"nextToken": token}


def delete_prompt(client) -> None:
    """Delete the triage prompt and all its versions, idempotently."""
    prompt_id = _find_prompt_id(client)
    if prompt_id is None:
        print(f"  prompt '{PROMPT_NAME}': not found (already removed). Fine.")
    else:
        try:
            # Deleting the prompt removes the draft AND every published version.
            client.delete_prompt(promptIdentifier=prompt_id)
            print(f"  prompt '{PROMPT_NAME}' (id {prompt_id}): DELETED (with all versions).")
        except ClientError as err:
            if err.response["Error"]["Code"] == "ResourceNotFoundException":
                print(f"  prompt '{PROMPT_NAME}': already gone — nothing to delete.")
            else:
                raise

    # Clean up the local pointer file so a later setup.py starts fresh.
    if PROMPT_ID_FILE.exists():
        PROMPT_ID_FILE.unlink()
        print(f"  removed local pointer {PROMPT_ID_FILE.relative_to(_ROOT)}")


def assert_no_other_resources() -> None:
    """Module 2 provisions nothing else idle-billed. State that plainly."""
    print("\nOther idle-billed resources from Module 2: NONE.")
    print("  - relay/triage.py: pay-per-token Converse calls — nothing persists.")
    print("  - Prompt Management prompt: not billed for storage (deleted above).")
    print("  The M1 $5 budget is KEPT on purpose (Module 1 owns it; it guards the")
    print("  whole course).")


def main() -> int:
    if sys.argv[1:]:
        print(f"Unknown argument(s): {' '.join(sys.argv[1:])}\n"
              "Usage: uv run python teardown.py",
              file=sys.stderr)
        return 1

    print("Tearing down Module 2 (idempotent).\n")
    print("Prompt Management:")
    delete_prompt(_client())
    assert_no_other_resources()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
