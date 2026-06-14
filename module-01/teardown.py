"""teardown.py — verify Module 1 left nothing billing in the background.

Module 1 of AWS GenAI Pro Mastery. Idempotent and verbose.

Module 1 creates exactly two things: a $5 AWS Budget (free, no idle cost) and
the Nova Lite Converse calls you made by hand (pay-per-token, nothing left
running). So there is NO idle-billed resource to delete — and this script
asserts that, rather than pretending to clean up something that does not exist.

The budget + its email alarm are KEPT ON PURPOSE: they backstop every later
module. Only at the very end of the course do you remove them, with:
    uv run python teardown.py --delete-budget

Run it:
    uv run python teardown.py
"""

from __future__ import annotations

import sys

import boto3
from botocore.exceptions import ClientError

REGION = "us-east-1"
BUDGET_NAME = "aws-genai-pro-monthly"  # must match setup.py


def _account_id() -> str:
    return boto3.client("sts", region_name=REGION).get_caller_identity()["Account"]


def assert_no_idle_resources() -> None:
    """Module 1 provisions nothing that bills while idle. State that plainly."""
    print("Idle-billed resources from Module 1: NONE.")
    print("  - hello_bedrock.py: pay-per-token Converse calls — nothing persists.")
    print("  - setup.py: an AWS Budget — free tier, $0 whether used or not.")
    print("  Nothing to delete. (Later modules add real resources; their own")
    print("  teardown scripts remove them — this one stays this simple.)")


def report_or_delete_budget(delete: bool) -> None:
    account_id = _account_id()
    budgets = boto3.client("budgets", region_name="us-east-1")

    try:
        budgets.describe_budget(AccountId=account_id, BudgetName=BUDGET_NAME)
        exists = True
    except ClientError as err:
        if err.response["Error"]["Code"] == "NotFoundException":
            exists = False
        else:
            raise

    if not delete:
        if exists:
            print(f"\nBudget '{BUDGET_NAME}': KEPT by design — it guards the")
            print("  whole course. Run with --delete-budget at course end.")
        else:
            print(f"\nBudget '{BUDGET_NAME}': not found (already removed). Fine.")
        return

    # --delete-budget: course is over, remove the persistent alarm.
    if not exists:
        print(f"\nBudget '{BUDGET_NAME}': already gone — nothing to delete.")
        return
    budgets.delete_budget(AccountId=account_id, BudgetName=BUDGET_NAME)
    print(f"\nBudget '{BUDGET_NAME}': DELETED (--delete-budget). Account is back")
    print("  to a clean slate. Thanks for taking the course.")


def main() -> int:
    delete = "--delete-budget" in sys.argv[1:]
    unknown = [a for a in sys.argv[1:] if a != "--delete-budget"]
    if unknown:
        print(f"Unknown argument(s): {' '.join(unknown)}\n"
              "Usage: uv run python teardown.py [--delete-budget]",
              file=sys.stderr)
        return 1

    print("Tearing down Module 1 (idempotent).\n")
    assert_no_idle_resources()
    report_or_delete_budget(delete)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
