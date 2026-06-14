"""setup.py — secure and budget the AWS account before the first token.

Module 1 of AWS GenAI Pro Mastery. Idempotent and verbose: safe to run twice,
and it tells you exactly what it creates and what it costs.

What it does:
  1. Creates a monthly $5 AWS Budget with an 80% email notification.
  2. Reports the model-access state (serverless auto-activation + the one-time
     Anthropic use-case form).

What it costs: $0. AWS Budgets gives you 2 action-enabled budgets free; this
script creates 1. The budget is the alarm that backstops the whole course.

Run it:
    export RELAY_BUDGET_EMAIL="you@example.com"
    uv run python setup.py
"""

from __future__ import annotations

import os
import sys

import boto3
from botocore.exceptions import ClientError

REGION = "us-east-1"

# The budget is PERSISTENT for the whole course — teardown.py keeps it on
# purpose. Naming it explicitly makes setup/teardown idempotent: we look it up
# by name rather than blindly creating duplicates.
BUDGET_NAME = "aws-genai-pro-monthly"
BUDGET_LIMIT_USD = "5.0"
NOTIFY_THRESHOLD_PCT = 80.0

# Nova activates automatically (serverless). Anthropic Claude needs a one-time
# use-case form. We only REPORT here; nothing blocks the M1 Nova Lite call.
NOVA_TEST_PROFILE = "us.amazon.nova-lite-v1:0"
ANTHROPIC_PROFILE = "us.anthropic.claude-haiku-4-5-20251001-v1:0"


def _account_id() -> str:
    return boto3.client("sts", region_name=REGION).get_caller_identity()["Account"]


def ensure_budget(email: str) -> None:
    """Create the $5 monthly budget + 80% email notification, idempotently."""
    account_id = _account_id()
    budgets = boto3.client("budgets", region_name="us-east-1")  # Budgets is global, anchored to us-east-1

    budget_def = {
        "BudgetName": BUDGET_NAME,
        "BudgetLimit": {"Amount": BUDGET_LIMIT_USD, "Unit": "USD"},
        "TimeUnit": "MONTHLY",
        "BudgetType": "COST",
    }
    notification = {
        "NotificationType": "ACTUAL",
        "ComparisonOperator": "GREATER_THAN",
        "Threshold": NOTIFY_THRESHOLD_PCT,
        "ThresholdType": "PERCENTAGE",
    }
    subscriber = {"SubscriptionType": "EMAIL", "Address": email}

    try:
        budgets.create_budget(
            AccountId=account_id,
            Budget=budget_def,
            NotificationsWithSubscribers=[
                {"Notification": notification, "Subscribers": [subscriber]}
            ],
        )
        print(f"  created budget '{BUDGET_NAME}': ${BUDGET_LIMIT_USD}/month, "
              f"email at {NOTIFY_THRESHOLD_PCT:.0f}% -> {email}")
        return
    except ClientError as err:
        if err.response["Error"]["Code"] != "DuplicateRecordException":
            raise

    # Budget already exists — reconcile the limit and the notification so a
    # second run still converges (e.g. you re-ran with a new email).
    print(f"  budget '{BUDGET_NAME}' already exists — reconciling")
    budgets.update_budget(AccountId=account_id, NewBudget=budget_def)

    existing = budgets.describe_notifications_for_budget(
        AccountId=account_id, BudgetName=BUDGET_NAME
    )["Notifications"]
    if not any(n["Threshold"] == NOTIFY_THRESHOLD_PCT for n in existing):
        budgets.create_notification(
            AccountId=account_id,
            BudgetName=BUDGET_NAME,
            Notification=notification,
            Subscribers=[subscriber],
        )
        print(f"  added {NOTIFY_THRESHOLD_PCT:.0f}% notification -> {email}")
    else:
        # Ensure the subscriber matches the current email.
        subs = budgets.describe_subscribers_for_notification(
            AccountId=account_id, BudgetName=BUDGET_NAME, Notification=notification
        )["Subscribers"]
        if subscriber not in subs:
            budgets.create_subscriber(
                AccountId=account_id,
                BudgetName=BUDGET_NAME,
                Notification=notification,
                Subscriber=subscriber,
            )
            print(f"  updated notification subscriber -> {email}")
        else:
            print(f"  notification already set -> {email}")


def report_model_access() -> None:
    """Report (not enforce) model-access state for the course."""
    print("\nModel access (us-east-1):")
    print("  Nova family: serverless auto-activation since Oct 2025 — there is")
    print("  NO 'request model access' console step anymore. The M1 lab call")
    print(f"  ({NOVA_TEST_PROFILE}) works out of the box.")
    print()
    print("  Anthropic Claude: one one-time use-case form unlocks the family")
    print(f"  ({ANTHROPIC_PROFILE} and friends). Submit it NOW so Claude is")
    print("  ready for later modules (the M13 judge especially):")
    print("    Console: Bedrock > Model access > Anthropic use-case details, OR")
    print("    API:     bedrock.put_use_case_for_model_access(...)")

    # Best-effort: list foundation models so you can confirm the client works.
    try:
        bedrock = boto3.client("bedrock", region_name=REGION)
        models = bedrock.list_foundation_models()["modelSummaries"]
        nova = [m for m in models if m["modelId"].startswith("amazon.nova")]
        anthropic = [m for m in models if m["modelId"].startswith("anthropic.")]
        print(f"\n  catalog reachable: {len(models)} foundation models "
              f"({len(nova)} Nova, {len(anthropic)} Anthropic).")
    except ClientError as err:
        print(f"\n  (could not list models: {err.response['Error']['Code']} — "
              "check your IAM policy includes bedrock:ListFoundationModels)")


def main() -> int:
    email = os.environ.get("RELAY_BUDGET_EMAIL")
    if not email:
        print(
            "RELAY_BUDGET_EMAIL is not set.\n"
            "The budget alarm needs an email to notify at 80% of $5. Set it:\n"
            '    export RELAY_BUDGET_EMAIL="you@example.com"\n'
            "then re-run `uv run python setup.py`.",
            file=sys.stderr,
        )
        return 1

    print("Setting up the AWS GenAI Pro account (idempotent).")
    print("Expected cost: $0 — AWS Budgets gives 2 budgets free; this uses 1.\n")
    print("Budget:")
    ensure_budget(email)
    report_model_access()
    print("\nDone. The $5 budget is PERSISTENT — it backstops the whole course.")
    print("Next: uv run python hello_bedrock.py \"What does CloudCart sell?\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
