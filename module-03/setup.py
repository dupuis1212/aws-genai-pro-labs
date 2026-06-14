"""setup.py — verify model access and ensure the triage prompt exists.

Module 3 of AWS GenAI Pro Mastery. Idempotent and verbose: safe to run twice,
and it tells you exactly what it checks/creates and what it costs.

Module 3 provisions NO new billed resource. What it does:

  1. VERIFY ACCESS to both Relay tiers' inference profiles (the new part). It
     sends one tiny Converse "ping" to the fast and the smart profile from
     relay/config.py and reports whether each is reachable. This is the fastest
     way to catch the two classic failures up front: a bare-regional-ID mistake
     (caught by config.py's `us.` profiles) and a missing model grant.
  2. ENSURE THE TRIAGE PROMPT exists (inherited from Module 2). triage.py still
     consumes the governed `relay-triage` prompt (version 1) from Bedrock Prompt
     Management; the 10-ticket regression suite depends on it. We reconcile the
     draft from prompts/triage_prompt.md (the git source of truth) and keep the
     immutable version 1.

What it costs:
  - Two Converse pings (fast + smart), maxTokens ~5 each: a fraction of a cent.
  - Prompt Management storage: $0 (not billed for stored prompts/versions, as of
    June 2026 — re-verify on the Bedrock pricing page).

The model IDs are NOT hard-coded here — setup.py reads them from relay/config.py,
the sole home of model-ID literals.

Run it:
    uv run python setup.py
    uv run python setup.py --skip-ping   # only ensure the prompt (no live call)
"""

from __future__ import annotations

import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

from relay import config

REGION = config.REGION

# Canonical prompt name in Prompt Management. We look it up by name to stay
# idempotent (create once, reconcile on re-run) rather than blindly duplicating.
PROMPT_NAME = "relay-triage"
VARIANT_NAME = "triage-v3-fewshot"
TEMPLATE_VARIABLE = "ticket"

# The prompt is governed with the SAME model the fast tier maps to — pulled from
# config.py (the sole home of model IDs), never re-typed here. Temperature 0
# because triage is classification.
TEMPERATURE = 0.0
MAX_TOKENS = 100

_ROOT = Path(__file__).resolve().parent
TEMPLATE_FILE = _ROOT / "prompts" / "triage_prompt.md"
PROMPT_ID_FILE = _ROOT / "prompts" / ".prompt_id"


def _agent_client():
    return boto3.client("bedrock-agent", region_name=REGION)


def _runtime_client():
    return boto3.client("bedrock-runtime", region_name=REGION)


# --- Step 1: verify access to both tiers' inference profiles ------------------
def verify_tier_access(client=None) -> bool:
    """Ping the fast and smart inference profiles with one tiny Converse each.

    Returns True if both are reachable. Never silently swallows a failure: an
    inaccessible profile prints the exact AWS error and the likely fix.
    """
    client = client or _runtime_client()
    all_ok = True
    print("Model access (one tiny Converse ping per tier):")
    for tier in ("fast", "smart"):
        profile = config.tier_profile(tier)
        try:
            client.converse(
                modelId=profile,
                messages=[{"role": "user", "content": [{"text": "ping"}]}],
                inferenceConfig={"maxTokens": 5, "temperature": 0.0},
            )
            print(f"  [ok] tier '{tier}': {profile} reachable.")
        except ClientError as err:
            all_ok = False
            code = err.response["Error"]["Code"]
            message = err.response["Error"]["Message"]
            print(f"  [FAIL] tier '{tier}': {profile} ({code}): {message}")
            if "inference profile" in message.lower():
                print("         -> looks like a bare-regional-ID issue, but the ID "
                      "is already a us. profile; check the model grant in this "
                      "account/Region.")
    return all_ok


# --- Step 2: ensure the triage prompt exists (inherited from Module 2) --------
def read_template() -> str:
    """Load the triage template (the git source of truth)."""
    text = TEMPLATE_FILE.read_text(encoding="utf-8")
    placeholder = "{{" + TEMPLATE_VARIABLE + "}}"
    if placeholder not in text:
        # Fail loudly: a template without its variable is a silent governance bug.
        raise SystemExit(
            f"Template {TEMPLATE_FILE} is missing the {placeholder} placeholder."
        )
    return text


def _variant(template_text: str) -> dict:
    """Build the single TEXT prompt variant: template + variable + model + config."""
    return {
        "name": VARIANT_NAME,
        "templateType": "TEXT",
        "templateConfiguration": {
            "text": {
                "text": template_text,
                "inputVariables": [{"name": TEMPLATE_VARIABLE}],
            }
        },
        "modelId": config.tier_profile("fast"),
        "inferenceConfiguration": {
            "text": {"temperature": TEMPERATURE, "maxTokens": MAX_TOKENS}
        },
    }


def _find_prompt_id(client) -> str | None:
    """Return the ID of an existing `relay-triage` prompt, or None."""
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


def ensure_prompt(client, template_text: str) -> str:
    """Create or reconcile the working draft of the triage prompt. Returns its ID."""
    variant = _variant(template_text)
    existing_id = _find_prompt_id(client)

    if existing_id is None:
        created = client.create_prompt(
            name=PROMPT_NAME,
            description="Relay triage classifier: ticket -> {intent, priority, sentiment} JSON.",
            defaultVariant=VARIANT_NAME,
            variants=[variant],
        )
        prompt_id = created["id"]
        print(f"  created prompt '{PROMPT_NAME}' (id {prompt_id})")
        print(f"    variant   : {VARIANT_NAME}")
        print(f"    model     : {config.tier_profile('fast')} @ temperature {TEMPERATURE}")
        print(f"    variable  : {{{{{TEMPLATE_VARIABLE}}}}}")
        return prompt_id

    # Already exists — reconcile the draft so the git template stays authoritative.
    print(f"  prompt '{PROMPT_NAME}' already exists (id {existing_id}) — reconciling draft")
    client.update_prompt(
        promptIdentifier=existing_id,
        name=PROMPT_NAME,
        description="Relay triage classifier: ticket -> {intent, priority, sentiment} JSON.",
        defaultVariant=VARIANT_NAME,
        variants=[variant],
    )
    return existing_id


def ensure_version_1(client, prompt_id: str) -> str:
    """Publish version 1 if no version exists yet. Returns the version ARN."""
    resp = client.list_prompts(promptIdentifier=prompt_id, maxResults=100)
    numbered = [
        s for s in resp.get("promptSummaries", [])
        if str(s.get("version", "")).isdigit()
    ]
    if numbered:
        latest = sorted(numbered, key=lambda s: int(s["version"]))[-1]
        print(f"  version {latest['version']} already published — keeping it (versions are immutable)")
        return latest["arn"]

    published = client.create_prompt_version(
        promptIdentifier=prompt_id,
        description="v1: role + JSON format constraints + 7 few-shot examples (temp 0).",
    )
    print(f"  published version {published['version']}")
    return published["arn"]


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    skip_ping = "--skip-ping" in argv
    leftover = [a for a in argv if a != "--skip-ping"]
    if leftover:
        print(f"Unknown argument(s): {' '.join(leftover)}\n"
              "Usage: uv run python setup.py [--skip-ping]", file=sys.stderr)
        return 1

    print("Setting up Module 3 — verify model access + ensure the triage prompt.")
    print("Expected cost: a fraction of a cent (two Converse pings); prompt storage $0.\n")

    # Step 1: model access (skippable for an offline dry run of step 2).
    if not skip_ping:
        try:
            access_ok = verify_tier_access()
        except NoCredentialsError:
            print("  [FAIL] no AWS credentials — set AWS_PROFILE=aws-genai-pro.",
                  file=sys.stderr)
            return 1
        if not access_ok:
            print("\nFix the model-access failures above before running the lab.",
                  file=sys.stderr)
            return 1
        print()
    else:
        print("Model access: SKIPPED (--skip-ping).\n")

    # Step 2: ensure the inherited triage prompt exists.
    template_text = read_template()
    client = _agent_client()
    print("Prompt (Bedrock Prompt Management):")
    try:
        prompt_id = ensure_prompt(client, template_text)
        version_arn = ensure_version_1(client, prompt_id)
    except ClientError as err:
        code = err.response["Error"]["Code"]
        message = err.response["Error"]["Message"]
        print(
            f"\nBedrock Prompt Management call failed ({code}):\n  {message}\n\n"
            "If this is AccessDenied, your course IAM role needs bedrock:CreatePrompt,\n"
            "CreatePromptVersion, ListPrompts, GetPrompt and UpdatePrompt. See lab.md.",
            file=sys.stderr,
        )
        return 1

    PROMPT_ID_FILE.write_text(prompt_id + "\n", encoding="utf-8")
    print(f"\n  recorded prompt ID -> {PROMPT_ID_FILE.relative_to(_ROOT)}")
    print(f"  version 1 ARN      -> {version_arn}")
    print("\nDone. Next:")
    print("  uv run python demo_llm.py \"Why was I charged twice for order #1042?\"")
    print("  uv run python run_prompt_tests.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
