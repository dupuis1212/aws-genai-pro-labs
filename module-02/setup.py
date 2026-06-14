"""setup.py — create the triage prompt in Amazon Bedrock Prompt Management.

Module 2 of AWS GenAI Pro Mastery. Idempotent and verbose: safe to run twice,
and it tells you exactly what it creates and what it costs.

What it does:
  1. Reads the governed template from prompts/triage_prompt.md (the git source of
     truth for the course — kept byte-synced with Prompt Management).
  2. Creates a parameterized prompt `relay-triage` with a {{ticket}} variable,
     pinned to Amazon Nova Micro at temperature 0 (a classifier's settings).
  3. Publishes VERSION 1 (an immutable snapshot — this is what governance means:
     versions you can pin, approve, and audit, not a mutable f-string).
  4. Records the prompt ID in prompts/.prompt_id so relay/triage.py can find it.

What it costs: $0. Bedrock Prompt Management does not bill for storing prompts or
versions (as of June 2026 — re-verify on the Bedrock pricing page). You only pay
per token at inference time, when triage.py actually calls Converse.

Run it:
    uv run python setup.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

REGION = "us-east-1"

# Canonical prompt name in Prompt Management. We look it up by name to stay
# idempotent (create once, reconcile on re-run) rather than blindly duplicating.
PROMPT_NAME = "relay-triage"
VARIANT_NAME = "triage-v3-fewshot"
TEMPLATE_VARIABLE = "ticket"

# The model + inference settings the prompt is governed with. Nova Micro via its
# us. INFERENCE PROFILE; temperature 0 because triage is classification. This is
# the SINGLE model ID in setup land — it mirrors the provisional constant in
# relay/triage.py (relay/config.py becomes the sole home for both at Module 3).
MODEL_ID = "us.amazon.nova-micro-v1:0"
TEMPERATURE = 0.0
MAX_TOKENS = 100

_ROOT = Path(__file__).resolve().parent
TEMPLATE_FILE = _ROOT / "prompts" / "triage_prompt.md"
PROMPT_ID_FILE = _ROOT / "prompts" / ".prompt_id"


def _client():
    return boto3.client("bedrock-agent", region_name=REGION)


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
        "modelId": MODEL_ID,
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
        print(f"    model     : {MODEL_ID} @ temperature {TEMPERATURE}")
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
    # Versions other than the mutable DRAFT show up in list_prompts when queried
    # by identifier. If a numbered version already exists, we keep it (immutable).
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


def main() -> int:
    print("Setting up Module 2 — Relay's triage prompt in Bedrock Prompt Management.")
    print("Expected cost: $0 — Prompt Management does not bill for stored prompts/versions.\n")

    template_text = read_template()
    client = _client()

    print("Prompt:")
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
    print("\nDone. relay/triage.py consumes this prompt by id + version 1.")
    print("Next: uv run python -m relay.triage data/tickets/ticket-001.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
