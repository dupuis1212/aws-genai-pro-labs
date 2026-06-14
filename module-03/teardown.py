"""teardown.py — leave the account clean after Module 3.

Module 3 of AWS GenAI Pro Mastery. Idempotent and verbose.

Module 3 creates NO new idle-billed resource. The FM integration layer is pure
code; every Converse call is pay-per-token and persists nothing. So teardown has
three jobs, all idempotent and safe to run twice:

  1. Remove the `relay-triage` prompt (and its version 1) from Bedrock Prompt
     Management. This is inherited from Module 2 — the prompt is the only standing
     cloud resource the running lab depends on. Storage is not billed, so this is
     hygiene, not cost control, but the course rule is one tested teardown per
     setup.
  2. Remove the OPTIONAL AppConfig application from the "Try it yourself" exercise
     (deporting the tier map into AWS AppConfig freeform config). If you never did
     that exercise, this is a no-op. AppConfig freeform config is free/cents; we
     delete it anyway so nothing lingers.
  3. State plainly that no other idle-billed resource exists.

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

from relay import config

REGION = config.REGION
PROMPT_NAME = "relay-triage"  # must match setup.py

# The OPTIONAL AppConfig application name used by the "Try it yourself" exercise.
# If the exercise was never done, no application by this name exists and the
# cleanup is a clean no-op.
APPCONFIG_APP_NAME = "relay-model-config"

_ROOT = Path(__file__).resolve().parent
PROMPT_ID_FILE = _ROOT / "prompts" / ".prompt_id"


def _agent_client():
    return boto3.client("bedrock-agent", region_name=REGION)


def _appconfig_client():
    return boto3.client("appconfig", region_name=REGION)


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
            client.delete_prompt(promptIdentifier=prompt_id)
            print(f"  prompt '{PROMPT_NAME}' (id {prompt_id}): DELETED (with all versions).")
        except ClientError as err:
            if err.response["Error"]["Code"] == "ResourceNotFoundException":
                print(f"  prompt '{PROMPT_NAME}': already gone — nothing to delete.")
            else:
                raise

    if PROMPT_ID_FILE.exists():
        PROMPT_ID_FILE.unlink()
        print(f"  removed local pointer {PROMPT_ID_FILE.relative_to(_ROOT)}")


def delete_appconfig(client) -> None:
    """Delete the optional Try-it-yourself AppConfig application, idempotently.

    Deleting an AppConfig application requires its profiles/environments to be
    gone first; we remove them in dependency order. If the application does not
    exist (the common case — the exercise is optional), this is a clean no-op.
    """
    app_id = _find_appconfig_app_id(client)
    if app_id is None:
        print(f"  AppConfig app '{APPCONFIG_APP_NAME}': not found "
              "(Try-it-yourself not done, or already removed). Fine.")
        return

    # Remove configuration profiles (and their hosted versions) first.
    try:
        profiles = client.list_configuration_profiles(ApplicationId=app_id).get("Items", [])
    except ClientError:
        profiles = []
    for profile in profiles:
        profile_id = profile["Id"]
        _delete_hosted_versions(client, app_id, profile_id)
        client.delete_configuration_profile(
            ApplicationId=app_id, ConfigurationProfileId=profile_id
        )
        print(f"    deleted configuration profile {profile_id}")

    # Remove environments.
    try:
        environments = client.list_environments(ApplicationId=app_id).get("Items", [])
    except ClientError:
        environments = []
    for environment in environments:
        client.delete_environment(ApplicationId=app_id, EnvironmentId=environment["Id"])
        print(f"    deleted environment {environment['Id']}")

    client.delete_application(ApplicationId=app_id)
    print(f"  AppConfig app '{APPCONFIG_APP_NAME}' (id {app_id}): DELETED.")


def _delete_hosted_versions(client, app_id: str, profile_id: str) -> None:
    try:
        versions = client.list_hosted_configuration_versions(
            ApplicationId=app_id, ConfigurationProfileId=profile_id
        ).get("Items", [])
    except ClientError:
        return
    for version in versions:
        client.delete_hosted_configuration_version(
            ApplicationId=app_id,
            ConfigurationProfileId=profile_id,
            VersionNumber=version["VersionNumber"],
        )


def _find_appconfig_app_id(client) -> str | None:
    try:
        paginator = client.get_paginator("list_applications")
        for page in paginator.paginate():
            for app in page.get("Items", []):
                if app["Name"] == APPCONFIG_APP_NAME:
                    return app["Id"]
    except ClientError:
        return None
    return None


def assert_no_other_resources() -> None:
    """Module 3 provisions nothing else idle-billed. State that plainly."""
    print("\nOther idle-billed resources from Module 3: NONE.")
    print("  - relay/llm.py: pay-per-token Converse/ConverseStream calls — nothing persists.")
    print("  - relay/config.py: pure code (the tier -> profile map). No resource.")
    print("  - Prompt Management prompt: not billed for storage (deleted above).")
    print("  The M1 $5 budget is KEPT on purpose (Module 1 owns it; it guards the")
    print("  whole course).")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv:
        print(f"Unknown argument(s): {' '.join(argv)}\n"
              "Usage: uv run python teardown.py",
              file=sys.stderr)
        return 1

    print("Tearing down Module 3 (idempotent).\n")
    print("Prompt Management:")
    delete_prompt(_agent_client())

    print("\nAppConfig (optional Try-it-yourself):")
    delete_appconfig(_appconfig_client())

    assert_no_other_resources()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
