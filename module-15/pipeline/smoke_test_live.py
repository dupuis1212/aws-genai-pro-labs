"""pipeline/smoke_test_live.py — the post-deploy smoke check the pipeline runs (Module 11).

A tiny, dependency-free POST -> poll-GET round-trip against the DEPLOYED Relay API. The
CodePipeline Smoke stage runs this (pipeline/smoke_buildspec.yml) after the Deploy stage; a
non-zero exit STOPS the pipeline before any further promotion — the rollback gate (skill
2.3.5). You can also run it by hand against a `cdk deploy`-ed API:

    uv run python pipeline/smoke_test_live.py https://<api-id>.execute-api.us-east-1.amazonaws.com/prod

Steps:
  1. POST /tickets with data/tickets/sample.json -> expect HTTP 202 + a ticket_id.
  2. Poll GET /tickets/{ticket_id} until the status is TERMINAL (answered / escalated /
     awaiting_approval / closed / failed) or the budget runs out.
  3. Exit 0 if the round-trip worked and the status is non-`failed`; exit non-zero (so the
     pipeline rolls back / stops) otherwise.

It uses only the standard library (urllib + json) so the Smoke CodeBuild image needs no
deps beyond Python. NO AWS SDK call — it talks HTTP to the public API, the way a CloudCart
client would.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE = _REPO_ROOT / "data" / "tickets" / "sample.json"

# Statuses that END the poll (the full frozen TicketRecord lifecycle, 06 §2). A ticket that
# reaches any of these is "done processing" — the worker advanced it past `received`.
TERMINAL = {"answered", "escalated", "awaiting_approval", "closed", "failed"}
POLL_SECONDS = 4
POLL_BUDGET_S = 90  # generous ceiling for an agent run + a cold start


def _post(base_url: str, body: dict) -> dict:
    req = urllib.request.Request(
        base_url.rstrip("/") + "/tickets",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        status = resp.status
        payload = json.loads(resp.read().decode("utf-8"))
    if status != 202:
        raise SystemExit(f"POST /tickets returned {status}, expected 202: {payload}")
    if "ticket_id" not in payload:
        raise SystemExit(f"POST /tickets response missing ticket_id: {payload}")
    return payload


def _get(base_url: str, ticket_id: str) -> dict:
    req = urllib.request.Request(
        base_url.rstrip("/") + f"/tickets/{ticket_id}", method="GET")
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def smoke(base_url: str) -> int:
    body = json.loads(SAMPLE.read_text(encoding="utf-8"))
    print(f"POST /tickets -> {base_url}")
    posted = _post(base_url, body)
    ticket_id = posted["ticket_id"]
    print(f"  202 Accepted, ticket_id={ticket_id} (status={posted.get('status')})")

    deadline = time.time() + POLL_BUDGET_S
    status = posted.get("status", "received")
    while time.time() < deadline:
        record = _get(base_url, ticket_id)
        status = record.get("status")
        print(f"  GET /tickets/{ticket_id} -> status={status}")
        if status in TERMINAL:
            break
        time.sleep(POLL_SECONDS)

    if status not in TERMINAL:
        print(f"SMOKE FAILED: ticket {ticket_id} never reached a terminal status "
              f"(last={status}).", file=sys.stderr)
        return 1
    if status == "failed":
        print(f"SMOKE FAILED: ticket {ticket_id} ended in `failed`.", file=sys.stderr)
        return 1
    print(f"SMOKE OK: ticket {ticket_id} reached `{status}`.")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1 or not argv[0].strip():
        print("Usage: uv run python pipeline/smoke_test_live.py <api-base-url>",
              file=sys.stderr)
        return 2
    base_url = argv[0].strip()
    try:
        return smoke(base_url)
    except urllib.error.HTTPError as err:
        print(f"SMOKE FAILED: HTTP {err.code} from the API: {err.read()[:200]!r}",
              file=sys.stderr)
        return 1
    except (urllib.error.URLError, OSError) as err:
        print(f"SMOKE FAILED: could not reach the API: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
