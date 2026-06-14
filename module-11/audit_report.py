"""audit_report.py — answer the auditor: cross the decision log with CloudTrail.

Module 10 of AWS GenAI Pro Mastery. The SOC 2 auditor asked four things about Relay:
what data it saw, who could access it, how long you keep it, and your documented limits.
This script answers the traceability half — "show me what the bot DID last hour, and who
called which sensitive API" — by joining the two governance trails the exam keeps
distinct:

  - the DECISION LOG (relay/agent.py -> decision_log.jsonl): the APPLICATION trail —
    what Relay DECIDED and why (which tools, on what REDACTED input, the outcome). This
    is where prompt-shaped content would live, so it is written redacted.
  - AWS CloudTrail (management events, FREE by default): the API trail — WHO (which IAM
    principal) called WHICH AWS API, and WHEN. CloudTrail records the API CALL, never the
    prompt CONTENT — the exam's favourite distinction. Data events would log object-level
    access too, but they are BILLED, so this report uses management events only and says
    so (no cost surprise).

Run it:
    uv run python audit_report.py --last 1h          # decision log + CloudTrail, 1 hour
    uv run python audit_report.py --last 24h          # last day
    uv run python audit_report.py --last 1h --no-cloudtrail   # decision log only (offline)

It degrades cleanly: with no AWS credentials (or --no-cloudtrail) it prints the decision
log alone and notes CloudTrail was skipped — so the offline smoke test can exercise the
join logic without a network. It READS only; it creates and deletes nothing.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import boto3
from botocore.exceptions import (
    BotoCoreError,
    ClientError,
    NoCredentialsError,
    ProfileNotFound,
)

from relay import config

REGION = config.REGION
_ROOT = Path(__file__).resolve().parent
DECISION_LOG = _ROOT / config.DECISION_LOG_FILE_NAME

# The sensitive AWS APIs an auditor cares about for Relay — the calls that touch the
# order book, the ticket store, the guardrail, and the IAM roles. CloudTrail logs these
# as MANAGEMENT events (free). A name list, in one place, so the report and the article
# agree on "what counts as sensitive".
SENSITIVE_EVENT_NAMES = (
    "PutItem", "UpdateItem", "DeleteItem",        # DynamoDB writes (relay-tickets)
    "CreateGuardrail", "UpdateGuardrail", "DeleteGuardrail",  # guardrail admin
    "CreateRole", "PutRolePolicy", "DeleteRole",  # IAM role admin (least-privilege)
    "PutObject", "DeleteObject",                  # S3 attachments writes/deletes
)


@dataclass
class DecisionRecord:
    """One parsed decision-log line (already redacted at write time)."""

    ts: str
    ticket_id: str
    status: str
    handed_off: bool
    gated: bool
    stop_reason: str
    actions: list = field(default_factory=list)


@dataclass
class AuditReport:
    """The joined report: decision records + CloudTrail events in the window."""

    since: dt.datetime
    decisions: list[DecisionRecord]
    cloudtrail_events: list[dict]
    cloudtrail_checked: bool


def parse_window(spec: str) -> dt.timedelta:
    """Parse a window like '1h', '24h', '30m', '7d' into a timedelta."""
    spec = spec.strip().lower()
    if not spec or not spec[:-1].isdigit():
        raise ValueError(f"Bad --last value {spec!r}. Use e.g. 1h, 30m, 24h, 7d.")
    n, unit = int(spec[:-1]), spec[-1]
    if unit == "m":
        return dt.timedelta(minutes=n)
    if unit == "h":
        return dt.timedelta(hours=n)
    if unit == "d":
        return dt.timedelta(days=n)
    raise ValueError(f"Bad --last unit in {spec!r}. Use m (minutes), h (hours), d (days).")


def _parse_ts(value: str) -> dt.datetime | None:
    """Parse the ISO-8601 'Z' timestamps the decision log writes."""
    try:
        return dt.datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=dt.timezone.utc
        )
    except (ValueError, TypeError):
        return None


def read_decision_log(
    since: dt.datetime, *, path: Path | None = None
) -> list[DecisionRecord]:
    """Read decision_log.jsonl and return records at/after `since` (newest first).

    Tolerant of a missing file (returns []) and a malformed line (skipped with a stderr
    note) — an audit report should not crash on one bad line. The records are already
    redacted at write time, so nothing here re-touches PII."""
    target = path or DECISION_LOG
    if not target.exists():
        return []
    out: list[DecisionRecord] = []
    for i, line in enumerate(target.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            print(f"[warn] decision log line {i} is not valid JSON — skipped.",
                  file=sys.stderr)
            continue
        ts = _parse_ts(obj.get("ts", ""))
        if ts is None or ts < since:
            continue
        out.append(DecisionRecord(
            ts=obj.get("ts", ""),
            ticket_id=obj.get("ticket_id", ""),
            status=obj.get("status", ""),
            handed_off=bool(obj.get("handed_off", False)),
            gated=bool(obj.get("gated", False)),
            stop_reason=obj.get("stop_reason", ""),
            actions=obj.get("actions", []),
        ))
    out.sort(key=lambda r: r.ts, reverse=True)
    return out


def _cloudtrail_client():
    return boto3.client("cloudtrail", region_name=REGION)


def read_cloudtrail(
    since: dt.datetime, *, client=None, event_names=SENSITIVE_EVENT_NAMES
) -> list[dict]:
    """Read CloudTrail MANAGEMENT events for the sensitive APIs since `since`.

    Uses LookupEvents (management events — FREE; no trail or data-event cost). Filters to
    the sensitive event names one at a time (LookupEvents allows a single attribute
    filter per call). Returns a flat list of {time, event, user, source} dicts. A
    ClientError is raised to the caller (the CLI degrades to decision-log-only). NEVER
    reads prompt content — CloudTrail does not carry it."""
    client = client or _cloudtrail_client()
    collected: list[dict] = []
    for name in event_names:
        resp = client.lookup_events(
            LookupAttributes=[{"AttributeKey": "EventName", "AttributeValue": name}],
            StartTime=since,
            MaxResults=20,
        )
        for ev in resp.get("Events", []):
            collected.append({
                "time": ev.get("EventTime"),
                "event": ev.get("EventName"),
                "user": ev.get("Username", "(unknown)"),
                "source": ev.get("EventSource", ""),
            })
    collected.sort(key=lambda e: str(e["time"]), reverse=True)
    return collected


def build_report(
    window: dt.timedelta,
    *,
    check_cloudtrail: bool = True,
    cloudtrail_client=None,
    decision_log_path: Path | None = None,
    now: dt.datetime | None = None,
) -> AuditReport:
    """Build the joined audit report for the window ending `now` (UTC)."""
    now = now or dt.datetime.now(dt.timezone.utc)
    since = now - window
    decisions = read_decision_log(since, path=decision_log_path)
    events: list[dict] = []
    checked = False
    if check_cloudtrail:
        events = read_cloudtrail(since, client=cloudtrail_client)
        checked = True
    return AuditReport(since=since, decisions=decisions,
                       cloudtrail_events=events, cloudtrail_checked=checked)


def render(report: AuditReport) -> str:
    """Render the report as plain text (PII-free — decisions are pre-redacted)."""
    lines: list[str] = []
    lines.append(f"Audit report — actions since {report.since:%Y-%m-%d %H:%M:%S} UTC")
    lines.append("=" * 64)

    lines.append("\nDECISION LOG (what Relay decided — the application 'why' trail):")
    if not report.decisions:
        lines.append("  (no decisions in this window)")
    for r in report.decisions:
        flags = []
        if r.handed_off:
            flags.append("handed_off")
        if r.gated:
            flags.append("GATED→awaiting_approval")
        flag_str = f" [{', '.join(flags)}]" if flags else ""
        lines.append(f"  {r.ts}  {r.ticket_id}  status={r.status}{flag_str}")
        for a in r.actions:
            tool = a.get("tool", "?")
            approved = a.get("approved")
            decided = ("proposed" if approved is None
                       else "approved" if approved else "rejected")
            # tool_input is already redacted at write time — safe to show.
            lines.append(f"      - {tool}({a.get('tool_input', {})}) [{decided}]")

    lines.append("\nCLOUDTRAIL (who called which AWS API — management events, FREE):")
    if not report.cloudtrail_checked:
        lines.append("  (skipped — run with credentials and without --no-cloudtrail)")
    elif not report.cloudtrail_events:
        lines.append("  (no sensitive management events in this window)")
    for e in report.cloudtrail_events:
        lines.append(f"  {e['time']}  {e['event']:<16} by {e['user']}  ({e['source']})")

    lines.append("\nNote: CloudTrail records the API CALL (who/what/when), not the prompt")
    lines.append("CONTENT. Prompt-shaped content lives in the decision log — and is")
    lines.append("written REDACTED, so neither trail leaks raw customer PII.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python audit_report.py",
        description="Cross Relay's decision log with CloudTrail to answer the auditor.",
    )
    parser.add_argument("--last", default="1h",
                        help="time window, e.g. 1h, 30m, 24h, 7d (default 1h)")
    parser.add_argument("--no-cloudtrail", action="store_true",
                        help="skip CloudTrail (decision log only — works offline)")
    args = parser.parse_args(argv)

    try:
        window = parse_window(args.last)
    except ValueError as err:
        print(f"{err}", file=sys.stderr)
        return 1

    try:
        report = build_report(window, check_cloudtrail=not args.no_cloudtrail)
    except (NoCredentialsError, ProfileNotFound):
        print("[note] No AWS credentials — showing the decision log only "
              "(CloudTrail skipped). Set AWS_PROFILE=aws-genai-pro for the full report.",
              file=sys.stderr)
        report = build_report(window, check_cloudtrail=False)
    except ClientError as err:
        print(f"[note] CloudTrail lookup failed "
              f"({err.response['Error']['Code']}) — showing the decision log only.",
              file=sys.stderr)
        report = build_report(window, check_cloudtrail=False)
    except BotoCoreError as err:
        print(f"[note] AWS problem ({err}) — showing the decision log only.",
              file=sys.stderr)
        report = build_report(window, check_cloudtrail=False)

    print(render(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
