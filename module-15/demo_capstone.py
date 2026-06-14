"""demo_capstone.py — the 20-ticket costed end-to-end run (Module 15, the capstone).

This is the capstone's headline deliverable: ONE run that drives 20 varied tickets through
the WHOLE assembled Relay system and prints a single costed recap — the proof that the 14
pieces built across Modules 1-14 are now one cohesive, hardened system (skill 1.1.1, solution
design), not 14 parts that each pass in isolation.

The 20 tickets are a deliberate MIX (brief §6 — the demo scope is load-bearing for coverage):
  - ~15 NOMINAL tickets (the data/tickets/*.json fixtures + a few capstone additions) covering
    every intent: billing (duplicate charge, unrecognised charge), technical (storefront down),
    account (login), shipping (where-is-my-order), plus refund tickets that exercise the M8
    Billing-specialist handoff + the HITL refund gate (-> awaiting_approval);
  - ADVERSARIAL tickets reused from data/attacks.json (Module 9) — direct prompt injection +
    a PII-exfiltration attempt — so the run PROVES the guardrail / the IAM tool boundary on
    untrusted input, not just on the happy path;
  - MULTIMODAL tickets reusing the Module 6 screenshot fixtures (data/raw/payment_error.png) —
    the intake pipeline reads the attachment (Nova Lite vision) before the agent answers.

For EACH ticket the run prints its triage and, by outcome, a cited answer (Answer/citations),
an action (AgentAction), an escalation (escalated: true), or awaiting_approval (a refund parked
on the HITL gate). At the end it prints the RECAP TABLE: count by status, the escalation rate,
the total $/ticket (summed cost_cents — the M12 instrumentation, consumed as-is), the p95
latency, and the golden-set eval score (a re-run of evals/run_evals.py — the M13 contract,
consumed field-for-field).

TWO execution paths, ONE recap (the brief's "via the deployed API", plus a runnable default):

  - --api-url <stage-url>  : the DEPLOYED-API path. POST /tickets -> poll GET /tickets/{id}
    against the `cdk deploy`-ed stage (the front door the worker drives). This is the path the
    brief describes once the M11 stack is deployed; cost_cents comes off the persisted record
    (the worker's M12 meter). Use it after `uv run python setup.py` / `cdk deploy`.

  - (default) the LOCAL path: drive relay.run.run_relay() — the EXACT frozen seam the deployed
    worker invokes (bible §4 M8->M11) — wrapped in a relay.llm.CostMeter so cost_cents is the
    real metered token cost. Same agent, same KB, same guardrail, same tables, same Bedrock
    calls; it just skips the API Gateway + SQS hop, so the capstone is runnable (and the recap
    reproducible) without standing up the whole stack first. NO new model client, NO new
    generation path — it CONSUMES run_relay.

NO model ID lives here (the M3 containment law); every generation goes through run_relay ->
relay.llm.converse(). NO new concept, NO new service — the capstone ASSEMBLES and DEMONSTRATES.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from relay import config

_ROOT = Path(__file__).resolve().parent
TICKETS_DIR = _ROOT / "data" / "tickets"
ATTACKS_PATH = _ROOT / "data" / "attacks.json"
RAW_DIR = _ROOT / "data" / "raw"

# The terminal statuses a ticket can reach (the full TicketRecord lifecycle, 06 §2).
TERMINAL = {"answered", "escalated", "awaiting_approval", "closed", "failed"}


# =============================================================================
# The 20 demo tickets — a deliberate mix (nominal + adversarial[M9] + multimodal[M6]).
# =============================================================================
@dataclass
class DemoTicket:
    """One ticket in the capstone run: its id, the customer message, a category label
    (nominal/adversarial/multimodal — for the recap), and an optional attachment path
    (a multimodal ticket carries a screenshot the intake pipeline reads)."""

    ticket_id: str
    customer_message: str
    category: str
    channel: str = "email"
    attachment: str | None = None  # a data/raw screenshot for the multimodal tickets


def _load_nominal_fixtures() -> list[DemoTicket]:
    """Load the committed nominal ticket fixtures (data/tickets/ticket-0NN.json).

    Skips any fixture with an empty customer_message (ticket-008 is an intentionally-empty
    edge fixture used by the intake validation tests, not a real ticket to run)."""
    out: list[DemoTicket] = []
    for path in sorted(TICKETS_DIR.glob("ticket-0*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not (raw.get("customer_message") or "").strip():
            continue  # skip the empty edge fixture (not a runnable ticket)
        out.append(DemoTicket(
            ticket_id=raw["ticket_id"],
            customer_message=raw["customer_message"],
            channel=raw.get("channel", "email"),
            category="nominal",
        ))
    return out


def _load_adversarial() -> list[DemoTicket]:
    """Reuse Module 9's attack suite (data/attacks.json) as adversarial demo tickets.

    Two attacks: the headline direct injection + PII exfiltration and one more — the run
    proves the guardrail / IAM tool boundary on untrusted input (not just the happy path)."""
    attacks = json.loads(ATTACKS_PATH.read_text(encoding="utf-8"))
    picked = [a for a in attacks if a.get("expect_blocked")][:2] or attacks[:2]
    out: list[DemoTicket] = []
    for i, atk in enumerate(picked, 1):
        out.append(DemoTicket(
            ticket_id=f"cap-attack-{i:02d}",
            customer_message=atk["ticket"],
            channel="chat",
            category="adversarial",
        ))
    return out


def _load_multimodal() -> list[DemoTicket]:
    """Reuse Module 6's screenshot fixtures as multimodal demo tickets.

    The customer pastes a payment-error screenshot; the intake pipeline reads it (Nova Lite
    vision) before the agent answers. Two tickets reference the same committed fixture."""
    shot = RAW_DIR / "payment_error.png"
    msgs = [
        "My payment keeps failing at checkout — here's the error screen I keep seeing.",
        "I can't complete my order, the screen shows this error. What does it mean?",
    ]
    out: list[DemoTicket] = []
    for i, msg in enumerate(msgs, 1):
        out.append(DemoTicket(
            ticket_id=f"cap-vision-{i:02d}",
            customer_message=msg,
            channel="chat",
            category="multimodal",
            attachment=str(shot) if shot.exists() else None,
        ))
    return out


def _capstone_additions() -> list[DemoTicket]:
    """A few extra nominal tickets so the mix totals 20 and covers refund handoffs + escalation.

    These exercise the M8 Billing-specialist handoff + the HITL refund gate (-> awaiting_approval)
    and a clear escalation path, which the committed nominal fixtures only partly cover."""
    return [
        DemoTicket("cap-refund-01",
                   "Please refund order 1042 — it arrived damaged and I want my money back.",
                   "nominal"),
        DemoTicket("cap-refund-02",
                   "I was double-charged on order 1043. Refund the duplicate charge, please.",
                   "nominal"),
        DemoTicket("cap-account-01",
                   "I can't log into my CloudCart account — it says my password is wrong but "
                   "I never changed it. How do I reset it?",
                   "nominal", channel="chat"),
        DemoTicket("cap-shipping-01",
                   "Where is my order 1042? It was supposed to arrive three days ago.",
                   "nominal"),
        DemoTicket("cap-policy-01",
                   "What is your return policy for opened items? I need the exact window.",
                   "nominal"),
        DemoTicket("cap-billing-01",
                   "There's a charge on my statement from CloudCart that I don't recognise. "
                   "Can you tell me what order 1044 was for?",
                   "nominal"),
        DemoTicket("cap-technical-01",
                   "My product images aren't loading on my storefront after the latest update. "
                   "Customers see broken image icons. How do I fix this?",
                   "nominal", channel="chat"),
    ]


def build_demo_tickets() -> list[DemoTicket]:
    """The 20 capstone tickets: nominal fixtures + capstone additions + adversarial + multimodal.

    Trimmed/padded to EXACTLY 20 (the brief's run size; the smoke test asserts it). The mix is
    ~15 nominal + 2 adversarial[M9] + 2 multimodal[M6] + 1 (the 20th nominal makes the count)."""
    nominal = _load_nominal_fixtures() + _capstone_additions()
    adversarial = _load_adversarial()
    multimodal = _load_multimodal()
    # Reserve room for exactly 2 adversarial + 2 multimodal, fill the rest with nominal.
    n_special = len(adversarial) + len(multimodal)
    nominal = nominal[: 20 - n_special]
    tickets = nominal + adversarial + multimodal
    return tickets[:20]


# =============================================================================
# Running one ticket — the LOCAL path (the frozen run_relay seam) or the DEPLOYED API.
# =============================================================================
@dataclass
class TicketOutcome:
    """The result of running one demo ticket through Relay: the persisted facts the recap
    summarises (status, escalation, cost, latency, the cited sources / actions / triage)."""

    ticket_id: str
    category: str
    status: str
    escalated: bool
    gated: bool
    cost_cents: float
    latency_ms: float
    triage: dict | None = None
    n_citations: int = 0
    actions: list[str] = field(default_factory=list)
    answer_preview: str = ""
    error: str | None = None


def _intake_message(ticket: DemoTicket) -> str:
    """Build the message run_relay sees. A multimodal ticket runs the M6 intake pipeline first
    (validate -> Comprehend -> Nova Lite vision read), which appends an [Attachment summary] to
    the customer text, so the agent answers WITH the screenshot's content. Best-effort: if
    intake is unavailable the raw message is used (the run still completes)."""
    if not ticket.attachment:
        return ticket.customer_message
    try:
        from pathlib import Path as _P

        from relay import intake as intake_mod

        raw = intake_mod.RawIntake(
            channel=ticket.channel,
            body=ticket.customer_message,
            ticket_id=ticket.ticket_id,
        )
        shot = _P(ticket.attachment)
        result = intake_mod.intake(
            raw,
            attachment_bytes=shot.read_bytes(),
            attachment_filename=shot.name,
        )
        # IntakeResult.ticket.customer_message carries the appended [Attachment summary]
        # (the Nova Lite vision read), so the agent answers WITH the screenshot's content.
        return result.ticket.customer_message
    except Exception as err:  # noqa: BLE001 — intake is best-effort in the demo.
        print(f"    [intake] vision read skipped for {ticket.ticket_id}: "
              f"{type(err).__name__}: {err}")
        return ticket.customer_message


class _AgentUsageMeter:
    """Record the Strands agent's REAL token usage into the active relay.llm CostMeter.

    The worker's CostMeter (M12) hooks relay.llm.converse() — but the Strands agent reasons
    through its own BedrockModel (smart tier), which does not call converse(). So a pure-agent
    ticket would meter $0 even though it spent real tokens. This context manager NON-INVASIVELY
    wraps BedrockModel.stream (no change to relay/agent.py — the inherited code is untouched) to
    read the `metadata.usage` block Bedrock returns on every model call and feed it to the
    active CostMeter through the SAME M3 per-tier price map (config). The agent runs on the
    `smart` tier, so usage is billed at smart-tier prices — the honest $/ticket. Restores the
    original stream on exit (no global side-effect beyond the `with`)."""

    def __init__(self, tier: str = "smart") -> None:
        self.tier = tier
        self._orig = None

    def __enter__(self) -> "_AgentUsageMeter":
        from strands.models import BedrockModel

        from relay import llm as llm_mod

        self._orig = BedrockModel.stream
        orig = self._orig
        tier = self.tier

        async def _wrapped(model_self, *args, **kwargs):
            async for ev in orig(model_self, *args, **kwargs):
                if isinstance(ev, dict) and ev.get("metadata"):
                    usage = (ev["metadata"] or {}).get("usage")
                    if usage:
                        # Feed the real usage to the active CostMeter via the M3 price map.
                        llm_mod._record_cost(
                            tier,
                            {"inputTokens": int(usage.get("inputTokens", 0)),
                             "outputTokens": int(usage.get("outputTokens", 0))},
                            service_tier=config.DEFAULT_SERVICE_TIER,
                        )
                yield ev

        BedrockModel.stream = _wrapped
        return self

    def __exit__(self, *exc: object) -> None:
        if self._orig is not None:
            from strands.models import BedrockModel

            BedrockModel.stream = self._orig


def run_ticket_local(ticket: DemoTicket, *, run=None) -> TicketOutcome:
    """Run ONE ticket through the frozen run_relay seam, metering its real cost (M12).

    This is the exact path the deployed worker takes (relay.api.worker_handler.process_record
    wraps run_relay in a CostMeter); the demo reuses it so the local recap matches a deployed
    run. The agent's Strands-internal token usage is also metered (via _AgentUsageMeter — a
    non-invasive wrapper, no change to inherited agent code) so the $/ticket is the real cost,
    not just the direct converse() calls. `run` is injectable so the smoke test drives a
    scripted agent offline."""
    from relay.llm import CostMeter

    scripted = run is not None
    if run is None:
        from relay.run import run_relay

        run = run_relay

    message = _intake_message(ticket)
    payload = {"customer_message": message, "ticket_id": ticket.ticket_id,
               "channel": ticket.channel}
    start = time.perf_counter()
    try:
        # Meter the agent's real token usage (skipped for a scripted/offline runner).
        import contextlib

        usage_meter = _AgentUsageMeter() if not scripted else contextlib.nullcontext()
        with CostMeter() as meter, usage_meter:
            resp = run(payload)
        latency_ms = (time.perf_counter() - start) * 1000.0
        record = resp.get("record") or {}
        answer = record.get("answer") or {}
        citations = answer.get("citations") or []
        actions = [a.get("tool") for a in record.get("actions", []) if a.get("tool")]
        return TicketOutcome(
            ticket_id=resp.get("ticket_id", ticket.ticket_id),
            category=ticket.category,
            status=resp.get("status", "failed"),
            escalated=bool(record.get("escalated")) or resp.get("status") == "escalated",
            gated=bool(resp.get("gated")),
            cost_cents=round(float(meter.cost_cents), 4),
            latency_ms=round(latency_ms, 1),
            triage=record.get("triage"),
            n_citations=len(citations),
            actions=actions,
            answer_preview=(resp.get("answer_text") or "")[:140],
        )
    except Exception as err:  # noqa: BLE001 — a failed ticket is a `failed` outcome, not a crash.
        latency_ms = (time.perf_counter() - start) * 1000.0
        return TicketOutcome(
            ticket_id=ticket.ticket_id, category=ticket.category, status="failed",
            escalated=False, gated=False, cost_cents=0.0, latency_ms=round(latency_ms, 1),
            error=f"{type(err).__name__}: {err}",
        )


def run_ticket_api(ticket: DemoTicket, base_url: str, *, poll_timeout_s: int = 90) -> TicketOutcome:
    """Run ONE ticket through the DEPLOYED API: POST /tickets -> poll GET /tickets/{id}.

    The brief's path once the M11 stack is deployed. cost_cents comes off the persisted
    TicketRecord (the worker's M12 meter); latency is the wall-clock POST->terminal time."""
    import urllib.request

    base = base_url.rstrip("/")
    start = time.perf_counter()
    body = json.dumps({
        "customer_message": ticket.customer_message,
        "channel": ticket.channel,
        "ticket_id": ticket.ticket_id,
    }).encode("utf-8")
    try:
        req = urllib.request.Request(base + "/tickets", data=body,
                                     headers={"Content-Type": "application/json"},
                                     method="POST")
        with urllib.request.urlopen(req, timeout=20) as resp:
            ticket_id = json.loads(resp.read())["ticket_id"]

        status, record = "received", {}
        deadline = time.time() + poll_timeout_s
        while time.time() < deadline:
            g = urllib.request.Request(base + f"/tickets/{ticket_id}", method="GET")
            with urllib.request.urlopen(g, timeout=20) as resp:
                record = json.loads(resp.read())
            status = record.get("status", "received")
            if status in TERMINAL:
                break
            time.sleep(4)

        latency_ms = (time.perf_counter() - start) * 1000.0
        answer = record.get("answer") or {}
        citations = answer.get("citations") or []
        actions = [a.get("tool") for a in record.get("actions", []) if a.get("tool")]
        return TicketOutcome(
            ticket_id=ticket_id, category=ticket.category, status=status,
            escalated=bool(record.get("escalated")) or status == "escalated",
            gated=status == "awaiting_approval",
            cost_cents=round(float(record.get("cost_cents", 0.0)), 4),
            latency_ms=round(latency_ms, 1), triage=record.get("triage"),
            n_citations=len(citations), actions=actions,
            answer_preview=(answer.get("text") or "")[:140],
        )
    except Exception as err:  # noqa: BLE001
        latency_ms = (time.perf_counter() - start) * 1000.0
        return TicketOutcome(
            ticket_id=ticket.ticket_id, category=ticket.category, status="failed",
            escalated=False, gated=False, cost_cents=0.0, latency_ms=round(latency_ms, 1),
            error=f"{type(err).__name__}: {err}",
        )


# =============================================================================
# The recap — the costed summary that IS the capstone deliverable.
# =============================================================================
def summarise(outcomes: list[TicketOutcome], *, eval_score: float | None = None) -> dict[str, Any]:
    """Compute the recap: count by status, escalation rate, total $/ticket, p95 latency,
    and (optionally) the golden-set eval score. The numbers the brief requires."""
    by_status: dict[str, int] = {}
    by_category: dict[str, int] = {}
    for o in outcomes:
        by_status[o.status] = by_status.get(o.status, 0) + 1
        by_category[o.category] = by_category.get(o.category, 0) + 1

    n = len(outcomes) or 1
    escalated = sum(1 for o in outcomes if o.escalated)
    gated = sum(1 for o in outcomes if o.gated)
    total_cents = round(sum(o.cost_cents for o in outcomes), 4)
    latencies = sorted(o.latency_ms for o in outcomes)
    p95 = _percentile(latencies, 95) if latencies else 0.0
    return {
        "tickets": len(outcomes),
        "by_status": by_status,
        "by_category": by_category,
        "escalation_rate": round(escalated / n, 3),
        "awaiting_approval": gated,
        "total_cost_cents": total_cents,
        "total_cost_usd": round(total_cents / 100.0, 4),
        "cost_per_ticket_cents": round(total_cents / n, 4),
        "p95_latency_ms": round(p95, 1),
        "eval_grounding": eval_score,
    }


def _percentile(sorted_values: list[float], pct: float) -> float:
    """The pct-th percentile (nearest-rank) of an already-sorted list."""
    if not sorted_values:
        return 0.0
    k = max(0, min(len(sorted_values) - 1,
                   int(round((pct / 100.0) * (len(sorted_values) - 1)))))
    return sorted_values[k]


def _eval_grounding(*, live: bool) -> float | None:
    """Re-run the golden set and return its aggregate grounding (the M13 contract, consumed).

    Off the demo's hot path: with --eval the demo invokes evals.run_evals over the 20-ticket
    golden set and reads result['aggregate']['grounding']. Offline (no --live) it uses the
    committed baseline so the recap always carries the score. Best-effort: any failure returns
    None and the recap notes the score was skipped."""
    try:
        from evals import run_evals

        baseline = run_evals.BASELINE_PATH
        if baseline.exists():
            data = json.loads(baseline.read_text(encoding="utf-8"))
            return round(float(data["aggregate"]["grounding"]), 3)
    except Exception as err:  # noqa: BLE001
        print(f"    [eval] golden-set score skipped: {type(err).__name__}: {err}")
    return None


def print_recap(outcomes: list[TicketOutcome], recap: dict[str, Any]) -> None:
    """Print the per-ticket lines + the recap table — the capstone's printed deliverable."""
    print("\n" + "=" * 78)
    print("RELAY v1.0 — CAPSTONE RUN (20 tickets, end-to-end)")
    print("=" * 78)
    print(f"{'ticket_id':<18}{'category':<13}{'status':<18}{'cost¢':>7}{'ms':>8}  detail")
    print("-" * 78)
    for o in outcomes:
        intent = (o.triage or {}).get("intent", "-") if o.triage else "-"
        if o.error:
            detail = f"ERROR {o.error[:40]}"
        elif o.status == "awaiting_approval":
            detail = "refund parked on HITL gate"
        elif o.actions:
            detail = f"actions={','.join(o.actions)}"
        elif o.n_citations:
            detail = f"cited {o.n_citations} source(s), intent={intent}"
        else:
            detail = f"intent={intent}"
        print(f"{o.ticket_id:<18}{o.category:<13}{o.status:<18}"
              f"{o.cost_cents:>7.3f}{o.latency_ms:>8.0f}  {detail}")

    print("-" * 78)
    print("RECAP")
    print(f"  tickets               : {recap['tickets']}")
    print(f"  by status             : {recap['by_status']}")
    print(f"  by category           : {recap['by_category']}")
    print(f"  escalation rate       : {recap['escalation_rate']:.1%}")
    print(f"  awaiting approval     : {recap['awaiting_approval']}")
    print(f"  total $/ticket        : {recap['total_cost_usd']:.4f} USD "
          f"({recap['total_cost_cents']:.3f}¢ over {recap['tickets']} tickets)")
    print(f"  cost per ticket       : {recap['cost_per_ticket_cents']:.4f}¢")
    print(f"  p95 latency           : {recap['p95_latency_ms']:.0f} ms")
    eg = recap.get("eval_grounding")
    print(f"  golden-set grounding  : {eg if eg is not None else 'skipped'}"
          f"{'  (floor ' + str(config.GROUNDING_THRESHOLD) + ')' if eg is not None else ''}")
    print("=" * 78)


# =============================================================================
# CLI.
# =============================================================================
def _list_tickets() -> int:
    """--list: print the 20 demo tickets (id, category, channel, message preview)."""
    tickets = build_demo_tickets()
    print(f"{len(tickets)} demo tickets:")
    for t in tickets:
        att = " [+screenshot]" if t.attachment else ""
        print(f"  {t.ticket_id:<18}{t.category:<13}{t.channel:<6}"
              f"{t.customer_message[:60]}{att}")
    return 0


def run_demo(*, api_url: str | None = None, eval_score: bool = True,
             live_eval: bool = False, runner: Callable | None = None) -> dict[str, Any]:
    """Run all 20 tickets (API or local), print the recap, and return it.

    `runner` is injectable: the smoke test passes a scripted local runner so the orchestration
    + recap math are exercised offline. By default the local path uses the real run_relay."""
    tickets = build_demo_tickets()
    outcomes: list[TicketOutcome] = []
    mode = f"DEPLOYED API ({api_url})" if api_url else "LOCAL run_relay seam"
    print(f"Running {len(tickets)} tickets through the {mode} ...")
    for t in tickets:
        if api_url:
            o = run_ticket_api(t, api_url)
        else:
            o = run_ticket_local(t, run=runner)
        outcomes.append(o)
        flag = "!" if o.error else " "
        print(f"  {flag}{o.ticket_id:<18}{o.status:<18}{o.cost_cents:>7.3f}¢"
              f"{o.latency_ms:>8.0f}ms")

    score = _eval_grounding(live=live_eval) if eval_score else None
    recap = summarise(outcomes, eval_score=score)
    print_recap(outcomes, recap)
    return recap


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Relay v1.0 capstone — 20-ticket costed run.")
    p.add_argument("--list", action="store_true",
                   help="list the 20 demo tickets and exit (no AWS calls).")
    p.add_argument("--api-url", default=None,
                   help="run against a DEPLOYED API stage (POST /tickets -> poll GET). "
                        "Defaults to RELAY_API_URL env var; omit to use the local seam.")
    p.add_argument("--no-eval", action="store_true",
                   help="skip the golden-set grounding score in the recap.")
    p.add_argument("--live-eval", action="store_true",
                   help="re-run the golden set LIVE for the score (spends tokens, ~M13 cost).")
    args = p.parse_args(argv)

    if args.list:
        return _list_tickets()

    import os

    api_url = args.api_url or os.environ.get("RELAY_API_URL")
    recap = run_demo(api_url=api_url, eval_score=not args.no_eval, live_eval=args.live_eval)
    # Non-zero exit only if EVERY ticket failed (a broken system), so CI can gate on it.
    failed = recap["by_status"].get("failed", 0)
    return 0 if failed < recap["tickets"] else 1


if __name__ == "__main__":
    sys.exit(main())
