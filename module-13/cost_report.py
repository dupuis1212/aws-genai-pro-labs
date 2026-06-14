"""cost_report.py — Relay's before/after cost & p95 report (Module 12, the graded deliverable).

Module 12 makes Relay's $/ticket MEASURABLE, then attacks it. This script is the lab's
NOTED output: it replays a reference set of CloudCart tickets twice — once BASELINE (no
levers: every ticket pays a full-price smart-tier answer) and once OPTIMIZED (the four
levers wired) — and prints a side-by-side table of `$/ticket` AND `p95` latency, plus the
delta. It is exactly the format the AIP-C01 Domain-4 questions ask for: a defensible,
numbered cost/latency trade-off.

    uv run python cost_report.py
        -> runs the reference tickets baseline + optimized against the live models and
           prints the before/after table. A SECOND identical question hits the semantic
           cache and reports cost_cents ≈ 0 with cache_hit=True.

    uv run python cost_report.py --offline
        -> the same table on a SCRIPTED model + an in-memory cache (no AWS, no cost) — the
           shape of the deliverable without spending a cent. The smoke test drives this path.

    uv run python cost_report.py --threshold 0.9 --ttl 3600
        -> sweep the semantic-cache similarity threshold / TTL (the "Try it yourself").

THE FOUR LEVERS this report exercises (brief §6):

  1. TIERED ROUTING (M3, measured here under COST): baseline forces every answer onto the
     smart tier; optimized lets the M3 router keep ~80% of tickets on the fast tier. The
     router was a COST decision before it was an architecture decision.
  2. PROMPT CACHING (relay/llm.py, cache_prompt=True): Relay's long system prompt is a
     reused INPUT prefix → cached provider-side at ≈ -90% on the cached tokens, with NO risk
     of a stale answer (it caches input, not output).
  3. SEMANTIC CACHE (relay/cache.py): a frequent, near-duplicate question is served from
     DynamoDB at cost ≈ 0 instead of a fresh converse() call — guarded by a strict
     similarity threshold + a TTL (never a blind cache).
  4. FLEX / BATCH (config, -50%): NOT exercised on this INTERACTIVE report — Flex/batch are
     latency-tolerant levers for the eval/backfill path only (the batch job lives in
     setup.py; the eval harness that backfills through it is Module 13). The report PRINTS
     the Flex/batch saving on the eval path as a separate line so the number is visible,
     never wired onto the interactive tickets (brief §9).

p95 is the 95th-percentile latency across the reference tickets — the number that catches
the tail a mean hides. Caching collapses the tail for repeated questions (a cache hit is
microseconds), so the optimized p95 drops alongside the cost.

NO model ID here — generation is relay.llm.converse() by tier; the cost line comes from the
M3 per-tier price map via the relay.llm.CostMeter (the API usage block is the source of
truth, never a guess).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from relay import cache as cache_module
from relay import config, llm
from relay.models import Answer

_ROOT = Path(__file__).resolve().parent

# Relay's reusable system prefix — the highest-leverage thing to PROMPT-CACHE (it is
# byte-identical on every ticket). In the real agent this is the much longer system prompt;
# here it stands in so the report's prompt-caching saving is concrete. Padded with the
# standing CloudCart support policy so it clears a model's minimum cacheable length.
_SYSTEM_PROMPT = (
    "You are Relay, CloudCart's support agent. Answer the customer's question concisely "
    "and accurately, grounded in CloudCart policy. Be empathetic, never promise a refund "
    "without the billing rules being met, and escalate anything you cannot resolve. "
    "CloudCart standing policy: orders ship in 2 business days; refunds are issued for "
    "items not delivered within 14 days; duplicate charges are reversed within one cycle; "
    "plan changes take effect at the next billing date; admins are added under Settings > "
    "Team. Keep replies under 120 words and end with a clear next step."
) * 2


# =============================================================================
# The reference ticket set.
# =============================================================================
def load_reference_tickets() -> list[dict]:
    """Load the reference tickets (data/tickets/ticket-0NN.json) the report replays.

    These are the same 10 CloudCart tickets the M2 prompt-regression suite uses — a
    realistic mix (a billing dispute, a 500-error outage, a how-to, ...). The report adds a
    DELIBERATE near-duplicate of the first question at the end so the optimized run shows a
    semantic-cache hit (cost ≈ 0). Falls back to a tiny built-in set if the fixtures are
    absent, so the report always runs.
    """
    tickets: list[dict] = []
    tdir = _ROOT / "data" / "tickets"
    for i in range(1, 11):
        path = tdir / f"ticket-{i:03d}.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            question = (data.get("customer_message") or "").strip()
            # The M2 fixtures include a DELIBERATE empty-message edge ticket (validation
            # bait — Bedrock rejects a blank content block). The cost report replays each
            # ticket as a REAL question, so skip any blank one rather than send an invalid
            # Converse call. We do NOT touch the inherited fixture (it earns its keep in the
            # intake-validation suite); we just do not replay a non-question here.
            if not question:
                continue
            tickets.append({
                "ticket_id": data.get("ticket_id", f"ticket-{i:03d}"),
                "question": question,
            })
    if not tickets:
        tickets = [
            {"ticket_id": "ref-1", "question": "Where is my order 1042?"},
            {"ticket_id": "ref-2", "question": "How do I add a second admin user?"},
        ]
    # A near-duplicate of the FIRST ticket — the semantic-cache demo (a hit on the optimized
    # run, cost ≈ 0). Reworded so it is NOT an exact-hash match: only the SEMANTIC lookup can
    # catch it, which is the point.
    first = tickets[0]
    tickets.append({
        "ticket_id": first["ticket_id"] + "-dup",
        "question": _reword(first["question"]),
        "is_duplicate_of": first["ticket_id"],
    })
    return tickets


def _reword(question: str) -> str:
    """A light reword so the duplicate is SEMANTICALLY close but not hash-identical."""
    return "Hi, quick follow-up: " + question.strip()


# =============================================================================
# A scripted model for --offline (no AWS, no cost) — deterministic usage.
# =============================================================================
def offline_converse_factory() -> Callable:
    """Build a converse() stand-in that returns a fixed answer with realistic token usage.

    It honours the SAME **params contract as relay.llm.converse (tier, cache_prompt,
    service_tier) so the report's optimized path is exercised offline: when cache_prompt is
    on, it reports cacheReadInputTokens so the cost line shows the prompt-caching discount.
    It records into the active CostMeter exactly like the real layer, so meter.cost_cents is
    populated offline too.
    """
    # Per-tier deterministic usage: a smart answer is bigger than a fast one (more output).
    # `sim_latency_s` is a SCRIPTED, deterministic latency so the offline p95 story is honest
    # (a smart-tier answer is slower than a fast one; a cached prefix shaves the input pass) —
    # it stands in for real model latency, which the live run measures for real.
    usage_by_tier = {
        "fast": {"inputTokens": 320, "outputTokens": 60, "sim_latency_s": 0.004},
        "smart": {"inputTokens": 320, "outputTokens": 180, "sim_latency_s": 0.012},
    }
    state = {"warm": set()}  # tiers whose prompt prefix is already cached (2nd call is warm)

    def _offline_converse(messages, *, tier="auto", stream=False, **params):
        concrete = "fast" if tier in ("fast", "auto") else tier
        base = usage_by_tier.get(concrete, usage_by_tier["smart"])
        cache_read = 0
        sim_latency = base["sim_latency_s"]
        if params.get("cache_prompt"):
            # First call WRITES the prefix to cache; later calls READ it at ≈ -90%.
            if concrete in state["warm"]:
                cache_read = 256  # most of the input is the cached system prefix
                sim_latency *= 0.7  # a cached input prefix shaves the input pass
            else:
                state["warm"].add(concrete)
        # Stand in for model latency so the offline p95 reflects the tier/cache choice.
        time.sleep(sim_latency)
        usage = {
            "inputTokens": base["inputTokens"],
            "outputTokens": base["outputTokens"],
            "totalTokens": base["inputTokens"] + base["outputTokens"],
            "cacheReadInputTokens": cache_read,
            "cacheWriteInputTokens": 0,
        }
        service_tier = params.get("service_tier") or config.DEFAULT_SERVICE_TIER
        # Mirror the real layer: total this call into the active CostMeter.
        llm._record_cost(concrete, usage, service_tier)
        return llm.ConverseResult(
            text="Thanks for reaching out — here is what I can do for you.",
            tier=concrete, usage=usage, stop_reason="end_turn",
            service_tier=service_tier,
        )

    return _offline_converse


# =============================================================================
# An in-memory cache for --offline (no DynamoDB) — same SemanticCache behaviour.
# =============================================================================
class _InMemoryTable:
    """A tiny dict-backed stand-in for a boto3 DynamoDB Table (get/put/delete/scan)."""

    def __init__(self) -> None:
        self._items: dict[str, dict] = {}

    def get_item(self, *, Key):  # noqa: N803 - boto3 kwarg name
        key = Key[config.CACHE_KEY]
        item = self._items.get(key)
        return {"Item": item} if item is not None else {}

    def put_item(self, *, Item):  # noqa: N803
        self._items[Item[config.CACHE_KEY]] = dict(Item)

    def delete_item(self, *, Key):  # noqa: N803
        self._items.pop(Key[config.CACHE_KEY], None)

    def scan(self, **kwargs):
        return {"Items": list(self._items.values())}


def offline_embedder() -> Callable:
    """A deterministic stand-in for the Titan embedder: a small bag-of-words vector.

    It is NOT Titan — it is a stable, normalized 1024-dim vector keyed off the question's
    word set, so two SEMANTICALLY close questions (sharing most words) score above the
    threshold and an unrelated one does not. Lets the offline report (and the smoke test)
    exercise the semantic-cache hit/miss decision without a real embeddings call.
    """

    def _embed(text: str) -> tuple[list[float], int]:
        vec = [0.0] * config.EMBED_DIMENSIONS
        for word in cache_module.normalize_question(text).split():
            h = int(hashlib.sha256(word.encode("utf-8")).hexdigest(), 16)
            vec[h % config.EMBED_DIMENSIONS] += 1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec], len(text.split())

    return _embed


# =============================================================================
# Running one ticket — baseline vs optimized.
# =============================================================================
@dataclass
class TicketResult:
    ticket_id: str
    cost_cents: float
    latency_ms: float
    cache_hit: bool = False
    tier: str = ""


@dataclass
class RunResult:
    label: str
    tickets: list[TicketResult] = field(default_factory=list)

    @property
    def total_cents(self) -> float:
        return sum(t.cost_cents for t in self.tickets)

    @property
    def cost_per_ticket(self) -> float:
        return self.total_cents / len(self.tickets) if self.tickets else 0.0

    @property
    def p95_ms(self) -> float:
        return _percentile([t.latency_ms for t in self.tickets], 95)

    @property
    def cache_hits(self) -> int:
        return sum(1 for t in self.tickets if t.cache_hit)


def _percentile(values: list[float], pct: int) -> float:
    """The pct-th percentile (nearest-rank) of a list — p95 is the tail metric we report."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return ordered[min(rank - 1, len(ordered) - 1)]


def run_baseline(tickets, *, converse) -> RunResult:
    """BASELINE: every ticket pays a full-price SMART-tier answer, no caching.

    This is the "before" — what Relay costs if you never measure or optimize: the smart tier
    for everything, the full system prompt billed every time, no cache. It is deliberately
    naive so the optimized run has something to beat (and it mirrors a real app that grew
    without a cost pass).
    """
    result = RunResult(label="baseline")
    for ticket in tickets:
        t0 = time.perf_counter()
        with llm.CostMeter() as meter:
            converse(_messages(ticket["question"]), tier="smart",
                     system=[{"text": _SYSTEM_PROMPT}])
        latency_ms = (time.perf_counter() - t0) * 1000
        result.tickets.append(TicketResult(
            ticket_id=ticket["ticket_id"], cost_cents=meter.cost_cents,
            latency_ms=latency_ms, tier="smart",
        ))
    return result


def run_optimized(tickets, *, converse, cache) -> RunResult:
    """OPTIMIZED: the M3 router picks the tier, the system prompt is prompt-cached, and a
    semantic-cache hit serves a repeat question at cost ≈ 0.

    The three levers wired on the INTERACTIVE path (Flex/batch stay off it, brief §9):
      - tiered routing: tier="auto" lets the router keep simple tickets on the fast tier;
      - prompt caching: cache_prompt=True caches the reused system prefix (≈ -90%);
      - semantic cache: a near-duplicate question is served from the cache (cost ≈ 0).
    """
    result = RunResult(label="optimized")
    for ticket in tickets:
        t0 = time.perf_counter()
        lookup = cache.lookup(ticket["question"])
        if lookup.hit:
            # CACHE HIT — no model call. cost_cents ≈ 0, and the latency is microseconds.
            latency_ms = (time.perf_counter() - t0) * 1000
            result.tickets.append(TicketResult(
                ticket_id=ticket["ticket_id"], cost_cents=0.0,
                latency_ms=latency_ms, cache_hit=True, tier="cache",
            ))
            continue
        with llm.CostMeter() as meter:
            res = converse(_messages(ticket["question"]), tier="auto",
                           system=[{"text": _SYSTEM_PROMPT}], cache_prompt=True)
        latency_ms = (time.perf_counter() - t0) * 1000
        # Store the fresh answer so the NEXT near-duplicate hits the cache (cost ≈ 0).
        cache.store(ticket["question"], _answer_from(res))
        result.tickets.append(TicketResult(
            ticket_id=ticket["ticket_id"], cost_cents=meter.cost_cents,
            latency_ms=latency_ms, tier=getattr(res, "tier", "auto"),
        ))
    return result


def _messages(question: str) -> list[dict]:
    return [{"role": "user", "content": [{"text": question}]}]


def _answer_from(res) -> Answer:
    """Wrap a converse() result's text in the frozen Answer schema for the cache.

    The report is illustrative (the grounded, cited answer comes from the Knowledge Base in
    the real flow); here we store the text with no citations so `grounded` reads False — the
    cache contract still round-trips through the frozen Answer/Citation schemas.
    """
    return Answer(text=getattr(res, "text", ""), citations=[], grounded=False)


# =============================================================================
# The Flex/batch eval-path saving (printed, never wired onto interactive tickets).
# =============================================================================
def eval_path_saving(baseline_cents: float) -> dict:
    """The -50% Flex/batch saving on the EVAL/BACKFILL path — a separate, visible line.

    Flex and batch are latency-tolerant levers (brief §9): they ride the eval/backfill path
    (the batch job in setup.py, the eval harness Module 13 backfills through it), NEVER
    Relay's interactive traffic. We report what the SAME volume of work would cost if it were
    a batch backfill instead of interactive answers, so the -50% is concrete — without ever
    applying it to a customer-facing ticket.
    """
    discounted = baseline_cents * (1 - config.BATCH_DISCOUNT)
    return {
        "interactive_cents": baseline_cents,
        "batch_flex_cents": discounted,
        "saving_pct": config.BATCH_DISCOUNT * 100,
    }


# =============================================================================
# Printing the before/after table (the deliverable).
# =============================================================================
def format_report(baseline: RunResult, optimized: RunResult) -> str:
    """Render the before/after table: $/ticket AND p95, with the deltas."""
    lines: list[str] = []
    lines.append("")
    lines.append("=" * 64)
    lines.append("  Relay cost & latency — before / after (Module 12)")
    lines.append("=" * 64)
    n = len(baseline.tickets)
    lines.append(f"  reference tickets : {n}")
    lines.append(f"  semantic-cache hits (optimized) : {optimized.cache_hits}")
    lines.append("")
    header = f"  {'metric':<22}{'baseline':>14}{'optimized':>14}{'delta':>12}"
    lines.append(header)
    lines.append("  " + "-" * 60)
    lines.append(_row("$/ticket (cents)", baseline.cost_per_ticket,
                      optimized.cost_per_ticket, lower_is_better=True))
    lines.append(_row("total cost (cents)", baseline.total_cents,
                      optimized.total_cents, lower_is_better=True))
    lines.append(_row("p95 latency (ms)", baseline.p95_ms,
                      optimized.p95_ms, lower_is_better=True))
    lines.append("")
    saving = eval_path_saving(baseline.total_cents)
    lines.append("  Flex/batch on the EVAL path (latency-tolerant, never interactive):")
    lines.append(f"    same volume as a batch backfill -> "
                 f"{saving['batch_flex_cents']:.4f}c "
                 f"(-{saving['saving_pct']:.0f}% vs {saving['interactive_cents']:.4f}c)")
    lines.append("=" * 64)
    return "\n".join(lines)


def _row(name: str, before: float, after: float, *, lower_is_better: bool) -> str:
    if before:
        pct = (after - before) / before * 100
    else:
        pct = 0.0
    arrow = "down" if (after < before) else ("up" if after > before else "flat")
    return (f"  {name:<22}{before:>14.4f}{after:>14.4f}"
            f"{pct:>10.1f}% {arrow}")


# =============================================================================
# CLI.
# =============================================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cost_report.py",
        description=("Replay Relay's reference tickets baseline vs optimized and print the "
                     "before/after $/ticket and p95 table (AWS GenAI Pro Mastery, Module 12)."),
    )
    parser.add_argument(
        "--offline", action="store_true",
        help="use a scripted model + an in-memory cache (no AWS, no cost) — the report shape "
             "without spending a cent. The smoke test drives this path.",
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="semantic-cache cosine similarity threshold (default config.CACHE_SIMILARITY_"
             "THRESHOLD) — the 'Try it yourself' hit-rate-vs-false-hit sweep.",
    )
    parser.add_argument(
        "--ttl", type=int, default=None,
        help="semantic-cache TTL in seconds (default config.CACHE_TTL_SECONDS).",
    )
    return parser


def build_runners(args):
    """Resolve the converse() + cache the report runs against (live or --offline).

    Returns (converse, cache). Offline: a scripted converse + an in-memory SemanticCache.
    Live: relay.llm.converse + a SemanticCache over the deployed DynamoDB table (the report
    needs `setup.py` to have created relay-cache + a real embedder).
    """
    if args.offline:
        converse = offline_converse_factory()
        cache = cache_module.SemanticCache(
            table=_InMemoryTable(), embed=offline_embedder(),
            threshold=args.threshold, ttl_seconds=args.ttl,
        )
    else:
        converse = llm.converse
        cache = cache_module.SemanticCache(threshold=args.threshold, ttl_seconds=args.ttl)
    return converse, cache


def run_report(args) -> RunResult:
    """Run both passes and print the table. Returns the optimized RunResult (for tests)."""
    tickets = load_reference_tickets()
    converse, cache = build_runners(args)
    baseline = run_baseline(tickets, converse=converse)
    optimized = run_optimized(tickets, converse=converse, cache=cache)
    print(format_report(baseline, optimized))
    return optimized


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_report(args)
    except Exception as err:  # noqa: BLE001 — print the real cause, exit non-zero.
        print(f"[cost_report] failed: {type(err).__name__}: {err}", file=sys.stderr)
        if not args.offline:
            print("Tip: run with --offline to see the report shape without AWS, or run "
                  "setup.py first (it creates the relay-cache table).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
