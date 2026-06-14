"""run_prompt_tests.py — the 10-ticket triage regression suite.

Module 2 of AWS GenAI Pro Mastery. This treats prompts like code: a fixed set of
10 reference CloudCart tickets (all 5 intents + edge cases — empty, bilingual,
shouting), with OBJECTIVE pass/fail criteria. Run it whenever you change the
prompt, the model, or the inference parameters; a green 10/10 is the gate for
shipping a new prompt version.

The three objective criteria per ticket:
  1. Valid JSON that validates against the Triage schema.
  2. intent == expected_intent.
  3. priority == expected_priority.

Sentiment is intentionally not asserted — it is the softest field, and the suite
stays objective. There is deliberately NO LLM-as-a-judge and no semantic metric
here: that is Module 13. This is a deterministic, assertion-based regression
suite, which is exactly what the exam's prompt-QA skill (1.6.4) is about.

This script makes REAL Converse calls (one per ticket, ~10 total). It needs the
prompt published (uv run python setup.py) and AWS credentials. Cost: ~10 Nova
Micro calls at temperature 0 — a fraction of a cent.

Run it:
    uv run python run_prompt_tests.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from relay.triage import (
    TriageError,
    estimate_cost,
    load_ticket,
    resolve_prompt_id,
    triage,
)

_ROOT = Path(__file__).resolve().parent
TICKETS_DIR = _ROOT / "data" / "tickets"
EXPECTED_FILE = TICKETS_DIR / "expected.json"


def load_expectations() -> list[dict]:
    data = json.loads(EXPECTED_FILE.read_text(encoding="utf-8"))
    return data["expectations"]


def run() -> int:
    expectations = load_expectations()

    # Resolve the prompt once up front so a missing setup fails clearly, not 10x.
    try:
        prompt_id = resolve_prompt_id()
    except TriageError as err:
        print(err, file=sys.stderr)
        return 2

    header = f"{'ticket':<12} {'expected':<22} {'got':<22} result"
    print(header)
    print("-" * len(header))

    passed = 0
    total_in = total_out = 0
    rows_failed: list[str] = []

    for exp in expectations:
        ticket_id = exp["ticket_id"]
        want_intent = exp["expected_intent"]
        want_priority = exp["expected_priority"]
        ticket = load_ticket(TICKETS_DIR / f"{ticket_id}.json")

        expected_str = f"{want_intent}/{want_priority}"
        try:
            result, usage = triage(ticket, prompt_id=prompt_id)
        except TriageError as err:
            print(f"{ticket_id:<12} {expected_str:<22} {'<invalid JSON>':<22} FAIL")
            rows_failed.append(f"{ticket_id}: {err}")
            continue

        total_in += usage["inputTokens"]
        total_out += usage["outputTokens"]

        got_str = f"{result.intent}/{result.priority}"
        ok = result.intent == want_intent and result.priority == want_priority
        mark = "PASS" if ok else "FAIL"
        print(f"{ticket_id:<12} {expected_str:<22} {got_str:<22} {mark}")
        if ok:
            passed += 1
        else:
            rows_failed.append(
                f"{ticket_id}: expected {expected_str}, got "
                f"{result.intent}/{result.priority}/{result.sentiment}"
            )

    n = len(expectations)
    cost = estimate_cost(total_in, total_out)
    print("-" * len(header))
    print(f"score: {passed}/{n}")
    print(f"tokens: in={total_in} out={total_out} | est. cost: ${cost:.6f}")

    if rows_failed:
        print("\nFailures:", file=sys.stderr)
        for line in rows_failed:
            print(f"  - {line}", file=sys.stderr)

    # Non-zero exit on any failure so this can gate a prompt change in CI.
    return 0 if passed == n else 1


if __name__ == "__main__":
    raise SystemExit(run())
