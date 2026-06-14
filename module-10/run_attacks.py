"""run_attacks.py — replay Relay's adversarial suite, baseline vs guarded, and MEASURE.

Module 9 of AWS GenAI Pro Mastery. The whole point of the module: you do not DECLARE a
GenAI agent safe, you MEASURE it. This script replays the 12 attacks in
data/attacks.json against Relay's input boundary two ways and prints the damage table
plus the before/after blocking rate.

    uv run python run_attacks.py --baseline
        Runs each attack with NO guardrail — the Module 8 world. Almost everything
        "passes" (reaches the agent), so the table is mostly damage. This is the
        "before" number.

    uv run python run_attacks.py --guarded
        Runs each attack through `relay-guardrail` via the standalone ApplyGuardrail
        API (relay.safety.apply_guardrail on the INPUT). Most attacks are now blocked;
        the table shows which LAYER caught each one, and the blocking rate jumps. Some
        attacks STILL pass on purpose — a guardrail is a probabilistic classifier, not
        a guarantee. That residual is the lesson, printed explicitly.

    uv run python run_attacks.py            # runs BOTH and prints the delta line
        # (measured over the malicious attacks only), e.g. "Blocking rate: 0/9 -> 8/9"
        # — one indirect injection still slips, the lab's pedagogical residual.

Offline by default? No — the --guarded / both modes make REAL ApplyGuardrail calls (one
per attack, a few cents total for ~12-24 text-unit evaluations as of June 2026). The
--baseline mode makes NO AWS call at all (it does not even need the guardrail) — it just
reports that, with no input control, every attack reaches the agent. The smoke test
covers the scoring logic offline with a fake guardrail.

This script holds NO model ID and creates nothing — it READS `relay-guardrail`
(resolved through relay.config from setup.py's markers). Run setup.py first for the
guarded mode.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

from relay import config

_ROOT = Path(__file__).resolve().parent
ATTACKS_FILE = _ROOT / "data" / "attacks.json"


def load_attacks(path: Path = ATTACKS_FILE) -> list[dict]:
    """Load the adversarial suite (12 attacks). Each entry is {id, category, ticket,
    expect_blocked, note?}."""
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class AttackOutcome:
    """One attack's result under one mode."""

    id: str
    category: str
    blocked: bool
    expect_blocked: bool
    caught_by: str  # which layer caught it (guarded mode), or ""

    @property
    def correct(self) -> bool:
        """True if the outcome matches the attack's expectation (blocked iff expected)."""
        return self.blocked == self.expect_blocked


def run_baseline(attacks: list[dict]) -> list[AttackOutcome]:
    """The "before" run: NO guardrail. Every attack reaches the agent (nothing blocked).

    This is the Module 8 world — Relay reads untrusted content with no input control, so
    the input boundary blocks nothing. We do not actually invoke the agent (that would
    burn tokens and, for the malicious ones, is the very thing we are trying to prevent);
    the measured fact is that the INPUT BOUNDARY catches nothing without a guardrail.
    """
    return [
        AttackOutcome(
            id=a["id"], category=a["category"], blocked=False,
            expect_blocked=a["expect_blocked"], caught_by="",
        )
        for a in attacks
    ]


def run_guarded(attacks: list[dict], *, apply_fn=None) -> list[AttackOutcome]:
    """The "after" run: each attack's text goes through `relay-guardrail` on the INPUT.

    `apply_fn(text, source)` defaults to relay.safety.apply_guardrail (a real
    ApplyGuardrail call). The smoke test passes a fake to score offline. An attack is
    "blocked" when the guardrail INTERVENED on the input; caught_by names the policy
    layer (prompt_attack / denied_topic / content_filter / pii_filter / ...).
    """
    if apply_fn is None:
        from relay import safety

        def apply_fn(text, source):  # noqa: ANN001 — local default
            return safety.apply_guardrail(text, source=source)

    from relay import safety  # for SOURCE_INPUT (and to keep imports lazy/offline-safe)

    outcomes: list[AttackOutcome] = []
    for a in attacks:
        result = apply_fn(a["ticket"], safety.SOURCE_INPUT)
        caught = ", ".join(result.caught_by()) if result.intervened else ""
        outcomes.append(
            AttackOutcome(
                id=a["id"], category=a["category"], blocked=result.intervened,
                expect_blocked=a["expect_blocked"], caught_by=caught,
            )
        )
    return outcomes


def blocking_rate(outcomes: list[AttackOutcome]) -> tuple[int, int]:
    """(# blocked, total) over the attacks that SHOULD be blocked (the real defense rate).

    We measure the rate over malicious attacks only (expect_blocked=True): a legitimate
    ticket passing is correct, not a "miss", so counting it would inflate the rate.
    """
    malicious = [o for o in outcomes if o.expect_blocked]
    blocked = sum(1 for o in malicious if o.blocked)
    return blocked, len(malicious)


def print_table(title: str, outcomes: list[AttackOutcome]) -> None:
    """Print the damage/defense table: attack, category, expected, result, caught-by."""
    print(f"\n{title}")
    print("-" * 96)
    print(f"{'attack':30}  {'category':24}  {'expect':7}  {'result':8}  caught by")
    print("-" * 96)
    for o in outcomes:
        expect = "BLOCK" if o.expect_blocked else "pass"
        result = "BLOCKED" if o.blocked else "passed"
        flag = "" if o.correct else "  <-- mismatch"
        print(f"{o.id:30}  {o.category:24}  {expect:7}  {result:8}  "
              f"{o.caught_by or '-'}{flag}")
    print("-" * 96)


def _print_summary(baseline: list[AttackOutcome] | None,
                   guarded: list[AttackOutcome] | None) -> None:
    """Print the blocking-rate line(s) and the honest residual note."""
    print("\n=== Blocking rate (over the malicious attacks) ===")
    if baseline is not None:
        b_blocked, b_total = blocking_rate(baseline)
        print(f"  baseline (no guardrail): {b_blocked}/{b_total} blocked")
    if guarded is not None:
        g_blocked, g_total = blocking_rate(guarded)
        print(f"  guarded  (relay-guardrail on input): {g_blocked}/{g_total} blocked")
    if baseline is not None and guarded is not None:
        b_blocked, b_total = blocking_rate(baseline)
        g_blocked, g_total = blocking_rate(guarded)
        print(f"\nBlocking rate: {b_blocked}/{b_total} -> {g_blocked}/{g_total}")
    if guarded is not None:
        slipped = [o for o in guarded if o.expect_blocked and not o.blocked]
        false_pos = [o for o in guarded if not o.expect_blocked and o.blocked]
        if slipped:
            print(f"\n{len(slipped)} malicious attack(s) STILL passed the guardrail — "
                  "this is expected.")
            print("  A guardrail is a probabilistic classifier, not a guarantee. The IAM")
            print("  tool boundary (Module 7) and post-validation are why a miss here is")
            print("  not catastrophic. You REDUCE and MEASURE injection; you do not 'solve' it.")
            for o in slipped:
                print(f"    - {o.id} ({o.category})")
        if false_pos:
            print(f"\n{len(false_pos)} LEGITIMATE ticket(s) were blocked (false "
                  "positives) — route these to a human review queue (HITL, Module 8).")
            for o in false_pos:
                print(f"    - {o.id}")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    do_baseline = "--baseline" in argv
    do_guarded = "--guarded" in argv
    leftover = [a for a in argv if a not in ("--baseline", "--guarded")]
    if leftover:
        print(f"Unknown argument(s): {' '.join(leftover)}\n"
              "Usage: uv run python run_attacks.py [--baseline] [--guarded]\n"
              "  (no flag) runs BOTH and prints the before/after blocking-rate delta.",
              file=sys.stderr)
        return 1
    # No flag -> run both (the headline before/after comparison).
    if not do_baseline and not do_guarded:
        do_baseline = do_guarded = True

    attacks = load_attacks()
    print(f"Adversarial suite: {len(attacks)} attacks from {ATTACKS_FILE.name} "
          f"(guardrail '{config.RELAY_GUARDRAIL_NAME}').")

    baseline = run_baseline(attacks) if do_baseline else None
    if baseline is not None:
        print_table("BASELINE — no guardrail (Module 8 world): every input reaches the "
                    "agent.", baseline)

    guarded = None
    if do_guarded:
        try:
            guarded = run_guarded(attacks)
        except ValueError as err:
            # Unresolved guardrail id (setup.py not run).
            print(f"\n{err}", file=sys.stderr)
            return 1
        except Exception as err:  # noqa: BLE001 — surface the real safety error clearly
            from relay.safety import SafetyError

            if isinstance(err, SafetyError):
                print(f"\nGuardrail call failed: {err}\n"
                      "Run setup.py to create relay-guardrail, and set "
                      "AWS_PROFILE=aws-genai-pro.", file=sys.stderr)
                return 1
            raise
        print_table("GUARDED — relay-guardrail on the input (ApplyGuardrail).", guarded)

    _print_summary(baseline, guarded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
