"""observability/inject_fault.py — the three VISIBLE, REVERSIBLE faults the lab diagnoses.

The runbook is only proven if you BREAK Relay on purpose and fix it. This script injects one
of three realistic GenAI failures, ONE AT A TIME, so the reader follows docs/runbook.md
(dashboard -> Logs Insights -> hypothesis -> remedy -> verify with run_evals). Every fault is
a FIXTURE, never opaque sabotage (brief §9):

  - `--list`     prints the three faults + their runbook entry.
  - `--fault X`  injects fault X (and records it in the .injected_fault marker).
  - `--restore`  undoes whatever is injected (idempotent — a clean state is a no-op).

The mechanism is in plain sight and commented in full — the reader can see exactly what
changed and undo it. The three faults (brief §6 step 7):

  1. context-overflow : writes a fixture ticket whose customer_message is HUGE (a giant
                        pasted log), so a triage/answer call overflows the model's context
                        window -> a size/validation error or a truncated answer. Diagnosis:
                        prompt size in the invocation logs. Remedy: dynamic chunking /
                        truncation (skill 5.2.1).
  2. kb-corruption    : OVERWRITES one CloudCart KB doc with a CONTRADICTORY version (a wrong
                        refund window) under data/faults/, so a re-sync would teach the KB a
                        falsehood -> grounding drops, citations point at the corrupted doc.
                        Diagnosis: grounding metric + the cited source. Remedy: restore the
                        doc + re-sync (skill 5.2.4 — retrieval drift).
  3. prompt-regression: swaps the answer system prompt for the committed degraded version
                        (data/degraded_prompt.md — the M13 gate demo prompt that answers from
                        memory, uncited) -> triage_ok / grounding fall. Diagnosis: diff the
                        prompt VERSIONS + output diffing. Remedy: revert the prompt (skill
                        5.2.3).

Everything is LOCAL + reversible: faults write under data/faults/ and flip the marker; no AWS
mutation, no model call, no model ID here. The reader APPLIES a fault to a real run by pointing
Relay at the fixture the fault produced (lab.md shows the exact command per fault); `--restore`
removes the fixtures and clears the marker. This keeps the smoke test fully offline and the
fault mechanism teachable.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from relay import config

# Repo root (this file is observability/inject_fault.py).
ROOT = Path(__file__).resolve().parent.parent
FAULTS_DIR = ROOT / "data" / "faults"
FAULT_STATE_FILE = ROOT / config.RELAY_FAULT_STATE_FILE_NAME

# Source artifacts the faults read from / corrupt.
DEGRADED_PROMPT = ROOT / "data" / "degraded_prompt.md"
KB_DOC_TO_CORRUPT = ROOT / "data" / "docs" / "billing-duplicate-charge.md"

def _rel(path: Path) -> str:
    """A repo-relative display path that never raises (tests may point FAULTS_DIR elsewhere)."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


# Each fault maps to a runbook entry (the article's "alarm -> runbook entry" rule applies to
# faults too: a fault you cannot look up is a fault you cannot fix). ONE place so --list, the
# runbook, and lab.md agree.
FAULT_RUNBOOK = {
    config.FAULT_CONTEXT_OVERFLOW: "Truncated answers / context-window overflow",
    config.FAULT_KB_CORRUPTION: "Vague answers / grounding drop (retrieval drift)",
    config.FAULT_PROMPT_REGRESSION: "Triage JSON suddenly wrong (prompt regression)",
}


# =============================================================================
# State marker — the visible record of what is injected.
# =============================================================================
def current_fault() -> str | None:
    """Return the currently-injected fault name, or None if Relay is clean."""
    if not FAULT_STATE_FILE.exists():
        return None
    name = FAULT_STATE_FILE.read_text(encoding="utf-8").strip()
    return name or None


def _write_state(fault: str | None) -> None:
    """Record (or clear) the injected fault. The marker is git-ignored runtime state."""
    if fault is None:
        FAULT_STATE_FILE.unlink(missing_ok=True)
    else:
        FAULT_STATE_FILE.write_text(fault + "\n", encoding="utf-8")


# =============================================================================
# Fault 1 — context-overflow (a HUGE ticket fixture).
# =============================================================================
def _giant_message(target_chars: int = 200_000) -> str:
    """A realistically huge customer_message: a pasted, repeated server log.

    ~200k characters is far past the prompt budget a fast-tier triage call is sized for, so the
    triage/answer call either errors on size or truncates — the context-window-overflow
    symptom. The content is an obvious repeated log line so the fixture is self-evidently a
    fault, not a real ticket.
    """
    line = ("2026-06-13T03:14:07Z ERROR checkout.service order=10422 "
            "stack=NullReferenceException at PaymentGateway.capture(...) retrying...\n")
    reps = target_chars // len(line) + 1
    preface = ("My checkout keeps failing and your support told me to paste the full log. "
               "Here is everything from the console, please read all of it:\n\n")
    return preface + line * reps


def inject_context_overflow() -> Path:
    """Write data/faults/context_overflow_ticket.json — a valid Ticket with a giant message.

    The fixture is a real, schema-valid Ticket (so intake/triage accept it); only its
    customer_message is oversized. lab.md runs Relay on THIS file to reproduce the overflow.
    """
    FAULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = FAULTS_DIR / "context_overflow_ticket.json"
    ticket = {
        "ticket_id": "fault-context-overflow",
        "channel": "email",
        "customer_message": _giant_message(),
        "created_at": "2026-06-13T03:14:07Z",
    }
    out.write_text(json.dumps(ticket, indent=2) + "\n", encoding="utf-8")
    size_kb = out.stat().st_size // 1024
    print(f"  wrote {_rel(out)} ({size_kb} KB customer_message).")
    print("  Reproduce: uv run python -m relay.run \"$(jq -r .customer_message "
          f"{_rel(out)})\"  # overflows the context window")
    print("  Diagnose : prompt/input size in the invocation logs "
          "(observability/queries/invocations_tokens_latency.logsinsights).")
    return out


def restore_context_overflow() -> None:
    """Remove the giant-ticket fixture. Idempotent."""
    out = FAULTS_DIR / "context_overflow_ticket.json"
    out.unlink(missing_ok=True)
    print("  removed the context-overflow ticket fixture.")


# =============================================================================
# Fault 2 — kb-corruption (a CONTRADICTORY KB doc).
# =============================================================================
def inject_kb_corruption() -> Path:
    """Write a CONTRADICTORY copy of a KB doc to data/faults/ + back up the original.

    The corruption flips the refund window from the true "5-10 business days" to a fabricated
    "instant / same minute" and removes the citations-worthy structure — exactly the kind of
    silently-wrong doc that makes Relay cite a falsehood with confidence. We DO NOT touch the
    real data/docs/ file on disk lightly: we copy the original to data/faults/<name>.orig and
    write the corrupted version to data/faults/<name>.corrupt. lab.md then shows how to swap
    the corrupted doc into docs/ + re-sync the KB to reproduce retrieval drift, and --restore
    puts the original back. Visible + reversible.
    """
    FAULTS_DIR.mkdir(parents=True, exist_ok=True)
    backup = FAULTS_DIR / (KB_DOC_TO_CORRUPT.name + ".orig")
    corrupt = FAULTS_DIR / (KB_DOC_TO_CORRUPT.name + ".corrupt")
    # Back up the pristine doc once (never overwrite an existing backup — that would lose the
    # real original if --inject is run twice).
    if not backup.exists():
        shutil.copy2(KB_DOC_TO_CORRUPT, backup)
        print(f"  backed up the original doc -> {_rel(backup)}.")
    corrupted_text = (
        "---\n"
        "title: Duplicate and double charges\n"
        "category: billing\n"
        "---\n\n"
        "# Duplicate and double charges\n\n"
        "Duplicate charges are refunded INSTANTLY — the money is back on the customer's "
        "card within the same minute, guaranteed.\n\n"   # <-- the fabricated falsehood
        "There is never any need to check whether a charge is a pending authorization or a "
        "real capture; just refund everything that looks duplicated, immediately, with no "
        "review.\n"
    )
    corrupt.write_text(corrupted_text, encoding="utf-8")
    print(f"  wrote the CONTRADICTORY doc -> {_rel(corrupt)} "
          "(fabricated 'instant refund').")
    print("  Reproduce: copy the .corrupt over data/docs/" + KB_DOC_TO_CORRUPT.name
          + ", re-run setup.py to re-sync the KB, then ask 'how long do refunds take?'.")
    print("  Diagnose : the EvalGrounding metric drops; the cited source is the corrupted doc "
          "(observability/queries/grounding_by_citation.logsinsights).")
    return corrupt


def restore_kb_corruption() -> None:
    """Restore the original KB doc from the backup + remove the fault artifacts. Idempotent."""
    backup = FAULTS_DIR / (KB_DOC_TO_CORRUPT.name + ".orig")
    corrupt = FAULTS_DIR / (KB_DOC_TO_CORRUPT.name + ".corrupt")
    if backup.exists():
        shutil.copy2(backup, KB_DOC_TO_CORRUPT)
        print(f"  restored data/docs/{KB_DOC_TO_CORRUPT.name} from the backup.")
        backup.unlink(missing_ok=True)
    else:
        print(f"  no backup found — data/docs/{KB_DOC_TO_CORRUPT.name} was not corrupted on "
              "disk (only the .corrupt fixture existed). Fine.")
    corrupt.unlink(missing_ok=True)
    print("  removed the kb-corruption fixtures. Re-run setup.py to re-sync the KB.")


# =============================================================================
# Fault 3 — prompt-regression (the degraded answer prompt).
# =============================================================================
def inject_prompt_regression() -> Path:
    """Stage the committed degraded prompt as the 'deployed' answer prompt under data/faults/.

    The degraded prompt (data/degraded_prompt.md — the M13 gate-demo prompt that answers from
    memory, uncited) is the realistic 'well-intentioned change that slips through review'. We
    copy it to data/faults/active_answer_prompt.md to represent the regressed deployment;
    lab.md shows how to publish it as a new Prompt Management VERSION to reproduce the
    regression in a real run, and --restore drops the staged copy. The diagnosis is OUTPUT
    DIFFING + diffing the prompt VERSIONS (skill 5.2.3), never chasing Lambda metrics.
    """
    if not DEGRADED_PROMPT.exists():
        raise FileNotFoundError(
            f"{_rel(DEGRADED_PROMPT)} is missing — it is the committed M13 "
            "gate-demo degraded prompt the prompt-regression fault reuses."
        )
    FAULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = FAULTS_DIR / "active_answer_prompt.md"
    shutil.copy2(DEGRADED_PROMPT, out)
    print(f"  staged the degraded answer prompt -> {_rel(out)} "
          "(answers from memory, uncited).")
    print("  Reproduce: publish it as a new Prompt Management VERSION, then re-run the "
          "golden set.")
    print("  Diagnose : OUTPUT DIFFING (golden-set answers before/after) + diff the prompt "
          "VERSIONS — NOT Lambda metrics (nothing changed in infra).")
    return out


def restore_prompt_regression() -> None:
    """Remove the staged degraded prompt. Idempotent (the committed prompt is untouched)."""
    out = FAULTS_DIR / "active_answer_prompt.md"
    out.unlink(missing_ok=True)
    print("  removed the staged degraded prompt. The real prompts/triage_prompt.md was never "
          "touched; revert the Prompt Management version to its prior revision.")


# =============================================================================
# Dispatch.
# =============================================================================
_INJECTORS = {
    config.FAULT_CONTEXT_OVERFLOW: inject_context_overflow,
    config.FAULT_KB_CORRUPTION: inject_kb_corruption,
    config.FAULT_PROMPT_REGRESSION: inject_prompt_regression,
}
_RESTORERS = {
    config.FAULT_CONTEXT_OVERFLOW: restore_context_overflow,
    config.FAULT_KB_CORRUPTION: restore_kb_corruption,
    config.FAULT_PROMPT_REGRESSION: restore_prompt_regression,
}


def inject(fault: str) -> None:
    """Inject one fault (refusing to stack a second over a live one). Records the marker."""
    if fault not in config.RELAY_INJECTED_FAULTS:
        raise ValueError(
            f"Unknown fault {fault!r}. Known: {', '.join(config.RELAY_INJECTED_FAULTS)}. "
            "Run with --list."
        )
    active = current_fault()
    if active and active != fault:
        raise RuntimeError(
            f"Fault {active!r} is already injected. Run `--restore` first — the lab injects "
            "ONE fault at a time so the diagnosis is unambiguous."
        )
    print(f"Injecting fault: {fault}  (runbook: {FAULT_RUNBOOK[fault]})")
    _INJECTORS[fault]()
    _write_state(fault)
    print(f"  marker {FAULT_STATE_FILE.name} = {fault}. Follow docs/runbook.md, then "
          "`--restore`.")


def restore() -> None:
    """Undo whatever fault is injected, then verify with run_evals (idempotent)."""
    active = current_fault()
    if active is None:
        print("Nothing to restore — Relay is clean (no .injected_fault marker).")
        return
    print(f"Restoring from fault: {active}")
    _RESTORERS[active]()
    _write_state(None)
    print("  marker cleared. Verify the return to baseline:")
    print("    uv run python evals/run_evals.py --fixture "
          "data/eval_fixtures/baseline_fixture.json \\")
    print(f"      --out evals/results/run-postfix-{active}.json --gate "
          "--baseline evals/results/run-baseline.json")


def list_faults() -> None:
    """Print the three injectable faults + their runbook entry (the --list output)."""
    active = current_fault()
    print("Injectable faults (one at a time; each maps to a docs/runbook.md entry):")
    for name in config.RELAY_INJECTED_FAULTS:
        mark = "  <-- INJECTED" if name == active else ""
        print(f"  {name:<18} runbook: {FAULT_RUNBOOK[name]}{mark}")
    if active is None:
        print("\nState: clean (no fault injected).")
    else:
        print(f"\nState: {active} is injected. `--restore` to undo.")


# =============================================================================
# CLI.
# =============================================================================
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="inject_fault.py",
        description="Inject one of three reversible GenAI faults the lab diagnoses, or restore.",
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true",
                       help="list the three injectable faults + their runbook entry.")
    group.add_argument("--fault", choices=list(config.RELAY_INJECTED_FAULTS),
                       help="inject this fault (one at a time).")
    group.add_argument("--restore", action="store_true",
                       help="undo whatever fault is injected (idempotent).")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        if args.list:
            list_faults()
        elif args.restore:
            restore()
        else:
            inject(args.fault)
    except (ValueError, RuntimeError, FileNotFoundError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
