"""evals/golden_set.py — the GOLDEN DATASET loader + its frozen schema (Module 13).

The golden dataset (skill 5.1.1) is Relay's reproducible quality yardstick: a small, curated
set of CloudCart tickets with the answer Relay SHOULD produce, so a number — not a vibe —
answers "did we just make Relay worse?". This module owns its FROZEN schema (06 §2 / bible
§3.4) and the loader the harness, the smoke test, and setup.py all read it through.

The frozen entry shape — reproduced field-for-field, no variation:

    { id: str,
      ticket: Ticket,                 # the frozen M2/M6/M10 Ticket schema
      expected_intent: <Triage intent literal>,
      expected_points: list[str],     # the facts a good answer must cover
      must_cite: bool }               # does a good answer require a citation?

20 entries (the brief's mix): 12 nominal, 4 edge cases, 2 adversarial (the M9 injection /
jailbreak family), 2 multimodal (an M6 screenshot attachment). It is a VERSIONED asset, not a
test fixture: it grows from the user-feedback loop (TicketRecord.feedback_rating -> triage the
low-rated answers -> add them here). The `ticket` is validated through the SAME frozen Ticket
schema Relay uses everywhere, so a golden ticket and a production ticket are the same object.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from relay.models import Ticket

# The golden set lives next to this module (evals/golden_set.json) so the harness, setup.py,
# and the smoke test resolve it the same way regardless of the working directory.
GOLDEN_SET_PATH = Path(__file__).resolve().parent / "golden_set.json"

# How many entries the contract pins (06 §2 / bible §3.4). The smoke test asserts EXACTLY this.
GOLDEN_SET_SIZE = 20


class GoldenEntry(BaseModel):
    """One golden-set entry — the FROZEN Évals contract (06 §2 / bible §3.4).

    `expected_intent` reuses the EXACT Triage intent literal (no `refund`, no rename) so a
    golden expectation can never diverge from the frozen Triage enum. `expected_points` are
    the facts a good answer must cover (the judge scores COVERAGE against them). `must_cite`
    says whether a good answer must carry at least one citation (a how-to from the docs must;
    a "thanks for the feedback" reply need not). No field beyond these five — the contract is
    closed; `model_config` forbids extras so a typo'd key fails loudly at load.
    """

    model_config = {"extra": "forbid"}

    id: str
    ticket: Ticket
    expected_intent: Literal["billing", "technical", "account", "shipping", "other"]
    expected_points: list[str]
    must_cite: bool


def load_golden_set(path: str | Path = GOLDEN_SET_PATH) -> list[GoldenEntry]:
    """Load + validate the golden set into a list of frozen GoldenEntry objects.

    Every entry is validated through GoldenEntry (which validates its `ticket` through the
    frozen Ticket schema), so a malformed golden file fails HERE with a clear Pydantic error
    rather than halfway through an eval run. Returns the entries in file order (the run table
    is stable and diffable).
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(
            f"{path}: golden set must be a JSON array of entries, got {type(raw).__name__}."
        )
    return [GoldenEntry.model_validate(item) for item in raw]


def categorize(entry: GoldenEntry) -> str:
    """Classify an entry as nominal / edge / adversarial / multimodal (for the run summary).

    Derived from the id prefix convention in golden_set.json (gold-NN-...): an `-adversarial`
    or `-injection`/`-jailbreak` id is adversarial; an entry whose ticket carries attachments
    is multimodal; an `-edge` id is an edge case; everything else is nominal. Used only for the
    human-readable breakdown the harness prints — never for scoring.
    """
    if entry.ticket.attachments:
        return "multimodal"
    name = entry.id.lower()
    if "adversarial" in name or "injection" in name or "jailbreak" in name:
        return "adversarial"
    if "edge" in name:
        return "edge"
    return "nominal"


def main(argv: list[str] | None = None) -> int:
    """`uv run python -m evals.golden_set` — validate + summarize the golden set."""
    entries = load_golden_set()
    counts: dict[str, int] = {}
    for entry in entries:
        counts[categorize(entry)] = counts.get(categorize(entry), 0) + 1
    print(f"Golden set: {len(entries)} entries (contract: {GOLDEN_SET_SIZE}).")
    for kind in ("nominal", "edge", "adversarial", "multimodal"):
        print(f"  {kind:<11}: {counts.get(kind, 0)}")
    must_cite = sum(1 for e in entries if e.must_cite)
    print(f"  must_cite  : {must_cite} of {len(entries)}")
    ok = len(entries) == GOLDEN_SET_SIZE
    print("OK" if ok else "MISMATCH: entry count != contract")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
