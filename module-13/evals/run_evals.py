"""evals/run_evals.py — orchestrate Relay's evals + the regression gate (Module 13).

This is where scores become a GATE (skills 5.1.4 / 5.1.8 / 5.1.9). It:

  1. runs the CANDIDATE (Relay's triage + kb.answer) over the 20-ticket golden set;
  2. scores every ticket with the LLM-as-a-judge (evals.judge — Claude Haiku 4.5 / Flex,
     never the candidate model);
  3. optionally folds in the Bedrock RAG-evaluation report (setup.py's job on `relay-kb`);
  4. prints the per-ticket table + the aggregate (the reporting IS the table, skill 5.1.8);
  5. writes the FROZEN results JSON (06 §2 / bible §3.4):

       uv run python evals/run_evals.py --out evals/results/run-<name>.json
       -> { run_name, config, scores: [ {id, triage_ok, grounding, coverage, citations} ],
            aggregate, cost_cents }

  6. with --gate --baseline <file>, FAILS (exit != 0) when aggregate grounding drops below the
     floor (config.EVAL_GROUNDING_FLOOR = the ONE 0.8 grounding constant) OR regresses more than
     config.EVAL_REGRESSION_MAX_DROP (>5 pts) vs the committed baseline. This is the stage the
     M11 CodePipeline runs after smoke, before promote — and the same harness validates a fresh
     deployment (skill 5.1.9).

OFFLINE BY DEFAULT for tests + building the baseline: candidate outputs and judge verdicts are
INJECTABLE (candidate_fn / judge_fn). The committed baseline and the smoke tests run with
deterministic fixtures (no AWS). With --live, the candidate is real triage + kb.answer and the
judge is a real Claude Haiku 4.5 / Flex call — the only path that spends tokens (the brief's
< $2). No model ID lives here; everything routes through relay.config + relay.llm.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from relay import config
from relay.models import AgentAction, Answer, Citation, Triage

from evals import judge as judge_mod
from evals.golden_set import GoldenEntry, categorize, load_golden_set

# Where named run results land (results/run-<name>.json). The baseline is run-baseline.json.
RESULTS_DIR = Path(__file__).resolve().parent / "results"
BASELINE_PATH = RESULTS_DIR / "run-baseline.json"


# =============================================================================
# Candidate outputs — what Relay produced for one ticket.
# =============================================================================
@dataclass
class CandidateOutput:
    """Relay's outputs for one golden ticket: the triage, the answer, the agent actions."""

    triage: Triage | None
    answer: Answer | None
    actions: list[AgentAction] = field(default_factory=list)


# A candidate function maps a GoldenEntry -> (CandidateOutput, candidate_cost_cents). The live
# one calls real triage + kb.answer (metered); the offline one reads a fixture.
CandidateFn = Callable[[GoldenEntry], "tuple[CandidateOutput, float]"]
# A judge function maps (entry, candidate) -> (JudgeVerdict, judge_cost_cents).
JudgeFn = Callable[[GoldenEntry, CandidateOutput], "tuple[judge_mod.JudgeVerdict, float]"]


# =============================================================================
# Live candidate + judge (spend tokens — used with --live only).
# =============================================================================
def live_candidate(entry: GoldenEntry) -> tuple[CandidateOutput, float]:
    """Run Relay for real on one golden ticket: triage + a grounded KB answer. Metered.

    Uses the SAME relay.triage + relay.kb the production agent uses (no re-implementation),
    each wrapped in a relay.llm.CostMeter so the candidate's spend is the REAL token cost, not
    a guess. The empty-message edge case (gold-13) skips the KB answer (nothing to answer).
    """
    from relay import kb as kb_mod
    from relay import triage as triage_mod
    from relay.llm import CostMeter

    message = entry.ticket.customer_message.strip()
    with CostMeter() as meter:
        triage_obj: Triage | None = None
        answer_obj: Answer | None = None
        if message:
            triage_obj, _ = triage_mod.triage(entry.ticket)
            answer_obj = kb_mod.answer(message, grounding_check=True)
    return CandidateOutput(triage=triage_obj, answer=answer_obj, actions=[]), meter.cost_cents


def live_judge(entry: GoldenEntry,
               candidate: CandidateOutput) -> tuple[judge_mod.JudgeVerdict, float]:
    """Score one ticket with the real LLM-as-a-judge (Claude Haiku 4.5 / Flex). Metered."""
    verdict, usage = judge_mod.score_ticket(
        ticket_message=entry.ticket.customer_message,
        expected_intent=entry.expected_intent,
        expected_points=entry.expected_points,
        must_cite=entry.must_cite,
        triage=candidate.triage,
        answer=candidate.answer,
        actions=candidate.actions,
    )
    cost = config.estimate_judge_cost(
        usage["inputTokens"], usage["outputTokens"], discount=0.0) * 100.0
    return verdict, cost


# =============================================================================
# One ticket's scored row (the frozen per-ticket result shape).
# =============================================================================
def score_one(entry: GoldenEntry, candidate: CandidateOutput,
              verdict: judge_mod.JudgeVerdict) -> dict[str, Any]:
    """Build the FROZEN per-ticket score row: {id, triage_ok, grounding, coverage, citations}.

    `triage_ok` is computed DETERMINISTICALLY here (the candidate's intent == expected_intent) —
    a fact we can check, not one to ask the judge to re-derive. `grounding` and `coverage` are
    the judge's 1-5 scores normalized onto [0,1] (so grounding lands on the gate's 0.8 scale).
    `citations` is the judge's citations_ok boolean. This is exactly the 06 §2 row shape.
    """
    triage_ok = (
        candidate.triage is not None
        and candidate.triage.intent == entry.expected_intent
    )
    return {
        "id": entry.id,
        "triage_ok": bool(triage_ok),
        "grounding": round(judge_mod.normalize_score(verdict.grounding.score), 3),
        "coverage": round(judge_mod.normalize_score(verdict.coverage.score), 3),
        "citations": bool(verdict.citations_ok),
    }


def aggregate_scores(scores: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate the per-ticket rows into the run's headline numbers.

    `grounding` is the mean grounding (the number the GATE reads). `triage_accuracy`,
    `coverage`, `citation_rate` are the other means — the reporting table's bottom line.
    """
    n = len(scores) or 1
    return {
        "triage_accuracy": round(sum(1 for s in scores if s["triage_ok"]) / n, 3),
        "grounding": round(sum(s["grounding"] for s in scores) / n, 3),
        "coverage": round(sum(s["coverage"] for s in scores) / n, 3),
        "citation_rate": round(sum(1 for s in scores if s["citations"]) / n, 3),
    }


# =============================================================================
# The run.
# =============================================================================
def run_evals(
    *,
    run_name: str,
    candidate_fn: CandidateFn,
    judge_fn: JudgeFn,
    golden: list[GoldenEntry] | None = None,
    rag_eval_report: dict | None = None,
) -> dict[str, Any]:
    """Run the candidate + judge over the golden set and build the FROZEN results dict.

    Returns { run_name, config, scores:[...], aggregate, cost_cents } (06 §2 / bible §3.4).
    `candidate_fn` / `judge_fn` are injected so the SAME code runs offline (fixtures, no AWS)
    and live (real triage/kb + a real judge). `rag_eval_report` (optional) is the Bedrock
    RAG-eval job's parsed report, folded into `config` for the side-by-side the article shows.
    """
    golden = golden if golden is not None else load_golden_set()
    scores: list[dict[str, Any]] = []
    total_cost_cents = 0.0

    for entry in golden:
        candidate, cand_cost = candidate_fn(entry)
        verdict, judge_cost = judge_fn(entry, candidate)
        scores.append(score_one(entry, candidate, verdict))
        total_cost_cents += cand_cost + judge_cost

    run_config = {
        # The exact models the run used — judge != candidates is visible in the artifact.
        "candidate_tiers": sorted(config.JUDGE_CANDIDATE_TIERS),
        "judge_tier": config.JUDGE_TIER,
        "judge_profile": config.JUDGE_PROFILE,
        "judge_service_tier": config.JUDGE_SERVICE_TIER,
        "grounding_floor": config.EVAL_GROUNDING_FLOOR,
        "regression_max_drop": config.EVAL_REGRESSION_MAX_DROP,
        "golden_set_size": len(golden),
    }
    if rag_eval_report is not None:
        run_config["rag_eval"] = rag_eval_report

    return {
        "run_name": run_name,
        "config": run_config,
        "scores": scores,
        "aggregate": aggregate_scores(scores),
        "cost_cents": round(total_cost_cents, 4),
    }


# =============================================================================
# The regression gate.
# =============================================================================
@dataclass
class GateResult:
    """The gate's verdict: passed + the human-readable reason(s)."""

    passed: bool
    reasons: list[str]


def evaluate_gate(result: dict[str, Any], baseline: dict[str, Any] | None) -> GateResult:
    """Apply the regression gate (contract 06 §2 / bible §3.4).

    FAILS when:
      - aggregate grounding < config.EVAL_GROUNDING_FLOOR (the ONE 0.8 grounding constant), OR
      - grounding regresses more than config.EVAL_REGRESSION_MAX_DROP (>5 pts) vs the baseline.
    Returns a GateResult; the CLI turns a failed gate into a non-zero exit (so the pipeline
    blocks). A missing baseline is allowed for the FLOOR check (you can gate on the floor alone),
    but the regression check is skipped with a noted reason — never silently treated as a pass.
    """
    grounding = float(result["aggregate"]["grounding"])
    reasons: list[str] = []
    passed = True

    if grounding < config.EVAL_GROUNDING_FLOOR:
        passed = False
        reasons.append(
            f"grounding regression: aggregate grounding {grounding:.3f} is below the floor "
            f"{config.EVAL_GROUNDING_FLOOR} (config.EVAL_GROUNDING_FLOOR)."
        )

    if baseline is not None:
        base_grounding = float(baseline["aggregate"]["grounding"])
        drop = base_grounding - grounding
        if drop > config.EVAL_REGRESSION_MAX_DROP:
            passed = False
            reasons.append(
                f"grounding regression: grounding dropped {drop:.3f} vs baseline "
                f"{base_grounding:.3f} (max allowed drop {config.EVAL_REGRESSION_MAX_DROP})."
            )
        elif passed:
            reasons.append(
                f"grounding {grounding:.3f} vs baseline {base_grounding:.3f} "
                f"(drop {drop:+.3f}) — within tolerance."
            )
    else:
        reasons.append("no baseline supplied — checked the grounding floor only.")

    if passed and grounding >= config.EVAL_GROUNDING_FLOOR and not reasons:
        reasons.append(f"grounding {grounding:.3f} >= floor {config.EVAL_GROUNDING_FLOOR}.")
    return GateResult(passed=passed, reasons=reasons)


# =============================================================================
# Reporting (the table that IS the report, skill 5.1.8).
# =============================================================================
def print_report(result: dict[str, Any], golden: list[GoldenEntry] | None = None) -> None:
    """Print the per-ticket table + the aggregate + the cost. The reporting deliverable."""
    by_id = {e.id: e for e in (golden or [])}
    print(f"\n=== Relay eval run: {result['run_name']} ===")
    print(f"{'ticket':<40} {'kind':<11} {'triage':<7} {'ground':<7} {'cover':<7} cite")
    print("-" * 80)
    for s in result["scores"]:
        kind = categorize(by_id[s["id"]]) if s["id"] in by_id else "-"
        print(
            f"{s['id']:<40} {kind:<11} "
            f"{('ok' if s['triage_ok'] else 'MISS'):<7} "
            f"{s['grounding']:<7.3f} {s['coverage']:<7.3f} "
            f"{'yes' if s['citations'] else 'no'}"
        )
    agg = result["aggregate"]
    print("-" * 80)
    print(
        f"AGGREGATE  triage_accuracy={agg['triage_accuracy']:.3f}  "
        f"grounding={agg['grounding']:.3f}  coverage={agg['coverage']:.3f}  "
        f"citation_rate={agg['citation_rate']:.3f}"
    )
    print(f"COST       cost_cents={result['cost_cents']:.4f} "
          f"(${result['cost_cents'] / 100:.4f})")
    if "rag_eval" in result["config"]:
        rag = result["config"]["rag_eval"]
        print(f"RAG-EVAL   {json.dumps(rag, ensure_ascii=False)}")


def write_result(result: dict[str, Any], out_path: str | Path) -> Path:
    """Write the frozen results JSON to out_path (creating results/ if needed)."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def load_baseline(path: str | Path = BASELINE_PATH) -> dict[str, Any]:
    """Load a committed results JSON (the baseline) for the gate comparison."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


# =============================================================================
# Offline candidate/judge from a committed fixture (baseline + tests + the gate demo).
# =============================================================================
def fixture_candidate_and_judge(fixture: dict[str, Any]):
    """Build a (candidate_fn, judge_fn) PAIR from a committed per-ticket fixture.

    The fixture maps each golden id -> the candidate's triage/answer/actions AND the judge's
    rubric scores, so a run is fully deterministic and offline (the committed baseline and the
    gate-demo degraded run are built this way — no tokens spent to ship a baseline). The SAME
    score_one/aggregate/gate code runs; only the source of the numbers differs.
    """
    def candidate_fn(entry: GoldenEntry) -> tuple[CandidateOutput, float]:
        f = fixture[entry.id]
        triage = Triage.model_validate(f["triage"]) if f.get("triage") else None
        answer = None
        if f.get("answer"):
            citations = [Citation.model_validate(c) for c in f["answer"].get("citations", [])]
            answer = Answer(text=f["answer"]["text"], citations=citations,
                            grounded=f["answer"]["grounded"])
        actions = [AgentAction.model_validate(a) for a in f.get("actions", [])]
        return CandidateOutput(triage=triage, answer=answer, actions=actions), \
            float(f.get("candidate_cost_cents", 0.0))

    def judge_fn(entry: GoldenEntry,
                 candidate: CandidateOutput) -> tuple[judge_mod.JudgeVerdict, float]:
        f = fixture[entry.id]
        verdict = judge_mod.JudgeVerdict.model_validate(f["verdict"])
        return verdict, float(f.get("judge_cost_cents", 0.0))

    return candidate_fn, judge_fn


# =============================================================================
# Fairness eval (skill 3.4.2 — the judge over twin pairs).
# =============================================================================
FAIRNESS_PAIRS_PATH = Path(__file__).resolve().parent.parent / "data" / "fairness_pairs.json"


def run_fairness(
    *,
    pairs: list[dict[str, Any]],
    score_a_fn: Callable[[dict[str, Any]], "tuple[judge_mod.FairnessVerdict, float]"],
    score_b_fn: Callable[[dict[str, Any]], "tuple[judge_mod.FairnessVerdict, float]"],
) -> dict[str, Any]:
    """Run the FAIRNESS rubric (3.4.2) over twin pairs; flag a divergence > tolerance.

    For each twin pair (same problem, different irrelevant customer attribute), the judge scores
    BOTH answers and we compare: a quality/tone gap larger than config.FAIRNESS_MAX_SCORE_DIVERGENCE
    means the answer quality varied with the attribute — a fairness failure. score_a_fn/score_b_fn
    are injected so this runs offline (fixture) or live (real judge). Returns a summary dict with
    per-pair gaps and an overall `fair` boolean.
    """
    rows: list[dict[str, Any]] = []
    total_cost = 0.0
    for pair in pairs:
        va, ca = score_a_fn(pair)
        vb, cb = score_b_fn(pair)
        total_cost += ca + cb
        gap = judge_mod.fairness_gap(va, vb)
        rows.append({
            "id": pair["id"],
            "attribute": pair.get("attribute", ""),
            "quality_a": va.quality, "quality_b": vb.quality,
            "tone_a": va.tone, "tone_b": vb.tone,
            "gap": gap,
            "fair": judge_mod.is_fair(va, vb),
        })
    return {
        "tolerance": config.FAIRNESS_MAX_SCORE_DIVERGENCE,
        "pairs": rows,
        "fair": all(r["fair"] for r in rows),
        "cost_cents": round(total_cost, 4),
    }


def fairness_fixture_scorers(fixture: dict[str, Any]):
    """Build offline (score_a_fn, score_b_fn) from a committed fairness fixture."""
    def score_a(pair):
        f = fixture[pair["id"]]["a"]
        return judge_mod.FairnessVerdict.model_validate(f["verdict"]), 0.006

    def score_b(pair):
        f = fixture[pair["id"]]["b"]
        return judge_mod.FairnessVerdict.model_validate(f["verdict"]), 0.006

    return score_a, score_b


def live_fairness_scorers():
    """Build live (score_a_fn, score_b_fn) that call the real judge on each twin's answer.

    Each twin's ANSWER comes from running Relay (kb.answer) on that twin's ticket; the judge
    then scores it with the fairness rubric. Returns the pair of scorer callables.
    """
    from relay import kb as kb_mod
    from relay.models import Ticket

    def _answer_text(ticket_dict: dict) -> str:
        ticket = Ticket.model_validate(ticket_dict)
        ans = kb_mod.answer(ticket.customer_message, grounding_check=True)
        return ans.text

    def score(side):
        def _score(pair):
            text = _answer_text(pair[side])
            verdict, usage = judge_mod.score_answer_for_fairness(
                ticket_message=pair[side]["customer_message"], answer_text=text)
            cost = config.estimate_judge_cost(
                usage["inputTokens"], usage["outputTokens"], discount=0.0) * 100.0
            return verdict, cost
        return _score

    return score("a"), score("b")


def print_fairness(report: dict[str, Any]) -> None:
    print("\n=== Fairness eval (twin pairs, tolerance "
          f"{report['tolerance']} pt) ===")
    print(f"{'pair':<30} {'attribute':<28} {'qA':>3} {'qB':>3} {'gap':>4} fair")
    print("-" * 76)
    for r in report["pairs"]:
        print(f"{r['id']:<30} {r['attribute'][:27]:<28} "
              f"{r['quality_a']:>3} {r['quality_b']:>3} {r['gap']:>4} "
              f"{'yes' if r['fair'] else 'NO'}")
    print("-" * 76)
    print(f"OVERALL fair={report['fair']}  cost_cents={report['cost_cents']:.4f}")


# =============================================================================
# CLI.
# =============================================================================
def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_evals.py",
        description="Run Relay's golden-set evals; optionally gate on grounding regression.",
    )
    p.add_argument("--out", default=str(RESULTS_DIR / "run-latest.json"),
                   help="where to write the run results JSON "
                        "(evals/results/run-<name>.json).")
    p.add_argument("--name", default=None,
                   help="run name (defaults to the --out file stem, e.g. 'baseline').")
    p.add_argument("--live", action="store_true",
                   help="run the REAL candidate (triage + kb.answer) and the REAL judge "
                        "(Claude Haiku 4.5 / Flex) — spends tokens (needs AWS + us-east-1).")
    p.add_argument("--fixture", default=None,
                   help="path to a committed per-ticket fixture JSON (offline candidate + "
                        "judge scores). Used to build the baseline and the gate demo.")
    p.add_argument("--gate", action="store_true",
                   help="apply the regression gate; exit != 0 on a grounding regression.")
    p.add_argument("--baseline", default=str(BASELINE_PATH),
                   help="baseline results JSON to compare against under --gate.")
    p.add_argument("--fairness", action="store_true",
                   help="run the fairness rubric (skill 3.4.2) over the twin pairs instead of "
                        "the golden set; exit != 0 if any pair diverges beyond the tolerance.")
    p.add_argument("--fairness-fixture",
                   default=None,
                   help="committed fairness fixture JSON (offline fairness run).")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    # Fairness mode (skill 3.4.2): score the twin pairs, not the golden set.
    if args.fairness:
        pairs = json.loads(FAIRNESS_PAIRS_PATH.read_text(encoding="utf-8"))
        if args.live:
            score_a, score_b = live_fairness_scorers()
        elif args.fairness_fixture:
            fixture = json.loads(Path(args.fairness_fixture).read_text(encoding="utf-8"))
            score_a, score_b = fairness_fixture_scorers(fixture)
        else:
            print("error: pass --live or --fairness-fixture <file> for a fairness run.",
                  file=sys.stderr)
            return 2
        report = run_fairness(pairs=pairs, score_a_fn=score_a, score_b_fn=score_b)
        print_fairness(report)
        if not report["fair"]:
            print("FAIRNESS FAILED — a twin pair diverged beyond the tolerance.",
                  file=sys.stderr)
            return 1
        print("FAIRNESS PASSED — answer quality holds across the controlled attributes.")
        return 0

    golden = load_golden_set()
    run_name = args.name or Path(args.out).stem.replace("run-", "")

    if args.live:
        candidate_fn: CandidateFn = live_candidate
        judge_fn: JudgeFn = live_judge
    elif args.fixture:
        fixture = json.loads(Path(args.fixture).read_text(encoding="utf-8"))
        candidate_fn, judge_fn = fixture_candidate_and_judge(fixture)
    else:
        print("error: pass --live (real run, spends tokens) or --fixture <file> "
              "(offline run from a committed fixture).", file=sys.stderr)
        return 2

    result = run_evals(run_name=run_name, candidate_fn=candidate_fn, judge_fn=judge_fn,
                       golden=golden)
    print_report(result, golden)
    out = write_result(result, args.out)
    print(f"\nwrote {out}")

    if args.gate:
        baseline = None
        baseline_path = Path(args.baseline)
        if baseline_path.exists():
            baseline = load_baseline(baseline_path)
        else:
            print(f"(no baseline at {baseline_path} — gating on the floor only)",
                  file=sys.stderr)
        gate = evaluate_gate(result, baseline)
        print("\n=== GATE ===")
        for reason in gate.reasons:
            print(f"  - {reason}")
        if gate.passed:
            print("GATE PASSED — grounding holds; deployment may proceed.")
            return 0
        print("GATE FAILED — grounding regression; deployment BLOCKED.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
