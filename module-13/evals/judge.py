"""evals/judge.py — Relay's LLM-as-a-judge (Module 13, skills 5.1.5 / 5.1.7 / 3.4.2).

A judge is a foundation model given an explicit RUBRIC (criteria + scale + what each criterion
means) and asked to score another model's output as structured JSON. It is the multi-perspective
lever the exam tests: cheaper and faster than a human panel, sharper than a string match (a
good support answer has fifty valid phrasings — no single label to diff against).

THE HARD INVARIANT (brief §9, enforced in relay.config): the judge is NEVER the model that
produced the answer. Relay answers with Amazon Nova (fast/smart/vision); the judge is Anthropic
Claude Haiku 4.5 — a different model family. Using the same model to judge itself imports
SELF-PREFERENCE BIAS (it rates "its own" style higher). Crossing vendors costs nothing and
kills the bias argument. The judge also runs on the FLEX service tier (-50%): an eval job
tolerates latency, so it never pays the interactive premium.

What this module gives the harness:
  - score_ticket(...)  : run the full rubric (triage + answer + agent) over one golden entry +
                         Relay's outputs, returning a Pydantic-validated JudgeVerdict.
  - score_pair(...)    : the FAIRNESS rubric (3.4.2) — score two TWIN tickets and flag a
                         quality gap larger than config.FAIRNESS_MAX_SCORE_DIVERGENCE.
  - calibration helpers: the judge is only trustworthy once it AGREES with human judgments on a
                         handful of hand-scored cases (the "calibrate before you trust" rule).

Output is ALWAYS validated through a Pydantic schema, with ONE retry that feeds the validation
error back to the judge (the same structured-output correction loop relay.triage uses). No
silent try/except — a judge that cannot produce a valid verdict raises, it does not return a
fabricated score.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from relay import config
from relay import llm
from relay.models import AgentAction, Answer, Triage

# The judge's scoring scale. 1 (poor) .. 5 (excellent), inclusive — small enough that a human
# can calibrate against it, granular enough to see a regression. ONE place so the rubric prose,
# the schema bounds, and the fairness tolerance agree.
SCORE_MIN = 1
SCORE_MAX = 5

# How many max output tokens the judge gets per verdict. A rubric verdict is a small JSON object
# plus a one-line rationale per criterion; this is generous without inviting an essay (verbosity
# bias works both ways — we ask for terse rationales).
_JUDGE_MAX_TOKENS = 600

# Deterministic judging: temperature 0 so the same answer scores the same way run to run (a
# wandering judge would manufacture "regressions"). The judge tier + service tier come from
# relay.config (the judge != candidate invariant lives there).
_JUDGE_INFERENCE = {"maxTokens": _JUDGE_MAX_TOKENS, "temperature": 0.0}


# =============================================================================
# The validated verdict schemas (the judge's structured output).
# =============================================================================
class CriterionScore(BaseModel):
    """One rubric criterion's score + a terse rationale. Bounded to [SCORE_MIN, SCORE_MAX]."""

    model_config = {"extra": "forbid"}

    score: int = Field(ge=SCORE_MIN, le=SCORE_MAX)
    rationale: str


class JudgeVerdict(BaseModel):
    """The judge's full verdict on one ticket (the structured output the harness consumes).

    The fields the run table reads are BOOLEANS + a grounding score, derived by the judge from
    the rubric:
      - triage_ok   : did Relay's triage intent match expected_intent?
      - coverage    : criterion score for how many expected_points the answer covered;
      - grounding   : criterion score for whether the answer is supported by cited sources
                      (the judge's read of grounding — compared in the article to the M9/RAG-eval
                      grounding, the "two grounding numbers should agree" check);
      - citations_ok: were citations present when must_cite required them?
      - tool_usage  : criterion score for whether the agent called the RIGHT tools and no
                      useless ones (skill 5.1.7 — tool-usage effectiveness, not task completion
                      alone);
      - task_completion : criterion score for whether the ticket's actual need was met.
    Scores are 1-5; the harness normalizes the two grounding-style scores onto [0,1].
    """

    model_config = {"extra": "forbid"}

    triage_ok: bool
    coverage: CriterionScore
    grounding: CriterionScore
    citations_ok: bool
    tool_usage: CriterionScore
    task_completion: CriterionScore
    overall_rationale: str


class FairnessVerdict(BaseModel):
    """The fairness rubric's verdict on ONE answer of a twin pair (skill 3.4.2).

    `quality` is the single quality score the fairness check compares ACROSS the pair: if the
    two twins' quality scores diverge by more than config.FAIRNESS_MAX_SCORE_DIVERGENCE, the
    answer quality varied with an irrelevant customer attribute — a fairness signal.
    """

    model_config = {"extra": "forbid"}

    quality: int = Field(ge=SCORE_MIN, le=SCORE_MAX)
    tone: int = Field(ge=SCORE_MIN, le=SCORE_MAX)
    rationale: str


# =============================================================================
# Rubric prompts (the explicit criteria + scale the judge is held to).
# =============================================================================
_SYSTEM_RUBRIC = (
    "You are a strict, fair evaluation judge for CloudCart's support agent, Relay. You did "
    "NOT write the answers you score and you have no stake in them. Score on the EVIDENCE in "
    "front of you, not on style or length: a terse correct answer beats a long vague one "
    "(do not reward verbosity). Return ONLY a single JSON object — no prose, no code fence. "
    f"All scores are integers from {SCORE_MIN} (poor) to {SCORE_MAX} (excellent)."
)


def _verdict_instructions(must_cite: bool) -> str:
    """The rubric body: the criteria, the scale, and the exact JSON shape required."""
    cite_rule = (
        "citations_ok: true ONLY if the answer carries at least one citation (this ticket "
        "REQUIRES a citation)."
        if must_cite else
        "citations_ok: true (this ticket does not require a citation; set true unless a "
        "citation is present but obviously wrong)."
    )
    return (
        "Evaluate Relay's handling of the ticket against these criteria and return this exact "
        "JSON object:\n"
        "{\n"
        '  "triage_ok": <true|false: did the triage intent match the EXPECTED intent?>,\n'
        '  "coverage": {"score": <1-5: how many of the EXPECTED POINTS the answer covered>, '
        '"rationale": "<one line>"},\n'
        '  "grounding": {"score": <1-5: is every claim supported by the cited sources / the '
        'retrieved context, with no invented promise?>, "rationale": "<one line>"},\n'
        f'  "citations_ok": <true|false: {cite_rule}>,\n'
        '  "tool_usage": {"score": <1-5: did the agent call the RIGHT tools and NO useless '
        'ones? 5 = exactly the needed tools; lower for missing OR superfluous tool calls>, '
        '"rationale": "<one line>"},\n'
        '  "task_completion": {"score": <1-5: was the customer\'s actual need met?>, '
        '"rationale": "<one line>"},\n'
        '  "overall_rationale": "<one line summarizing the verdict>"\n'
        "}\n"
        "Score grounding LOW if the answer makes a promise or states a fact not supported by "
        "the cited sources, even if it sounds plausible. Score tool_usage LOW if the agent "
        "called tools the ticket did not need (e.g. looking up unrelated orders)."
    )


def _fairness_instructions() -> str:
    return (
        "Evaluate the QUALITY and TONE of this single answer on its own merits and return this "
        "exact JSON object:\n"
        "{\n"
        '  "quality": <1-5: how well the answer solves the customer\'s problem>,\n'
        '  "tone": <1-5: how professional and helpful the tone is>,\n'
        '  "rationale": "<one line>"\n'
        "}\n"
        "Judge ONLY the answer's quality and tone. Ignore the customer's name, region, business "
        "size, language fluency, or emotional tone — those are irrelevant to how good the "
        "answer should be."
    )


# =============================================================================
# The judge call (single call site, validated output, one retry).
# =============================================================================
def _candidate_block(
    *,
    ticket_message: str,
    expected_intent: str,
    expected_points: list[str],
    must_cite: bool,
    triage: Triage | None,
    answer: Answer | None,
    actions: list[AgentAction],
) -> str:
    """Render the ticket + Relay's outputs into the user block the judge scores."""
    triage_str = triage.model_dump_json() if triage is not None else "null"
    answer_str = answer.model_dump_json() if answer is not None else "null"
    actions_str = json.dumps([a.model_dump() for a in actions], ensure_ascii=False)
    points = "\n".join(f"  - {p}" for p in expected_points) or "  (none)"
    return (
        f"TICKET:\n{ticket_message}\n\n"
        f"EXPECTED INTENT: {expected_intent}\n"
        f"EXPECTED POINTS a good answer must cover:\n{points}\n"
        f"CITATION REQUIRED: {must_cite}\n\n"
        f"RELAY'S TRIAGE: {triage_str}\n"
        f"RELAY'S ANSWER: {answer_str}\n"
        f"RELAY'S AGENT ACTIONS: {actions_str}\n"
    )


def _call_judge(system: str, user: str, schema: type[BaseModel],
                *, converse_fn=None) -> tuple[BaseModel, dict]:
    """Run the judge once, validate, retry ONCE on a schema miss. Returns (verdict, usage).

    `converse_fn` defaults to relay.llm.converse — the SINGLE Bedrock call site — pinned to the
    judge tier + the Flex service tier (config). Tests inject a fake converse to stay offline.
    The retry feeds the Pydantic error back to the judge (structured-output correction); a
    second failure RAISES JudgeError — never a silent or fabricated verdict.
    """
    converse_fn = converse_fn or llm.converse
    prompt = user
    total_in = total_out = 0
    last_error = ""

    for attempt in (1, 2):
        result = converse_fn(
            [{"role": "user", "content": [{"text": prompt}]}],
            tier=config.JUDGE_TIER,
            system=[{"text": system}],
            inferenceConfig=dict(_JUDGE_INFERENCE),
            service_tier=config.JUDGE_SERVICE_TIER,
        )
        total_in += int(result.usage.get("inputTokens", 0))
        total_out += int(result.usage.get("outputTokens", 0))
        candidate = _extract_json(result.text)
        try:
            verdict = schema.model_validate_json(candidate)
        except ValidationError as err:
            last_error = str(err)
            if attempt == 2:
                break
            prompt = (
                f"{user}\n\n"
                "Your previous reply was rejected by schema validation:\n"
                f"--- your reply ---\n{result.text}\n--- end ---\n"
                f"Validation error:\n{last_error}\n\n"
                "Return ONLY the corrected JSON object — the exact keys and value types "
                "described above, nothing else."
            )
            continue
        usage = {"inputTokens": total_in, "outputTokens": total_out,
                 "totalTokens": total_in + total_out}
        return verdict, usage

    raise JudgeError(
        "The judge produced output that failed schema validation even after one retry.\n"
        f"Last validation error:\n{last_error}"
    )


class JudgeError(RuntimeError):
    """The judge could not produce a schema-valid verdict (even after one retry)."""


def _extract_json(text: str) -> str:
    """Isolate the JSON object from the judge's reply (strip a stray fence / prose)."""
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    return text.strip()


# =============================================================================
# Public scoring API (what evals.run_evals calls).
# =============================================================================
def score_ticket(
    *,
    ticket_message: str,
    expected_intent: str,
    expected_points: list[str],
    must_cite: bool,
    triage: Triage | None,
    answer: Answer | None,
    actions: list[AgentAction] | None = None,
    converse_fn=None,
) -> tuple[JudgeVerdict, dict]:
    """Score one ticket against the full rubric. Returns (verdict, judge token usage).

    The harness passes Relay's real triage/answer/actions for one golden entry; the judge
    returns a validated JudgeVerdict. `triage_ok` short-circuits the model only for the boolean
    intent match the harness already knows — the judge still SEES the triage so its overall
    rationale is informed, but the harness trusts its own deterministic intent comparison for
    the table's triage_ok (a judge should not be asked to re-derive a fact we can compute).
    """
    actions = actions or []
    user = _candidate_block(
        ticket_message=ticket_message,
        expected_intent=expected_intent,
        expected_points=expected_points,
        must_cite=must_cite,
        triage=triage,
        answer=answer,
        actions=actions,
    )
    system = f"{_SYSTEM_RUBRIC}\n\n{_verdict_instructions(must_cite)}"
    verdict, usage = _call_judge(system, user, JudgeVerdict, converse_fn=converse_fn)
    return verdict, usage  # type: ignore[return-value]


def score_answer_for_fairness(
    *,
    ticket_message: str,
    answer_text: str,
    converse_fn=None,
) -> tuple[FairnessVerdict, dict]:
    """Score ONE answer of a fairness twin pair (skill 3.4.2). Returns (verdict, usage)."""
    system = f"{_SYSTEM_RUBRIC}\n\n{_fairness_instructions()}"
    user = f"TICKET:\n{ticket_message}\n\nRELAY'S ANSWER:\n{answer_text}\n"
    verdict, usage = _call_judge(system, user, FairnessVerdict, converse_fn=converse_fn)
    return verdict, usage  # type: ignore[return-value]


def fairness_gap(a: FairnessVerdict, b: FairnessVerdict) -> int:
    """The larger of the two twins' quality/tone score gaps — the fairness divergence."""
    return max(abs(a.quality - b.quality), abs(a.tone - b.tone))


def is_fair(a: FairnessVerdict, b: FairnessVerdict) -> bool:
    """True when the twin answers diverge by at most the allowed tolerance (config)."""
    return fairness_gap(a, b) <= config.FAIRNESS_MAX_SCORE_DIVERGENCE


def normalize_score(score: int) -> float:
    """Map a 1-5 rubric score onto [0, 1] (so grounding/coverage land on the gate's scale).

    1 -> 0.0 .. 5 -> 1.0, linearly. The gate compares aggregate grounding against the 0.8 floor
    (config.EVAL_GROUNDING_FLOOR), and the judge's 1-5 grounding score normalizes onto that
    scale: a 5 is 1.0 (fully grounded), a 4 is 0.75 (already below the 0.8 floor — a warning),
    a 3 is 0.5. Strict on purpose: an answer the judge is unsure is grounded should not pass.
    """
    score = max(SCORE_MIN, min(SCORE_MAX, int(score)))
    return (score - SCORE_MIN) / (SCORE_MAX - SCORE_MIN)


# =============================================================================
# Calibration — the judge is only trusted once it agrees with humans.
# =============================================================================
def calibration_agreement(judge_scores: list[int], human_scores: list[int],
                          *, tolerance: int = 1) -> float:
    """Fraction of cases where the judge agrees with a human score within `tolerance`.

    The "calibrate before you trust" rule (brief §6 step 3, misconception box): before the
    judge's scores gate anything, you hand-score ~5 cases and check the judge LANDS NEAR the
    human. Returns the agreement fraction in [0, 1]; the lab calls a judge calibrated at >= 0.8.
    Pure + offline — no model call (the human scores are committed, the judge scores recorded).
    """
    if not judge_scores or len(judge_scores) != len(human_scores):
        raise ValueError("judge_scores and human_scores must be non-empty and equal length.")
    agree = sum(1 for j, h in zip(judge_scores, human_scores) if abs(j - h) <= tolerance)
    return agree / len(judge_scores)
