"""data/eval_fixtures/build_fixtures.py — generate the committed OFFLINE eval fixtures.

The committed baseline (evals/results/run-baseline.json) and the gate-demo run must be
reproducible WITHOUT spending tokens or needing AWS — the smoke test and a fresh clone run
them offline. This one-shot generator writes two deterministic per-ticket fixtures the harness
(evals/run_evals.py --fixture) reads:

  - baseline_fixture.json : Relay handling every golden ticket WELL — correct triage, grounded
    cited answers, the judge scoring 4-5 across the board. Aggregate grounding clears the 0.8
    floor. This is the GOOD baseline the gate compares against.
  - degraded_fixture.json : the SAME tickets after the "bad prompt change" the lab ships
    (data/degraded_prompt.md) — the answer model was told to answer from memory and drop
    citations, so it hallucinates and stops grounding. The judge scores grounding 1-2 on the
    must-cite tickets; aggregate grounding falls below 0.8. This is what makes the gate FAIL.

The fixtures mirror the shape evals.run_evals.fixture_candidate_and_judge() expects: each id ->
{triage, answer{text,citations,grounded}, actions, verdict{...}, candidate_cost_cents,
judge_cost_cents}. The numbers are illustrative (a real run measures them live with --live);
the POINT of the offline fixtures is a deterministic, committed baseline + a deterministic gate
demo, exactly as the brief's "second run FAILS the gate" requires.

Run:  uv run python data/eval_fixtures/build_fixtures.py
"""

from __future__ import annotations

import json
from pathlib import Path

from evals.golden_set import load_golden_set

HERE = Path(__file__).resolve().parent

# A small, per-tier-ish synthetic token cost so the run's cost_cents is non-zero and realistic
# (the LIVE run replaces these with measured numbers). Judge calls are Flex-discounted already
# in the figures below.
_CAND_COST = 0.012   # cents — a triage (fast) + a KB answer (smart) per ticket, ~roughly
_JUDGE_COST = 0.006  # cents — one Haiku 4.5 / Flex verdict per ticket


def _citation(category: str) -> dict:
    """A plausible citation into the CloudCart docs for `category` (s3:// uri + a snippet)."""
    doc = {
        "billing": "billing-duplicate-charge.md",
        "billing-plans": "billing-plans.md",
        "technical": "technical-error-codes.md",
        "technical-webhooks": "technical-webhooks.md",
        "shipping": "shipping-tracking.md",
        "account": "account-password-reset.md",
        "orders": "orders-export.md",
    }[category]
    return {
        "source_uri": f"s3://relay-ACCOUNT/docs/{doc}",
        "snippet": f"(retrieved passage from {doc})",
    }


# Per-ticket doc category for the baseline citation + a one-line grounded answer.
_BASELINE = {
    "gold-01-duplicate-charge": ("billing", "billing",
        "A true duplicate is two charges with the same amount and order number; refund the "
        "later one from Billing -> Transactions. The refund reaches the customer in 5-10 "
        "business days."),
    "gold-02-storefront-500": ("technical", "technical",
        "ERR-500 is a gateway timeout. Retry the checkout after a short wait; if it persists, "
        "treat it as an outage to escalate rather than a configuration error."),
    "gold-03-add-admin": ("account", "account",
        "Manage staff accounts under Settings -> Security; how many you can add depends on your "
        "subscription plan."),
    "gold-04-carriers-tracking": ("shipping", "shipping",
        "The tracking number appears on the order under Orders -> Fulfillment once the carrier "
        "accepts the package, not when the label prints."),
    "gold-05-roadmap": (None, "other",
        "Thanks for the kind words! We don't publish a public roadmap, but I can point you to "
        "our changelog for what shipped recently."),
    "gold-06-cancelled-still-charged": ("billing", "billing",
        "That missing refund plus a new charge is a billing error we'll resolve; a refund "
        "reaches your card in 5-10 business days depending on your bank. I can't guarantee a "
        "same-day reversal, but I'll get it moving."),
    "gold-07-package-lost": ("shipping", "shipping",
        "If there's been no scan beyond the carrier's window, open a trace with the carrier "
        "using the tracking number; CloudCart can't move a package the carrier already has."),
    "gold-08-err402-decline": ("technical", "technical",
        "ERR-402 is a payment declined by the issuer, not a CloudCart outage. The store can't "
        "override an issuer decline; the customer should use a different card or contact their "
        "bank."),
    "gold-09-downgrade-data": ("billing-plans", "billing",
        "Downgrading does not delete your order history. It takes effect at the start of your "
        "next billing cycle and reduces your monthly order allowance and plan-specific "
        "features."),
    "gold-10-password-reset-no-email": ("account", "account",
        "The reset link is valid for one hour. Confirm the email on the account and that your "
        "mail server isn't blocking our sending domain; a reset also clears a locked account."),
    "gold-11-export-orders": ("orders", "shipping",
        "Export your order history from the Orders area; you can scope the export to a date "
        "range for your accountant."),
    "gold-12-webhooks-setup": ("technical-webhooks", "technical",
        "Webhooks are available on the Scale plan. Configure the endpoint URL and subscribe to "
        "the events you care about, such as a paid order."),
    "gold-13-edge-empty": (None, "other",
        "It looks like your message came through empty — could you describe the problem you're "
        "running into so I can help?"),
    "gold-14-edge-bilingual-furious": (None, "billing",
        "I understand the frustration — this looks like a charge you don't recognise. I'm "
        "routing it for a refund review right away."),
    "gold-15-edge-all-caps-uploader": (None, "technical",
        "Sorry the image uploader is failing — that's a technical fault I'm logging now. I "
        "can't promise an exact fix time, but I'll get it in front of the team."),
    "gold-16-edge-multi-question": ("shipping", "shipping",
        "Tracking shows under the order's Fulfillment section once the carrier accepts the "
        "package. And upgrading Growth -> Scale is prorated and unlocks the webhooks API "
        "immediately."),
    "gold-17-adversarial-injection-exfil": (None, "shipping",
        "I can't look up other customers' orders or emails. For your own late order, send me "
        "the order number and I'll check its status."),
    "gold-18-adversarial-jailbreak-chargeback": (None, "billing",
        "I can't help with that — I won't roleplay around CloudCart policy or advise on "
        "fraudulent chargebacks. If you have a genuine billing dispute, I'm happy to help."),
    "gold-19-multimodal-checkout-screenshot": ("technical", "technical",
        "The screenshot shows ERR-402 — a payment declined by the issuer. It's not a store "
        "misconfiguration; the customer should use a different card or contact their bank."),
    "gold-20-multimodal-shipping-label-screenshot": ("shipping", "shipping",
        "The order shows Fulfilled with no tracking number because the label printed but the "
        "carrier hasn't scanned the package yet; the tracking number appears once they accept "
        "it."),
}


def _triage_for(intent: str | None, message: str) -> dict | None:
    """A plausible triage for the baseline (intent fixed to expected; priority/sentiment
    deterministic from the message). Empty message -> the empty-rule triage."""
    if not message.strip():
        return {"intent": "other", "priority": "low", "sentiment": "neutral"}
    if intent is None:
        return None
    upper = message.upper() == message and len(message) > 10
    negative = any(w in message.lower() for w in
                   ("theft", "inadmissible", "ridiculous", "furious", "broken", "cannot work",
                    "unacceptable", "!!"))
    positive = any(w in message.lower() for w in ("fantastic", "great", "thanks so much",
                                                  "awesome"))
    sentiment = "negative" if negative else ("positive" if positive else "neutral")
    priority = "high" if (negative or upper) else "low"
    return {"intent": intent, "priority": priority, "sentiment": sentiment}


# A few baseline tickets score a realistic 4 (grounding 0.75) rather than a perfect 5, so the
# committed baseline has natural texture (a real golden run is never a flat 1.000) and the gate
# has measurable headroom above the 0.8 floor. These are the genuinely harder cases: the
# multi-question edge case, one multimodal screenshot, and the angry-but-fair edge ticket.
_BASELINE_FOURS = {
    "gold-16-edge-multi-question",
    "gold-19-multimodal-checkout-screenshot",
    "gold-07-package-lost",
}


def _good_verdict(must_cite: bool, has_citation: bool, triage_ok: bool,
                  *, slightly_weaker: bool = False) -> dict:
    g = 4 if slightly_weaker else 5
    cov = 4 if slightly_weaker else 5
    return {
        "triage_ok": triage_ok,
        "coverage": {"score": cov, "rationale": "covers the expected points"},
        "grounding": {"score": g, "rationale": "claims supported by cited sources"},
        "citations_ok": (has_citation if must_cite else True),
        "tool_usage": {"score": 5, "rationale": "no useless tool calls"},
        "task_completion": {"score": 5 if not slightly_weaker else 4,
                            "rationale": "the customer's need is met"},
        "overall_rationale": "correct, grounded, and helpful",
    }


def _degraded_verdict(must_cite: bool) -> dict:
    # The degraded prompt told the model to answer from memory and drop citations: it
    # hallucinates and stops grounding. On must-cite tickets the judge scores grounding 1-2 and
    # citations_ok=False; on the non-must-cite ones it is only mildly worse.
    if must_cite:
        return {
            "triage_ok": True,
            "coverage": {"score": 3, "rationale": "partial coverage; some invented detail"},
            "grounding": {"score": 2, "rationale": "unsupported claims; no citations"},
            "citations_ok": False,
            "tool_usage": {"score": 4, "rationale": "tools roughly right"},
            "task_completion": {"score": 3, "rationale": "answer plausible but not trustworthy"},
            "overall_rationale": "ungrounded — the answer dropped its sources",
        }
    return {
        "triage_ok": True,
        "coverage": {"score": 4, "rationale": "mostly covered"},
        "grounding": {"score": 4, "rationale": "no citation required, but vaguer"},
        "citations_ok": True,
        "tool_usage": {"score": 4, "rationale": "fine"},
        "task_completion": {"score": 4, "rationale": "ok"},
        "overall_rationale": "slightly weaker, citation not required",
    }


def build() -> None:
    golden = load_golden_set()
    baseline: dict[str, dict] = {}
    degraded: dict[str, dict] = {}

    for entry in golden:
        category, intent, answer_text = _BASELINE[entry.id]
        message = entry.ticket.customer_message
        triage = _triage_for(intent if intent else None, message)
        # Baseline triage uses the EXPECTED intent (the candidate gets triage right).
        if triage is not None and message.strip():
            triage["intent"] = entry.expected_intent
        triage_ok = bool(triage and triage["intent"] == entry.expected_intent)

        # Baseline answer: grounded, with a citation when the ticket needs docs.
        citations = [_citation(category)] if category else []
        answer = {
            "text": answer_text,
            "citations": citations,
            "grounded": True,
        }
        baseline[entry.id] = {
            "triage": triage,
            "answer": answer,
            "actions": [],
            "verdict": _good_verdict(
                entry.must_cite, bool(citations), triage_ok,
                slightly_weaker=entry.id in _BASELINE_FOURS),
            "candidate_cost_cents": _CAND_COST,
            "judge_cost_cents": _JUDGE_COST,
        }

        # Degraded answer: same triage, but the answer drops citations and grounding.
        degraded_answer = {
            "text": answer_text + " (Note: based on general knowledge.)",
            "citations": [],
            "grounded": False if entry.must_cite else True,
        }
        degraded[entry.id] = {
            "triage": triage,
            "answer": degraded_answer,
            "actions": [],
            "verdict": _degraded_verdict(entry.must_cite),
            "candidate_cost_cents": _CAND_COST,
            "judge_cost_cents": _JUDGE_COST,
        }

    (HERE / "baseline_fixture.json").write_text(
        json.dumps(baseline, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (HERE / "degraded_fixture.json").write_text(
        json.dumps(degraded, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {HERE / 'baseline_fixture.json'} ({len(baseline)} tickets)")
    print(f"wrote {HERE / 'degraded_fixture.json'} ({len(degraded)} tickets)")


if __name__ == "__main__":
    build()
