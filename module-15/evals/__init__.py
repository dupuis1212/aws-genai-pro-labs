"""evals — Relay's evaluation harness (Module 13 of AWS GenAI Pro Mastery).

Relay ships, runs, and is optimized (M1-M12). It even knows what it costs per ticket (M12).
What it does NOT yet know is whether it is any GOOD — whether the last prompt change (M12's
compression) made an answer worse. "Looks fine after reading five tickets" is not an answer.
This package is the answer: an objective, reproducible quality measurement with a gate that
BLOCKS a regression before it deploys.

  - evals.golden_set : `golden_set.json` — the GOLDEN DATASET (skill 5.1.1): 20 CloudCart
                   tickets (12 nominal, 4 edge, 2 adversarial[M9], 2 multimodal[M6]), each a
                   frozen `{id, ticket: Ticket, expected_intent, expected_points[], must_cite}`
                   (the frozen Évals contract, 06 §2 / bible §3.4). It is a VERSIONED ASSET,
                   not a throwaway test set — it grows from the user-feedback loop
                   (TicketRecord.feedback_rating, the new POST /feedback endpoint).
  - evals.judge  : the LLM-AS-A-JUDGE (skills 5.1.5/5.1.7). A rubric-driven scorer that reads
                   a ticket + Relay's triage/answer/actions and returns a Pydantic-validated
                   JSON verdict (triage correct? expected points covered? grounded? cited when
                   required? right tool used?). The judge is Anthropic Claude Haiku 4.5 on the
                   FLEX tier — a DIFFERENT model family from every Relay CANDIDATE (Amazon Nova
                   fast/smart/vision), so it cannot prefer its own answers (self-preference
                   bias). It validates its own output and retries ONCE on a schema miss — no
                   silent try/except. It also carries the FAIRNESS rubric (skill 3.4.2): the
                   same judge over twin-ticket pairs, flagging a quality gap across irrelevant
                   customer attributes.
  - evals.run_evals : the ORCHESTRATOR (skills 5.1.4/5.1.8/5.1.9). Runs the candidate over the
                   golden set, scores every ticket with the judge, reads the Bedrock RAG-eval
                   report, prints the per-ticket + aggregate table (the reporting IS the table),
                   totals cost_cents, writes `results/run-<name>.json`, and — with --gate — FAILS
                   (exit != 0) when aggregate grounding drops below 0.8 or regresses > 5 pts vs
                   the committed baseline. The same harness validates a fresh deployment and the
                   same gate stage wires into the M11 CodePipeline (after smoke, before promote).

No model ID lives here — the judge's ID is the "judge" tier in relay.config, reached through
the single relay.llm.converse() call site (config.JUDGE_TIER). The RAG-evaluation JOB is a
Bedrock Model Evaluations job (no job surcharge — you pay only the tokens it consumes); it is
created in setup.py and torn down in teardown.py. Flex/batch ride the eval/backfill path only —
never Relay's interactive traffic (brief §9).
"""

__all__ = [
    "golden_set",
    "judge",
    "run_evals",
]
