"""observability/ — Relay's ops layer (Module 14).

Module 14 of AWS GenAI Pro Mastery gives Relay EYES. Through Module 13 Relay was
deployed (M11), cost-instrumented (M12), and evaluated (M13) — but in production it
was BLIND: no centralized invocation logs, no dashboard, no alarms, no procedure when
something breaks. A GenAI app does not fail with a stack trace; it DEGRADES in silence
(vaguer answers, drifting retrieval, a quietly doubling bill). This package is the cure.

It is NOT a `relay/` submodule (like `evals/`, it sits beside the package), so it changes
no frozen contract and `relay/__init__.__all__` is untouched. It OBSERVES the existing
Relay calls; it never makes a generation call of its own (no model ID lives here — the
grep gate covers it). Three pieces:

  - metrics.py             : the EMF / PutMetricData emitter. Relay's worker
                             (relay/api/worker_handler.py) and the eval harness
                             (evals/run_evals.py) call it BY ADDITION to publish the
                             custom metrics the dashboard reads ($/ticket, escalation,
                             guardrail block rate, eval grounding, tool latency, tokens).
                             All metric names + the namespace live in relay.config.
  - setup_observability.py : turns ON Bedrock model-invocation logging (-> CloudWatch
                             Logs, FREE on the Bedrock side), builds the `relay-ops`
                             dashboard (8 widgets), and creates the four alarms (p95
                             latency, throttling, cost anomaly, grounding<0.8 reusing the
                             one M9/M13 0.8 constant) wired to an SNS email topic. Prints
                             the dashboard URL.
  - inject_fault.py        : the three VISIBLE, REVERSIBLE faults the lab diagnoses
                             (context-overflow, kb-corruption, prompt-regression) +
                             `--restore`. A fault is a documented fixture, never opaque
                             sabotage (brief §9): `--list` shows them, `--restore` undoes
                             them, and the mechanism is commented in full.

The Logs Insights queries the runbook references live under queries/ (so the runbook cites
a PRECISE query, not "look at the logs"). The runbook itself is docs/runbook.md.

Model Invocation Logs (Bedrock request/response payloads) are NOT CloudTrail (M10
management-event audit) — the exam's favourite distinction. CloudWatch generative AI
observability (GA re:Invent 2025) is the native LLM/agent tracing path; this package holds
the names and wiring, never a homemade pre-GA metric-filter parser of prompt text.
"""

from __future__ import annotations

__all__ = [
    "metrics",
    "setup_observability",
    "inject_fault",
]
