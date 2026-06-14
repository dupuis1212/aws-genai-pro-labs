# The "bad prompt change" — what the regression gate catches (Module 13 demo)

This file is the *deliberately degraded* answer-system prompt the lab uses to prove the
regression gate works. It is the kind of well-intentioned change that slips through code review
and quietly breaks Relay: an engineer, chasing the Module 12 cost win, decides the Knowledge
Base round-trip is "too slow" and rewrites the answer prompt to let the model reply **from its
own memory, without retrieving or citing sources**.

It reads perfectly reasonable. It is also exactly how a support agent starts hallucinating
refund timelines and inventing policy — ungrounded, uncited, confident, wrong.

## The degraded answer-system prompt (DO NOT ship this)

> You are Relay, CloudCart's support agent. Answer the customer's question directly and
> concisely from your own knowledge. Do not waste time retrieving documents or adding
> citations — customers want a fast, friendly answer, not a wall of links. Be confident and
> reassuring; if you are not sure of an exact figure, give your best estimate so the customer
> is not left waiting.

Two instructions do the damage:

1. **"from your own knowledge ... do not retrieve or cite"** — the answer stops being grounded
   in the CloudCart docs, so the judge's **grounding** score collapses on every `must_cite`
   ticket and **citations** drops to false.
2. **"give your best estimate"** — invites invented refund windows and policy, which the judge
   scores as ungrounded even when they sound plausible.

## How the lab demonstrates the catch (offline, no tokens)

`data/eval_fixtures/degraded_fixture.json` is the committed, deterministic result of running the
golden set through this degraded prompt (built by `data/eval_fixtures/build_fixtures.py`): the
answers drop their citations, the judge scores grounding 1-2 on the must-cite tickets, and the
aggregate grounding falls from the baseline **0.963** to **0.400**.

```
uv run python evals/run_evals.py \
  --out evals/results/run-degraded.json \
  --name degraded \
  --fixture data/eval_fixtures/degraded_fixture.json \
  --gate --baseline evals/results/run-baseline.json
```

The gate fails twice over — below the **0.8** floor (`config.EVAL_GROUNDING_FLOOR`, the same
0.8 the Module 9 grounding escalation and the Module 14 alarm use) **and** more than a 5-point
drop versus the baseline — and `run_evals.py` exits non-zero. In the pipeline (the eval-gate
stage the Module 13 CDK change wires in after smoke, before promote), a non-zero exit **blocks
the deploy**. The bad prompt never reaches a customer.

## Running it live (real tokens)

To see the real model degrade rather than the committed fixture, replace `relay/kb.py`'s answer
prompt with the degraded text above (or point `kb.answer` at it), then run
`uv run python evals/run_evals.py --live --out evals/results/run-degraded.json --gate`. The lab
keeps the fixture path as the default so a fresh clone can prove the gate **without spending a
cent** — the same reason the baseline is committed.
