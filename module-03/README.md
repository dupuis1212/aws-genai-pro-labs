# Module 3 — The FM Integration Layer: Model Selection, Routing, Streaming, and Resilience

**What:** Relay gets a real **FM integration layer**. Module 2's triage called one
model directly, with no retry and no streaming. Module 3 writes Relay's **design
doc**, then builds and **freezes** `relay/llm.py` — a single `converse()` that
routes fast/smart by complexity, streams with **ConverseStream**, retries
throttling with **exponential backoff + jitter**, and falls back across Regions —
and refactors triage to call it. This is the LLM contract the rest of the course
consumes.

The `converse(messages, *, tier="auto", stream=False, **params)` signature is
**frozen** here and is byte-identical through Module 15. All model IDs live in one
place — `relay/config.py` — mapped from a tier to an **inference profile**; no
other file in `relay/` names a model.

**How to run** (region us-east-1, profile `AWS_PROFILE=aws-genai-pro`; no AWS key
in code or `.env`):

```bash
uv sync

# 1. Verify access to both tiers' inference profiles (one tiny Converse ping each)
#    and ensure the inherited relay-triage prompt exists (idempotent).
uv run python setup.py

# 2. See the layer route, stream, and cost a call.
uv run python demo_llm.py "Why was I charged twice for order #1042?"   # -> smart, streamed
uv run python demo_llm.py "hi" --no-stream                              # -> fast path

# 3. Prove non-regression: the Module 2 triage suite still passes 10/10
#    (triage now routes through converse(tier="fast")).
uv run python run_prompt_tests.py

# 4. Offline tests (no credentials, no network) — incl. the throttling-backoff path.
uv run pytest
uv run python -m pytest tests/ -k retry

# 5. Leave the account clean (idempotent; no new idle-billed resource exists).
uv run python teardown.py
```

Full step-by-step walkthrough — the design doc, the router, the resilience path,
and the AppConfig / frontier "Try it yourself" — is in [`lab.md`](lab.md).

Files (NEW or MODIFIED in Module 3):

- `relay/config.py` — **NEW.** The sole home of model-ID literals: the tier ->
  inference-profile map (`fast`/`smart`/`frontier`) and a per-tier price map.
- `relay/llm.py` — **NEW and FROZEN.** `converse(messages, *, tier="auto",
  stream=False, **params)`: the unique Bedrock call site — routing, ConverseStream,
  backoff, cross-Region fallback, graceful degradation. No silent `try/except`.
- `relay/triage.py` — **MODIFIED.** Dropped its model constant; now calls
  `converse(tier="fast")` and reports cost via `config.estimate_cost`.
- `docs/relay-design.md` — **NEW.** Relay's design doc (graded artifact).
- `demo_llm.py` — **NEW.** CLI showing the router decision, streamed delivery, cost.
- `relay/models.py`, `prompts/`, `data/tickets/`, `run_prompt_tests.py` —
  **inherited from Module 2, unchanged.**
- `setup.py` / `teardown.py` — verify model access / leave the account clean.
- `tests/smoke_test.py` — offline by default (Stubber); guards the frozen signature,
  the router, the backoff/degrade paths, streaming, and the M2 triage flow. Two
  opt-in live calls.
