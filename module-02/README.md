# Module 2 — Prompt Engineering in Production: Templates, Structured Output, and Prompt Management

**What:** Relay gets its first real job — **triage**. A raw CloudCart support
ticket comes in; a validated `Triage {intent, priority, sentiment}` object comes
out, every time. This module introduces the `relay/` package (its first two
schemas, `Ticket` and `Triage`), a triage prompt versioned and governed in
**Amazon Bedrock Prompt Management**, structured JSON output validated by
Pydantic with one validation retry, and a 10-ticket regression suite that turns
"it looks better" into "10/10 tickets pass".

There is no `relay/llm.py` or `relay/config.py` yet — those arrive in Module 3.
In Module 2, the single model ID lives as one **provisional** constant at the top
of `relay/triage.py`.

**How to run** (region us-east-1, profile `AWS_PROFILE=aws-genai-pro`):

```bash
uv sync

# 1. Create the parameterized triage prompt in Bedrock Prompt Management
#    and publish version 1 (idempotent; prints the prompt ID + version ARN).
uv run python setup.py

# 2. Triage one ticket: prints the validated Triage JSON + tokens/cost.
uv run python -m relay.triage data/tickets/ticket-001.json

# 3. Run the 10-ticket regression suite: prints a per-ticket table + final score.
uv run python run_prompt_tests.py

# 4. Offline tests (no credentials, no network).
uv run pytest

# 5. Remove the Prompt Management prompt and all its versions (idempotent).
uv run python teardown.py
```

Full step-by-step walkthrough — including the v1→v3 prompt iteration, the
inference-parameter table, and the "Try it yourself" extensions — is in
[`lab.md`](lab.md).

Files:

- `relay/models.py` — the frozen `Ticket` (4 fields) and `Triage` schemas.
- `relay/triage.py` — `triage(ticket)`: load the prompt by id+version, Converse
  (Nova Micro, temperature 0), `Triage.model_validate_json`, 1 **validation**
  retry.
- `prompts/triage_prompt.md` — the git mirror of the Prompt Management template
  (byte-synced; the repo stays the source of truth for the course).
- `data/tickets/` — 10 reference tickets (5 intents + edge cases).
- `run_prompt_tests.py` — the prompt regression suite.
- `setup.py` / `teardown.py` — idempotent Prompt Management lifecycle.
- `tests/smoke_test.py` — offline by default (Stubber); one opt-in live call.
