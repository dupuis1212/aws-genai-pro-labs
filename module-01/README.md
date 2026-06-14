# Module 1 — Bedrock, Models, and Your First Converse Call

**What:** the bootstrap module of AWS GenAI Pro Mastery. You secure and budget
an AWS account, confirm model access, and make your first **Amazon Bedrock
Converse** call to Amazon Nova Lite — the fundamental gesture every later module
builds on. There is no `relay/` package yet; that starts in Module 2.

**How to run** (region us-east-1, profile `AWS_PROFILE=aws-genai-pro`):

```bash
uv sync
export RELAY_BUDGET_EMAIL="you@example.com"
uv run python setup.py                                # $5 budget + 80% alarm (idempotent, $0)
uv run python hello_bedrock.py "What does CloudCart sell?"  # first Converse call + cost line
uv run pytest                                         # offline tests, no credentials
uv run python teardown.py                             # asserts nothing idle-billed; keeps budget
```

Full step-by-step walkthrough — including the "AWS in 30 minutes" links for
beginners and the "Try it yourself" extensions — is in [`lab.md`](lab.md).

Files: `hello_bedrock.py` (the Converse call), `setup.py` / `teardown.py`
(idempotent account hygiene), `iam/course-policy.json` (least-privilege policy),
`tests/smoke_test.py` (offline by default).
