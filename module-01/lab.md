# Module 1 lab — Secure your account and make your first Converse call

> **This lab cost me less than $0.01 on June 2026 prices** — the measured
> Nova Lite spend across every call in this lab (two `hello_bedrock.py` runs at
> temperature 0.2 and 0.9, plus the one live smoke-test call) was **$0.00003**:
> ~115 input + ~96 output tokens at $0.06 / $0.24 per million. Cheap, not free.
>
> **Teardown reminder:** Module 1 leaves nothing billing in the background.
> Run `uv run python teardown.py` to confirm — it keeps the $5 budget on
> purpose (it backstops the whole course).

**Goal:** prepare a secured, budget-alarmed AWS account for the 15 labs, then
validate the full chain with a first **Converse** call to Amazon Nova Lite.

Region for the whole course: **us-east-1**. Profile: `AWS_PROFILE=aws-genai-pro`
(any profile works as long as it resolves to your course account; on a machine
where the default profile already points there, you can skip the export).

---

## AWS in 30 minutes (only if you're starting from zero)

This course does not re-teach AWS basics. If you don't yet have an account, an
IAM user, and a working CLI, spend 30 minutes here first:

- **Create an AWS account** — https://aws.amazon.com/free/ (then secure the
  root user with MFA and never use it again).
- **Create an IAM user/role for the course** — IAM user guide:
  https://docs.aws.amazon.com/IAM/latest/UserGuide/id_users_create.html
- **Install and configure the AWS CLI** (`aws configure --profile aws-genai-pro`)
  — https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html

Everyone else: continue.

---

## Step 1 — A dedicated, least-privilege identity

Use a course-only IAM user or role, never the root user, and never a hardcoded
key. The policy this lab needs is in [`iam/course-policy.json`](iam/course-policy.json):
it grants only `bedrock:Converse*`/`InvokeModel*` on Nova + Claude profiles,
AWS Budgets, SNS for notifications, and `sts:GetCallerIdentity`. Attach it:

```bash
aws iam create-policy \
  --policy-name aws-genai-pro-course \
  --policy-document file://iam/course-policy.json \
  --profile aws-genai-pro
# then attach the resulting ARN to your course user/role
```

The code never reads an AWS key from a file. `boto3` uses the default session,
which respects `AWS_PROFILE` if set, otherwise your default credentials. There
is **no `.env` for AWS** anywhere in this course.

## Step 2 — Budget alarm before the first token

```bash
export RELAY_BUDGET_EMAIL="you@example.com"
uv sync
uv run python setup.py
```

`setup.py` is idempotent and verbose. It creates a **$5/month AWS Budget** with
an **80% email notification**, prints exactly what it made, and confirms the
cost is **$0** (AWS Budgets gives you 2 budgets free; this uses 1). Run it twice
— the second run reconciles instead of duplicating.

> The budget is **persistent**. It is the one thing Module 1 deliberately leaves
> behind, because it guards every later module's spend.

## Step 3 — Model access

`setup.py` also reports model access. Two facts that trip up old tutorials:

- **There is no "request model access" console step anymore** (gone since Oct
  2025). Serverless foundation models like **Amazon Nova** auto-activate, so the
  Step 4 call works immediately.
- **Anthropic Claude** still needs a **one-time use-case form**. Submit it now
  (Bedrock console > Model access > Anthropic use-case details, or the
  `PutUseCaseForModelAccess` API). It unlocks Claude for later modules — the
  Module 13 LLM-as-a-judge especially.

## Step 4 — Your first Converse call

```bash
uv run python hello_bedrock.py "What does CloudCart sell?"
```

Expected output (yours will differ in wording and token counts):

```text
CloudCart is a hosted e-commerce platform. It lets small merchants set up and
run online stores without managing their own servers or payment plumbing.

tokens: in=23 out=58 | est. cost: $0.00002
```

The cost line is computed from the API's `usage` block times a documented
per-token price (Nova Lite, us-east-1, as of June 2026 — re-verify on the
[Bedrock pricing page](https://aws.amazon.com/bedrock/pricing/)). The model ID
is a single constant at the top of `hello_bedrock.py`: `us.amazon.nova-lite-v1:0`.
That `us.` prefix is an **inference profile** — the only way to call recent
models on-demand. A bare `amazon.nova-lite-v1:0` fails with "Retry with an
inference profile"; the script catches exactly that and tells you how to fix it.

## Step 5 — Play with inference parameters

Edit `inferenceConfig` in `hello_bedrock.py` and re-run the same question:

- `"temperature": 0.0` — deterministic, near-identical answers every run.
- `"temperature": 0.9` — more varied phrasing, occasional tangents.

Watch the token counts and the cost line move with the answer length.

> **InvokeModel comparison (read-only — the only time you'll see it).** Before
> Converse existed, you called each model with `bedrock_runtime.invoke_model()`
> and a **model-specific JSON body** — Nova, Claude, and Llama each had a
> different payload and a different response shape. **Converse** replaced all of
> that with one unified message shape across every model. The only place this
> course uses `invoke_model` is Titan **embeddings** in Module 4 (Converse
> cannot embed). For text generation: always Converse.

## Step 6 — Try it yourself

1. **Same code, different model.** Once the Anthropic form is approved, change
   `MODEL_ID` to `us.anthropic.claude-haiku-4-5-20251001-v1:0` and re-run. The
   request body does not change at all — that's the point of Converse: one shape,
   every model.
2. **Tighten the alarm.** In `setup.py`, change `NOTIFY_THRESHOLD_PCT` to `50.0`
   and re-run to get notified at $2.50 instead of $4.00.

## Step 7 — Teardown

```bash
uv run python teardown.py
```

It asserts there is **no idle-billed resource** (there isn't — Module 1 only
made pay-per-token calls and a free budget) and keeps the budget. At the very
end of the whole course, run `uv run python teardown.py --delete-budget`.

---

## Run the tests (no credentials needed)

```bash
uv run pytest          # offline by default — Stubber, no AWS, no network
RELAY_LIVE_TESTS=1 uv run pytest -m live   # one real sub-cent Bedrock call
```
