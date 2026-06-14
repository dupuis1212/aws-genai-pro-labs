# Module 2 lab — Relay's triage: structured output, Prompt Management, and a regression suite

> **This lab cost me $0.0032 on June 2026 prices.** Triage runs on **Amazon
> Nova Micro** ($0.035 / $0.14 per million input/output tokens, us-east-1). That
> figure includes the prompt iteration to a green 10/10 plus several full passes
> of the 10-ticket regression suite — measured from the Converse `usage` block,
> never guessed. A clean run-through of the lab as written (setup + one triage +
> one suite + teardown) is about **$0.0006**: ~13,000 input + ~220 output tokens
> per suite pass, with a seven-example few-shot prompt riding along on every call.
> **Amazon Bedrock Prompt Management does not bill for stored prompts or
> versions** (re-verify on the
> [Bedrock pricing page](https://aws.amazon.com/bedrock/pricing/)).
>
> **Teardown reminder:** run `uv run python teardown.py` when you're done. It
> deletes the `relay-triage` prompt and all its versions, and confirms Module 2
> left nothing else billing. The M1 $5 budget stays — it backstops the course.

**Goal:** build Relay's triage — a raw CloudCart ticket in, a validated
`Triage {intent, priority, sentiment}` object out — with the prompt **versioned
in Amazon Bedrock Prompt Management** and a **10-ticket regression suite** that
gives you an objective 10/10 instead of a vibe.

Region for the whole course: **us-east-1**. Profile: `AWS_PROFILE=aws-genai-pro`
(any profile that resolves to your course account works).

---

## Step 1 — The `relay/` package and its first two schemas

Module 2 introduces the cumulative `relay/` package. It ships exactly two pieces:
`relay/models.py` (schemas) and `relay/triage.py` (the triage flow). No
`relay/llm.py` or `relay/config.py` yet — those arrive in Module 3.

`relay/models.py` freezes the first two schemas of the whole course. Reproduce
them **field-for-field** — they are a contract every later module depends on:

```python
class Ticket(BaseModel):           # exactly 4 fields in Module 2
    ticket_id: str
    channel: Literal["email", "chat"]
    customer_message: str
    created_at: str

class Triage(BaseModel):           # complete and frozen — never extended
    intent: Literal["billing", "technical", "account", "shipping", "other"]
    priority: Literal["low", "normal", "high", "urgent"]
    sentiment: Literal["negative", "neutral", "positive"]
```

`Triage`'s three enums are LAW: 5 intents, 4 priorities, 3 sentiments. No
`refund` intent, no `severity` rename. (Module 6 and Module 10 each add one field
to `Ticket` — by addition only — but not yet.)

```bash
uv sync
```

## Step 2 — Write the triage prompt (v1 → v3)

A production prompt is not one shot; you iterate it against real tickets. The
file `prompts/triage_prompt.md` is the **git source of truth** for the course —
it is byte-synced with Prompt Management in Step 3.

- **v1 (naive):** "Classify this ticket's intent, priority, and sentiment."
  Result on ticket-001: the model writes a paragraph — *"This looks like a
  billing issue, and the customer seems upset, so I'd say high priority…"* — then
  `Triage.model_validate_json` explodes. No structure, no JSON.
- **v2 (constrained):** add the **role**, the **format constraint** ("Return a
  SINGLE JSON object… the first character MUST be `{`"), and the allowed values
  per field. Now most tickets parse — but the edge cases drift: the empty
  ticket-008 gets a hallucinated `"technical"`, and the all-caps ticket-010 gets
  `"urgent"` when it's really `"high"` (annoying, not money-on-fire).
- **v3 (few-shot):** add three **worked examples** spanning a calm low-priority
  question, an angry high-priority billing charge, and a store-down urgent. The
  examples pin the priority boundary and the "classify the problem, not the
  pleasantries" rule. This is where the regression suite earns its keep: run
  live against Nova Micro, v3 lands at **7/10**, not 10/10. The model
  over-escalates priority on an angry-but-not-on-fire ticket (the all-caps
  uploader bug 010 and the lost-parcel 007 both come back `urgent` instead of
  `high`), and it hallucinates a `billing` intent for the **empty** ticket 008
  instead of `other`. Tone is not priority, and an empty ticket is not a guess.
- **v3.1 (sharpened):** tighten the `urgent` vs `high` boundary in prose
  ("`urgent` is only for the WHOLE store down or sales failing across the
  board"), lift the empty-ticket rule to a **FIRST RULE** at the very top with
  the exact JSON to return, and add four more few-shot examples — the all-caps
  bug, the lost parcel, a how-to question, and the empty ticket — bringing the
  shipped `prompts/triage_prompt.md` to **seven** worked examples in total. Now
  the suite is a clean **10/10** — and it is the suite, not a vibe, that told us
  when we were done.

> **Few-shot is not free.** Those examples (~700 tokens here) ride along on
> *every* triage call. At a million tickets a month that is ~700M input tokens
> ≈ **$25/mo** just for the examples — a real, recurring cost you trade against
> accuracy. (Module 12 makes repeated prompt prefixes nearly free with prompt
> caching.)

## Step 3 — Govern the prompt in Bedrock Prompt Management

Versioning a prompt **in your app code** (an f-string in git) is not the same as
**governing** it. The exam — and production — want immutable versions you can
pin, approve, and audit. That is **Amazon Bedrock Prompt Management**.

```bash
uv run python setup.py
```

`setup.py` is idempotent and verbose. It:

1. reads `prompts/triage_prompt.md` (keeping git authoritative),
2. creates a parameterized prompt `relay-triage` with a `{{ticket}}` variable,
   bound to Nova Micro at **temperature 0**,
3. publishes **version 1** — an immutable snapshot,
4. records the prompt ID in `prompts/.prompt_id` so `relay/triage.py` finds it.

Re-run it any time; it reconciles the draft instead of duplicating, and keeps the
already-published version 1 untouched (that's what immutable means).

> Your **code pins an identifier + version number, never the prompt text.** A
> reviewer approves version 2 before anyone consumes it; CloudTrail records who
> created and read each version. That is governance, not a git diff.

## Step 4 — Triage a ticket (structured output + validation + 1 retry)

```bash
uv run python -m relay.triage data/tickets/ticket-001.json
```

Expected output:

```text
{"intent":"billing","priority":"high","sentiment":"negative"}

tokens: in=1305 out=19 | est. cost: $0.000048
```

`relay/triage.py` loads the template **by id + version 1** from Prompt
Management, renders `{{ticket}}`, calls **Converse** (Nova Micro, temperature 0),
and then — this is the load-bearing part — does **not trust the output**. It runs
`Triage.model_validate_json`. If validation fails, it **retries exactly once**,
feeding the Pydantic error back into the prompt so the model can correct itself.
If the second attempt still fails, it raises `TriageError` carrying the raw text
— **never a silent pass**.

This retry is a **validation retry**. Network retries, throttling backoff, and
cross-Region fallback belong to the FM integration layer — that's Module 3.

> **Why not tool calling for the JSON?** You can also get structured output via
> tool calling; Relay adopts that when it becomes an agent in Module 7. The
> Module 2 path is deliberately prompt-constrained + `model_validate_json` + one
> retry.

### Inference parameters — and why triage is temperature 0

| Parameter | What it controls | Triage (classify) | Drafting (a reply) | Classic trap |
|---|---|---|---|---|
| `temperature` | Randomness of the next-token distribution | **0** | ~0.7 | Thinking 0 = strict determinism (it isn't — see below) |
| `topP` (nucleus) | Smallest token set whose cumulative prob ≥ p | 1.0 (moot at temp 0) | ~0.9 | Tuning `topP` *and* `temperature` hard at once and fighting yourself |
| `topK` | Cap on the number of candidate tokens | unused for triage | small for tighter style | Assuming every model exposes it (Nova via Converse leans on temperature/topP) |
| `maxTokens` | Hard cap on output length | small (~100 — one JSON line) | larger for prose | Setting it so low the JSON gets truncated mid-object |

**Triage runs at temperature 0. Creativity in a classifier is a bug, not a
feature.** A reply you *draft* for a customer wants some warmth (~0.7); a label
you *assign* wants the single most-likely answer, every time.

> ⚠️ **Common misconception: "Temperature 0 makes the model deterministic, so I
> don't need output validation."** Wrong twice. Temperature 0 reduces variance
> but guarantees neither strict determinism (floating-point and routing still
> wobble) nor schema compliance (the model can still emit `"refund"`, prose, or a
> truncated brace). **Only programmatic validation guarantees the format** — which
> is exactly why `triage()` validates and retries even at temperature 0.

## Step 5 — Test prompts like code: the 10-ticket regression suite

```bash
uv run python run_prompt_tests.py
```

Expected output:

```text
ticket       expected               got                    result
-----------------------------------------------------------------------
ticket-001   billing/high           billing/high           PASS
ticket-002   technical/urgent       technical/urgent       PASS
ticket-003   account/low            account/low            PASS
ticket-004   shipping/low           shipping/low           PASS
ticket-005   other/low              other/low              PASS
ticket-006   billing/urgent         billing/urgent         PASS
ticket-007   shipping/high          shipping/high          PASS
ticket-008   other/low              other/low              PASS
ticket-009   billing/urgent         billing/urgent         PASS
ticket-010   technical/high         technical/high         PASS
-----------------------------------------------------------------------
score: 10/10
tokens: in=13092 out=220 | est. cost: $0.000489
```

The 10 reference tickets cover all **5 intents** plus the edge cases that break
naive prompts: an **empty** ticket (008), a **bilingual + furious**
French/English billing rant (009), and an **all-caps shouting** technical
complaint (010). The pass criteria are **objective**: valid JSON, expected
intent, expected priority. (Sentiment is the softest field, so the suite doesn't
assert it.) Re-run this whenever you change the prompt, the model, or the
parameters — a green 10/10 is the gate for shipping a new prompt version.

> No LLM-as-a-judge and no semantic metric here — that is Module 13. This is a
> deterministic, assertion-based regression suite, which is exactly what the
> exam's prompt-QA skill (1.6.4) means.

## Step 6 — Try it yourself

1. **Publish a version 2 and compare.** Add an eighth few-shot example to
   `prompts/triage_prompt.md` (e.g. a `shipping` delay that's only `normal`
   priority), re-run `uv run python setup.py` to publish version 2, point
   `_PROMPT_VERSION` in `relay/triage.py` at `"2"`, and run the suite again. Did
   10/10 hold? Did any token counts move? That before/after comparison is how you
   roll out a prompt change safely. (Tooled A/B testing is Module 13.)
2. **Break the temperature on purpose.** Set `_TEMPERATURE = 0.9` in
   `relay/triage.py` and run `run_prompt_tests.py` a few times. Watch the edge
   cases (008/009/010) start to wobble and the score dip below 10/10 — a live
   demonstration that a classifier wants temperature 0, and that validation
   catches the fallout when it doesn't.

## Step 7 — Teardown

```bash
uv run python teardown.py
```

It deletes the `relay-triage` prompt and all its versions (idempotent — safe to
run twice), removes the local `prompts/.prompt_id` pointer, and confirms Module 2
leaves nothing else billing. The $5 budget stays.

---

## Run the tests (no credentials needed)

```bash
uv run pytest          # offline by default — Stubber, no AWS, no network
RELAY_LIVE_TESTS=1 uv run pytest -m live   # ONE real sub-cent triage call (needs setup.py)
```

The offline suite validates the frozen schemas, the 10 ticket fixtures, and the
full triage flow (happy path, prose-stripping, the one-shot validation retry, and
the "raise, don't swallow" failure) with both Bedrock clients stubbed.
