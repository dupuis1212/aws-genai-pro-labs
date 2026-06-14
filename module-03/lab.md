# Module 3 lab — Relay's FM integration layer: routing, streaming, and resilience

> **This lab cost me $0.01 on June 2026 prices.** A clean run-through as written —
> `setup.py` (run twice, four tiny Converse pings), a few `demo_llm.py` calls (one
> **smart-tier** Amazon Nova 2 Lite streamed answer plus a couple of **fast-tier**
> Nova Micro calls), the re-run 10-ticket triage suite, the two live tests, and
> teardown — totalled **$0.0065** measured from the Converse `usage` blocks. Almost
> all of that is the single optional **frontier** Claude Sonnet 4.5 call in the
> Try-it-yourself ($0.0049); skip that and the whole lab is under a fifth of a cent.
> The fast-tier (Nova Micro) traffic — including the 13k-token triage suite — costs
> about $0.0005. Every figure is computed from the Converse `usage` block, never
> guessed. Prices are as of June 2026 — re-verify on the
> [Bedrock pricing page](https://aws.amazon.com/bedrock/pricing/).
>
> **Teardown reminder:** run `uv run python teardown.py` when you're done. Module 3
> creates **no new idle-billed resource** — it removes the inherited `relay-triage`
> prompt and the optional AppConfig app (if you did Try-it-yourself), then confirms
> the account is clean. The M1 $5 budget stays — it backstops the course.

**Goal:** write Relay's **design doc**, then build and **freeze** `relay/llm.py` —
one `converse()` that routes fast/smart, streams, retries with backoff, and falls
back cross-Region — and refactor Module 2's triage to call it. This is the LLM
contract every later module consumes.

Region for the whole course: **us-east-1**. Profile: `AWS_PROFILE=aws-genai-pro`
(any profile that resolves to your course account works). No AWS key in code or
`.env` — credentials come from the profile.

---

## Step 1 — Carry the cumulative state forward

Module 3 starts from Module 2's `relay/` package byte-for-byte (`models.py`,
`triage.py`, `prompts/`, `data/tickets/`, `run_prompt_tests.py`) and adds the FM
integration layer on top.

```bash
uv sync
```

`relay/models.py` is untouched — `Ticket` (4 fields) and `Triage` (3 enums) are
LAW and stay identical. Only `relay/triage.py` changes in this module, and only to
route through the new layer.

## Step 2 — Write the design doc first

Before code, fill in `docs/relay-design.md` from the template provided. It is a
graded artifact: context, target architecture (a Mermaid diagram of the LLM
layer), the contracts, the dated decisions (why two tiers, why Converse
everywhere, why us-east-1), tier selection with evidence, and costs. The rule:
**nothing past Module 3 is described as built** — the downstream components are
drawn as forward references, not as done.

Writing the doc first is the point. It is where you decide that triage is fast-tier
work, that a billing dispute is smart-tier, and that a `frontier` tier exists only
as a reference. The code then implements a decision you already made on paper.

## Step 3 — `relay/config.py`: the only place a model ID lives

`relay/config.py` is the **sole home of model-ID literals** in the whole repo. It
maps a Relay *tier* to an **inference profile**:

```python
TIERS = {
    "fast":     "us.amazon.nova-micro-v1:0",   # triage, router floor, tests
    "smart":    "us.amazon.nova-2-lite-v1:0",  # complex answers, agent reasoning
    "frontier": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",  # reference only
}
```

Every ID is a `us.` (or `global.`) **inference profile**, never a bare regional
ID. That prefix IS the **cross-Region inference** profile: it routes across the
Regions in its geography for capacity. A bare regional ID fails on-demand with
"Retry with an inference profile" — that is the nominal mode of these models, not
a disaster-recovery toggle. The grep gate proves nothing leaks out of this file:

```bash
grep -rE '(us|global|eu)\.(amazon|anthropic)\.' relay/ | grep -v config.py
# -> empty
```

## Step 4 — `relay/llm.py`: the frozen `converse()` contract

This is the load-bearing file of the course. The signature is **frozen** here and
is byte-identical through Module 15:

```python
converse(messages, *, tier="auto", stream=False, **params)
```

It is the **unique Bedrock call site** for all Relay code. Three jobs:

- **Routing.** `tier="auto"` runs the complexity router (Step 5). `fast`/`smart`/
  `frontier` force a tier.
- **Streaming.** `stream=True` uses **ConverseStream** and yields text deltas;
  `stream=False` returns the whole reply.
- **Resilience.** Throttling and 5xx get **exponential backoff + jitter**, then a
  cross-Region profile fallback, then a tier degrade (`smart -> fast`). No silent
  `try/except` — a non-retryable error surfaces immediately as `LLMError`.

```python
# Retries are capped at 2 beyond the first attempt. Past that you are burning
# latency on a loop that is not getting better — degrade or fail, never spin.
_MAX_RETRIES = 2
_BACKOFF_BASE, _BACKOFF_CAP = 0.5, 8.0   # sleep = min(CAP, BASE * 2**attempt) + jitter
```

Note we disable botocore's own retries (`max_attempts=1`) so the backoff is
explicit and teachable, not delegated invisibly to the SDK.

## Step 5 — The complexity router (`tier="auto"`)

The router is deliberately simple and **explainable**: a keyword/length heuristic,
floored at `fast` (~80% of CloudCart tickets), escalating to `smart` only on a
reasoning signal. It returns *why*, so you can see the decision.

```python
route([{"role": "user", "content": [{"text": "hi"}]}])
# RouteDecision(tier='fast', reason='no complexity signal — default fast tier')

route([{"role": "user", "content": [{"text": "Why was I charged twice for order #1042?"}]}])
# RouteDecision(tier='smart', reason="matched complexity keyword 'charged twice'")
```

This is **routing by content/complexity** — classify, then send once. It is *not*
**model cascading** (always try the small model, escalate on a bad result — a
second call on failure). The exam loves that distinction: routing picks once;
cascading retries bigger. Relay routes. Routing **by metrics** (pick a tier from
live latency/error telemetry) is named but not built here.

## Step 6 — Refactor triage to go through the layer

`relay/triage.py` drops its provisional model constant and calls
`converse(tier="fast")`. The validation flow is unchanged (one validation retry,
no silent pass); it just inherits the layer's backoff and fallback for free. Prove
non-regression — the Module 2 suite must still pass **10/10**:

```bash
uv run python setup.py            # pings fast+smart, ensures the relay-triage prompt
uv run python run_prompt_tests.py # the inherited 10-ticket regression suite
```

Expected tail:

```text
-----------------------------------------------------------------------
score: 10/10
tokens: in=13092 out=220 | est. cost: $0.000489
```

Same 10/10 as Module 2 — refactoring the call path changed nothing the customer
sees, which is exactly the point of a non-regression gate.

## Step 7 — See it work: `demo_llm.py`

```bash
uv run python demo_llm.py "Why was I charged twice for order #1042?"
```

Expected shape:

```text
router: tier=smart, reason=matched complexity keyword 'charged twice'

response (streaming):

I'm sorry for the double charge. Here's what I'd check on order #1042: ...
[streams token by token]

tokens: in=78 out=126 | tier=smart | est. cost: $0.000339
```

And the fast path, no streaming:

```bash
uv run python demo_llm.py "hi" --no-stream
```

```text
router: tier=fast, reason=no complexity signal — default fast tier

response:

Hi! How can I help with your CloudCart store today?

tokens: in=41 out=18 | tier=fast | est. cost: $0.000004
```

> **When does streaming matter?** For a long answer in front of a human, streaming
> cuts perceived latency by showing the first tokens immediately
> (time-to-first-token). For a parser like triage — a one-line JSON object you
> validate — streaming buys nothing; you need the whole object before you can act.
> That is why triage runs `stream=False` and the customer-facing demo streams.

## Step 8 — Prove the resilience path (offline)

You do not need a real outage to see the backoff. The offline test stubs a
`ThrottlingException` and asserts the layer retries with backoff instead of
propagating:

```bash
uv run python -m pytest tests/ -k retry
```

The suite also proves graceful degradation: a `smart` call throttled on **both**
its primary and `global.` profiles falls back to `fast` and still answers.

## Step 9 — Try it yourself

1. **Switch the model without a redeploy — AWS AppConfig.** Deport the tier map
   into an **AWS AppConfig** freeform configuration (`relay-model-config`
   application) and have `config.py` load it at startup, falling back to the
   in-code map. Change the smart tier in AppConfig, restart, and watch the new
   model take over — no code change, no redeploy. This is skill 1.2.2's whole
   point: model selection is configuration, not code. (AppConfig freeform config
   is free/cents; `teardown.py` deletes the `relay-model-config` app for you.)
2. **Add a `frontier` tier and measure the delta.** The `frontier` tier
   (`us.anthropic.claude-sonnet-4-5-20250929-v1:0`) is already in `config.py`. Run
   the same three tricky tickets through `--tier smart` and `--tier frontier` and
   compare the cost line. Is the frontier answer enough better to justify ~10x the
   price? For CloudCart triage, almost never — which is why it is a reference
   tier, not production traffic.

## Step 10 — Teardown

```bash
uv run python teardown.py
```

Idempotent. It deletes the `relay-triage` prompt and all versions, removes the
optional `relay-model-config` AppConfig app (a no-op if you skipped Try-it-
yourself), and confirms Module 3 left nothing idle-billed. The $5 budget stays.

---

## Run the tests (no credentials needed)

```bash
uv run pytest          # offline by default — Stubber, no AWS, no network
RELAY_LIVE_TESTS=1 uv run pytest -m live   # TWO real sub-cent calls (fast stream + smart)
```

The offline suite guards the **frozen `converse()` signature**, the canonical
tiers, the model-ID containment grep gate, the deterministic router, the
backoff-on-throttling and degrade-to-fast paths, the streaming delta path, and the
full Module 2 triage flow now routed through `converse(tier="fast")`. The two live
tests make exactly one fast ConverseStream and one smart Converse — together well
under a tenth of a cent.
