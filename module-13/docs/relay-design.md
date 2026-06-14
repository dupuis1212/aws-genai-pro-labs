# Relay — Design Doc (v0.3, Module 3)

> A living design doc. It states what Relay IS today, the target it is being built
> toward, and the decisions (with dates and reasons) that got us here. Each module
> updates it. **As of Module 3, only the triage path and the FM integration layer
> exist** — everything past the dashed line below is planned, not built. Nothing in
> this doc describes a component as "done" before its module ships it.
>
> Last updated: 13 June 2026 (Module 3). Owner: the course. Region: us-east-1.

## 1. Context

CloudCart is a SaaS e-commerce company. Support tickets arrive by email and chat,
sometimes with screenshots. **Relay** is the GenAI agent that triages each ticket,
answers from CloudCart's own documentation with citations, takes safe actions in
business systems, and escalates to a human when needed — deployed serverless,
guarded, evaluated, observed, and cost-optimized on AWS.

On a sale Monday, CloudCart sees ~3,000 tickets/hour. The Module 2 triage worked
in a demo but called one model directly, with no network retry, no streaming, and
no way to route a trivial "where is my order?" away from the same model that
handles a billing dispute. Module 3 fixes the foundation: a single FM integration
layer that every future part of Relay calls.

## 2. Target architecture (final state, Module 15)

```mermaid
graph TD
    subgraph Built by Module 3
        T["triage.py / future callers"] --> C["llm.converse()"]
        C --> R{"router (tier=auto)"}
        R -->|fast| PF["us.amazon.nova-micro-v1:0"]
        R -->|smart| PS["us.amazon.nova-2-lite-v1:0"]
        C --> RB["retry: exponential backoff + jitter"]
        RB --> FB["fallback: alternate cross-Region profile, then degrade tier"]
        PF -.->|IDs live ONLY in| CFG["relay/config.py"]
        PS -.->|IDs live ONLY in| CFG
    end

    subgraph Planned (later modules — NOT built yet)
        API["API Gateway + Lambda + SQS (M11)"]
        INTAKE["intake.py: validate / normalize / vision / PII (M6, M10)"]
        AGENT["Strands agent + tools + MCP (M7), AgentCore Runtime (M8)"]
        KB["Knowledge Base relay-kb (M5)"]
        GUARD["Guardrails (M9)"]
        OBS["Observability: dashboard, alarms, X-Ray (M14)"]
    end

    API -.-> INTAKE -.-> AGENT
    AGENT -.-> C
    AGENT -.-> KB
    GUARD -.-> C
    OBS -.-> C
```

The diagram's top box is real today. Everything in the bottom box is a forward
reference — drawn so the integration layer's place is clear, not because it
exists. Each component arrives in its own module and calls `llm.converse()`; none
of them ever names a model ID.

## 3. The FM integration layer (this module)

Relay has exactly one Bedrock generation call site: `relay/llm.py`'s

```python
converse(messages, *, tier="auto", stream=False, **params)
```

This signature is **frozen** from Module 3 through Module 15. Later modules grow
its body by addition (image content blocks, a guardrail parameter, prompt-caching
and Flex selection through `**params`) but never change the signature. Consumers
depend on exactly this shape.

What the layer does:

- **Routing.** `tier="auto"` runs a small, explainable complexity router that
  reads the request and picks `fast` or `smart`. Explicit `fast` / `smart` /
  `frontier` skip the router. This is routing *by content/complexity* — classify,
  then send once. It is **not** model cascading (always try small, escalate on a
  bad result); cascading doubles latency on hard cases, and Relay does not do it.
- **Streaming.** `stream=True` uses **ConverseStream** and yields text deltas as
  they arrive — for long answers read by a human (time-to-first-token). `stream=
  False` returns the whole reply — for parsers like triage, where streaming buys
  nothing.
- **Resilience.** Throttling and 5xx are retried with **exponential backoff +
  jitter**, never an immediate loop. When a profile is throttled past its retries,
  the call falls back to the tier's alternate cross-Region profile, then degrades
  the tier (`smart -> fast`) as a last resort. No silent `try/except`.

## 4. Contracts (stable, reproduced field-for-field from the Relay spec)

| Contract | Definition (frozen) |
|---|---|
| `converse` signature | `converse(messages, *, tier="auto", stream=False, **params)` — the UNIQUE Bedrock call site |
| Tiers | `"fast"` (triage, simple) · `"smart"` (complex answers, agent) · `"auto"` (the complexity router) · `"frontier"` (reference / Try-it only) |
| Model-ID containment | tier -> inference profile map lives in `relay/config.py` ONLY; no `us.`/`global.` ID anywhere else in `relay/` |
| `Ticket` (M2) | `{ticket_id, channel: "email"\|"chat", customer_message, created_at}` — exactly 4 fields |
| `Triage` (M2) | `{intent: 5 values, priority: 4 values, sentiment: 3 values}` — complete |

`Ticket` and `Triage` are unchanged by Module 3. Triage was only refactored to
call `converse(tier="fast")` and drop its provisional model constant.

## 5. Tier selection — decided with evidence

| Tier | Inference profile | Role in Relay | Why |
|---|---|---|---|
| `fast` | `us.amazon.nova-micro-v1:0` | triage, the router's floor, tests | Cheapest capable model; ~80% of CloudCart tickets are fast-tier work |
| `smart` | `us.amazon.nova-2-lite-v1:0` | complex answers, agent reasoning | Reasoning + large context for disputes and multi-step questions |
| `frontier` | `us.anthropic.claude-sonnet-4-5-20250929-v1:0` | reference / Try-it-yourself only | A ceiling to measure the cost delta against; not production traffic |

Prices "as of June 2026" — re-verify on the Bedrock pricing page. The cost line in
`demo_llm.py` is computed from the API usage block, never guessed.

## 6. Decisions (dated, with reasons)

- **2026-06-13 — Two production tiers, fast + smart.** Routing is a cost decision
  before it is an architecture decision. Most tickets are simple; paying a smart
  model for "where is my order?" is waste. A third `frontier` tier exists only as
  a reference point, never as default traffic.
- **2026-06-13 — Converse / ConverseStream everywhere; never the legacy
  single-prompt invoke path.** One API across all text models gives one message
  shape, one streaming path, and one place to add tool calling and guardrails
  later. The single exception in the whole course is Titan embeddings (Module 4),
  which has no Converse path and returns a vector, not text.
- **2026-06-13 — Inference profiles, never bare regional IDs.** A recent model
  invoked on-demand with a bare regional ID fails ("Retry with an inference
  profile"). The `us.`/`global.` prefix is the **cross-Region inference** profile;
  it routes across the Regions in its geography for capacity. That is the NOMINAL
  mode, not a disaster-recovery toggle.
- **2026-06-13 — us-east-1 for the whole course.** One Region keeps IAM, quotas,
  and pricing reasoning simple. Cross-Region behavior is handled by the inference
  profiles, not by a multi-Region application architecture.
- **2026-06-13 — The layer owns retries, not the SDK.** botocore's adaptive retry
  is disabled so backoff is explicit and teachable. Retries are capped at 2 beyond
  the first attempt; past that, degrade or fail — never loop.

## 7. Costs

Module 3's lab cost is dominated by a handful of fast-tier calls plus a few smart
and one frontier call in the Try-it-yourself. Expected total: **under $1**. No
idle-billed resource is created — `setup.py` only verifies access and (re)creates
the Module 2 Prompt Management prompt, which is not billed for storage. The
optional AppConfig "Try it yourself" uses freeform configuration (cents at most)
and is removed at teardown.

## 8. Open questions / forward references

- Routing **by metrics** (pick a tier from live latency/error telemetry) is named
  but not built here — the router reads the request, not a metrics feed.
- **Circuit breakers** (e.g. via Step Functions) are theory in this module.
- A managed config switch (**AWS AppConfig**) can override the tier map without a
  redeploy; shown as a pattern and a Try-it-yourself, not wired into the layer.
- The model **customization lifecycle** (LoRA, SageMaker Model Registry, rollback)
  is explained but deliberately not built — Relay reaches for RAG first (Module 4).
