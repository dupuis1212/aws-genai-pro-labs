# Module 7 — Agentic AI on AWS: Strands Agents, Tool Calling, and MCP

**What:** Until now Relay could read and answer — triage (M2), cited answers from the
Knowledge Base (M5), multimodal intake (M6) — but it could not **act**. A customer asks
*"where is my order #1042?"* and a doc-only system can only paraphrase the shipping
policy; the real answer is in the **order book**, not the docs.

Module 7 makes Relay an **agent**: a **Strands** agent whose foundation model runs the
**ReAct loop** (reason → act → observe) and **decides which tool to call** — `search_kb`
for a how-to, `lookup_order` for a real order status, `create_ticket` to record the
outcome — without us scripting each path. The two business tools are served by a
stateless **CloudCart MCP server on AWS Lambda**; `search_kb` stays a local tool.

```bash
uv run python -m relay.agent "Where is order 1042? It was supposed to arrive Monday."
# --- agent actions (the ReAct loop) ---
#   1. lookup_order({'order_id': '1042'}) -> {"order_id": "1042", "status": "in_transit", ...}
#   2. create_ticket({'ticket_id': 'ticket-…', 'status': 'answered', ...}) -> stored …
# --- final answer ---
# Order 1042 is in transit with CloudCart Express and is now estimated to arrive
# 2026-06-15 (a carrier delay pushed it past the original Monday estimate).
# --- TicketRecord persisted to relay-tickets ---
# { "ticket_id": "...", "status": "answered", "actions": [ ... ], "cost_cents": 0.0, ... }

uv run python -m relay.agent "How do refunds work?"   # -> the agent chooses search_kb
```

This module **freezes** two schemas, by **addition only** (06 §2 / bible §3.1),
reproduced field-for-field:

```python
class AgentAction(BaseModel):     # frozen M7 — exactly 4 fields
    tool: str
    tool_input: dict
    result: str
    approved: bool | None = None  # EFFECTIVE only at M8 — ALWAYS None at M7

class TicketRecord(BaseModel):    # frozen M7
    ticket_id: str
    status: Literal["received", "triaged", "awaiting_approval",
                    "answered", "escalated", "closed", "failed"]   # full enum frozen now
    triage: Triage | None
    answer: Answer | None
    actions: list[AgentAction]
    escalated: bool
    cost_cents: float             # 0.0 placeholder at M7 -> really populated at M12
    updated_at: str
```

The full 7-status enum is present from the moment it is frozen, even though **M7 only
ever writes four** (`received | triaged | answered | failed`). `approved` is `None`
everywhere. `cost_cents` is a `0.0` placeholder. There is **no `feedback_rating`** (that
is Module 13).

## Guarded execution (skill 2.1.3)

> An agent without a tool-call budget is a billing incident waiting to happen.

Three layers of guardrail, demonstrated in the lab:

1. **Stop condition** — `limits={"turns": MAX_ITERATIONS}` caps the ReAct loop. A model
   stuck on a failing tool stops at N turns (`stop_reason == "limit_turns"`) instead of
   looping forever and burning tokens. The record is recorded as `failed`, not a crash.
2. **Timeout** — a wall-clock ceiling on the run (on the bedrock-runtime client), so a
   hung call cannot pin the process.
3. **IAM resource boundaries** — the MCP Lambda's execution role can **read only
   `relay-orders`** and **write only `relay-tickets`** (explicit table ARNs, no `*`). A
   write anywhere else is denied by IAM, not just by convention.

## MCP: one protocol for your tools (skill 2.1.7)

The two business tools are served by a **stateless MCP server** (`mcp_server/`,
`FastMCP`) deployed to **AWS Lambda** behind a Function URL. The agent is the MCP
**client** (`relay/tools.py`): it connects, **discovers** the server's tools at runtime,
and hands them to the Strands agent — add `get_shipping_policy` on the server and the
agent picks it up with no client change. `search_kb` stays a **local** Strands tool over
the Knowledge Base (the 1.5.6 retrieval-as-a-tool pattern). Lambda is for the **stateless,
lightweight** MCP server; a heavy/stateful tool server would go on **ECS** (theory).

## How to run

(region us-east-1, profile `AWS_PROFILE=aws-genai-pro`; no AWS key in code or `.env`)

```bash
uv sync   # adds strands-agents~=1.43 and mcp~=1.27 (re-verified on PyPI, MCP spec 2025-06-18)

# 0. Stand up the agent's tables (relay-orders seeded 25, relay-tickets), the IAM-bounded
#    MCP Lambda + Function URL, on top of the inherited Knowledge Base (search_kb backend).
uv run python setup.py

# 1. An order-status ticket -> the agent calls lookup_order, answers, writes a TicketRecord:
uv run python -m relay.agent "Where is order 1042? It was supposed to arrive Monday."

# 2. A documentation question -> the agent chooses search_kb instead of lookup_order:
uv run python -m relay.agent "How do refunds work?"

# 3. Run the MCP server LOCALLY (no Lambda) and point the agent at it:
uv run python -m mcp_server                                  # http://127.0.0.1:8000/mcp
RELAY_MCP_URL=http://127.0.0.1:8000/mcp uv run python -m relay.agent "Where is order 1042?"

# 4. Offline tests (no credentials, no network) — the frozen AgentAction/TicketRecord
#    contract, the store on moto DynamoDB, the tools, the FULL agent ReAct loop driven by
#    a scripted model, the stop-condition guardrail, and the IAM boundary. Cumulative M2–M7.
uv run pytest

# 5. Up to seven sub-cent real calls (budgeted), incl. ONE capped live agent run:
RELAY_LIVE_TESTS=1 uv run pytest -m live

# 6. Remove the MCP Lambda + role; KEEP the (on-demand, ~$0) tables + KB that Module 8 reuses:
uv run python teardown.py
uv run python teardown.py --delete-tables   # also drop relay-orders + relay-tickets
```

Full step-by-step walkthrough — the ReAct loop, the Strands-vs-Step-Functions-vs-Bedrock
Agents choice, anatomy of a reliable tool, the MCP topology, and the two guardrail demos
(a runaway loop cut by the stop condition; a write outside `relay-tickets` refused by IAM)
— is in [`lab.md`](lab.md).

## Files (NEW or MODIFIED in Module 7)

- `relay/agent.py` — **NEW.** Relay as a Strands ReAct agent: `build_agent(...)` (the
  SMART-tier model from `relay.config`, the Relay system prompt, the tools, the
  `AgentAction` journal hook — injectable so tests drive it with a scripted model) and
  `handle(...)` (run under a max-iterations stop condition + timeout, journal every tool
  call, persist a `TicketRecord` to `relay-tickets`). CLI:
  `python -m relay.agent "<customer message>"`.
- `relay/tools.py` — **NEW.** `search_kb` (a local Strands `@tool` over the KB
  `Retrieve`, the 1.5.6 pattern) plus the MCP-client wiring that discovers
  `lookup_order` / `create_ticket` from the CloudCart MCP server. Validates parameters
  and returns clean, model-facing errors — no silent `try/except`.
- `mcp_server/` — **NEW.** The stateless CloudCart MCP server: `store.py` (DynamoDB data
  access for the two tools), `server.py` (`FastMCP` wrapping them as MCP tools, stateless
  streamable-HTTP), `app.py` (the Lambda Function-URL → ASGI adapter), `__main__.py`
  (local dev server). No model ID, no Bedrock call — pure business I/O.
- `data/orders.json` — **NEW.** 25 seeded CloudCart orders (including order **1042**, the
  brief's demo) with real statuses (`in_transit`, `delivered`, `processing`, …).
- `relay/models.py` — **MODIFIED (additive).** Adds `AgentAction` and `TicketRecord`. The
  M2–M6 schemas are untouched; no `feedback_rating` (M13), no field re-typed.
- `relay/config.py` — **MODIFIED (additive).** Appends the table names
  (`relay-orders` / `relay-tickets` + their keys) and `resolve_mcp_url(...)`. The M3
  tier map and the M4 embedder are untouched (the agent runs on the existing `smart`
  tier — no new model).
- `relay/__init__.py` — **MODIFIED (additive).** Tracks the new `tools` and `agent`
  submodules.
- `setup.py` / `teardown.py` — **MODIFIED (additive).** setup creates + seeds the tables,
  the IAM-bounded MCP Lambda role, the Lambda, and the Function URL (records `.mcp_url`),
  on top of the inherited KB; teardown removes the Lambda + role and keeps the on-demand
  tables + KB (Module 8 reuses them), with `--delete-tables` / `--delete-kb` opt-outs.
- `relay/llm.py`, `relay/kb.py`, `relay/intake.py`, `relay/triage.py`, `ingest/`,
  `prompts/`, `data/docs/`, `data/raw/`, `data/tickets/`, `compare_chunking.py`,
  `compare_retrieval.py`, `freshness_test.py` — **inherited from Module 6, byte-identical.**
- `tests/smoke_test.py` — offline by default (cumulative Modules 2–7); live calls opt-in
  (`RELAY_LIVE_TESTS=1`) with a documented budget (≤7 calls; one is a capped live agent run).
