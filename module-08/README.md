# Module 8 — Multi-Agent Systems and Bedrock AgentCore: Runtime, Memory, and HITL

**What:** At the end of Module 7, Relay **acts** — but it runs on your laptop, forgets
every conversation the moment it ends, and has a tool that could execute a refund with no
human in the loop. A customer writes *"this is the third time I'm asking — just refund
order #1042 already."* The Module 7 agent does not know it is the third time (it
remembers nothing), it could refund blindly, and it only runs while your terminal is
open.

Module 8 fixes all three. Relay is deployed as a managed service on **Bedrock AgentCore
Runtime** (microVM, sessions up to 8 h, idle free) with **AgentCore Memory** (short-term
session + long-term cross-session). It hands off billing/refund tickets to a **Billing
specialist**, and a refund is **gated behind human approval**: the action is proposed,
the ticket parks in `awaiting_approval`, and a human approves or rejects it.

```bash
uv run python -m relay.run "this is the third time I'm asking — just refund order 1042"
# --- Relay invocation ---
#   handed off: True -> the Billing specialist
#   gated     : True (refund awaiting human approval)
# --- agent actions (the trail across the handoff) ---
#   1. lookup_order({'order_id': '1042'}) [approved] -> {"status": "in_transit", ...}
#   2. refund({'order_id': '1042', 'amount_cents': 12900, ...}) [PROPOSED (awaiting approval)]
#   3. create_ticket({'ticket_id': '...', 'status': 'awaiting_approval', ...}) [approved]
# --- final answer ---
# I'm sorry for the trouble. I've submitted a refund of $129.00 for order 1042 for
# review; you'll get a confirmation shortly.
# [HITL] A refund is AWAITING APPROVAL — nothing was charged back yet.

uv run python -m relay.approve <ticket_id> --approve   # execute the refund -> answered
uv run python -m relay.approve <ticket_id> --reject    # abandon + escalate -> escalated
```

## What this module builds (on top of Module 7)

- **`relay/specialists.py` (NEW)** — the **Billing specialist**: a second **Strands**
  agent with its own refund-tone system prompt and a `refund` tool. The generalist
  **hands off** billing/refund tickets to it (supervisor/handoff topology). It shares the
  generalist's CloudCart tools and AgentAction journal.
- **`relay/agent.py` (MODIFIED, by addition)** — `handle_with_handoff(...)` routes a
  refund ticket to the specialist, and the **HITL gate** records a proposed `refund` as
  `AgentAction(approved=None)` and parks the `TicketRecord` in `awaiting_approval` — it is
  **not executed**.
- **`relay/approve.py` (NEW)** — the local/programmatic human decision: `approve(...)`
  sets `AgentAction.approved = True/False`, then **executes** the refund (→ `answered`) or
  **escalates** (→ `escalated`). The public approval endpoint + event bus are Module 11.
- **`relay/run.py` (NEW)** — Relay's invocation entrypoint, deployed on **AgentCore
  Runtime**. `run_relay(payload) -> response` is the frozen invoke contract (Module 11's
  worker reuses it). It wires the handoff, the gate, and **AgentCore Memory**.
- **`agentcore/` (NEW)** — the `agentcore`-CLI deploy config (`agentcore.yaml` + README).
- **`setup.py` / `teardown.py` (MODIFIED)** — setup creates the AgentCore **Memory**
  store; teardown **purges** it (the long-term store is the only idle-billed item).

## Frozen contracts USED here (no new schema — bible §3.1)

Module 8 adds **no field**. It makes the frozen-since-M7 `AgentAction.approved`
**effective** and exercises the frozen `TicketRecord` status `awaiting_approval`:

```python
class AgentAction(BaseModel):     # frozen M7 — exactly 4 fields, UNCHANGED at M8
    tool: str
    tool_input: dict
    result: str
    approved: bool | None = None  # None = proposed/awaiting · True = approved · False = rejected

# TicketRecord (frozen M7) statuses — awaiting_approval is exercised here for the first time:
#   received | triaged | awaiting_approval | answered | escalated | closed | failed
```

The **Billing specialist** name is canonical (06 §5.4) — no synonym.

## AgentCore (runtime) vs Strands (framework)

A common misconception: *"Bedrock AgentCore replaces Strands Agents."* It does not.
**AgentCore** is the managed **runtime** where the agent runs (microVM, sessions, memory,
identity); **Strands** is the **framework** that defines the reasoning loop, the tools,
and the handoff. They **compose**: you write the agent in Strands (`relay/agent.py`,
`relay/specialists.py`) and deploy it on AgentCore Runtime (`relay/run.py` +
`agentcore/`).

## Run it

```bash
export AWS_PROFILE=aws-genai-pro          # us-east-1 everywhere; no keys in code/.env
uv sync                                   # installs strands-agents, mcp, bedrock-agentcore
uv run python setup.py                    # tables + MCP Lambda + AgentCore Memory
# deploy on AgentCore Runtime with the agentcore CLI — see agentcore/README.md:
#   agentcore configure --config-file agentcore/agentcore.yaml && agentcore launch
uv run python -m relay.run "this is the third time I'm asking — just refund order 1042"
uv run python -m relay.approve <ticket_id> --approve
uv run pytest                             # 160 offline tests (no AWS calls)
RELAY_LIVE_TESTS=1 uv run pytest -m live  # opt-in, capped (~$0.06, see lab.md)
uv run python teardown.py                 # purges AgentCore Memory; then: agentcore destroy
```

## Boundaries (what this module does NOT do)

- No input/output **guardrails**, injection defense, or grounding — Module 9.
- No **public API**, no `POST /tickets/{id}/approve` endpoint, no `relay-events`
  EventBridge bus — Module 11. Here the approval is local/programmatic (`relay/approve.py`).
- No **Bedrock Agents "classic"** in code (theory only); no AgentCore **Observability**
  (Module 14); no agent-trajectory **eval** (Module 13); no preview AgentCore components
  (Agent Registry / Payments) — GA only (Runtime / Memory).

See `lab.md` for the full step-by-step, the measured cost, and "Try it yourself".
