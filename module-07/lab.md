# Module 7 lab — agentic AI on AWS: Strands agents, tool calling, and MCP

> **This lab cost me about $0.02 on June 2026 prices** (well under the syllabus budget
> of < $2 for Module 7). Every token figure below is read from the API response, never
> guessed — a single instrumented agent run measured ~3,600 in / ~150 out tokens across
> 3 smart-tier model calls (lookup_order → create_ticket → final answer) = ~$0.0013. The spend is a handful of **Amazon Nova 2 Lite**
> (smart-tier) agent runs — each a short ReAct loop of a few model calls — plus the
> inherited KB/embedding calls. DynamoDB on-demand, the MCP Lambda, and the Function URL
> bill **~$0 idle**. Measured breakdown:
>
> | Item | Real usage observed | Cost |
> |---|---|---|
> | agent runs (smart tier, ~5 runs × 3 model calls) | ~3,600 in / ~180 out per run | ~$0.0077 |
> | live smoke KB RetrieveAndGenerate (1) | ~1,500 in / ~200 out | $0.00095 |
> | live smoke `converse` (fast + smart) | ≤64 out each | $0.00019 |
> | live smoke Nova Lite vision (1) | ~808 in / ~38 out | $0.00006 |
> | KB ingestion (inherited, Titan, 3 re-syncs × 7 docs) | small docs | $0.0003 |
> | live smoke + standalone Titan embeddings | ~few hundred tokens | ~$0 |
> | DynamoDB on-demand (orders seed ×3 + ticket writes) | a few hundred tiny ops | <$0.001 |
> | Lambda (MCP server, deploys + a few invocations) + Function URL | cold + warm starts | <$0.001 |
> | **Total (measured)** | | **≈ $0.01 → $0.02** |
>
> The biggest line is the agent itself: a ReAct loop is **several smart-tier model calls per
> ticket** (that is the cost the stop condition bounds). Nova 2 Lite is ~$0.30 in / ~$2.50
> out per million tokens (AS OF JUNE 2026 — re-verify on the
> [Bedrock pricing page](https://aws.amazon.com/bedrock/pricing/)).
>
> - **No idle billing.** DynamoDB `relay-orders` / `relay-tickets` are **on-demand**
>   (PAY_PER_REQUEST → ~$0 idle); the MCP **Lambda** and its **Function URL** bill only per
>   invocation; the inherited KB + S3 Vectors are ~$0 idle. The agent runs **locally** — no
>   always-on runtime (that is Module 8).
>
> **Teardown reminder:** run `uv run python teardown.py` when you're done. It **deletes the
> MCP server Lambda + its Function URL + its IAM role** and **keeps** the on-demand tables and
> the Knowledge Base — Module 8 reuses them, and they bill ~$0 idle. Add `--delete-tables`
> for a clean slate (you will re-seed on the next setup). The M1 $5 budget stays; it
> backstops the course.

**Goal:** turn Relay into a **Strands** agent with the tools `search_kb`, `lookup_order`,
and `create_ticket` — the two business tools served by a **stateless CloudCart MCP server
on AWS Lambda** — under **stop conditions, a timeout, and an IAM-bounded Lambda role**.
Because a system that can only generate text cannot resolve a ticket; it has to read
business systems and write to them.

Region for the whole course: **us-east-1**. Profile: `AWS_PROFILE=aws-genai-pro`. No AWS
key in code or `.env`.

---

## Step 1 — Carry the cumulative state forward

Module 7 starts from Module 6's `relay/` package byte-for-byte (`models.py`, `config.py`,
`llm.py`, `triage.py`, `kb.py`, `intake.py`) plus the inherited `ingest/` pipeline, the
data bucket, and the **Module 5 Knowledge Base `relay-kb`** (the `search_kb` tool retrieves
from it — if it is gone, run Module 5's setup).

```bash
uv sync   # adds strands-agents~=1.43 and mcp~=1.27 (re-verified on PyPI; MCP spec 2025-06-18)
aws sts get-caller-identity   # the account the bucket name is suffixed with
```

> **Pins (re-verified on generation day, 13 Jun 2026):** `strands-agents` latest **1.43.0**
> → pinned `~=1.43`; `mcp` latest **1.27.2** → pinned `~=1.27` (Model Context Protocol spec
> revision **2025-06-18**). Strands moves fast — re-verify these on PyPI when you run the
> lab; `uv.lock` is committed for reproducibility.

---

## Step 2 — Stand up the agent's resources

`setup.py` is idempotent and verbose. On top of the inherited KB it creates:

- **DynamoDB `relay-orders`** (on-demand) — the CloudCart order book, **seeded with the 25
  orders** in `data/orders.json` (including order **1042**, the demo order);
- **DynamoDB `relay-tickets`** (on-demand) — where the agent persists a `TicketRecord`;
- the **IAM-bounded MCP Lambda role** `relay-mcp-lambda-role` (read `relay-orders`, write
  `relay-tickets`, own logs — nothing else);
- the **CloudCart MCP server** packaged from `mcp_server/` and deployed to the Lambda
  `relay-mcp-server`, fronted by a **Function URL** (recorded in `.mcp_url`).

```bash
uv run python setup.py
# ... orders table CREATED ... relay-orders SEEDED 25 orders ...
# ... IAM role 'relay-mcp-lambda-role' CREATED (read orders, write tickets ONLY) ...
# ... Lambda 'relay-mcp-server' CREATED ... Function URL CREATED ... recorded -> .mcp_url
```

> The MCP Lambda bundles only what the **server** needs (`mcp`, `pydantic`, `starlette`,
> `relay.config`/`relay.models`) — **not** `strands-agents` (that is the client side).
> `setup.py` already resolves those deps **for the Lambda target** — it runs
> `uv pip install --python-platform x86_64-manylinux2014 --python-version 3.12` into a temp
> dir before zipping, so the compiled `pydantic_core` wheel matches the Lambda runtime no
> matter what host OS you build on (macOS/arm64 included). You do **not** need to hand-build
> the zip or use a container. Re-run `setup.py` (or `setup.py --skip-kb`) to redeploy the
> function code after a change.

---

## Step 3 — From completion to agent: the reasoning–action loop

`relay/agent.py` builds a **Strands agent**. The foundation model (the inherited **`smart`
tier**, Nova 2 Lite — no new model) runs the **ReAct loop**: it reasons, decides to call a
tool, observes the result, and repeats until it can answer. We do **not** script the path.

```bash
uv run python -m relay.agent "Where is order 1042? It was supposed to arrive Monday."
```

The agent reasons that an **order-status** question needs the **order book**, calls
`lookup_order(order_id="1042")` (served over MCP by the Lambda), reads back the real
status (`in_transit`, ETA `2026-06-15`), calls `create_ticket(...)` to record the outcome,
and answers — citing the real status, not a doc generality. Every tool call is journaled as
an **`AgentAction`** and the whole thing is persisted as a **`TicketRecord`** in
`relay-tickets`, with its `actions[]` log.

A **documentation** question takes a different path with no code change — the model picks
the doc tool:

```bash
uv run python -m relay.agent "How do refunds work?"
# --- agent actions ---  1. search_kb({'query': 'how do refunds work'}) -> [1] ...
```

---

## Step 4 — Reliable tools: definitions, validation, guarded execution

**A tool's docstring is its spec; its type hints are its schema.** Strands' `@tool` (and
FastMCP's `@mcp.tool()`) turn `lookup_order(order_id: str) -> str` + the docstring into the
exact tool definition the model sees — one source of truth, no hand-written JSON schema.

**Validate parameters; return clean errors to the model.** A blank/missing order id, or an
unknown order, returns a short **model-facing message** (`"No order '9999' found ... it may
be mistyped"`), not a stack trace and never a silent empty result — so the model can
recover (ask the customer, try another tool). No silent `try/except`. (See
`mcp_server/store.py` and `relay/tools.py`.)

**`AgentAction` (frozen schema) is the action journal.** Each tool call records
`{tool, tool_input, result, approved}` — and `approved` is **`None`** everywhere at Module
7 (the human-approval flow is Module 8).

**Guarded execution — three layers (skill 2.1.3):**

1. **Stop condition** — `limits={"turns": MAX_ITERATIONS}` (`MAX_ITERATIONS = 6`).
2. **Timeout** — a wall-clock ceiling on the bedrock-runtime client.
3. **IAM resource boundaries** — the MCP Lambda role reads only `relay-orders`, writes only
   `relay-tickets`. Full least-privilege per component is Module 10.

> **An agent without a tool-call budget is a billing incident waiting to happen.** A ReAct
> loop is several **billed** model calls per ticket; the stop condition is what makes that
> cost bounded.

---

## Step 5 — MCP: one protocol for your tools

Without a protocol, N agents × M tools means bespoke glue everywhere. The **Model Context
Protocol (MCP)** standardizes the wire: a client connects to a server and **discovers** its
tools at runtime. The two business tools live on a **stateless MCP server** (`mcp_server/`,
`FastMCP`, `stateless_http=True`) deployed to **AWS Lambda** — exactly the **lightweight,
stateless** case Lambda fits. A heavy/stateful tool server (a warm model, a big cache) would
go on **ECS** (theory, not built here).

Relay is the **MCP client** (`relay/tools.py`): it opens a streamable-HTTP connection,
lists the server's tools, and hands `lookup_order` / `create_ticket` to the agent.
`search_kb` stays a **local** Strands tool over the Knowledge Base (the 1.5.6
retrieval-as-a-tool pattern — your own read path, no business side effect).

You can run the server locally to develop without redeploying:

```bash
uv run python -m mcp_server                                  # http://127.0.0.1:8000/mcp
RELAY_MCP_URL=http://127.0.0.1:8000/mcp uv run python -m relay.agent "Where is order 1042?"
```

> **Restricted-org accounts (Function URL `AuthType=NONE`).** `setup.py` fronts the MCP
> Lambda with a **public** Function URL (`AuthType=NONE`) for lab simplicity, and the agent
> connects to it unsigned. If your account is in an AWS Organization whose **SCP blocks
> public Lambda Function URLs**, that URL returns **403 Forbidden** and the agent falls back
> to the doc-only path. Two ways through: (a) use the **local MCP server** above
> (`RELAY_MCP_URL=http://127.0.0.1:8000/mcp` — same MCP protocol, same real DynamoDB tables,
> same Bedrock agent loop, only the transport host differs), which is exactly what the live
> smoke test uses; or (b) switch the URL to `AuthType=AWS_IAM` and sign requests with SigV4.
> The deployed Lambda itself is verified — a SigV4-signed `initialize` returns HTTP 200.

> **⚠️ MCP is not an agent-to-agent protocol.** MCP standardizes the connection
> **agent ↔ tools/data** (an agent is the *client* of tool *servers*). Communication
> *between agents* (handoffs, A2A) is a different mechanism — Module 8. And **tool calling
> does not mean the tool runs at the model provider**: it is **your** code (this Lambda) that
> executes, which is exactly why the IAM boundaries matter.

---

## Step 6 — Demonstrate the guardrails

**(a) A runaway loop, cut by the stop condition.** Feed the agent a request that makes it
keep calling a tool that never "finishes", and lower the cap to watch the stop condition
fire:

```bash
# Try-it #2 below shows max_iterations=1; the smoke test proves the cut with a looping model:
uv run pytest -k test_stop_condition_cuts_a_runaway_agent -q
# -> stop_reason == "limit_turns", status == "failed", <= max_iterations tool calls.
```

Without the cap the loop is unbounded — and so is the bill.

**(b) A write outside `relay-tickets`, refused by IAM.** The Lambda role can write only
`relay-tickets`. To see the boundary, attempt a write to another table **from the Lambda's
role**:

```bash
# From the MCP Lambda's execution role (relay-mcp-lambda-role): a PutItem to relay-orders or
# any other table is denied —
aws dynamodb put-item --table-name relay-orders \
    --item '{"order_id": {"S": "x"}}' --region us-east-1
# AccessDeniedException: not authorized to perform: dynamodb:PutItem on resource
#   arn:aws:dynamodb:us-east-1:<acct>:table/relay-orders
```

The boundary is enforced by IAM, not by convention — the offline test
`test_mcp_lambda_role_is_bounded_to_orders_read_tickets_write` proves the policy shape (read
orders, write tickets, no `*`).

---

## Step 7 — Offline tests, then teardown

```bash
uv run pytest                          # offline: no creds, no network (Modules 2–7)
RELAY_LIVE_TESTS=1 uv run pytest -m live   # up to 7 sub-cent calls (1 is a capped agent run)
uv run python teardown.py              # delete MCP Lambda + role; KEEP tables + KB (M8 reuses)
```

The offline tests cover the frozen `AgentAction` / `TicketRecord` contract, the
`mcp_server.store` data layer on a **moto** DynamoDB backend (lookup + idempotent ticket
write + clean errors), the tools, the **full agent ReAct loop driven by a scripted model**
(no Bedrock call) producing a `TicketRecord` with ≥1 `AgentAction`, the **stop-condition
guardrail**, the **IAM boundary** policy, and `setup.py`/`teardown.py` idempotency. The
live marker makes at most **seven** real calls — the inherited six plus **one capped live
agent run** (it skips cleanly if the MCP server / tables are not set up).

Teardown removes the MCP Lambda + Function URL + role and **keeps** the on-demand tables and
the KB (Module 8 reuses both, ~$0 idle). `--delete-tables` drops `relay-orders` /
`relay-tickets`; `--delete-kb` tears down the inherited KB too.

---

## Try it yourself

1. **Add a tool on the MCP server and watch discovery.** In `mcp_server/server.py`, add a
   `get_shipping_policy()` `@mcp.tool()` (return CloudCart's shipping windows). Redeploy
   (`uv run python setup.py --skip-kb`) and ask the agent *"what's your shipping policy?"* —
   it picks up the new tool with **no change to `relay/tools.py` or the agent**. That is the
   point of MCP: tools are discovered, not hard-coded.
2. **Shrink the budget and watch it degrade.** Call `agent_mod.handle(..., max_iterations=1)`
   (or lower `relay.agent.MAX_ITERATIONS`) and ask the order-status question. With one turn
   the agent cannot both look up the order **and** answer — the stop condition fires, the
   record is `failed`, and you see exactly why the budget has to leave room for the loop the
   task needs.
