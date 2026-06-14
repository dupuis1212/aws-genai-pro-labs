"""relay/agent.py — Relay as a Strands ReAct agent that ACTS, not just answers.

Module 7 of AWS GenAI Pro Mastery. Until now Relay could read and answer (triage in
M2, cited answers from the Knowledge Base in M5, multimodal intake in M6) — but it
could not DO anything. A customer asks "where is my order #1042?" and a doc-only system
can only paraphrase the shipping policy; the answer is in the ORDER BOOK, not the docs.

This module makes Relay an **agent**: a Strands agent whose foundation model runs the
**ReAct loop** (reason -> act -> observe, repeat) and DECIDES which tool to call —
`search_kb` for a how-to, `lookup_order` for a real order status, `create_ticket` to
record the outcome — without us scripting each path. The two business tools are served
by the CloudCart **MCP server** on Lambda (mcp_server/); `search_kb` is a local tool.

What this file owns:

  - build_agent(...)  : assemble the Strands agent — the SMART tier model (resolved
                        from relay.config — no model ID here), the Relay system prompt,
                        the tools, and the AgentAction journal hook. INJECTABLE (model,
                        tools) so the offline test drives it with a scripted model.
  - handle(...)       : run the agent on one ticket UNDER GUARDRAILS — a max-iterations
                        STOP CONDITION and a wall-clock TIMEOUT — journal every tool
                        call as a frozen AgentAction, then PERSIST a TicketRecord to the
                        `relay-tickets` table. Returns the TicketRecord.

GUARDED EXECUTION (skill 2.1.3) — three layers, because an agent without a tool-call
budget is a billing incident waiting to happen:
  1. STOP CONDITION: `limits={"turns": MAX_ITERATIONS}` caps the ReAct loop. A model
     stuck calling a failing tool stops at N turns (stop_reason "limit_turns") instead
     of looping forever and burning tokens. The lab DEMOnstrates this.
  2. TIMEOUT: a wall-clock ceiling on the whole run, so a hung tool/model call cannot
     pin the process indefinitely.
  3. IAM RESOURCE BOUNDARIES: enforced one layer out — the MCP server's Lambda role can
     read ONLY relay-orders and write ONLY relay-tickets (setup.py). A tool that tried
     to touch anything else is denied by IAM, not just by convention. The lab shows the
     refusal.

This is single-agent only. There is NO multi-agent handoff, NO managed runtime/memory,
and NO human approval here — `AgentAction.approved` stays None everywhere (all of that
is Module 8). The agent runs LOCALLY.

Run it:
    # against the deployed MCP Lambda (setup.py recorded its URL):
    uv run python -m relay.agent "Where is order 1042? It was supposed to arrive Monday."
    # a documentation question -> the agent chooses search_kb, not lookup_order:
    uv run python -m relay.agent "How do refunds work?"
    # against a local MCP server (uv run python -m mcp_server):
    RELAY_MCP_URL=http://127.0.0.1:8000/mcp uv run python -m relay.agent "Where is order 1042?"
"""

from __future__ import annotations

import datetime as dt
import sys
import uuid
from dataclasses import dataclass

from botocore.config import Config as BotoConfig

from strands import Agent
from strands.hooks import AfterToolCallEvent, HookProvider, HookRegistry
from strands.models import BedrockModel

from relay import config, tools
from relay.models import AgentAction, TicketRecord

# --- Guardrails: the agent's execution budget (skill 2.1.3) -------------------
# The stop condition. A ReAct loop is N model calls; capping turns caps the cost and
# stops a runaway. 6 is comfortably more than the 2–3 turns a real CloudCart ticket
# needs (search_kb OR lookup_order, then create_ticket, then a final answer), so a
# healthy run never hits it — but a model stuck on a failing tool does.
MAX_ITERATIONS = 6

# Wall-clock ceiling for one full agent run (seconds). A second guardrail in case a
# single tool/model call hangs (the turn cap only fires BETWEEN cycles). Passed to the
# bedrock-runtime client so a stuck Converse call cannot pin the run forever.
AGENT_TIMEOUT_S = 60

# The Relay tier the agent reasons on. A NAME, resolved through relay.config — the
# model ID lives ONLY in config (the grep gate proves it). The agent is the "complex"
# workload (multi-step reasoning over tools), so the SMART tier, exactly as the spec
# pins it.
AGENT_TIER = "smart"

# Relay's system prompt — its role, its tools, and its operating rules. Kept here (not
# in a prompt store) because it is code-coupled to the tool set; M8 will give the
# Billing specialist its own.
SYSTEM_PROMPT = (
    "You are Relay, CloudCart's customer-support agent. CloudCart is an e-commerce "
    "SaaS platform. You help customers by REASONING about their request and CALLING "
    "TOOLS — you do not guess.\n\n"
    "Your tools:\n"
    "  - search_kb(query): search the help-center docs for how-to / policy answers "
    "(refunds, plans, password resets, error codes, shipping policy).\n"
    "  - lookup_order(order_id): get the REAL, live status of a specific order. Use "
    "this for 'where is my order?' — the order book is NOT in the docs.\n"
    "  - create_ticket(ticket_id, status, summary): record what you did, ONCE, at the "
    "end.\n\n"
    "Rules:\n"
    "  1. Choose the right tool. An order-status question -> lookup_order. A how-to / "
    "policy question -> search_kb. Do not call lookup_order for a documentation "
    "question, or search_kb for a specific order's status.\n"
    "  2. Answer from what the tools return. Cite the order's real status or the doc, "
    "do not invent shipping dates or policies.\n"
    "  3. If a tool returns an error or 'not found', tell the customer plainly and, if "
    "useful, ask them for the missing detail — do not retry the same failing call in a "
    "loop.\n"
    "  4. When you have resolved the request, call create_ticket once with a short "
    "summary, then give the customer your final answer."
)


# =============================================================================
# The AgentAction journal — every tool call recorded (skill 2.1.6, the audit trail).
# =============================================================================
class ActionJournal(HookProvider):
    """A Strands hook that records every tool call as a frozen AgentAction.

    The ReAct loop fires AfterToolCallEvent each time a tool finishes (success or
    error). We capture the tool NAME, the INPUT the model passed, and the tool's
    RESULT text — the auditable record of what the agent DID, which lands in the
    TicketRecord.actions[] log. `approved` stays None at Module 7 (no approval flow).
    """

    def __init__(self) -> None:
        self.actions: list[AgentAction] = []

    def register_hooks(self, registry: HookRegistry, **kwargs) -> None:
        registry.add_callback(AfterToolCallEvent, self._on_after_tool)

    def _on_after_tool(self, event: AfterToolCallEvent) -> None:
        tool_use = event.tool_use or {}
        name = tool_use.get("name", "(unknown)")
        tool_input = tool_use.get("input", {}) or {}
        if not isinstance(tool_input, dict):
            tool_input = {"value": tool_input}
        result_text = _result_text(event.result, event.exception)
        self.actions.append(
            AgentAction(
                tool=name,
                tool_input=tool_input,
                result=result_text,
                approved=None,  # EFFECTIVE only at Module 8 — always None at M7.
            )
        )


def _result_text(result, exception) -> str:
    """Flatten a Strands tool result (or an exception) to one string for the journal."""
    if exception is not None:
        return f"error: {exception}"
    if result is None:
        return ""
    # A tool result is {"toolUseId", "status", "content": [{"text": ...}, ...]}.
    if isinstance(result, dict):
        content = result.get("content", [])
        if isinstance(content, list):
            parts = [block.get("text", "") for block in content
                     if isinstance(block, dict) and "text" in block]
            joined = " ".join(p for p in parts if p).strip()
            status = result.get("status")
            if status == "error" and not joined:
                return "error"
            return joined or str(result)
    return str(result)


# =============================================================================
# Building the agent — injectable so tests drive it with a scripted model.
# =============================================================================
def _bedrock_model() -> BedrockModel:
    """The SMART-tier Bedrock model for the agent. Model ID from relay.config ONLY.

    The bedrock-runtime client carries the wall-clock guardrail (read/connect timeout)
    so a stuck model call cannot hang the run past AGENT_TIMEOUT_S.
    """
    return BedrockModel(
        model_id=config.tier_profile(AGENT_TIER),
        region_name=config.REGION,
        boto_client_config=BotoConfig(
            read_timeout=AGENT_TIMEOUT_S,
            connect_timeout=10,
            retries={"max_attempts": 2, "mode": "standard"},
        ),
    )


def build_agent(*, model=None, extra_tools=None, journal: ActionJournal | None = None):
    """Assemble Relay's Strands agent. Returns (agent, journal).

    Args:
        model: a Strands model. Defaults to the SMART-tier BedrockModel. Tests pass a
            scripted model so the whole loop runs offline with no Bedrock call.
        extra_tools: the MCP business tools (lookup_order / create_ticket) discovered
            from the CloudCart MCP server, to add alongside the local search_kb. When
            omitted, the agent has only search_kb (used by the doc-only demo + tests).
        journal: an ActionJournal to record tool calls into (a fresh one is created if
            omitted) — returned so the caller can read actions[] after the run.

    The agent gets the Relay system prompt and `callback_handler=None` (we do not stream
    to stdout from inside the library; the CLI prints the loop itself).
    """
    journal = journal or ActionJournal()
    tool_list = [tools.search_kb]
    if extra_tools:
        tool_list.extend(extra_tools)

    agent = Agent(
        model=model if model is not None else _bedrock_model(),
        tools=tool_list,
        system_prompt=SYSTEM_PROMPT,
        hooks=[journal],
        callback_handler=None,
    )
    return agent, journal


# =============================================================================
# Handling one ticket — run under guardrails, journal, persist a TicketRecord.
# =============================================================================
def final_text(agent_result) -> str:
    """Extract the agent's final answer text from its last message."""
    message = getattr(agent_result, "message", {}) or {}
    parts = [
        block.get("text", "")
        for block in message.get("content", [])
        if isinstance(block, dict) and "text" in block
    ]
    return "".join(parts).strip()


@dataclass
class HandleResult:
    """The outcome of handling one ticket: the persisted record + the agent's answer.

    `record` is the frozen TicketRecord written to relay-tickets; `answer_text` is the
    agent's final prose to the customer (it rides in the agent's last message, not on
    the record schema, so we surface it here for the CLI/demo); `stop_reason` is the
    Strands stop reason ("end_turn", "limit_turns", ...) so a caller can see WHY a run
    ended.
    """

    record: TicketRecord
    answer_text: str
    stop_reason: str


def handle(
    customer_message: str,
    *,
    ticket_id: str | None = None,
    agent=None,
    journal: ActionJournal | None = None,
    persist=None,
    max_iterations: int = MAX_ITERATIONS,
) -> HandleResult:
    """Run the agent on one ticket under guardrails; persist + return a HandleResult.

    Steps:
      1. Run the Strands agent on the customer message with a STOP CONDITION
         (`limits={"turns": max_iterations}`). Every tool call is journaled as an
         AgentAction by the ActionJournal hook.
      2. Decide the outcome status from the run: `answered` on a clean finish,
         `failed` if the stop condition (or an error) cut the run before a final answer.
      3. Build the frozen TicketRecord (actions[] = the journal, cost_cents = 0.0
         placeholder at M7, escalated = False) and PERSIST it to relay-tickets via
         `persist` (mcp_server.store.create_ticket by default; injectable for tests).

    Args:
        customer_message: the ticket text the customer sent.
        ticket_id: the record's id (a generated id when omitted).
        agent / journal: a prebuilt (agent, journal) — e.g. one already wired with MCP
            tools inside a `with mcp_business_tools()` block. When omitted, a doc-only
            agent (search_kb) is built.
        persist: a callable(ticket_id, *, status, summary, actions) -> stored record
            (dict or TicketRecord). Defaults to mcp_server.store.create_ticket. Tests
            pass a fake to stay offline / assert the write.
        max_iterations: the turn cap (the stop condition). Lower it to 1 to watch the
            agent degrade (the lab's "Try it yourself" #2).

    Returns a HandleResult(record, answer_text, stop_reason).
    """
    ticket_id = ticket_id or f"ticket-{uuid.uuid4().hex[:8]}"
    if agent is None:
        agent, journal = build_agent(journal=journal)
    if journal is None:
        raise ValueError(
            "handle() needs the ActionJournal that build_agent returned so it can read "
            "actions[]. Pass both `agent` and `journal`, or neither."
        )

    # Tell the model the ticket id so its own create_ticket call targets the SAME row
    # that handle()'s authoritative write (with the full actions[] journal) overwrites —
    # idempotent on ticket_id, so there is exactly one relay-tickets row, not two.
    prompt = (
        f"Ticket id for this conversation: {ticket_id}\n"
        f"Customer message: {customer_message}"
    )

    status = "failed"
    summary = ""
    answer_text = ""
    stop_reason = ""
    try:
        result = agent(prompt, limits={"turns": max_iterations})
        answer_text = final_text(result)
        stop_reason = getattr(result, "stop_reason", "") or ""
        if stop_reason == "limit_turns":
            # The STOP CONDITION fired: the loop was cut at the turn cap. Record it as
            # a failure (the agent did not reach a clean final answer) — not a crash.
            status = "failed"
            summary = (
                f"Stopped at the {max_iterations}-turn limit (stop condition) without "
                "a final answer."
            )
        else:
            status = "answered"
            summary = (answer_text[:200] if answer_text
                       else "Agent finished without text output.")
    except Exception as err:  # noqa: BLE001 — record the failure, never swallow it.
        status = "failed"
        summary = f"Agent run failed: {type(err).__name__}: {err}"

    record = _persist(
        persist=persist,
        ticket_id=ticket_id,
        status=status,
        summary=summary,
        actions=journal.actions,
    )
    # _persist returns the stored record as a dict; rebuild the frozen TicketRecord so
    # the caller gets a typed object (and we re-validate the round trip).
    typed = TicketRecord.model_validate(record) if isinstance(record, dict) else record
    return HandleResult(record=typed, answer_text=answer_text, stop_reason=stop_reason)


def _persist(*, persist, ticket_id, status, summary, actions) -> dict:
    """Write the AUTHORITATIVE TicketRecord through `persist` (store.create_ticket).

    The agent may also call the create_ticket TOOL itself mid-run — but that call only
    carries ticket_id/status/summary, not the actions[] journal (which the hook only
    finishes assembling after the run). So handle() does the authoritative write HERE,
    with the FULL journal, through the SAME store path. Because the store write is
    IDEMPOTENT on ticket_id (PutItem upserts) and handle() passes the model the same
    ticket id, this overwrites the agent's own row rather than creating a second one —
    one writer, one row. Defaulting to mcp_server.store.create_ticket keeps a single
    relay-tickets writer.
    """
    if persist is None:
        from mcp_server import store
        persist = store.create_ticket
    return persist(
        ticket_id,
        status=status,
        summary=summary,
        actions=[a.model_dump() for a in actions],
    )


# =============================================================================
# CLI — run one ticket end to end, printing the agent loop, answer, and TicketRecord.
# =============================================================================
def _print_result(outcome: HandleResult) -> None:
    record = outcome.record
    print("\n--- agent actions (the ReAct loop) ---")
    if not record.actions:
        print("  (no tool calls — the agent answered directly)")
    for i, action in enumerate(record.actions, 1):
        result = action.result.replace("\n", " ")
        if len(result) > 160:
            result = result[:157] + "..."
        print(f"  {i}. {action.tool}({action.tool_input}) -> {result}")
    print(f"\n--- final answer (stop reason: {outcome.stop_reason or 'n/a'}) ---")
    print(outcome.answer_text or "(the agent produced no text answer)")
    print("\n--- TicketRecord persisted to relay-tickets ---")
    print(record.model_dump_json(indent=2))


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _print_only_persist(ticket_id, *, status, summary, actions):
    """A persist function that does NOT write DynamoDB — used when no MCP server is
    configured (the doc-only fallback). Returns a record dict so handle() still works."""
    record = TicketRecord(
        ticket_id=ticket_id, status=status, triage=None, answer=None,
        actions=[AgentAction.model_validate(a) for a in actions], escalated=False,
        cost_cents=0.0, updated_at=_now_iso(),
    )
    print("[note] No relay-tickets write (no MCP server) — record not persisted.",
          file=sys.stderr)
    return record.model_dump(mode="json")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print(
            'Usage: uv run python -m relay.agent "<customer message>"\n'
            'Example: uv run python -m relay.agent '
            '"Where is order 1042? It was supposed to arrive Monday."',
            file=sys.stderr,
        )
        return 1

    customer_message = argv[0]
    ticket_id = f"ticket-{uuid.uuid4().hex[:8]}"

    # Connect the MCP client, discover the business tools, build the agent WITH them,
    # and handle the ticket inside the connection's lifetime. If no MCP URL is
    # configured, degrade to the doc-only agent so a pure documentation question still
    # works (no order lookups / ticket writes) — and say so.
    try:
        with tools.mcp_business_tools() as biz_tools:
            agent, journal = build_agent(extra_tools=biz_tools)
            outcome = handle(customer_message, ticket_id=ticket_id,
                             agent=agent, journal=journal)
    except ValueError as err:
        # resolve_mcp_url could not find a URL: degrade to the local-only agent.
        print(f"[note] MCP server not configured ({err}).", file=sys.stderr)
        print("[note] Running with search_kb only (no order lookups / ticket writes).",
              file=sys.stderr)
        agent, journal = build_agent()
        outcome = handle(customer_message, ticket_id=ticket_id, agent=agent,
                         journal=journal, persist=_print_only_persist)
    except Exception as err:  # noqa: BLE001
        print(f"Agent run could not start: {type(err).__name__}: {err}",
              file=sys.stderr)
        return 1

    _print_result(outcome)
    return 0 if outcome.record.status == "answered" else 1


if __name__ == "__main__":
    raise SystemExit(main())
