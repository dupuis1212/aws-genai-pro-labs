"""relay/run.py — Relay's invocation entrypoint, deployed on AgentCore Runtime.

Module 8 of AWS GenAI Pro Mastery. Module 7's agent ran on your laptop: it stopped when
you closed the terminal and forgot every conversation. Module 8 deploys it as a managed
service on **Bedrock AgentCore Runtime** (a microVM with sessions up to 8 h, idle FREE,
per-session isolation) with **AgentCore Memory** (short-term session events + long-term
cross-session records). This file is the seam between the two worlds:

  - run_relay(payload) -> response : the PURE, framework-agnostic invocation contract.
    ONE JSON payload in, ONE JSON response out. This is the FROZEN shape M11's
    worker_handler invokes the deployed agent through — it must not drift.
  - the AgentCore Runtime app : the BedrockAgentCoreApp `@app.entrypoint` wrapper that
    AgentCore Runtime calls in the deployed microVM. It just adapts the runtime's
    payload to run_relay() and runs the HTTP server. Built lazily so importing
    relay.run stays offline for tests.

The FROZEN invoke contract (bible §2.2 M8 — do not change the keys, M11 depends on it):

    payload  = {
        "customer_message": str,            # required — the ticket text
        "ticket_id":        str | None,     # optional — generated when absent
        "triage_intent":    str | None,     # optional — the M2 Triage.intent, for routing
        "customer_id":      str | None,     # optional — the AgentCore Memory actor id
        "session_id":       str | None,     # optional — the AgentCore Memory session id
    }
    response = {
        "ticket_id":   str,                 # the handled ticket's id
        "status":      str,                 # the TicketRecord status (awaiting_approval,
                                            #   answered, escalated, failed)
        "answer_text": str,                 # the agent's prose to the customer
        "handed_off":  bool,                # routed to the Billing specialist?
        "gated":       bool,                # a refund is awaiting human approval?
        "record":      dict,                # the full TicketRecord (model_dump)
    }

Memory, used here (skill 2.1.1 managed-memory facet):
  - SHORT-TERM (session): the in-progress conversation thread. AgentCore Memory stores
    the turn-by-turn events under a session id, so a SECOND message in the same session
    SEES the first — the agent remembers within the conversation.
  - LONG-TERM (cross-session): per-customer facts/preferences distilled from past
    tickets, under a customer-keyed namespace. This is the only idle-billed store
    (~$0.75/1K records/month) — purged at teardown.
  - WHAT WE STORE: short useful facts and summaries. WHAT WE DO NOT: raw PII in long-term
    memory (a customer's name/email/card never go to the durable store as-is) — PII
    redaction at intake is Module 10; this file keeps the long-term writes to non-PII
    facts (the order id, the resolution), 1 comment marking the M10 boundary.

This is the reasoning + invocation layer; AgentCore is the RUNTIME that hosts it — they
COMPOSE (AgentCore does not replace Strands). No model ID and no bare model-invoke path
here; the agent resolves its tier through relay.config.
"""

from __future__ import annotations

import json
import re
import sys
import uuid

from relay import agent as agent_module
from relay import config
from relay import tools


# =============================================================================
# AgentCore Memory helpers (skill 2.1.1 managed-memory facet) — short + long term.
# =============================================================================
# Inlined here (relay/run.py is the M8 entrypoint; the bible §2.2 M8 row adds exactly
# specialists.py / approve.py / run.py / agentcore/, no separate memory module). All
# three helpers are BEST-EFFORT: a memory outage degrades the run to stateless, it
# NEVER fails the ticket. They talk to AgentCore Memory through the wrapper the caller
# passes (or resolve the memory id from config); tests pass a fake or omit it.
def _memory_client(memory):
    """Resolve the AgentCore Memory client wrapper, or None if memory is not set up.

    `memory` may be: an object exposing get_last_k_turns/create_event/
    retrieve_memories (an AgentCore MemoryClient, or a test fake) — used as-is; or
    None — then we try to build one from the resolved memory id, returning None
    (stateless) if AgentCore Memory is not configured. Never raises."""
    if memory is not None:
        return memory
    try:
        config.resolve_memory_id()
    except ValueError:
        return None  # no Memory store -> run stateless (degraded, not an error)
    try:
        from bedrock_agentcore.memory import MemoryClient

        return MemoryClient(region_name=config.REGION)
    except Exception:  # noqa: BLE001 — SDK missing/offline -> stateless, never fail.
        return None


def load_session_memory(memory, *, session_id, customer_id) -> str:
    """Return a short recap of the in-session conversation so far (short-term memory).

    Reads the recent turn events AgentCore Memory stored for this session so a follow-up
    message "remembers" the earlier one. Returns "" when there is no memory store, no
    session, or no prior turns (a first message). Best-effort — any error -> ""."""
    client = _memory_client(memory)
    if client is None or not session_id:
        return ""
    try:
        mem_id = config.resolve_memory_id()
        turns = client.get_last_k_turns(
            memory_id=mem_id, actor_id=str(customer_id or "anonymous"),
            session_id=str(session_id), k=4,
        )
    except Exception:  # noqa: BLE001
        return ""
    if not turns:
        return ""
    lines: list[str] = []
    for turn in turns:
        for msg in (turn if isinstance(turn, list) else [turn]):
            role = (msg.get("role") or "").lower() if isinstance(msg, dict) else ""
            text = msg.get("content", {}).get("text", "") if isinstance(msg, dict) else ""
            if text:
                lines.append(f"  {role or 'msg'}: {text}")
    return "\n".join(lines)


def record_session_turn(memory, *, session_id, customer_id, customer_message,
                        answer_text) -> None:
    """Append this turn (customer message + Relay answer) to short-term session memory.

    So the NEXT message in the same session can recall it. Best-effort — a failure is
    logged to stderr and swallowed (the ticket already succeeded)."""
    client = _memory_client(memory)
    if client is None or not session_id:
        return
    try:
        mem_id = config.resolve_memory_id()
        client.create_event(
            memory_id=mem_id, actor_id=str(customer_id or "anonymous"),
            session_id=str(session_id),
            messages=[(customer_message, "USER"), (answer_text or "", "ASSISTANT")],
        )
    except Exception as err:  # noqa: BLE001
        print(f"[memory] session write skipped: {type(err).__name__}", file=sys.stderr)


def record_long_term_fact(memory, *, customer_id, fact) -> None:
    """Record a NON-PII fact in long-term (cross-session) memory for a customer.

    Long-term memory is the durable, idle-billed store (purged at teardown). We write
    ONLY non-PII facts here (the order/resolution), never the raw customer message —
    PII redaction of inputs is Module 10. Best-effort; a failure is swallowed."""
    client = _memory_client(memory)
    if client is None or not customer_id or not fact:
        return
    try:
        mem_id = config.resolve_memory_id()
        # A long-term fact is stored as an event in a per-customer "facts" session;
        # AgentCore's long-term semantic strategy distils durable records from these
        # events (the namespace is configured on the strategy, not per event). The
        # CreateEvent sessionId must match [a-zA-Z0-9][a-zA-Z0-9-_]* (no slashes), so we
        # build a valid id from the namespace template rather than pass the raw template.
        session_id = _long_term_session_id(customer_id)
        client.create_event(
            memory_id=mem_id, actor_id=str(customer_id),
            session_id=session_id, messages=[(fact, "ASSISTANT")],
        )
    except Exception as err:  # noqa: BLE001
        print(f"[memory] long-term write skipped: {type(err).__name__}", file=sys.stderr)


def _long_term_session_id(customer_id) -> str:
    """A CreateEvent-valid session id for a customer's long-term "facts" stream.

    AgentCore requires sessionId to match [a-zA-Z0-9][a-zA-Z0-9-_]* — the long-term
    namespace template (support/customer/<id>/facts) carries slashes, so we flatten it
    to a valid id. The semantic strategy still distils long-term records from these
    events; the namespace is a strategy-level concept, not the event's session id."""
    raw = config.MEMORY_LONG_TERM_NAMESPACE.format(actor_id=str(customer_id))
    flat = re.sub(r"[^a-zA-Z0-9-_]", "-", raw).strip("-")
    return flat or f"facts-{customer_id}"


# =============================================================================
# The PURE invocation contract — run_relay(payload) -> response (FROZEN, M11 reuses).
# =============================================================================
def run_relay(payload: dict, *, biz_tools=None, memory=None) -> dict:
    """Handle one invocation. ONE dict in, ONE dict out (the frozen contract above).

    Args:
        payload: the invoke payload (see the module docstring for the frozen keys).
        biz_tools: the shared CloudCart MCP tools (lookup_order/create_ticket). When
            None, run_relay opens an MCP connection itself for the duration of the run.
            Tests pass a list of local tools to stay offline.
        memory: an AgentCore Memory client wrapper (relay.memory_helpers). When None,
            memory is resolved from config; tests pass a fake or omit it (memory is
            best-effort — a memory outage degrades to a stateless run, it never fails
            the ticket).

    Steps:
      1. validate the payload (a missing customer_message is a clear error);
      2. LOAD short-term session memory so a follow-up message sees the conversation;
      3. run the agent WITH HANDOFF + the HITL gate (relay.agent.handle_with_handoff);
      4. RECORD the turn to session memory and a non-PII fact to long-term memory;
      5. return the frozen response dict.
    """
    customer_message = (payload or {}).get("customer_message")
    if not customer_message or not str(customer_message).strip():
        raise ValueError("payload.customer_message is required and must be non-empty.")

    ticket_id = (payload or {}).get("ticket_id") or f"ticket-{uuid.uuid4().hex[:8]}"
    triage_intent = (payload or {}).get("triage_intent")
    customer_id = (payload or {}).get("customer_id")
    session_id = (payload or {}).get("session_id")

    # --- 2. Short-term memory: recall the in-session conversation so far -----------
    # Best-effort: a memory error degrades to a stateless run; it never fails the run.
    prior_context = load_session_memory(memory, session_id=session_id,
                                        customer_id=customer_id)
    message = customer_message
    if prior_context:
        # Prepend a short recap so the agent "remembers" the earlier turn(s) — this is
        # the lab's "the agent remembers the previous question" demo.
        message = (
            "Earlier in this conversation:\n" + prior_context
            + "\n\nNew message from the same customer:\n" + customer_message
        )

    # --- 3. Run the agent with the M8 handoff + HITL gate --------------------------
    # If biz_tools were not supplied, open an MCP connection for the run so the agent
    # has lookup_order/create_ticket; degrade to a tool-light run when MCP is absent.
    if biz_tools is not None:
        outcome = _run_with_tools(message, ticket_id, triage_intent, biz_tools)
    else:
        outcome = _run_resolving_tools(message, ticket_id, triage_intent)

    # --- 4. Record memory (best-effort, non-PII for long-term) --------------------
    record_session_turn(memory, session_id=session_id, customer_id=customer_id,
                        customer_message=customer_message,
                        answer_text=outcome.answer_text)
    # LONG-TERM: store only a NON-PII fact (the ticket outcome), never the raw message.
    # PII redaction of inputs is Module 10; here we deliberately keep the durable write
    # to the resolution summary, which carries no customer name/email/card.
    record_long_term_fact(memory, customer_id=customer_id,
                          fact=f"Ticket {ticket_id}: status {outcome.record.status}"
                               + (f", handed off to the {config.BILLING_SPECIALIST_NAME}"
                                  if outcome.handed_off else ""))

    return {
        "ticket_id": outcome.record.ticket_id,
        "status": outcome.record.status,
        "answer_text": outcome.answer_text,
        "handed_off": outcome.handed_off,
        "gated": outcome.gated,
        "record": outcome.record.model_dump(mode="json"),
    }


def _run_with_tools(message, ticket_id, triage_intent, biz_tools):
    """Run handle_with_handoff giving BOTH the generalist and the specialist the shared
    MCP business tools (so a handoff keeps the same toolset)."""
    from relay import specialists

    generalist = agent_module.build_agent(extra_tools=biz_tools)
    specialist = specialists.build_billing_specialist(extra_tools=biz_tools)
    return agent_module.handle_with_handoff(
        message, ticket_id=ticket_id, triage_intent=triage_intent,
        generalist=generalist, specialist=specialist,
    )


def _run_resolving_tools(message, ticket_id, triage_intent):
    """Open an MCP connection for the run; degrade to a tool-light run if none is set."""
    try:
        with tools.mcp_business_tools() as biz_tools:
            return _run_with_tools(message, ticket_id, triage_intent, biz_tools)
    except ValueError:
        # No MCP URL configured — degrade (a doc/refund-only run still works).
        return agent_module.handle_with_handoff(
            message, ticket_id=ticket_id, triage_intent=triage_intent,
        )


# =============================================================================
# The AgentCore Runtime app — the @app.entrypoint AgentCore invokes in the microVM.
# =============================================================================
def build_app():
    """Build the BedrockAgentCoreApp that AgentCore Runtime serves. Imported lazily so
    relay.run stays offline for tests (the bedrock-agentcore SDK is only needed when the
    deployed runtime actually runs). The agentcore CLI deploys THIS app."""
    from bedrock_agentcore.runtime import BedrockAgentCoreApp

    app = BedrockAgentCoreApp()

    @app.entrypoint
    def invoke(payload, context=None):
        """AgentCore Runtime entrypoint: adapt the runtime payload to run_relay().

        AgentCore passes the request body as `payload` and a RequestContext as
        `context` (it carries the session id AgentCore manages). We map them onto the
        frozen run_relay contract and return its JSON response.
        """
        # AgentCore manages a per-invocation session id; prefer the payload's, else the
        # runtime context's, so short-term memory is keyed by the live session.
        if "session_id" not in (payload or {}) and context is not None:
            sid = getattr(context, "session_id", None)
            if sid:
                payload = {**(payload or {}), "session_id": sid}
        return run_relay(payload)

    return app


# =============================================================================
# CLI — invoke run_relay locally (the lab's headline `uv run python -m relay.run "..."`).
# =============================================================================
def _print_response(response: dict) -> None:
    print("\n--- Relay invocation ---")
    print(f"  ticket_id : {response['ticket_id']}")
    print(f"  status    : {response['status']}")
    print(f"  handed off: {response['handed_off']}"
          + (f" -> the {config.BILLING_SPECIALIST_NAME}" if response['handed_off']
             else ""))
    print(f"  gated     : {response['gated']}"
          + (" (refund awaiting human approval)" if response['gated'] else ""))
    print("\n--- agent actions (the trail across the handoff) ---")
    actions = response["record"].get("actions", [])
    if not actions:
        print("  (no tool calls)")
    for i, action in enumerate(actions, 1):
        approved = action.get("approved")
        mark = {None: "PROPOSED (awaiting approval)", True: "approved",
                False: "rejected"}.get(approved, str(approved))
        result = (action.get("result") or "").replace("\n", " ")
        if len(result) > 140:
            result = result[:137] + "..."
        print(f"  {i}. {action['tool']}({action['tool_input']}) "
              f"[{mark}] -> {result}")
    print("\n--- final answer ---")
    print(response["answer_text"] or "(no text answer)")
    if response["gated"]:
        print(
            "\n[HITL] A refund is AWAITING APPROVAL — nothing was charged back yet.\n"
            f"       Approve: uv run python -m relay.approve {response['ticket_id']} "
            "--approve\n"
            f"       Reject : uv run python -m relay.approve {response['ticket_id']} "
            "--reject"
        )


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) < 1 or not argv[0].strip():
        print(
            'Usage: uv run python -m relay.run "<customer message>" '
            '[--customer <id>] [--session <id>] [--intent <triage intent>]\n'
            'Example: uv run python -m relay.run '
            '"this is the third time I\'m asking — just refund order 1042"',
            file=sys.stderr,
        )
        return 1

    customer_message = argv[0]
    payload: dict = {"customer_message": customer_message}
    # Tiny flag parser (no argparse ceremony for a demo entrypoint).
    rest = argv[1:]
    for flag, key in (("--customer", "customer_id"), ("--session", "session_id"),
                      ("--intent", "triage_intent")):
        if flag in rest:
            i = rest.index(flag)
            if i + 1 < len(rest):
                payload[key] = rest[i + 1]

    try:
        response = run_relay(payload)
    except ValueError as err:
        print(f"[run] {err}", file=sys.stderr)
        return 1
    except Exception as err:  # noqa: BLE001
        print(f"[run] invocation failed: {type(err).__name__}: {err}", file=sys.stderr)
        return 1

    _print_response(response)
    # answered / awaiting_approval are healthy outcomes; failed is not.
    return 0 if response["status"] != "failed" else 1


if __name__ == "__main__":
    # When AgentCore Runtime starts the container it serves the app (BedrockAgentCoreApp);
    # the agentcore CLI does that via build_app(). The runtime signals its context with an
    # env var, so we serve the HTTP app ONLY then — never from an ordinary shell, where
    # `python -m relay.run "..."` must call the CLI (and print usage when given no args).
    import os

    if os.environ.get("BEDROCK_AGENTCORE_RUNTIME") or \
            os.environ.get("RELAY_SERVE_AGENTCORE") == "1":
        build_app().run()
    else:
        raise SystemExit(main())
