"""relay/specialists.py — the Billing specialist (a Strands agent Relay hands off to).

Module 8 of AWS GenAI Pro Mastery. Until now Relay was ONE agent (M7): a generalist
that searches the docs, looks up orders, and writes tickets. That is usually enough —
a single well-tooled agent beats a swarm, and every extra agent costs a model turn
(latency), duplicated context (tokens), and debugging surface (complexity). You add an
agent only when the SPECIALIZATION is genuinely justified.

A REFUND is that case. It needs a different TONE (apologetic, precise about money), its
own RULES (what is refundable, the window, partial vs full), and a guard rail the
generalist must not carry: it must PROPOSE the refund for a human, never execute it
blindly. So Module 8 introduces the **Billing specialist** (the canonical name, 06 §5.4
— no synonym): a second Strands agent with its OWN system prompt and a `refund` tool.

Topology (T5.1): SUPERVISOR / HANDOFF. The generalist Relay is the supervisor; when a
ticket is a billing/refund case it HANDS OFF to the Billing specialist (relay.agent does
the routing). The two share the SAME CloudCart tools (lookup_order / create_ticket over
the MCP server) and the SAME AgentCore Memory — the handoff ROUTES the work, it does not
clone it. AWS **Agent Squad** is the blueprint term for orchestrating several such
agents; it is a concept the article discusses, not a dependency of this lab.

What this file owns:

  - refund(order_id, amount_cents, reason) : the SENSITIVE @tool the specialist calls
    to PROPOSE a refund. It does NOT move money — it returns a structured proposal the
    HITL gate (relay.agent) records as an AgentAction(approved=None) and parks in
    `awaiting_approval`. Execution happens only after a human approves (relay.approve).
  - build_billing_specialist(...) : assemble the Billing specialist agent — the SMART
    tier (resolved from relay.config, no model ID here), its refund-tone system prompt,
    its tools (refund + the shared lookup_order/create_ticket), and the same
    AgentAction journal hook the generalist uses. INJECTABLE (model, tools) so the
    offline test drives it with a scripted model.

AgentCore (the managed RUNTIME) and Strands (the FRAMEWORK) compose: this file defines
the specialist's reasoning loop and tools in Strands; relay.run deploys the whole thing
on AgentCore Runtime. AgentCore does NOT replace Strands.

No model ID and no model-invoke path appear here (the grep gate proves it); the tier is
a NAME resolved through relay.config.
"""

from __future__ import annotations

import json

from botocore.config import Config as BotoConfig

from strands import Agent, tool
from strands.models import BedrockModel

from relay import config
from relay.agent import ActionJournal

# The Billing specialist's reasoning tier — a NAME, resolved through relay.config.
# Refund reasoning is the "complex" workload, so the SMART tier, exactly as the spec
# pins it. The model ID lives ONLY in config (the grep gate proves it).
SPECIALIST_TIER = config.BILLING_SPECIALIST_TIER

# The specialist's own wall-clock guardrail (seconds), mirrored from the generalist —
# a stuck Converse call cannot pin the handoff forever.
SPECIALIST_TIMEOUT_S = 60

# The Billing specialist's system prompt — refund TONE and RULES, distinct from the
# generalist's. Kept here (code-coupled to the refund tool) the same way Relay's prompt
# lives in relay.agent. The pivotal rule: PROPOSE, never execute — the refund is a
# proposal a human approves (the HITL gate, skill 2.1.5).
BILLING_SPECIALIST_SYSTEM_PROMPT = (
    "You are CloudCart's Billing specialist. A frustrated customer has been handed "
    "off to you because their ticket is about money — a charge, a refund, or a "
    "billing dispute. Be calm, precise, and empathetic; never argue about money.\n\n"
    "Your tools:\n"
    "  - lookup_order(order_id): get the REAL order — its status, total, and dates — "
    "before you decide anything. Always look up the order first.\n"
    "  - refund(order_id, amount_cents, reason): PROPOSE a refund for review. This "
    "does NOT move money. It records a proposal that a human teammate approves or "
    "rejects. Call it ONCE, with the exact amount in cents and a one-line reason.\n"
    "  - create_ticket(ticket_id, status, summary): record the outcome.\n\n"
    "Refund rules:\n"
    "  1. Look up the order first. Refund at most the order's total; for a partial "
    "issue (one item, a delay credit) propose only the affected amount.\n"
    "  2. A refund is a PROPOSAL, not an action. Call refund(...) to propose it, then "
    "tell the customer their refund request has been submitted for review and will be "
    "confirmed shortly. NEVER promise the money is already back — a human approves it.\n"
    "  3. If the order cannot be found or the request is not actually about a refund, "
    "say so plainly; do not propose a refund you cannot justify.\n"
    "  4. After proposing, call create_ticket once with a short summary."
)


# =============================================================================
# refund — the SENSITIVE tool the specialist PROPOSES (never executes here).
# =============================================================================
@tool
def refund(order_id: str, amount_cents: int, reason: str) -> str:
    """Propose a refund for a CloudCart order. This does NOT move money.

    Use this ONLY when a customer is owed a refund and you have looked up the order.
    The refund is PROPOSED for human review — a teammate approves or rejects it. Tell
    the customer their refund request was submitted for review; do not say it is done.

    Args:
        order_id: the CloudCart order to refund (e.g. "1042").
        amount_cents: the refund amount in CENTS (e.g. 12900 for $129.00). At most the
            order total; use the affected amount for a partial refund.
        reason: a one-line reason for the refund (e.g. "carrier lost the package").

    Returns a confirmation that the refund was PROPOSED and is awaiting human approval.
    """
    oid = str(order_id).strip().lstrip("#").strip()
    if not oid:
        return "order_id is required to propose a refund (e.g. '1042')."
    try:
        cents = int(amount_cents)
    except (TypeError, ValueError):
        return (
            "amount_cents must be a whole number of cents (e.g. 12900 for $129.00). "
            f"Got {amount_cents!r}."
        )
    if cents <= 0:
        return "A refund amount must be greater than zero cents."
    reason_text = (reason or "").strip() or "(no reason given)"
    # A structured PROPOSAL the HITL gate records — it does not touch any payment
    # system. Execution is deferred to relay.approve after a human approves.
    proposal = {
        "proposed": True,
        "order_id": oid,
        "amount_cents": cents,
        "reason": reason_text,
        "status": "awaiting_approval",
    }
    return (
        "Refund PROPOSED and submitted for human approval (not yet executed): "
        + json.dumps(proposal)
    )


# The canonical name of the sensitive tool this module owns (06 §2 / config). Exposed
# so the agent and the tests can assert the HITL contract without a magic string.
SPECIALIST_TOOL_NAMES = (config.REFUND_TOOL_NAME,)


# =============================================================================
# Building the Billing specialist — injectable so tests drive it with a scripted model.
# =============================================================================
def _bedrock_model() -> BedrockModel:
    """The SMART-tier Bedrock model for the specialist. Model ID from relay.config ONLY."""
    return BedrockModel(
        model_id=config.tier_profile(SPECIALIST_TIER),
        region_name=config.REGION,
        boto_client_config=BotoConfig(
            read_timeout=SPECIALIST_TIMEOUT_S,
            connect_timeout=10,
            retries={"max_attempts": 2, "mode": "standard"},
        ),
    )


def build_billing_specialist(
    *, model=None, extra_tools=None, journal: ActionJournal | None = None
):
    """Assemble the Billing specialist Strands agent. Returns (agent, journal).

    Args:
        model: a Strands model. Defaults to the SMART-tier BedrockModel. Tests pass a
            scripted model so the whole handoff runs offline with no Bedrock call.
        extra_tools: the SHARED CloudCart business tools (lookup_order / create_ticket)
            discovered from the MCP server, added alongside this module's `refund`. The
            specialist shares the generalist's tools — the handoff routes work, it does
            not give the specialist a private toolset.
        journal: the ActionJournal to record tool calls into — PASS THE GENERALIST'S
            journal so the handoff's actions land in the SAME TicketRecord.actions[]
            log (one auditable trail across both agents). A fresh one is created if
            omitted.

    The specialist gets its own refund-tone system prompt and `callback_handler=None`.
    """
    journal = journal or ActionJournal()
    tool_list = [refund]
    if extra_tools:
        tool_list.extend(extra_tools)

    agent = Agent(
        model=model if model is not None else _bedrock_model(),
        tools=tool_list,
        system_prompt=BILLING_SPECIALIST_SYSTEM_PROMPT,
        hooks=[journal],
        callback_handler=None,
    )
    return agent, journal
