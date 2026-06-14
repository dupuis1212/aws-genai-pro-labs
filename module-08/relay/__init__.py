"""relay — CloudCart's GenAI support agent.

Introduced in Module 2 of AWS GenAI Pro Mastery. This is the cumulative `relay/`
package the rest of the course builds on. By Module 3 it ships four pieces:

  - relay.models : the frozen Pydantic v2 schemas Ticket and Triage.
  - relay.config : the SOLE home of model-ID literals — the tier -> inference
                   profile map. Added in Module 3.
  - relay.llm    : converse() — the UNIQUE Bedrock call site (routing, streaming,
                   retries, cross-Region fallback). Added and FROZEN in Module 3;
                   its signature is byte-identical from M3 through M15.
  - relay.triage : raw ticket -> validated Triage JSON via Bedrock Prompt
                   Management. Module 3 refactored it to call
                   converse(tier="fast") instead of carrying its own model ID.
  - relay.kb     : retrieve() (Retrieve) + answer() (RetrieveAndGenerate) over the
                   managed Bedrock Knowledge Base `relay-kb` -> a grounded, cited
                   Answer. Added in Module 5. Introduces the Citation/Answer
                   schemas in relay.models, by addition.
  - relay.intake : raw email/chat (± screenshot) -> a validated, normalized Ticket.
                   Validate-before-generate gates, signature/quoted-thread
                   normalization, Amazon Comprehend entity extraction, an attachment
                   upload to attachments/, and an Amazon Nova Lite vision read of the
                   screenshot appended under [Attachment summary]. Added in Module 6;
                   freezes the Attachment schema and adds Ticket.attachments, by
                   addition. It does NOT redact PII (that is Module 10).
  - relay.tools  : the agent's tools — search_kb (a LOCAL Strands @tool over the
                   Knowledge Base, the 1.5.6 retrieval-as-a-tool pattern) plus the
                   MCP-client wiring that discovers lookup_order / create_ticket from
                   the CloudCart MCP server. Added in Module 7.
  - relay.agent  : Relay as a Strands ReAct agent. The SMART-tier model decides which
                   tool to call; every call is journaled as a frozen AgentAction and a
                   TicketRecord is persisted to relay-tickets. Stop condition (max
                   iterations) + timeout + an IAM-bounded MCP Lambda are the execution
                   guardrails. Added in Module 7; freezes the AgentAction / TicketRecord
                   schemas in relay.models, by addition. Single-agent only — no handoff,
                   no managed runtime/memory, no human approval (all Module 8).

  - relay.specialists : the Billing specialist — a second Strands agent (its own
                   refund-tone system prompt + a `refund` tool) the generalist HANDS
                   OFF billing/refund tickets to (supervisor/handoff topology). Added
                   in Module 8. Shares the generalist's tools and AgentAction journal.
  - relay.approve : the human-in-the-loop decision point. approve(ticket_id, decision)
                   reads an `awaiting_approval` TicketRecord, sets AgentAction.approved
                   = True/False, then EXECUTES the refund (-> answered) or escalates
                   (-> escalated). Added in Module 8 — local/programmatic; the public
                   approval endpoint + the approval event bus are Module 11.
  - relay.run    : Relay's invocation entrypoint, deployed on Bedrock AgentCore Runtime.
                   run_relay(payload) -> response is the FROZEN invoke contract M11's
                   worker reuses; it wires the handoff, the HITL gate, and AgentCore
                   Memory (short-term session + long-term cross-session). Added in M8.

Module 8 makes AgentAction.approved EFFECTIVE (None/True/False) and exercises the frozen
TicketRecord status `awaiting_approval` — BY USE, with NO schema change (the field and
the status were frozen at Module 7). Later modules extend this package by ADDITION only
(the guardrail layer in Module 9, and so on). Nothing here is rewritten downstream.
"""

__all__ = [
    "models", "config", "llm", "triage", "kb", "intake", "tools", "agent",
    "specialists", "approve", "run",
]
