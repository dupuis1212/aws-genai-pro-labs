"""relay — CloudCart's GenAI support agent.

Introduced in Module 2 of AWS GenAI Pro Mastery. This is the first appearance of
the cumulative `relay/` package the rest of the course builds on. Module 2 ships
exactly two pieces of it:

  - relay.models : the frozen Pydantic v2 schemas Ticket and Triage.
  - relay.triage : raw ticket -> validated Triage JSON via Bedrock Prompt
                   Management + a single Converse call (Nova Micro, temperature 0).

Later modules extend this package by ADDITION only (relay.llm and relay.config
arrive in Module 3, relay.kb in Module 5, and so on). Nothing here is rewritten
downstream.
"""

__all__ = ["models", "triage"]
