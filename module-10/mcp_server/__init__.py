"""mcp_server — the stateless CloudCart MCP server (Module 7).

This package is the **Model Context Protocol (MCP)** server that exposes CloudCart's
two business tools to Relay's agent:

  - lookup_order(order_id)            -> read the order book (DynamoDB `relay-orders`)
  - create_ticket(ticket_id, ...)     -> write a TicketRecord (DynamoDB `relay-tickets`)

It is **stateless**: every request carries everything it needs, the handler holds no
session state between calls, and the only durable state lives in DynamoDB. That is
exactly why it fits AWS Lambda (skill 2.1.7): a stateless, short-lived, event-driven
tool server costs ~$0 idle. (A heavy, long-running, stateful tool server — a big model,
a warm cache, a websocket fan-out — would go on ECS instead; that is the article's
theory contrast, not built here.)

Layout:
  - store.py   : the DynamoDB data-access layer (the SOLE place the two tools' table
                 reads/writes live). Pure-ish, client-injectable, unit-testable on moto.
  - server.py  : the FastMCP server — wraps store.py functions as MCP @tools and mounts
                 the streamable-HTTP transport. Runs locally (`python -m mcp_server`) or
                 inside the Lambda.
  - app.py     : the AWS Lambda entrypoint (handler) that serves the FastMCP
                 streamable-HTTP app behind a Lambda Function URL.

The agent (relay.agent) is the MCP CLIENT: relay.tools builds an MCP client over this
server's URL and hands the discovered `lookup_order` / `create_ticket` tools to the
Strands agent. `search_kb` stays a LOCAL Strands tool over the Knowledge Base (the
1.5.6 retrieval-as-a-tool pattern) — only the two business tools move onto MCP.

No model IDs, no inference profiles, and no direct model-invoke path live here — this
server does NOT call a foundation model. It is pure business I/O over DynamoDB; the agent
(the MCP client) is the only thing that talks to Bedrock, through relay.llm / Strands.
"""

from mcp_server import store  # noqa: F401  (re-export for `from mcp_server import store`)

__all__ = ["store", "server", "app"]
