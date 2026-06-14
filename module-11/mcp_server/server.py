"""mcp_server/server.py — the CloudCart MCP server (FastMCP), stateless.

Module 7. This is the **Model Context Protocol** server the agent connects to as a
CLIENT (skill 2.1.7). It exposes CloudCart's two business tools — `lookup_order` and
`create_ticket` — over MCP, backed by the DynamoDB data access in `store.py`.

Why MCP at all? Without a protocol, every agent needs a bespoke integration for every
tool: N agents × M tools of glue. MCP standardizes the wire — one client speaks to any
MCP server, tools are DISCOVERED at runtime (the agent lists the server's tools, it
does not hard-code them). Add `get_shipping_policy` here and the agent picks it up with
no code change (the lab's "Try it yourself" #1).

STATELESS, on purpose (the Lambda fit, skill 2.1.7): `stateless_http=True` means each
request is self-contained — no server-side MCP session to keep alive between calls — so
a short-lived Lambda invocation can serve a request and exit. The only durable state is
in DynamoDB. A heavy/stateful tool server (a warm model, a big cache) would run on ECS
instead — that is the article's theory contrast, not this server.

The two tools' DOCSTRINGS and TYPE HINTS are the tool SPEC the model sees (skill 2.1.6):
FastMCP turns the signature into a JSON schema and the docstring into the description,
exactly like a Strands @tool. Validation + clean, model-facing errors live in store.py.

Run it locally for dev (no Lambda, no deploy):
    uv run python -m mcp_server
    # serves the streamable-HTTP transport at http://127.0.0.1:8000/mcp
    # then point the agent at it:
    RELAY_MCP_URL=http://127.0.0.1:8000/mcp uv run python -m relay.agent "Where is order 1042?"

No foundation model is called here and no model ID appears — this is pure tool I/O.
"""

from __future__ import annotations

import json

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from mcp_server import store
from relay import config

# DNS-rebinding protection (a Host-header allowlist) is the MCP transport's defense for a
# server bound to localhost in a browser context. FastMCP turns it ON by default for the
# 127.0.0.1 dev bind, allowing ONLY localhost Host headers — which would reject the Lambda
# Function URL host (`<id>.lambda-url.us-east-1.on.aws`) with HTTP 421 "Invalid Host
# header". This server runs behind a Lambda Function URL (AWS controls routing and the
# host is a fixed AWS domain) or on localhost for dev — neither is the browser-rebinding
# threat the check guards against — so we disable it explicitly. Same object serves both
# the local `python -m mcp_server` and the Lambda, so it must accept either host.
_TRANSPORT_SECURITY = TransportSecuritySettings(enable_dns_rebinding_protection=False)

# One stateless FastMCP server. `stateless_http=True` is what makes it Lambda-friendly:
# no long-lived per-client MCP session is held server-side. The mount path matches
# config.MCP_SERVER_PATH so the client (relay.tools) and server agree by construction.
mcp = FastMCP(
    name="cloudcart-tools",
    instructions=(
        "CloudCart business tools for the Relay support agent: look up an order's real "
        "status, and create/persist a support ticket. Read-only on orders; writes only "
        "tickets."
    ),
    stateless_http=True,
    transport_security=_TRANSPORT_SECURITY,
)


@mcp.tool()
def lookup_order(order_id: str) -> str:
    """Look up the real status of a CloudCart order by its order id.

    Use this to answer "where is my order?" / "has my order shipped?" — anything that
    needs the live order book, which is NOT in the help-center docs. Returns the
    order's status, key dates, total, and line items.

    Args:
        order_id: the CloudCart order id, e.g. "1042" (a leading "#" is fine).

    Returns a human-readable summary of the order, or a clear message if no such order
    exists (so you can tell the customer to re-check the number).
    """
    try:
        order = store.lookup_order(order_id)
    except store.OrderNotFound as err:
        return str(err)
    except store.ToolInputError as err:
        return f"Cannot look up the order: {err}"
    return json.dumps(order, ensure_ascii=False)


@mcp.tool()
def create_ticket(
    ticket_id: str,
    status: str = "answered",
    summary: str | None = None,
) -> str:
    """Create/persist a CloudCart support ticket record for this conversation.

    Call this once, at the END, after you have answered or resolved the customer's
    request, to record what happened. Writing is idempotent on ticket_id (a retry
    overwrites the same row, never duplicates it).

    Args:
        ticket_id: the ticket's id (the record's primary key).
        status: the outcome — one of "received", "triaged", "answered", "failed".
            Use "answered" when you resolved it, "failed" if you could not.
        summary: a one-line note of what you did, for a human scanning the queue.

    Returns a confirmation that the ticket was stored (or a clear error if the input
    was invalid).
    """
    try:
        record = store.create_ticket(
            ticket_id, status=status, summary=summary
        )
    except store.ToolInputError as err:
        return f"Cannot create the ticket: {err}"
    except store.StoreError as err:
        return f"Failed to store the ticket: {err}"
    return (
        f"Ticket {record['ticket_id']} stored in {config.RELAY_TICKETS_TABLE} "
        f"with status {record['status']!r}."
    )


def main() -> int:
    """Serve the MCP server locally over streamable HTTP (dev only).

    Mounts the streamable-HTTP transport so a Strands MCP client can connect at
    http://127.0.0.1:8000/mcp. In production the SAME `mcp` object is served by the
    Lambda handler in app.py — no separate code path.
    """
    # FastMCP serves streamable HTTP at "/mcp" by default; that matches
    # config.MCP_SERVER_PATH. Host/port are dev defaults (127.0.0.1:8000).
    mcp.run(transport="streamable-http")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
