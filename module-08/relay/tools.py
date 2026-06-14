"""relay/tools.py — Relay's agent tools: one local, two over MCP.

Module 7 of AWS GenAI Pro Mastery. An agent is only as useful as the tools it can
call. Relay gets exactly three (canonical names, 06 §5.4 — no synonyms):

  - search_kb(query)          LOCAL Strands @tool over the Bedrock Knowledge Base
                              `relay-kb` (relay.kb.retrieve). This is the 1.5.6 pattern:
                              RETRIEVAL EXPOSED AS A STANDARDIZED TOOL the model can
                              choose to call. It stays a local tool (not on MCP) on
                              purpose — it is Relay's own read path into its own KB,
                              with no business side effect.
  - lookup_order(order_id)    served by the CloudCart MCP SERVER (mcp_server/, on
  - create_ticket(...)        Lambda). These TWO business tools — read the order book,
                              write a ticket — move onto **MCP** (skill 2.1.7): a
                              standardized client/server wire, tools DISCOVERED at
                              runtime, so the same agent could consume any MCP server.

WHAT MAKES A TOOL RELIABLE (skills 2.1.6 + 2.1.3), shown on search_kb here and on the
MCP tools in mcp_server/store.py:

  - The DOCSTRING is the spec and the TYPE HINTS are the schema. Strands' @tool turns
    `search_kb(query: str) -> str` + this docstring into the tool definition the model
    sees — no hand-written JSON schema, one source of truth.
  - PARAMETERS ARE VALIDATED and ERRORS ARE RETURNED TO THE MODEL, cleanly. A blank
    query or a KB that is not set up does NOT crash the agent loop with a stack trace;
    the tool returns a short, model-facing message ("no query given", "KB not set up")
    so the model can recover (ask the user, try another tool, give up gracefully). No
    silent `try/except` that hides the failure.

No model ID and no direct model-invoke path appear here — search_kb retrieves through
relay.kb (the Retrieve API), and the MCP tools do pure DynamoDB I/O. The agent
(relay.agent) is the only thing that calls a foundation model.
"""

from __future__ import annotations

import contextlib

from strands import tool

from relay import config, kb

# How many KB passages search_kb returns to the model. Small: the model reads a few
# grounded snippets and cites them, it does not need the whole corpus. A name in one
# place so the tool and the lab agree.
SEARCH_KB_TOP_K = 4


# =============================================================================
# search_kb — the LOCAL retrieval tool (skill 1.5.6: retrieval as a standardized tool).
# =============================================================================
@tool
def search_kb(query: str) -> str:
    """Search CloudCart's help-center documentation for an answer.

    Use this for HOW-TO and POLICY questions — refunds, plan changes, password resets,
    error codes, shipping policy — anything that is explained in the docs. Do NOT use
    it to look up a specific customer's order status; that is `lookup_order`.

    Args:
        query: the customer's question, in natural language.

    Returns the most relevant documentation passages, each with its source URI, for you
    to answer from and cite. Returns a short note if nothing relevant is found.
    """
    if not query or not query.strip():
        return "No query was given. Pass the customer's question text to search_kb."

    try:
        passages = kb.retrieve(query.strip(), top_k=SEARCH_KB_TOP_K)
    except kb.KBError as err:
        # A model-facing, recoverable message — not a crash. Most common first-run
        # cause: the KB is not set up yet (run setup.py / Module 5's setup).
        return (
            "The knowledge base could not be searched right now "
            f"({err}). Answer from what you already know, or ask the customer for "
            "more detail."
        )

    if not passages:
        return (
            f"No documentation found for {query.strip()!r}. There may be no help-center "
            "article on this; tell the customer you could not find a documented answer."
        )

    lines: list[str] = []
    for i, passage in enumerate(passages, 1):
        snippet = " ".join(passage.text.split())
        if len(snippet) > 600:
            snippet = snippet[:597] + "..."
        source = passage.source_uri or "(unknown source)"
        lines.append(f"[{i}] {source}\n{snippet}")
    return "\n\n".join(lines)


# =============================================================================
# The MCP client — discover lookup_order / create_ticket from the CloudCart server.
# =============================================================================
# The two BUSINESS tools live on the MCP server (mcp_server/, on Lambda). Relay is the
# MCP CLIENT: it opens a streamable-HTTP connection to the server's URL, LISTS the
# tools the server advertises, and hands them to the Strands agent. Tools are
# DISCOVERED, not hard-coded — add `get_shipping_policy` on the server and the agent
# sees it with no change here (the lab's "Try it yourself" #1).
def mcp_client(url: str | None = None):
    """Build a Strands MCP client for the CloudCart MCP server (not yet connected).

    The URL is resolved through relay.config.resolve_mcp_url (explicit arg ->
    RELAY_MCP_URL env -> the .mcp_url file setup.py writes after deploying the Lambda;
    set RELAY_MCP_URL=http://127.0.0.1:8000/mcp to use a local `python -m mcp_server`).

    Returns a strands MCPClient. Use it as a context manager so the connection is
    opened and torn down cleanly:

        with mcp_client() as client:
            tools = client.list_tools_sync()
            ...

    Imports of the streamable-HTTP transport are deferred to call time so importing
    relay.tools stays light and offline (the smoke test imports this module without a
    network).
    """
    from mcp.client.streamable_http import streamablehttp_client
    from strands.tools.mcp import MCPClient

    resolved = config.resolve_mcp_url(url)
    # The transport callable is what MCPClient invokes to open the connection. We close
    # over the resolved URL; Strands manages the async lifecycle on a background thread.
    return MCPClient(lambda: streamablehttp_client(resolved))


@contextlib.contextmanager
def mcp_business_tools(url: str | None = None):
    """Context manager yielding the discovered MCP business tools (list).

    Opens the MCP client, lists the server's tools, and yields them as Strands tool
    objects ready to hand to an Agent. The connection stays open for the duration of
    the `with` block (the agent must call the tools INSIDE it), then is torn down.

        with mcp_business_tools() as biz_tools:
            agent = build_agent(extra_tools=biz_tools)
            agent("Where is order 1042?")

    On a connection failure it raises (the URL is wrong, or the server/Lambda is down)
    — an explicit failure the caller surfaces, never a silent empty tool list.
    """
    client = mcp_client(url)
    with client:
        yield client.list_tools_sync()


# The canonical tool names this module is responsible for wiring (06 §5.4). Exposed so
# the agent and the tests can assert the contract without magic strings.
LOCAL_TOOL_NAMES = ("search_kb",)
MCP_TOOL_NAMES = ("lookup_order", "create_ticket")
ALL_TOOL_NAMES = LOCAL_TOOL_NAMES + MCP_TOOL_NAMES
