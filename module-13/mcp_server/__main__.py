"""`python -m mcp_server` — run the CloudCart MCP server locally (dev).

Serves the SAME FastMCP server the Lambda hosts, over streamable HTTP at
http://127.0.0.1:8000/mcp. Point the agent at it with:

    RELAY_MCP_URL=http://127.0.0.1:8000/mcp uv run python -m relay.agent "Where is order 1042?"
"""

from mcp_server.server import main

raise SystemExit(main())
