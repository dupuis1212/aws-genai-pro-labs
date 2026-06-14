"""mcp_server/app.py — the AWS Lambda entrypoint for the CloudCart MCP server.

Module 7. setup.py packages `mcp_server/` + its deps into a Lambda whose handler is
`mcp_server.app.handler`, fronted by a **Lambda Function URL**. The agent's MCP client
(relay.tools) connects to that URL and discovers `lookup_order` / `create_ticket`.

The MCP server itself (server.py) is a Starlette ASGI app (FastMCP's streamable-HTTP
transport). A Lambda Function URL delivers HTTP as a payload-v2 event, so this module
is a tiny, self-contained **Function-URL → ASGI adapter**: it translates the event into
an ASGI `http` scope, drives the ASGI app, and translates the response back. Keeping the
adapter here (no extra dependency) is deliberate — the only NEW runtime deps Module 7
adds are `strands-agents` and `mcp`; we do not pull in a web-adapter package.

FastMCP's streamable-HTTP transport runs a background task group started by the ASGI
**lifespan**. So this module keeps ONE persistent event loop per warm container, starts
the lifespan once, and submits each request to it (a cold start pays the startup; warm
invocations reuse it). The server is STATELESS (server.py sets `stateless_http=True`), so
no per-request session state crosses invocations.

Why this is Lambda-shaped (skill 2.1.7): each Function-URL request is self-contained and
a warm/cold Lambda serves it and idles at ~$0. Heavy/stateful tool servers belong on ECS
(theory, not built).

No model ID, no Bedrock call here — the Lambda runs DynamoDB I/O only, under an IAM role
bounded to relay-orders (read) + relay-tickets (write) (setup.py, skill 2.1.3).
"""

from __future__ import annotations

import asyncio
import base64
import threading

from mcp_server.server import mcp

# Build the streamable-HTTP ASGI app ONCE per container (warm-start reuse). FastMCP
# mounts the transport at "/mcp" (config.MCP_SERVER_PATH).
_asgi_app = mcp.streamable_http_app()


class _AppRunner:
    """Owns one event loop + a started ASGI lifespan, reused across warm invocations.

    FastMCP's streamable-HTTP session manager needs its lifespan task group running, so
    we start the lifespan once on a dedicated loop thread and drive each request on that
    same loop. Thread-safe: the handler submits coroutines via run_coroutine_threadsafe.
    """

    def __init__(self, app) -> None:
        self._app = app
        self._loop = asyncio.new_event_loop()
        self._lifespan_started = threading.Event()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._lifespan_started.wait(timeout=30)

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._start_lifespan())
        self._loop.run_forever()

    async def _start_lifespan(self) -> None:
        # Drive the ASGI lifespan startup so the session manager's task group is live.
        self._lifespan_send: asyncio.Queue = asyncio.Queue()
        self._lifespan_recv: asyncio.Queue = asyncio.Queue()

        async def receive():
            return await self._lifespan_recv.get()

        async def send(message):
            await self._lifespan_send.put(message)

        scope = {"type": "lifespan", "asgi": {"version": "3.0"}}
        # Run the lifespan in the background; it stays alive for the loop's lifetime.
        self._loop.create_task(self._app(scope, receive, send))
        await self._lifespan_recv.put({"type": "lifespan.startup"})
        msg = await self._lifespan_send.get()  # wait for startup complete/failed
        if msg["type"] == "lifespan.startup.failed":
            raise RuntimeError(f"ASGI lifespan startup failed: {msg.get('message')}")
        self._lifespan_started.set()

    def handle(self, scope, receive, send) -> None:
        """Run one HTTP request to completion on the persistent loop (blocking)."""
        future = asyncio.run_coroutine_threadsafe(
            self._app(scope, receive, send), self._loop
        )
        future.result(timeout=29)


# One runner per warm container (lazily started on the first invocation).
_runner: _AppRunner | None = None


def _get_runner() -> _AppRunner:
    global _runner
    if _runner is None:
        _runner = _AppRunner(_asgi_app)
    return _runner


def handler(event: dict, context=None) -> dict:
    """AWS Lambda handler for a Function URL (payload format 2.0).

    Adapts the Function-URL HTTP event to the Starlette ASGI app (driven on the runner's
    persistent loop, so FastMCP's lifespan task group is live) and back.
    """
    http = event.get("requestContext", {}).get("http", {})
    method = http.get("method", "GET")
    raw_path = event.get("rawPath", "/") or "/"
    raw_query = event.get("rawQueryString", "") or ""

    headers = event.get("headers", {}) or {}
    scope_headers = [
        (k.lower().encode("latin-1"), str(v).encode("latin-1"))
        for k, v in headers.items()
    ]

    body = event.get("body", "") or ""
    body_bytes = base64.b64decode(body) if event.get("isBase64Encoded") \
        else body.encode("utf-8")

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": method,
        "scheme": "https",
        "path": raw_path,
        "raw_path": raw_path.encode("utf-8"),
        "query_string": raw_query.encode("utf-8"),
        "headers": scope_headers,
        "server": ("lambda", 443),
        "client": ("lambda", 0),
    }

    response: dict = {"status": 500, "headers": [], "body": b""}
    body_iter = iter((
        {"type": "http.request", "body": body_bytes, "more_body": False},
    ))

    async def receive() -> dict:
        try:
            return next(body_iter)
        except StopIteration:
            return {"type": "http.disconnect"}

    async def send(message: dict) -> None:
        if message["type"] == "http.response.start":
            response["status"] = message["status"]
            response["headers"] = message.get("headers", [])
        elif message["type"] == "http.response.body":
            response["body"] += message.get("body", b"")

    _get_runner().handle(scope, receive, send)

    out_headers: dict[str, str] = {
        raw_key.decode("latin-1"): raw_val.decode("latin-1")
        for raw_key, raw_val in response["headers"]
    }

    raw_body: bytes = response["body"]
    try:
        text_body = raw_body.decode("utf-8")
        is_b64 = False
    except UnicodeDecodeError:
        text_body = base64.b64encode(raw_body).decode("ascii")
        is_b64 = True

    return {
        "statusCode": response["status"],
        "headers": out_headers,
        "body": text_body,
        "isBase64Encoded": is_b64,
    }
