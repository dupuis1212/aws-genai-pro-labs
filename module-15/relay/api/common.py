"""relay/api/common.py — shared HTTP plumbing for the four Relay API Lambda handlers.

Module 11. API Gateway hands a Lambda an event dict and expects a response dict
(`{statusCode, headers, body}`); SQS hands the worker a `{Records: [...]}` batch. This
module is the ONE place that shape lives, so the four handlers stay short and the smoke
test can build the same events the test asserts against.

It also carries the REQUEST-VALIDATION helpers (skill 2.4.1): API Gateway does a first,
cheap structural validation with a JSON-Schema request model (defined in the CDK stack,
so a malformed payload is rejected at the edge BEFORE a Lambda cold-starts). These helpers
are the SECOND layer — they parse + validate the body into the frozen Ticket schema and
return a CLEAN 400 on bad input, never a stack trace. Defence in depth: the model catches
the obvious junk, the handler enforces the business contract.

No foundation-model call, no model ID, no AWS resource name here — pure request/response
adaptation. The handlers import the resource names from relay.config.
"""

from __future__ import annotations

import json
from typing import Any

# A neutral CORS/JSON header set every response carries. A browser-based CloudCart console
# (the article's Amplify survey) needs CORS; the lab keeps it permissive (the WAF / tighter
# origin policy is the "In production" note, not wired here).
JSON_HEADERS = {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*",
}


def response(status_code: int, body: dict | list) -> dict:
    """Build an API Gateway proxy response (the `{statusCode, headers, body}` shape).

    `body` is JSON-encoded; the handler passes a plain dict/list. This is the SOLE place
    the proxy-integration response shape is built, so every handler answers consistently.
    """
    return {
        "statusCode": status_code,
        "headers": dict(JSON_HEADERS),
        "body": json.dumps(body),
    }


def error(status_code: int, message: str, **extra: Any) -> dict:
    """A clean error response: `{"error": "<message>", ...}` with the given status.

    Used for 400 (bad request / validation), 404 (no such ticket), 409 (not in a state
    that can be approved), 500 (unexpected). The message is meant for the API caller —
    actionable, never a raw exception or stack trace (the handler logs the detail).
    """
    payload = {"error": message}
    payload.update(extra)
    return response(status_code, payload)


class BadRequest(ValueError):
    """A request body failed validation. The handler turns it into a clean 400.

    The message is caller-facing (e.g. "channel must be 'email' or 'chat'"), so the API
    consumer can fix the request — the second validation layer behind API Gateway's own
    JSON-Schema request model.
    """


def parse_json_body(event: dict) -> dict:
    """Return the request body as a dict (decoding base64 if API Gateway encoded it).

    API Gateway proxy events carry the body as a STRING under `event["body"]` (or None for
    an empty body), with `isBase64Encoded` set when binary. A missing/blank body is a
    BadRequest; non-object JSON (a bare list/number) is a BadRequest — the endpoints all
    expect a JSON object. This never raises a bare JSONDecodeError; it raises BadRequest
    with a clean message the handler returns as 400.
    """
    raw = event.get("body")
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        raise BadRequest("Request body is required and must be a JSON object.")
    if event.get("isBase64Encoded"):
        import base64

        try:
            raw = base64.b64decode(raw).decode("utf-8")
        except Exception as err:  # noqa: BLE001 — turn any decode failure into a 400.
            raise BadRequest("Request body is not valid base64-encoded UTF-8.") from err
    if isinstance(raw, (dict, list)):
        parsed = raw  # already-decoded (a direct invoke / test passes a dict)
    else:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as err:
            raise BadRequest(f"Request body is not valid JSON: {err.msg}.") from err
    if not isinstance(parsed, dict):
        raise BadRequest("Request body must be a JSON object, not a list or scalar.")
    return parsed


def path_param(event: dict, name: str) -> str:
    """Return a required path parameter (e.g. {ticket_id}) or raise BadRequest.

    API Gateway proxy events put `{ticket_id}` under `event["pathParameters"]`. A missing
    parameter is a routing/config bug surfaced as a clean 400 rather than a KeyError.
    """
    params = event.get("pathParameters") or {}
    value = params.get(name)
    if not value or not str(value).strip():
        raise BadRequest(f"Path parameter {name!r} is required.")
    return str(value).strip()
