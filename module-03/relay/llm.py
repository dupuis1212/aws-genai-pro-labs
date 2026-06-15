"""relay/llm.py — Relay's FM integration layer. The UNIQUE Bedrock call site.

Module 3 of AWS GenAI Pro Mastery introduces — and FREEZES — this file. Every
generation call in Relay, now and for the rest of the course, goes through one
function:

    converse(messages, *, tier="auto", stream=False, **params)

This signature is byte-identical from Module 3 through Module 15. The body grows
by addition in later modules (image content blocks in M6, a guardrail parameter in
M9, prompt-caching/Flex selection via **params in M12) — but the signature never
changes, and `triage.py`, `kb.py`, `agent.py` and every other consumer depend on
exactly this shape.

What this layer does that a raw boto3 `converse()` call does not:

  - Routing. `tier="auto"` runs a small, explainable complexity router (see
    route()) that picks "fast" or "smart" per request. `tier="fast"` / "smart" /
    "frontier" force a tier. The tier -> inference profile mapping lives in
    relay/config.py ONLY — no model ID appears in this file.
  - Streaming. `stream=True` uses ConverseStream and yields text deltas as they
    arrive (time-to-first-token, for long answers in front of a human). `stream=
    False` returns the whole reply (for parsers like triage, where streaming buys
    nothing).
  - Resilience. Throttling and 5xx are retried with EXPONENTIAL BACKOFF + JITTER,
    never an immediate loop (an immediate loop makes throttling worse). After the
    retries on one profile are exhausted by throttling, the call FALLS BACK to the
    tier's alternate cross-Region profile when one exists, then degrades the tier
    (smart -> fast) as a last resort. No silent try/except: every failure path is
    explicit and surfaces a clear error.

A note on cross-Region inference: the `us.`/`global.` profile IDs in config.py
ALREADY route across the Regions in their geography for capacity — that is the
nominal mode. Our explicit "fallback to the alternate profile" is a second,
coarser lever for when an entire profile is being throttled, not the primary DR
mechanism.

Return shapes:
  - stream=False -> ConverseResult(text, tier, usage, stop_reason)
  - stream=True  -> a generator yielding text deltas; after exhaustion its
                    .result attribute holds the final ConverseResult (so callers
                    get tokens/cost without a second call).
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Iterator

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError

from relay import config

# --- Resilience tuning (taught explicitly, not buried) ------------------------
# Cap retries at 2 network retries beyond the first attempt. Past that you are
# burning tokens-of-latency on a loop that is not getting better; degrade or fail.
_MAX_RETRIES = 2

# Exponential backoff base and cap (seconds). Sleep = min(CAP, BASE * 2**attempt)
# then add jitter so a fleet of clients does not retry in lockstep (the classic
# thundering-herd that turns one throttle into a storm).
_BACKOFF_BASE = 0.5
_BACKOFF_CAP = 8.0

# Error codes that are worth retrying: throttling and transient server errors.
# Everything else (validation, access denied) is a bug or a permission problem —
# retrying it just wastes calls, so we raise immediately.
_RETRYABLE_CODES = frozenset(
    {
        "ThrottlingException",
        "TooManyRequestsException",
        "ServiceUnavailableException",
        "ModelTimeoutException",
        "InternalServerException",
        "ModelNotReadyException",
    }
)

# Keys this layer forwards from **params into the Converse request. A caller may pass
# inferenceConfig / system / toolConfig / additionalModelRequestFields; any OTHER key
# in **params is ignored (this layer owns modelId, messages, tiers, retries, backoff).
_INFERENCE_KEYS = {"inferenceConfig", "system", "toolConfig", "additionalModelRequestFields"}


class LLMError(RuntimeError):
    """Raised when a Converse call cannot be completed after retries and fallback.

    Carries the last underlying error so the failure is debuggable, never silent.
    """

    def __init__(self, message: str, *, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.__cause__ = cause


@dataclass
class RouteDecision:
    """Why the router picked a tier — surfaced in demos and logs, not hidden."""

    tier: str
    reason: str


@dataclass
class ConverseResult:
    """The non-streaming return value, and the streaming generator's final state."""

    text: str
    tier: str
    usage: dict[str, int]
    stop_reason: str
    route: RouteDecision | None = None


# --- The complexity router (tier="auto") --------------------------------------
# Heuristic, deliberately simple and EXPLAINABLE. The exam distinguishes:
#   - routing BY CONTENT/COMPLEXITY  (this): classify the request, then send it to
#     the right tier ONCE.
#   - model cascading                (not this): always try the small model, then
#     ESCALATE on a poor/failed result — a second call on failure.
# Relay routes; it does not cascade. Cascading doubles latency on the hard cases.
#
# Routing-by-metrics (pick the tier from live latency/error telemetry) stays
# theory in Module 3 — this router reads the request, not a metrics feed.

# Words that signal a request needs reasoning, multi-step work, or careful
# wording in front of a customer -> escalate to the smart tier.
_SMART_KEYWORDS = (
    "refund",
    "charged twice",
    "double charge",
    "duplicate charge",
    "dispute",
    "chargeback",
    "cancel my subscription",
    "explain",
    "why was",
    "why am i",
    "reconcile",
    "discrepancy",
    "compare",
    "step by step",
    "troubleshoot",
    "not working after",
    "integration",
    "api error",
    "webhook",
)

# Above this many characters, a request is likely a detailed problem report that
# benefits from the smart tier. Short tickets ("hi", "where is my order?") are
# fast-tier work.
_LONG_REQUEST_CHARS = 320


def _last_user_text(messages: list[dict[str, Any]]) -> str:
    """Concatenate the text blocks of the final user message (for routing only)."""
    for message in reversed(messages):
        if message.get("role") == "user":
            parts = [
                block.get("text", "")
                for block in message.get("content", [])
                if "text" in block
            ]
            return " ".join(parts)
    return ""


def route(messages: list[dict[str, Any]]) -> RouteDecision:
    """Pick "fast" or "smart" for tier="auto" — and say why.

    Pure, deterministic, and unit-testable on fixed inputs (no model call). The
    floor is "fast" (~80% of CloudCart tickets); we escalate to "smart" only when
    a keyword or length signal says the request needs reasoning.
    """
    text = _last_user_text(messages)
    lowered = text.lower()

    for keyword in _SMART_KEYWORDS:
        if keyword in lowered:
            return RouteDecision("smart", f"matched complexity keyword {keyword!r}")

    if len(text) >= _LONG_REQUEST_CHARS:
        return RouteDecision(
            "smart", f"long request ({len(text)} chars >= {_LONG_REQUEST_CHARS})"
        )

    return RouteDecision(
        config.DEFAULT_TIER, "no complexity signal — default fast tier"
    )


# --- Client cache -------------------------------------------------------------
# One bedrock-runtime client per process. We disable botocore's own adaptive
# retries (mode="standard", max_attempts=1) because THIS layer owns the retry
# policy — teaching backoff explicitly, not delegating it invisibly to the SDK.
_clients: dict[str, Any] = {}


def _runtime_client():
    client = _clients.get("runtime")
    if client is None:
        client = boto3.client(
            "bedrock-runtime",
            region_name=config.REGION,
            config=BotoConfig(retries={"max_attempts": 1, "mode": "standard"}),
        )
        _clients["runtime"] = client
    return client


def _backoff_sleep(attempt: int) -> None:
    """Sleep exponentially with full jitter before retry `attempt` (1-indexed)."""
    ceiling = min(_BACKOFF_CAP, _BACKOFF_BASE * (2 ** attempt))
    time.sleep(random.uniform(0, ceiling))


def _resolve_tier(tier: str, messages: list[dict[str, Any]]) -> tuple[str, RouteDecision | None]:
    """Turn the requested tier into a concrete tier + the route reason (if auto)."""
    if tier == "auto":
        decision = route(messages)
        return decision.tier, decision
    # An explicit tier ("fast"/"smart"/"frontier") is validated by tier_profile().
    config.tier_profile(tier)  # raises ValueError on a typo
    return tier, None


def _profile_chain(tier: str) -> list[str]:
    """Profiles to try for a tier, in order: primary, then alternate (if any)."""
    chain = [config.tier_profile(tier)]
    alt = config.ALT_PROFILES.get(tier)
    if alt and alt not in chain:
        chain.append(alt)
    return chain


def _degrade(tier: str) -> str | None:
    """The graceful-degradation step: smart -> fast. Fast has nowhere to fall."""
    return "fast" if tier == "smart" else None


def _request_kwargs(modelId: str, messages: list[dict[str, Any]],
                    params: dict[str, Any]) -> dict[str, Any]:
    """Assemble the Converse/ConverseStream request from messages + passthrough params."""
    kwargs: dict[str, Any] = {"modelId": modelId, "messages": messages}
    for key, value in params.items():
        if key in _INFERENCE_KEYS:
            kwargs[key] = value
    return kwargs


def _call_once(modelId: str, messages: list[dict[str, Any]],
               params: dict[str, Any]) -> dict[str, Any]:
    """One non-streaming Converse call against one profile (no retry here)."""
    client = _runtime_client()
    return client.converse(**_request_kwargs(modelId, messages, params))


def _with_resilience(invoke, *, tier: str):
    """Run `invoke(profile)` with backoff+jitter, profile fallback, then degrade.

    `invoke` takes an inference-profile ID and returns whatever the call returns.
    The retry/fallback policy is identical for streaming and non-streaming; only
    the inner `invoke` differs. Raises LLMError when every avenue is exhausted —
    never swallows the failure.
    """
    tried_tier = tier
    last_error: Exception | None = None

    while tried_tier is not None:
        for profile in _profile_chain(tried_tier):
            for attempt in range(_MAX_RETRIES + 1):
                try:
                    return invoke(profile), tried_tier
                except ClientError as err:
                    code = err.response["Error"]["Code"]
                    last_error = err
                    if code not in _RETRYABLE_CODES:
                        # Validation/access errors are not transient. Surface the
                        # real cause (often the bare-ID trap) instead of retrying.
                        raise LLMError(
                            f"Converse failed on profile {profile} "
                            f"({code}): {err.response['Error']['Message']}",
                            cause=err,
                        ) from err
                    if attempt < _MAX_RETRIES:
                        _backoff_sleep(attempt + 1)
                    # else: fall through to the next profile in the chain.
            # This profile is exhausted by throttling/5xx; try the next profile.
        # All profiles for this tier exhausted; degrade the tier (smart -> fast).
        tried_tier = _degrade(tried_tier)

    raise LLMError(
        f"Converse exhausted retries, cross-Region fallback, and tier degradation "
        f"for tier {tier!r}. Last error: {last_error}",
        cause=last_error,
    )


def _stream_deltas(response: dict[str, Any], result_box: ConverseResult) -> Iterator[str]:
    """Yield text deltas from a ConverseStream response; fill result_box at the end."""
    text_parts: list[str] = []
    for event in response["stream"]:
        if "contentBlockDelta" in event:
            delta = event["contentBlockDelta"]["delta"]
            if "text" in delta:
                chunk = delta["text"]
                text_parts.append(chunk)
                yield chunk
        elif "messageStop" in event:
            result_box.stop_reason = event["messageStop"].get("stopReason", "")
        elif "metadata" in event:
            usage = event["metadata"].get("usage", {})
            result_box.usage = {
                "inputTokens": usage.get("inputTokens", 0),
                "outputTokens": usage.get("outputTokens", 0),
                "totalTokens": usage.get("totalTokens", 0),
            }
    result_box.text = "".join(text_parts)


class _StreamingResult:
    """A generator of text deltas that also exposes the final ConverseResult.

    Iterate it for tokens; read `.result` after iteration for tokens/cost. This
    keeps `converse(stream=True)` a single call with a clean return contract.
    """

    def __init__(self, gen: Iterator[str], result: ConverseResult) -> None:
        self._gen = gen
        self.result = result

    def __iter__(self) -> Iterator[str]:
        return self._gen


def converse(messages, *, tier="auto", stream=False, **params):
    """Relay's single Bedrock entrypoint. Signature FROZEN M3 -> M15.

    Args:
        messages: Converse-format messages, e.g.
            [{"role": "user", "content": [{"text": "..."}]}].
        tier: "auto" (complexity router picks fast/smart), or an explicit
            "fast" / "smart" / "frontier".
        stream: False -> ConverseResult; True -> a _StreamingResult you iterate
            for text deltas, with `.result` holding the final ConverseResult.
        **params: passed through to Converse — inferenceConfig, system, toolConfig,
            additionalModelRequestFields. (Later modules read more keys here, e.g.
            a guardrail in M9 and caching/Flex in M12 — by addition, signature
            unchanged.)

    Raises:
        LLMError on a non-retryable error, or after retries + cross-Region
        fallback + tier degradation are all exhausted. Never a silent pass.
    """
    concrete_tier, decision = _resolve_tier(tier, messages)

    if stream:
        result = ConverseResult(text="", tier=concrete_tier, usage={}, stop_reason="",
                                route=decision)

        def invoke_stream(profile: str) -> dict[str, Any]:
            client = _runtime_client()
            return client.converse_stream(**_request_kwargs(profile, messages, params))

        response, used_tier = _with_resilience(invoke_stream, tier=concrete_tier)
        result.tier = used_tier
        return _StreamingResult(_stream_deltas(response, result), result)

    def invoke(profile: str) -> dict[str, Any]:
        return _call_once(profile, messages, params)

    response, used_tier = _with_resilience(invoke, tier=concrete_tier)
    text = response["output"]["message"]["content"][0]["text"]
    usage = response.get("usage", {})
    return ConverseResult(
        text=text,
        tier=used_tier,
        usage={
            "inputTokens": usage.get("inputTokens", 0),
            "outputTokens": usage.get("outputTokens", 0),
            "totalTokens": usage.get("totalTokens", 0),
        },
        stop_reason=response.get("stopReason", ""),
        route=decision,
    )
