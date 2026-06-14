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

MODULE 6 ADDITION — multimodal (image) content blocks. The Converse API carries
TEXT and IMAGES in the SAME message: a `content` list can hold a `{"text": ...}`
block and an `{"image": {"format, "source": {"bytes": ...}}}` block side by side.
The `converse()` SIGNATURE does not change — image blocks ride inside `messages`,
which already flow through unchanged. This module just adds the helper
`image_block(data, media_type)` so the ONE place that knows the Converse image
shape (and the admitted formats) is this LLM layer, not scattered across callers
(relay.intake builds its vision message with it). This is the Converse-native path
— NOT the legacy single-prompt invoke path with base64 in a model-specific body,
which 07 §3.3 forbids; with an SDK you pass RAW bytes and boto3 base64-encodes them.
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

# Keys this layer manages itself — callers pass everything else through **params
# straight into the Converse request (e.g. inferenceConfig, system, toolConfig).
_INFERENCE_KEYS = {"inferenceConfig", "system", "toolConfig", "additionalModelRequestFields"}

# --- Module 9: the guardrail parameter (by addition; signature UNCHANGED) ------
# converse()'s SIGNATURE is frozen byte-identical M3->M15, so the Module 9 guardrail
# rides in through the existing **params — a caller passes `guardrail=<id>` (and
# optionally `guardrail_version=`, `guardrail_trace=`) and this layer translates it
# into the Converse/ConverseStream `guardrailConfig` block. This is the IN-LINE attach
# mode: Bedrock evaluates the guardrail on the INPUT before the model sees it and on the
# OUTPUT before it returns — one round trip, both directions. (The STANDALONE mode —
# ApplyGuardrail on arbitrary text, e.g. the KB grounding check — lives in
# relay/safety.py, the only other tolerated bedrock-runtime caller.)
#
# These three keys are CONSUMED here (popped out of params) so they never leak into the
# raw Converse request as unknown members; everything else still passes through. Passing
# no `guardrail` leaves the call exactly as it was before Module 9 — additive, opt-in.
_GUARDRAIL_PARAM = "guardrail"
_GUARDRAIL_VERSION_PARAM = "guardrail_version"
_GUARDRAIL_TRACE_PARAM = "guardrail_trace"

# --- Module 12: prompt caching + the service tier (by addition; signature UNCHANGED) ---
# Module 12 (the token economy) adds two cost levers to converse() the SAME additive way
# Module 9 added the guardrail: they ride in through the EXISTING **params, never a new
# positional/keyword in the frozen signature, never a parallel bedrock-runtime client.
#
#   cache_prompt=True : insert a Converse CACHE POINT after the system prompt, so Bedrock
#       caches that reused PREFIX provider-side and bills it at ~10% on the next call
#       (PROMPT_CACHE_INPUT_DISCOUNT). This is the ONLY M12 lever wired onto the interactive
#       path — it caches reused INPUT, so it can NEVER serve a stale answer (the prompt-vs-
#       semantic-cache distinction the exam tests). The usage block then reports
#       cacheReadInputTokens / cacheWriteInputTokens, which the cost line consumes.
#   service_tier="flex"|"standard"|"priority" : the Bedrock service tier (re:Invent 2025,
#       -50% on FLEX). Relay's interactive path stays on DEFAULT_SERVICE_TIER (standard);
#       FLEX is for the eval/backfill path ONLY (brief §9 — never on interactive traffic).
#       Translated here into the Converse top-level `serviceTier` member (its `type` enum:
#       priority|default|flex|reserved; "standard" maps to the API's "default") and popped
#       out of params so it never reaches the raw request as an unknown key.
#
# Both keys are CONSUMED here (popped out of params) so they never reach the raw Converse
# request as unknown members; everything else still passes through. Passing neither leaves
# the call byte-identical to its Module 9 behaviour — additive, opt-in.
_CACHE_PROMPT_PARAM = "cache_prompt"
_SERVICE_TIER_PARAM = "service_tier"

# An INTERNAL marker key (never sent to Converse) that _call_once writes back onto the caller's
# params dict when it had to DEGRADE an unsupported service tier to the model's default — so
# converse() can bill the call at the tier it ACTUALLY ran on, not the one it asked for. It is
# stripped from the request in _request_kwargs alongside the other consumed **params keys.
_ACTUAL_SERVICE_TIER_KEY = "_actual_service_tier"

# The Converse cache-point block. Bedrock reads `{"cachePoint": {"type": "default"}}` as a
# checkpoint: everything BEFORE it (the system prompt / the reused context) is the cacheable
# prefix. We attach it to the END of the `system` field so Relay's long, reused system
# prompt is the cached prefix — the highest-leverage place (it is identical across tickets).
_CACHE_POINT_BLOCK = {"cachePoint": {"type": "default"}}


# The Bedrock Converse API expresses the service tier on the TOP-LEVEL `serviceTier`
# structure, via its `type` member (enum: priority | default | flex | reserved) — NOT on
# `performanceConfig.latency`, which is a different knob (latency optimization: standard |
# optimized). The course tier names (standard | flex | priority) map onto the API enum below;
# "standard" is the API's "default". (Live-verified June 2026 against the botocore Converse
# model — the earlier `performanceConfig.latency` mapping was rejected with a ValidationException
# because that field only accepts [standard, optimized].)
_SERVICE_TIER_TO_API_TYPE: dict[str, str] = {
    config.SERVICE_TIER_STANDARD: "default",
    config.SERVICE_TIER_FLEX: "flex",
    config.SERVICE_TIER_PRIORITY: "priority",
}


def _service_tier_config(params: dict[str, Any]) -> dict[str, Any] | None:
    """Build the Converse `serviceTier` block from the service_tier **param, or None.

    Pops `service_tier` out of `params` (so it never reaches the raw Converse request as an
    unknown key). Returns the `serviceTier` structure Bedrock expects ({"type": <api enum>}),
    or None when the caller asked for the default Standard tier (the interactive path) — in
    which case the request is byte-identical to its pre-M12 shape. FLEX (-50%) is for
    latency-tolerant eval/backfill jobs ONLY; the caller (cost_report.py / the batch path /
    the M13 judge) opts in, the interactive worker never does (brief §9). A typo'd tier raises
    immediately, never a silent fallback to the wrong (billed) tier.
    """
    tier = params.pop(_SERVICE_TIER_PARAM, None)
    if tier is None or tier == config.DEFAULT_SERVICE_TIER:
        return None
    api_type = _SERVICE_TIER_TO_API_TYPE.get(tier)
    if api_type is None:
        raise ValueError(
            f"Unknown service_tier {tier!r}. Known: "
            f"{', '.join(sorted(_SERVICE_TIER_TO_API_TYPE))}. "
            "FLEX (-50%) is for latency-tolerant eval/backfill jobs only, never "
            "Relay's interactive traffic."
        )
    return {"type": api_type}


def _apply_prompt_cache(kwargs: dict[str, Any], params: dict[str, Any]) -> None:
    """Insert a Converse CACHE POINT at the end of the system prompt when cache_prompt=True.

    Pops `cache_prompt` out of `params`. When truthy AND a `system` field is present, appends
    `{"cachePoint": {"type": "default"}}` to the system blocks so the reused system prefix is
    cached provider-side (≈ -90% on the cached input on the next call). Caching an INPUT
    prefix never staleness an answer — that is what separates prompt caching from the
    semantic cache (relay/cache.py). A no-op when cache_prompt is falsy or there is no system
    prompt to cache, so the request stays byte-identical for callers that do not opt in.
    """
    want_cache = params.pop(_CACHE_PROMPT_PARAM, False)
    if not want_cache:
        return
    system = kwargs.get("system")
    if not system:
        return  # nothing reusable to cache (no system prefix) — leave the request as-is.
    # Copy so we never mutate the caller's list across retries/fallback profiles.
    blocks = list(system)
    if blocks and blocks[-1] == _CACHE_POINT_BLOCK:
        return  # already has a cache point at the end (idempotent).
    blocks.append(dict(_CACHE_POINT_BLOCK))
    kwargs["system"] = blocks


def _usage_dict(usage: dict[str, Any]) -> dict[str, int]:
    """Normalize a Converse `usage` block into the result's token counts.

    Module 12 ADDS (by addition) the two prompt-caching counters Bedrock reports when a
    cache point is in play, so the cost line can bill cached input at the discounted rate:

      - cacheReadInputTokens  : input tokens served FROM the prompt cache (≈ -90%);
      - cacheWriteInputTokens : input tokens written INTO the cache on a cold call (these
                                count as normal input; the SAVING lands on the NEXT call).

    Both default to 0, so a non-cached call's usage dict carries the same three core keys it
    always did (input/output/total) — additive, never a shape break for older readers.
    """
    return {
        "inputTokens": usage.get("inputTokens", 0),
        "outputTokens": usage.get("outputTokens", 0),
        "totalTokens": usage.get("totalTokens", 0),
        # ADDED M12 (default 0 when no prompt cache is in play):
        "cacheReadInputTokens": usage.get("cacheReadInputTokens", 0),
        "cacheWriteInputTokens": usage.get("cacheWriteInputTokens", 0),
    }


def _guardrail_config(params: dict[str, Any]) -> dict[str, Any] | None:
    """Build the Converse `guardrailConfig` from the guardrail-related **params, or None.

    Pops `guardrail` / `guardrail_version` / `guardrail_trace` out of `params` (so they
    do not reach the raw Converse request as unknown keys) and assembles the
    guardrailConfig block Bedrock expects. Returns None when no `guardrail` was passed —
    the call then behaves exactly as it did before Module 9 (additive, opt-in).

    The version defaults through relay.config.resolve_guardrail_version (the published
    version, never DRAFT for traffic); trace defaults to "enabled" so an intervention's
    reason is inspectable in demos/logs (the lab prints which layer caught what).
    """
    guardrail_id = params.pop(_GUARDRAIL_PARAM, None)
    version = params.pop(_GUARDRAIL_VERSION_PARAM, None)
    trace = params.pop(_GUARDRAIL_TRACE_PARAM, None)
    if not guardrail_id:
        return None
    return {
        "guardrailIdentifier": guardrail_id,
        "guardrailVersion": config.resolve_guardrail_version(version),
        # "enabled" surfaces the assessment trace so a caller can SEE which policy
        # intervened (content filter vs denied topic vs prompt-attack vs grounding).
        "trace": trace or "enabled",
    }


# --- Module 6: multimodal (image) content blocks ------------------------------
# The Converse ImageBlock format enum (verified against the botocore bedrock-runtime
# model, boto3 1.43): one of png / jpeg / gif / webp. We map the MIME types the
# intake gate admits onto these Converse format strings, so there is ONE source of
# truth for "which image types Relay accepts" and it lives in this LLM layer.
#
# The vision model itself (Amazon Nova Lite, the multimodal/vision profile in
# config.py) is resolved by tier exactly like every other call — no model ID here.
# Image SIZE / count limits are a model-side concern documented in the lab; the
# gate in relay.intake enforces the byte ceiling before an image ever reaches here.
IMAGE_MEDIA_TYPE_TO_FORMAT: dict[str, str] = {
    "image/png": "png",
    "image/jpeg": "jpeg",
    "image/gif": "gif",
    "image/webp": "webp",
}


def image_block(data: bytes, media_type: str) -> dict[str, Any]:
    """Build one Converse IMAGE content block from raw image bytes + its MIME type.

    Returns `{"image": {"format": <png|jpeg|gif|webp>, "source": {"bytes": data}}}`
    — the Converse-native shape. With an AWS SDK you pass RAW bytes; boto3
    base64-encodes them for the wire (so we do NOT base64 here, and we never build a
    model-specific single-prompt invoke payload — that legacy path is banned, 07 §3.3).

    A caller drops this block into a message's `content` list alongside a `{"text":
    ...}` block; converse(messages=...) sends both in one multimodal message. Raises
    ValueError for an unsupported media type — the same admitted set the intake gate
    checks, so a bad type fails loudly, never silently.
    """
    fmt = IMAGE_MEDIA_TYPE_TO_FORMAT.get(media_type)
    if fmt is None:
        raise ValueError(
            f"Unsupported image media type {media_type!r}. Converse accepts "
            f"{', '.join(sorted(IMAGE_MEDIA_TYPE_TO_FORMAT))}."
        )
    return {"image": {"format": fmt, "source": {"bytes": data}}}


class LLMError(RuntimeError):
    """Raised when a Converse call cannot be completed after retries and fallback.

    Carries the last underlying error so the failure is debuggable, never silent.
    """

    def __init__(self, message: str, *, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.__cause__ = cause


# --- Module 12: the per-ticket cost meter (by addition; converse() unchanged) -------
# A ticket is SEVERAL converse() calls (triage fast + answer generation smart + agent tool
# loops), so the ticket's cost is the SUM over all of them (the cost-decomposition diagram
# in the article). cost_cents lives on TicketRecord (frozen M7), but the agent/store that
# build the record are INHERITED and never rewritten — so Module 12 needs to total a
# ticket's token spend WITHOUT touching those files or the frozen converse() signature.
#
# The CostMeter is a context manager that, while active, records EVERY converse() call's
# (tier, usage) on a process-local stack. The Module 12 worker_handler wraps the agent run
# in `with CostMeter() as meter: run_relay(...)`, then reads `meter.cost_cents` and writes
# it onto the persisted TicketRecord. converse() just appends to the active meter (a no-op
# when none is active), so its body grows by ADDITION and its signature is untouched.
#
# This is in-process accounting, not a billing source of truth — the API usage block IS the
# truth, and that is exactly what the meter sums (never a guess). It is reset per `with`
# block, so two tickets never cross-contaminate, and nested meters (a meter inside a meter)
# each see only their own calls.
_cost_meters: list["CostMeter"] = []


class CostMeter:
    """Sum the cost of every converse() call made inside a `with` block (skill 4.1.1).

    Usage:

        with CostMeter() as meter:
            run_relay(payload)          # any number of converse() calls, any tier
        record.cost_cents = meter.cost_cents

    Records one entry per call: {tier, input_tokens, output_tokens, cached_input_tokens,
    service_tier, cost_usd}. `cost_usd` / `cost_cents` total the lot through the M3 price
    map (config.estimate_cost_discounted), honouring prompt-cache reads and a Flex/batch
    discount. Thread-of-execution local (a simple stack) — the lab and the worker are
    single-threaded per ticket; the article notes a real fleet keys this by request id.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __enter__(self) -> "CostMeter":
        _cost_meters.append(self)
        return self

    def __exit__(self, *exc: object) -> None:
        # Always pop, even on an exception, so a failing ticket does not leak the meter
        # into the next run (no silent cross-contamination).
        if _cost_meters and _cost_meters[-1] is self:
            _cost_meters.pop()

    def record(self, tier: str, usage: dict[str, int], *, service_tier: str) -> None:
        """Record one converse() call's token usage and its computed cost (USD)."""
        input_tokens = int(usage.get("inputTokens", 0))
        output_tokens = int(usage.get("outputTokens", 0))
        cached = int(usage.get("cacheReadInputTokens", 0))
        discount = config.FLEX_DISCOUNT if service_tier == config.SERVICE_TIER_FLEX else 0.0
        cost = config.estimate_cost_discounted(
            tier, input_tokens, output_tokens,
            discount=discount, cached_input_tokens=cached,
        )
        self.calls.append({
            "tier": tier,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cached_input_tokens": cached,
            "service_tier": service_tier,
            "cost_usd": cost,
        })

    @property
    def cost_usd(self) -> float:
        """Total cost in US dollars across every recorded call."""
        return sum(call["cost_usd"] for call in self.calls)

    @property
    def cost_cents(self) -> float:
        """Total cost in CENTS — what TicketRecord.cost_cents holds (frozen M7 field)."""
        return self.cost_usd * 100.0

    @property
    def call_count(self) -> int:
        return len(self.calls)


def _record_cost(tier: str, usage: dict[str, int], service_tier: str) -> None:
    """Append one call's usage to the active CostMeter, if any (a no-op otherwise).

    Called at the END of every converse() so the meter sees each call's REAL usage block.
    No active meter -> nothing happens (the common case: a one-off call outside a ticket)."""
    if _cost_meters:
        _cost_meters[-1].record(tier, usage, service_tier=service_tier)


@dataclass
class RouteDecision:
    """Why the router picked a tier — surfaced in demos and logs, not hidden."""

    tier: str
    reason: str


@dataclass
class ConverseResult:
    """The non-streaming return value, and the streaming generator's final state.

    Module 9 adds `guardrail_action` BY ADDITION (default None): "GUARDRAIL_INTERVENED"
    when the in-line `relay-guardrail` blocked or masked this call, else None / "NONE".
    It is read off the Converse `stopReason` ("guardrail_intervened"), so a caller
    (run_attacks.py) can tell an attack was caught at the model boundary without parsing
    the trace. Adding a defaulted field does not change converse()'s frozen SIGNATURE.
    """

    text: str
    tier: str
    usage: dict[str, int]
    stop_reason: str
    route: RouteDecision | None = None
    guardrail_action: str | None = None  # ADDED M9 (by addition; default None)
    # ADDED M12 (by addition; default standard): which Bedrock service tier this call
    # billed on — "standard" (interactive) or "flex" (-50%, eval/backfill only). Lets the
    # cost meter apply the Flex discount and a reader see the call's billing tier.
    service_tier: str = "standard"


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
    # An explicit tier ("fast"/"smart"/"frontier"/"vision") is validated by tier_profile().
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
    """Assemble the Converse/ConverseStream request from messages + passthrough params.

    Module 9: if a `guardrail` was passed, the (popped) guardrail keys are translated
    into the `guardrailConfig` block that attaches `relay-guardrail` to BOTH sides of
    this call (input + output). With no guardrail, the request is byte-for-byte what it
    was before Module 9 — the guardrail is opt-in and additive.
    """
    # Work on a copy: the config builders POP their keys, and the same params dict is
    # reused across retries/fallback profiles, so we must not mutate the caller's dict.
    local = dict(params)
    guardrail_config = _guardrail_config(local)
    # Module 12: pop the service tier BEFORE the passthrough loop so it never leaks raw.
    service_tier_config = _service_tier_config(local)
    kwargs: dict[str, Any] = {"modelId": modelId, "messages": messages}
    for key, value in local.items():
        if key in _INFERENCE_KEYS:
            kwargs[key] = value
    if guardrail_config is not None:
        kwargs["guardrailConfig"] = guardrail_config
    if service_tier_config is not None:
        kwargs["serviceTier"] = service_tier_config
    # Module 12: insert the prompt cache point (after `system` is in kwargs). Pops
    # cache_prompt; a no-op unless the caller opted in AND there is a system prefix.
    _apply_prompt_cache(kwargs, local)
    return kwargs


# A model that does not offer the requested service tier rejects the Converse request with a
# ValidationException whose message names the service tier. As of June 2026 the FLEX tier is
# offered by the Nova candidates (e.g. Nova 2 Lite) but NOT by the Claude judge (Claude Haiku
# 4.5 serves only the DEFAULT tier) and NOT by Nova Micro — so an eval call that asks for FLEX
# on a model without it must DEGRADE to that model's default tier rather than fail. We detect
# this one specific, non-transient validation error and retry ONCE without the service tier;
# the caller then bills the call at DEFAULT (no Flex -50% it never received). This is graceful
# degradation in the spirit of the smart->fast tier degrade — never a silent wrong-billing.
_SERVICE_TIER_UNSUPPORTED_MARKER = "service tier"


def _is_unsupported_service_tier_error(err: ClientError) -> bool:
    """True when Converse rejected the request because the model lacks the service tier."""
    if err.response.get("Error", {}).get("Code") != "ValidationException":
        return False
    message = err.response.get("Error", {}).get("Message", "").lower()
    return _SERVICE_TIER_UNSUPPORTED_MARKER in message and "not supported" in message


def _call_once(modelId: str, messages: list[dict[str, Any]],
               params: dict[str, Any]) -> dict[str, Any]:
    """One non-streaming Converse call against one profile (no retry here).

    If the model does not offer the requested service tier (e.g. FLEX on the Claude judge),
    retry ONCE on the model's DEFAULT tier and record the degradation on `params` so the cost
    meter bills DEFAULT (not the Flex discount the call never got). Other errors propagate.
    """
    client = _runtime_client()
    try:
        return client.converse(**_request_kwargs(modelId, messages, params))
    except ClientError as err:
        if not (_SERVICE_TIER_PARAM in params and _is_unsupported_service_tier_error(err)):
            raise
        degraded = dict(params)
        degraded.pop(_SERVICE_TIER_PARAM, None)
        params[_ACTUAL_SERVICE_TIER_KEY] = config.DEFAULT_SERVICE_TIER
        return client.converse(**_request_kwargs(modelId, messages, degraded))


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
            result_box.usage = _usage_dict(usage)
    result_box.text = "".join(text_parts)
    # Module 12: now that the stream is drained and usage is known, total this call into the
    # active CostMeter (a no-op when none is active). The service tier was stamped on the
    # result_box by converse() before streaming began.
    _record_cost(result_box.tier, result_box.usage, result_box.service_tier)


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
    # Module 12: read (without consuming) which service tier this call billed on, so the
    # cost meter can apply the Flex -50% discount. _request_kwargs pops it from a COPY of
    # params, so this peek does not change the request. Defaults to Standard (interactive).
    service_tier = (params or {}).get(_SERVICE_TIER_PARAM) or config.DEFAULT_SERVICE_TIER

    if stream:
        result = ConverseResult(text="", tier=concrete_tier, usage={}, stop_reason="",
                                route=decision)

        def invoke_stream(profile: str) -> dict[str, Any]:
            client = _runtime_client()
            return client.converse_stream(**_request_kwargs(profile, messages, params))

        response, used_tier = _with_resilience(invoke_stream, tier=concrete_tier)
        result.tier = used_tier
        # Module 12: record cost AFTER the stream is drained (usage arrives in the final
        # metadata event), so the meter sees the real token counts. _stream_deltas fills
        # result.usage, then calls _record_cost — the result_box carries the service tier.
        result.service_tier = service_tier  # type: ignore[attr-defined]
        return _StreamingResult(_stream_deltas(response, result), result)

    def invoke(profile: str) -> dict[str, Any]:
        return _call_once(profile, messages, params)

    response, used_tier = _with_resilience(invoke, tier=concrete_tier)
    text = response["output"]["message"]["content"][0]["text"]
    usage = _usage_dict(response.get("usage", {}))
    stop_reason = response.get("stopReason", "")
    # Module 9: when the in-line guardrail blocks/masks, Converse returns stopReason
    # "guardrail_intervened" and `text` carries the guardrail's blocked-output message.
    # Surface that as guardrail_action so a caller can tell the attack was caught at the
    # model boundary without parsing the trace block. (No guardrail -> None.)
    guardrail_action = (
        "GUARDRAIL_INTERVENED" if stop_reason == "guardrail_intervened" else None
    )
    # Module 12: total this call into the active per-ticket CostMeter (a no-op when none is
    # active). The usage block is the source of truth; the Flex discount applies only when
    # this call billed on the Flex tier. If _call_once had to DEGRADE an unsupported service
    # tier (e.g. FLEX on the Claude judge) to the model's default, it recorded the actual tier
    # back on `params`, so we bill the tier the call really ran on — never a phantom discount.
    billed_service_tier = params.get(_ACTUAL_SERVICE_TIER_KEY, service_tier)
    _record_cost(used_tier, usage, billed_service_tier)
    return ConverseResult(
        text=text,
        tier=used_tier,
        usage=usage,
        stop_reason=stop_reason,
        route=decision,
        guardrail_action=guardrail_action,
    )
