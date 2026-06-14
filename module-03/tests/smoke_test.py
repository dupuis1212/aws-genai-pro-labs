"""smoke_test.py — Module 3 lab tests (cumulative: covers Modules 2 and 3).

OFFLINE BY DEFAULT. Every test here except the ones marked `live` runs with NO
AWS credentials and makes NO network call:
  - schema tests are pure Pydantic (the M2 Ticket/Triage contract);
  - the router tests are pure functions on fixed inputs (no model call);
  - the converse() tests drive relay.llm with a botocore Stubber on the
    bedrock-runtime client (Converse, ConverseStream, and a ThrottlingException
    that proves the backoff path);
  - the triage tests drive relay.triage with Stubbers for the bedrock-agent
    (get_prompt) and the bedrock-runtime (converse) clients.
That is the course convention — anyone can `uv run pytest` on a fresh clone.

TWO tests are marked `live` and make real calls:
    RELAY_LIVE_TESTS=1 uv run pytest -m live
LIVE-CALL BUDGET: exactly TWO calls total —
  1. one ConverseStream on the FAST tier (us.amazon.nova-micro-v1:0), and
  2. one Converse on the SMART tier (us.amazon.nova-2-lite-v1:0),
both with maxTokens<=64. Together that is well under $0.001 (a tenth of a cent)
as of June 2026. They need AWS credentials and us-east-1.
"""

from __future__ import annotations

import datetime as dt
import inspect
import os
import re
import sys
from pathlib import Path

import boto3
import pytest
from botocore.stub import ANY, Stubber

# Import the lab package (module-03/ ships the cumulative relay/ package).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from relay import config  # noqa: E402
from relay import llm  # noqa: E402
from relay import triage as triage_mod  # noqa: E402
from relay.models import Ticket, Triage  # noqa: E402

_ROOT = Path(__file__).resolve().parent.parent
TICKETS_DIR = _ROOT / "data" / "tickets"
RELAY_DIR = _ROOT / "relay"


# ===========================================================================
# Module 2 contract — schemas and ticket fixtures (still LAW at M3)
# ===========================================================================
def test_ticket_has_exactly_four_fields():
    # The M2 contract: Ticket is EXACTLY 4 fields. No attachments, no
    # pii_redacted (those are added by addition in M6 / M10).
    assert set(Ticket.model_fields) == {
        "ticket_id", "channel", "customer_message", "created_at"
    }


def test_triage_literals_are_the_frozen_enums():
    # The three enums are LAW (06 §2 / bible §3.1). Guard them field-for-field.
    def literals(field: str) -> set[str]:
        ann = Triage.model_fields[field].annotation
        return set(ann.__args__)

    assert literals("intent") == {
        "billing", "technical", "account", "shipping", "other"
    }
    assert literals("priority") == {"low", "normal", "high", "urgent"}
    assert literals("sentiment") == {"negative", "neutral", "positive"}


def test_triage_rejects_an_unknown_intent():
    with pytest.raises(ValueError):
        Triage.model_validate_json(
            '{"intent": "refund", "priority": "high", "sentiment": "negative"}'
        )


def test_all_ten_ticket_fixtures_are_valid_tickets():
    files = sorted(TICKETS_DIR.glob("ticket-*.json"))
    assert len(files) == 10
    for f in files:
        ticket = triage_mod.load_ticket(f)
        assert ticket.channel in ("email", "chat")
        assert ticket.ticket_id == f.stem


# ===========================================================================
# Module 3 — the frozen converse() signature and the model-ID containment law
# ===========================================================================
def test_converse_signature_is_frozen_byte_identical():
    """The contract: converse(messages, *, tier="auto", stream=False, **params).

    This signature is byte-identical M3 -> M15. Any drift breaks every downstream
    consumer, so guard it exactly.
    """
    sig = inspect.signature(llm.converse)
    params = list(sig.parameters.values())

    assert params[0].name == "messages"
    assert params[0].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert params[0].default is inspect.Parameter.empty

    tier = sig.parameters["tier"]
    assert tier.kind is inspect.Parameter.KEYWORD_ONLY
    assert tier.default == "auto"

    stream = sig.parameters["stream"]
    assert stream.kind is inspect.Parameter.KEYWORD_ONLY
    assert stream.default is False

    assert any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params), (
        "converse must accept **params"
    )
    # No extra positional/keyword parameters were invented.
    assert [p.name for p in params] == ["messages", "tier", "stream", "params"]


def test_tiers_are_the_canonical_set():
    # Canonical tiers (06 §2): fast / smart, plus the frontier reference tier.
    # "auto" is the router request, never a key in the map.
    assert "auto" not in config.TIERS
    assert set(config.TIERS) == {"fast", "smart", "frontier"}


def test_tier_map_points_at_inference_profiles_not_bare_ids():
    # The real M1 trap and the grep gate: every ID is a us./global. profile.
    for tier, profile in config.TIERS.items():
        assert profile.startswith(("us.", "global.")), (tier, profile)
        assert not profile.startswith(("amazon.", "anthropic.")), (tier, profile)


def test_fast_and_smart_map_to_nova_micro_and_nova2_lite():
    # The frozen tier choices from the GROUND TRUTH / bible §1.
    assert config.tier_profile("fast") == "us.amazon.nova-micro-v1:0"
    assert config.tier_profile("smart") == "us.amazon.nova-2-lite-v1:0"


def test_tier_profile_rejects_unknown_tier():
    with pytest.raises(ValueError):
        config.tier_profile("turbo")


def test_no_model_id_literal_outside_config_py():
    """Grep gate: a us./global. profile ID may appear ONLY in relay/config.py."""
    pattern = re.compile(r"(us|global|eu)\.(amazon|anthropic)\.")
    offenders = []
    for path in RELAY_DIR.glob("*.py"):
        if path.name == "config.py":
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if pattern.search(line):
                offenders.append(f"{path.name}:{lineno}: {line.strip()}")
    assert offenders == [], "model IDs leaked outside config.py:\n" + "\n".join(offenders)


def test_no_legacy_invoke_path_anywhere_in_relay():
    """Converse/ConverseStream ONLY — the legacy single-prompt invoke path is

    banned in the relay package (Titan embeddings in Module 4's ingest/ is the sole
    exception in the whole course, and lives outside relay/). The forbidden tokens
    are assembled from fragments so this guard does not itself trip the repo-wide
    grep gate for them.
    """
    forbidden = ["invoke" + "_model", "Invoke" + "Model"]
    for path in RELAY_DIR.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"legacy invoke path in {path.name}: {token}"


# ===========================================================================
# Module 3 — the complexity router (tier="auto"), pure and deterministic
# ===========================================================================
def _user(text: str) -> list[dict]:
    return [{"role": "user", "content": [{"text": text}]}]


def test_router_defaults_to_fast_on_a_simple_short_request():
    decision = llm.route(_user("hi"))
    assert decision.tier == "fast"
    assert "fast" in decision.reason


def test_router_escalates_to_smart_on_a_billing_keyword():
    decision = llm.route(_user("Why was I charged twice for order #1042?"))
    assert decision.tier == "smart"


def test_router_escalates_to_smart_on_a_long_request():
    # A long request with NO complexity keyword: it must escalate on LENGTH alone,
    # so the reason is the length branch (not a keyword match).
    long_text = "I have a question about my account settings and preferences. " * 8
    decision = llm.route(_user(long_text))
    assert decision.tier == "smart"
    assert "long request" in decision.reason


def test_router_is_deterministic():
    msg = _user("Please refund the duplicate charge.")
    assert llm.route(msg).tier == llm.route(msg).tier == "smart"


# ===========================================================================
# Module 3 — converse() over a stubbed bedrock-runtime client (offline)
# ===========================================================================
def _converse_response(reply: str, in_tok: int, out_tok: int) -> dict:
    return {
        "output": {"message": {"role": "assistant", "content": [{"text": reply}]}},
        "stopReason": "end_turn",
        "usage": {
            "inputTokens": in_tok,
            "outputTokens": out_tok,
            "totalTokens": in_tok + out_tok,
        },
        "metrics": {"latencyMs": 180},
    }


def _stub_runtime(monkeypatch) -> tuple[object, Stubber]:
    """Make llm.converse use a stubbed bedrock-runtime client (no creds/network)."""
    client = boto3.client("bedrock-runtime", region_name="us-east-1")
    stubber = Stubber(client)
    monkeypatch.setattr(llm, "_clients", {"runtime": client})
    return client, stubber


def test_converse_nonstreaming_returns_result(monkeypatch):
    client, stub = _stub_runtime(monkeypatch)
    stub.add_response(
        "converse",
        _converse_response("Hello! How can I help with your CloudCart store?", 30, 12),
        {"modelId": config.tier_profile("fast"), "messages": ANY},
    )
    with stub:
        result = llm.converse(_user("hi"), tier="fast")
    assert isinstance(result, llm.ConverseResult)
    assert result.tier == "fast"
    assert result.text.startswith("Hello!")
    assert result.usage["inputTokens"] == 30
    assert result.usage["outputTokens"] == 12


def test_converse_auto_routes_to_smart_profile(monkeypatch):
    client, stub = _stub_runtime(monkeypatch)
    stub.add_response(
        "converse",
        _converse_response("I can help with that billing dispute.", 50, 20),
        {"modelId": config.tier_profile("smart"), "messages": ANY},
    )
    with stub:
        result = llm.converse(
            _user("Why was I charged twice for order #1042?"), tier="auto"
        )
    assert result.tier == "smart"
    assert result.route is not None and result.route.tier == "smart"


def test_converse_passes_inference_config_through_params(monkeypatch):
    client, stub = _stub_runtime(monkeypatch)
    inference = {"maxTokens": 100, "temperature": 0.0}
    stub.add_response(
        "converse",
        _converse_response('{"intent": "billing"}', 40, 8),
        {
            "modelId": config.tier_profile("fast"),
            "messages": ANY,
            "inferenceConfig": inference,
        },
    )
    with stub:
        result = llm.converse(_user("hi"), tier="fast", inferenceConfig=inference)
    assert result.usage["inputTokens"] == 40


def test_converse_retries_on_throttling_then_succeeds(monkeypatch):
    """The backoff path: a ThrottlingException is retried, not propagated."""
    client, stub = _stub_runtime(monkeypatch)
    # First call throttled...
    stub.add_client_error(
        "converse",
        service_error_code="ThrottlingException",
        service_message="Too many requests",
        http_status_code=429,
    )
    # ...retry succeeds.
    stub.add_response(
        "converse",
        _converse_response("Recovered after backoff.", 25, 10),
        {"modelId": config.tier_profile("fast"), "messages": ANY},
    )

    # Make backoff instant so the test does not actually sleep.
    sleeps: list[int] = []
    monkeypatch.setattr(llm, "_backoff_sleep", lambda attempt: sleeps.append(attempt))

    with stub:
        result = llm.converse(_user("hi"), tier="fast")
    assert result.text == "Recovered after backoff."
    # Exactly one backoff happened (one throttle before the success).
    assert sleeps == [1]


def test_converse_raises_llmerror_on_nonretryable(monkeypatch):
    """A ValidationException is NOT retried — surfaced immediately as LLMError."""
    client, stub = _stub_runtime(monkeypatch)
    stub.add_client_error(
        "converse",
        service_error_code="ValidationException",
        service_message="Invalid request",
        http_status_code=400,
    )
    with stub:
        with pytest.raises(llm.LLMError):
            llm.converse(_user("hi"), tier="fast")


def test_converse_exhausts_then_degrades_smart_to_fast(monkeypatch):
    """Graceful degradation: smart throttled on all profiles -> fall to fast.

    smart has a primary AND a global. alternate profile; both are throttled
    through every retry, so the layer degrades to the fast tier and succeeds.
    """
    client, stub = _stub_runtime(monkeypatch)
    monkeypatch.setattr(llm, "_backoff_sleep", lambda attempt: None)

    # smart primary: 3 attempts (1 + 2 retries) all throttled.
    # smart alternate (global.): 3 attempts all throttled.
    for _ in range(6):
        stub.add_client_error(
            "converse",
            service_error_code="ThrottlingException",
            service_message="Too many requests",
            http_status_code=429,
        )
    # Degraded to fast: succeeds.
    stub.add_response(
        "converse",
        _converse_response("Degraded to fast tier and answered.", 22, 9),
        {"modelId": config.tier_profile("fast"), "messages": ANY},
    )
    with stub:
        result = llm.converse(_user("explain my invoice"), tier="smart")
    assert result.tier == "fast"
    assert result.text.startswith("Degraded")


def _stream_response(deltas: list[str], in_tok: int, out_tok: int) -> dict:
    events: list[dict] = [{"messageStart": {"role": "assistant"}}]
    events.append({"contentBlockStart": {"start": {}, "contentBlockIndex": 0}})
    for chunk in deltas:
        events.append(
            {"contentBlockDelta": {"delta": {"text": chunk}, "contentBlockIndex": 0}}
        )
    events.append({"contentBlockStop": {"contentBlockIndex": 0}})
    events.append({"messageStop": {"stopReason": "end_turn"}})
    events.append(
        {
            "metadata": {
                "usage": {
                    "inputTokens": in_tok,
                    "outputTokens": out_tok,
                    "totalTokens": in_tok + out_tok,
                },
                "metrics": {"latencyMs": 210},
            }
        }
    )
    return {"stream": iter(events)}


def test_converse_streaming_yields_deltas_then_fills_result(monkeypatch):
    # ConverseStream returns a botocore EventStream, which the Stubber cannot model
    # (its response shape is not a plain dict). So we replace converse_stream on a
    # real client with a fake that returns our event iterator under "stream" — the
    # exact shape relay.llm consumes.
    client = boto3.client("bedrock-runtime", region_name="us-east-1")
    captured: dict = {}

    def fake_converse_stream(**kwargs):
        captured.update(kwargs)
        return _stream_response(["Let ", "me ", "check ", "that."], 33, 14)

    monkeypatch.setattr(client, "converse_stream", fake_converse_stream)
    monkeypatch.setattr(llm, "_clients", {"runtime": client})

    streaming = llm.converse(_user("hi"), tier="fast", stream=True)
    chunks = list(streaming)

    assert captured["modelId"] == config.tier_profile("fast")
    assert chunks == ["Let ", "me ", "check ", "that."]
    assert streaming.result.text == "Let me check that."
    assert streaming.result.usage["outputTokens"] == 14
    assert streaming.result.tier == "fast"


# ===========================================================================
# Module 2 triage flow — now routed through relay.llm.converse (still offline)
# ===========================================================================
_TEMPLATE_TEXT = "Classify this ticket. Output ONLY JSON.\n\nTicket: {{ticket}}"


def _get_prompt_response() -> dict:
    return {
        "name": "relay-triage",
        "id": "PROMPTID01",
        "arn": "arn:aws:bedrock:us-east-1:111122223333:prompt/PROMPTID01",
        "version": "1",
        "variants": [
            {
                "name": "triage-v3-fewshot",
                "templateType": "TEXT",
                "templateConfiguration": {
                    "text": {
                        "text": _TEMPLATE_TEXT,
                        "inputVariables": [{"name": "ticket"}],
                    }
                },
            }
        ],
        "createdAt": dt.datetime(2026, 6, 1),
        "updatedAt": dt.datetime(2026, 6, 1),
    }


def _stubbed_agent_client() -> tuple[object, Stubber]:
    client = boto3.client("bedrock-agent", region_name="us-east-1")
    stubber = Stubber(client)
    stubber.add_response(
        "get_prompt",
        _get_prompt_response(),
        {"promptIdentifier": "PROMPTID01", "promptVersion": "1"},
    )
    return client, stubber


def test_triage_returns_validated_triage_offline(monkeypatch):
    agent, agent_stub = _stubbed_agent_client()
    runtime, runtime_stub = _stub_runtime(monkeypatch)
    runtime_stub.add_response(
        "converse",
        _converse_response(
            '{"intent": "billing", "priority": "high", "sentiment": "negative"}',
            in_tok=210, out_tok=18,
        ),
        {"modelId": config.tier_profile("fast"), "messages": ANY,
         "inferenceConfig": ANY},
    )

    ticket = triage_mod.load_ticket(TICKETS_DIR / "ticket-001.json")
    with agent_stub, runtime_stub:
        result, usage = triage_mod.triage(ticket, prompt_id="PROMPTID01",
                                          agent_client=agent)
    assert isinstance(result, Triage)
    assert result.intent == "billing"
    assert result.priority == "high"
    assert usage["inputTokens"] == 210
    assert usage["outputTokens"] == 18


def test_triage_strips_prose_around_json(monkeypatch):
    agent, agent_stub = _stubbed_agent_client()
    runtime, runtime_stub = _stub_runtime(monkeypatch)
    runtime_stub.add_response(
        "converse",
        _converse_response(
            'Sure! Here is the JSON:\n'
            '{"intent": "technical", "priority": "urgent", "sentiment": "negative"}\n'
            'Hope that helps!',
            in_tok=205, out_tok=40,
        ),
        {"modelId": config.tier_profile("fast"), "messages": ANY,
         "inferenceConfig": ANY},
    )
    ticket = triage_mod.load_ticket(TICKETS_DIR / "ticket-002.json")
    with agent_stub, runtime_stub:
        result, _ = triage_mod.triage(ticket, prompt_id="PROMPTID01",
                                      agent_client=agent)
    assert result.intent == "technical"
    assert result.priority == "urgent"


def test_triage_retries_once_on_invalid_then_succeeds(monkeypatch):
    agent, agent_stub = _stubbed_agent_client()
    runtime, runtime_stub = _stub_runtime(monkeypatch)
    # Attempt 1: invalid intent -> validation fails.
    runtime_stub.add_response(
        "converse",
        _converse_response(
            '{"intent": "refund", "priority": "high", "sentiment": "negative"}',
            in_tok=210, out_tok=18,
        ),
        {"modelId": config.tier_profile("fast"), "messages": ANY,
         "inferenceConfig": ANY},
    )
    # Attempt 2 (retry with the error fed back): valid.
    runtime_stub.add_response(
        "converse",
        _converse_response(
            '{"intent": "billing", "priority": "high", "sentiment": "negative"}',
            in_tok=260, out_tok=18,
        ),
        {"modelId": config.tier_profile("fast"), "messages": ANY,
         "inferenceConfig": ANY},
    )
    ticket = triage_mod.load_ticket(TICKETS_DIR / "ticket-001.json")
    with agent_stub, runtime_stub:
        result, usage = triage_mod.triage(ticket, prompt_id="PROMPTID01",
                                          agent_client=agent)
    assert result.intent == "billing"
    # Usage is summed across BOTH calls so the cost line includes the retry.
    assert usage["inputTokens"] == 210 + 260
    assert usage["outputTokens"] == 18 + 18


def test_triage_raises_after_two_invalid_attempts_no_silent_pass(monkeypatch):
    agent, agent_stub = _stubbed_agent_client()
    runtime, runtime_stub = _stub_runtime(monkeypatch)
    for _ in range(2):
        runtime_stub.add_response(
            "converse",
            _converse_response("not json at all", in_tok=200, out_tok=5),
            {"modelId": config.tier_profile("fast"), "messages": ANY,
             "inferenceConfig": ANY},
        )
    ticket = triage_mod.load_ticket(TICKETS_DIR / "ticket-001.json")
    with agent_stub, runtime_stub:
        with pytest.raises(triage_mod.TriageError) as excinfo:
            triage_mod.triage(ticket, prompt_id="PROMPTID01", agent_client=agent)
    assert excinfo.value.raw_output == "not json at all"


def test_triage_runs_on_the_fast_tier_via_converse(monkeypatch):
    """Triage must go through converse(tier="fast") — guard the modelId used."""
    agent, agent_stub = _stubbed_agent_client()
    runtime, runtime_stub = _stub_runtime(monkeypatch)
    runtime_stub.add_response(
        "converse",
        _converse_response(
            '{"intent": "billing", "priority": "high", "sentiment": "negative"}',
            210, 18,
        ),
        {
            "modelId": config.tier_profile("fast"),  # the fast profile, from config
            "messages": ANY,
            "inferenceConfig": {"maxTokens": 100, "temperature": 0.0},
        },
    )
    ticket = triage_mod.load_ticket(TICKETS_DIR / "ticket-001.json")
    with agent_stub, runtime_stub:
        result, _ = triage_mod.triage(ticket, prompt_id="PROMPTID01",
                                      agent_client=agent)
    assert result.intent == "billing"


def test_triage_estimate_cost_uses_fast_tier_price():
    """estimate_cost delegates to config's per-tier price map (fast tier)."""
    cost = triage_mod.estimate_cost(1000, 1000)
    fast = config.PRICE_PER_1K["fast"]
    assert cost == pytest.approx(fast["input"] + fast["output"])


def test_resolve_prompt_id_errors_without_setup(monkeypatch, tmp_path):
    monkeypatch.delenv("RELAY_TRIAGE_PROMPT_ID", raising=False)
    monkeypatch.setattr(triage_mod, "_PROMPT_ID_FILE", tmp_path / "nope")
    with pytest.raises(triage_mod.TriageError) as excinfo:
        triage_mod.resolve_prompt_id()
    assert "setup.py" in str(excinfo.value)


# ===========================================================================
# The LIVE tests (opt-in) — budget: 2 calls total
# ===========================================================================
@pytest.mark.live
@pytest.mark.skipif(
    os.environ.get("RELAY_LIVE_TESTS") != "1",
    reason="set RELAY_LIVE_TESTS=1 to make TWO real (sub-cent) Bedrock calls",
)
def test_live_fast_tier_streaming():
    """ONE real ConverseStream on the FAST tier. < $0.0005 as of June 2026."""
    streaming = llm.converse(
        _user("In one short sentence, what is a support ticket?"),
        tier="fast",
        stream=True,
        inferenceConfig={"maxTokens": 64, "temperature": 0.2},
    )
    chunks = list(streaming)
    assert "".join(chunks) == streaming.result.text
    assert streaming.result.text.strip() != ""
    assert streaming.result.tier == "fast"
    assert streaming.result.usage["outputTokens"] > 0


@pytest.mark.live
@pytest.mark.skipif(
    os.environ.get("RELAY_LIVE_TESTS") != "1",
    reason="set RELAY_LIVE_TESTS=1 to make TWO real (sub-cent) Bedrock calls",
)
def test_live_smart_tier_nonstreaming():
    """ONE real Converse on the SMART tier (auto-routed). < $0.0005 as of June 2026."""
    result = llm.converse(
        _user("Why was I charged twice for order #1042? Explain the dispute steps."),
        tier="auto",
        inferenceConfig={"maxTokens": 64, "temperature": 0.2},
    )
    assert result.tier == "smart"  # the router escalated on the billing keywords
    assert result.text.strip() != ""
    assert result.usage["inputTokens"] > 0
