"""smoke_test.py — Module 4 lab tests (cumulative: covers Modules 2, 3 and 4).

OFFLINE BY DEFAULT. Every test here except the ones marked `live` runs with NO
AWS credentials and makes NO network call:
  - schema tests are pure Pydantic (the M2 Ticket/Triage contract — UNCHANGED at
    M4: no new schema is introduced);
  - the router tests are pure functions on fixed inputs (no model call);
  - the converse() tests drive relay.llm with a botocore Stubber on the
    bedrock-runtime client (Converse, ConverseStream, and a ThrottlingException
    that proves the backoff path);
  - the triage tests drive relay.triage with Stubbers for the bedrock-agent
    (get_prompt) and the bedrock-runtime (converse) clients;
  - the Module 4 chunker tests are pure and deterministic on a fixed doc;
  - the Module 4 ingestion tests stub Titan embeddings (invoke_model) and run the
    S3 Vectors bucket/index/PutVectors lifecycle on a moto backend; the kNN query
    (which moto does not implement) is driven by a botocore Stubber.
That is the course convention — anyone can `uv run pytest` on a fresh clone.

TESTS marked `live` make real calls:
    RELAY_LIVE_TESTS=1 uv run pytest -m live
LIVE-CALL BUDGET: at most FOUR calls total —
  Modules 2/3 (2 calls): one ConverseStream on the FAST tier
    (us.amazon.nova-micro-v1:0) and one Converse on the SMART tier
    (us.amazon.nova-2-lite-v1:0), both maxTokens<=64.
  Module 4 (2 calls): two Amazon Titan Text Embeddings V2 invoke_model calls
    (one for a tiny doc chunk, one for a query) — embeddings, not text, ~$0 each.
Together that is well under $0.001 (a tenth of a cent) as of June 2026. They need
AWS credentials and us-east-1. The Module-4 live test does NOT touch S3 Vectors
(no bucket/index is created from the test) — it only exercises the embedder.
"""

from __future__ import annotations

import datetime as dt
import inspect
import io
import json
import os
import re
import sys
from pathlib import Path

import boto3
import pytest
from botocore.response import StreamingBody
from botocore.stub import ANY, Stubber

# Import the lab packages (module-04/ ships the cumulative relay/ package plus the
# new ingest/ pipeline).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from relay import config  # noqa: E402
from relay import llm  # noqa: E402
from relay import triage as triage_mod  # noqa: E402
from relay.models import Ticket, Triage  # noqa: E402
from ingest import chunkers as chunkers_mod  # noqa: E402
from ingest import embed as embed_mod  # noqa: E402
from ingest import run as run_mod  # noqa: E402
from ingest import upsert as upsert_mod  # noqa: E402
import compare_chunking  # noqa: E402

_ROOT = Path(__file__).resolve().parent.parent
TICKETS_DIR = _ROOT / "data" / "tickets"
DOCS_DIR = _ROOT / "data" / "docs"
RELAY_DIR = _ROOT / "relay"
INGEST_DIR = _ROOT / "ingest"


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
# Module 4 — NO new Pydantic schema (the contract guard)
# ===========================================================================
def test_module4_introduces_no_new_pydantic_schema():
    """M4 is RAG ingestion only: Ticket/Triage are UNCHANGED, nothing else exists.

    Answer/Citation are M5; Attachment is M6. Guarding this here keeps the bible
    §3.1 "zero Pydantic schema in M4" invariant honest.
    """
    import relay.models as models_mod
    from pydantic import BaseModel

    schema_names = {
        name for name, obj in vars(models_mod).items()
        if isinstance(obj, type) and issubclass(obj, BaseModel) and obj is not BaseModel
    }
    assert schema_names == {"Ticket", "Triage"}
    # And Ticket is still exactly 4 fields (no early attachments/pii_redacted).
    assert set(Ticket.model_fields) == {
        "ticket_id", "channel", "customer_message", "created_at"
    }


# ===========================================================================
# Module 4 — config additions (resource names + the pinned embedder)
# ===========================================================================
def test_resource_names_are_field_for_field_frozen():
    # 06 §2 / bible §3.3 — reproduced field-for-field.
    assert config.relay_bucket("111122223333") == "relay-111122223333"
    assert config.relay_vector_bucket("111122223333") == "relay-vectors-111122223333"
    assert config.RELAY_INDEX == "relay-docs"
    assert config.RELAY_BUCKET_PREFIXES == ("docs/", "attachments/", "vectors/")


def test_embedder_is_titan_v2_pinned_at_1024_dims():
    # The vector contract: Titan Text Embeddings V2, 1024 dims, cosine — NEVER
    # swapped (M5 KB and M12 cache reuse this index). The Nova successor stays out.
    assert config.EMBED_MODEL_ID == "amazon.titan-embed-text-v2:0"
    assert config.EMBED_DIMENSIONS == 1024
    assert config.EMBED_DISTANCE_METRIC == "cosine"
    assert "nova" not in config.EMBED_MODEL_ID.lower()


def test_tier_map_unchanged_by_module4():
    # M4 only ADDS resource names; the M3 tier map is untouched (bible §2.2).
    assert set(config.TIERS) == {"fast", "smart", "frontier"}
    assert config.tier_profile("fast") == "us.amazon.nova-micro-v1:0"
    assert config.tier_profile("smart") == "us.amazon.nova-2-lite-v1:0"


# ===========================================================================
# Module 4 — the three chunkers, pure and deterministic
# ===========================================================================
_DOC = """---
title: Exporting your order history
category: orders
---

# Exporting your order history

CloudCart keeps a record of every order. You can export it at any time.

## Where the export lives

The export tool is under Settings -> Data & Privacy -> Export data. It is not on
the Orders page, which is why customers cannot find it.

## Exporting to CSV

Tick Orders, choose a date range, click Start export. You get an email with a
secure download link valid for 24 hours.
"""

_SOURCE_URI = "s3://relay-111122223333/docs/orders-export.md"


def test_parse_document_reads_front_matter():
    doc = chunkers_mod.parse_document(_DOC, _SOURCE_URI)
    assert doc.category == "orders"
    assert doc.title == "Exporting your order history"
    assert doc.source_uri == _SOURCE_URI
    assert "Where the export lives" in doc.body


def test_chunkers_are_deterministic():
    # Same input -> byte-identical chunks, for every strategy. This is what makes
    # the offline test and the fair comparison possible.
    for strategy in ("fixed", "hierarchical", "semantic"):
        a = chunkers_mod.chunk_document(_DOC, _SOURCE_URI, strategy)
        b = chunkers_mod.chunk_document(_DOC, _SOURCE_URI, strategy)
        assert [c.text for c in a] == [c.text for c in b]
        assert a, f"{strategy} produced no chunks"


def test_every_chunk_carries_canonical_metadata():
    # The vector-metadata canon (bible §3.3): {category, source_uri, chunk_index}.
    for strategy in ("fixed", "hierarchical", "semantic"):
        chunks = chunkers_mod.chunk_document(_DOC, _SOURCE_URI, strategy)
        for i, chunk in enumerate(chunks):
            meta = chunk.metadata()
            assert meta["category"] == "orders"
            assert meta["source_uri"] == _SOURCE_URI
            assert meta["chunk_index"] == i


def test_hierarchical_splits_on_markdown_headings():
    chunks = chunkers_mod.chunk_document(_DOC, _SOURCE_URI, "hierarchical")
    headings = [c.heading for c in chunks]
    assert any("Where the export lives" in h for h in headings)
    assert any("Exporting to CSV" in h for h in headings)
    # The "Where the export lives" chunk contains its section body, with the trail.
    where = next(c for c in chunks if "Where the export lives" in c.heading)
    assert "Settings -> Data & Privacy" in where.text


def test_semantic_never_splits_mid_sentence():
    chunks = chunkers_mod.chunk_document(_DOC, _SOURCE_URI, "semantic")
    # Every chunk ends at a sentence boundary (terminal punctuation), never mid-word.
    for chunk in chunks:
        assert chunk.text.rstrip()[-1] in ".!?", chunk.text[-40:]


def test_fixed_size_overlap_repeats_boundary_text():
    # Overlap means consecutive fixed chunks share a tail/head window. Use a small
    # window so the doc yields >=2 chunks deterministically.
    doc = chunkers_mod.parse_document(_DOC, _SOURCE_URI)
    chunks = chunkers_mod.fixed_size(doc, chunk_chars=200, overlap_chars=60)
    assert len(chunks) >= 2
    # The end of chunk 0 should reappear at the start of chunk 1 (overlap window).
    tail = chunks[0].text[-40:]
    assert any(seg and seg in chunks[1].text for seg in (tail[-20:], tail[:20]))


def test_unknown_strategy_raises():
    with pytest.raises(ValueError):
        chunkers_mod.chunk_document(_DOC, _SOURCE_URI, "windowed")


def test_shipped_docs_all_parse_and_chunk():
    # The real corpus: every shipped doc has front matter and yields chunks under
    # every strategy.
    docs = sorted(DOCS_DIR.glob("*.md"))
    assert len(docs) >= 6
    for path in docs:
        text = path.read_text(encoding="utf-8")
        doc = chunkers_mod.parse_document(text, f"s3://b/docs/{path.name}")
        assert doc.category != "uncategorized", path.name
        for strategy in ("fixed", "hierarchical", "semantic"):
            assert chunkers_mod.chunk_document(text, "s3://b/docs/x", strategy)


def test_questions_reference_real_doc_stems():
    # Every question's hand-labelled relevant_docs must be a real doc stem.
    questions = json.loads((_ROOT / "data" / "questions.json").read_text("utf-8"))
    stems = {p.stem for p in DOCS_DIR.glob("*.md")}
    assert len(questions) >= 8
    for q in questions:
        for ref in q["relevant_docs"]:
            assert ref in stems, f"{ref} not in {stems}"
    # At least one question carries an exact identifier (ERR-402).
    assert any("ERR-402" in q["question"] for q in questions)


# ===========================================================================
# Module 4 — Titan embeddings (the SOLE invoke_model), stubbed offline
# ===========================================================================
def _titan_invoke_response(dims: int = 1024, tokens: int = 7) -> dict:
    body = json.dumps(
        {"embedding": [0.01] * dims, "inputTextTokenCount": tokens}
    ).encode("utf-8")
    return {
        "body": StreamingBody(io.BytesIO(body), len(body)),
        "contentType": "application/json",
    }


def _stub_titan(monkeypatch, n_calls: int = 1) -> tuple[object, Stubber]:
    client = boto3.client("bedrock-runtime", region_name="us-east-1")
    stubber = Stubber(client)
    for _ in range(n_calls):
        stubber.add_response(
            "invoke_model",
            _titan_invoke_response(),
            {
                "modelId": config.EMBED_MODEL_ID,
                "body": ANY,
                "accept": "application/json",
                "contentType": "application/json",
            },
        )
    return client, stubber


def test_embed_one_returns_a_1024_vector_via_invoke_model(monkeypatch):
    client, stub = _stub_titan(monkeypatch, n_calls=1)
    with stub:
        vector, tokens = embed_mod.embed_one("How do I export orders?", client=client)
    assert len(vector) == config.EMBED_DIMENSIONS
    assert tokens == 7


def test_embed_texts_batches_and_sums_tokens(monkeypatch):
    client, stub = _stub_titan(monkeypatch, n_calls=3)
    with stub:
        result = embed_mod.embed_texts(["a", "b", "c"], client=client)
    assert result.count == 3
    assert result.input_tokens == 21  # 7 per call x 3
    assert all(len(v) == config.EMBED_DIMENSIONS for v in result.vectors)


def test_embed_rejects_dimension_mismatch(monkeypatch):
    client = boto3.client("bedrock-runtime", region_name="us-east-1")
    stub = Stubber(client)
    stub.add_response(
        "invoke_model",
        _titan_invoke_response(dims=512),  # WRONG dims -> must be rejected
        {"modelId": config.EMBED_MODEL_ID, "body": ANY,
         "accept": "application/json", "contentType": "application/json"},
    )
    with stub:
        with pytest.raises(ValueError):
            embed_mod.embed_one("x", client=client)


# ===========================================================================
# Module 4 — S3 Vectors lifecycle on moto + the kNN query via Stubber
# ===========================================================================
ACCOUNT = "111122223333"
VECTOR_BUCKET = config.relay_vector_bucket(ACCOUNT)
INDEX = config.RELAY_INDEX


@pytest.fixture
def s3vectors_backend():
    """A moto-backed s3vectors client with the relay-docs index ready.

    moto implements create/delete bucket+index, put_vectors, and list_vectors —
    the upsert lifecycle. It does NOT implement query_vectors; the kNN tests use a
    botocore Stubber for that one call.
    """
    from moto import mock_aws

    with mock_aws():
        client = boto3.client("s3vectors", region_name="us-east-1")
        client.create_vector_bucket(vectorBucketName=VECTOR_BUCKET)
        client.create_index(
            vectorBucketName=VECTOR_BUCKET, indexName=INDEX,
            dataType="float32", dimension=config.EMBED_DIMENSIONS,
            distanceMetric=config.EMBED_DISTANCE_METRIC,
            metadataConfiguration={"nonFilterableMetadataKeys": ["snippet"]},
        )
        yield client


def test_upsert_chunks_writes_namespaced_vectors(s3vectors_backend):
    doc = chunkers_mod.parse_document(_DOC, _SOURCE_URI)
    chunks = chunkers_mod.hierarchical(doc)
    embeddings = [[0.02] * config.EMBED_DIMENSIONS for _ in chunks]
    written = upsert_mod.upsert_chunks(
        VECTOR_BUCKET, INDEX, "hierarchical", "orders-export",
        chunks, embeddings, client=s3vectors_backend,
    )
    assert written == len(chunks)
    # The keys are namespaced by strategy and carry strategy+metadata.
    listed = s3vectors_backend.list_vectors(
        vectorBucketName=VECTOR_BUCKET, indexName=INDEX, returnMetadata=True,
    )
    keys = [v["key"] for v in listed["vectors"]]
    assert all(k.startswith("hierarchical#orders-export#") for k in keys)
    a_meta = listed["vectors"][0]["metadata"]
    assert a_meta["strategy"] == "hierarchical"
    assert a_meta["category"] == "orders"
    assert a_meta["source_uri"] == _SOURCE_URI


def test_upsert_length_mismatch_raises(s3vectors_backend):
    doc = chunkers_mod.parse_document(_DOC, _SOURCE_URI)
    chunks = chunkers_mod.hierarchical(doc)  # >1 chunk
    assert len(chunks) > 1
    with pytest.raises(ValueError):
        # One embedding for several chunks -> guarded mismatch.
        upsert_mod.upsert_chunks(
            VECTOR_BUCKET, INDEX, "hierarchical", "orders-export",
            chunks, [[0.0] * config.EMBED_DIMENSIONS], client=s3vectors_backend,
        )


def test_ingest_run_end_to_end_offline(monkeypatch, s3vectors_backend, tmp_path):
    """Full ingest pass: chunk -> embed (stubbed) -> upsert (moto), no creds."""
    # A tiny docs dir so the embed stub count is predictable.
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "orders-export.md").write_text(_DOC, encoding="utf-8")

    n_chunks = len(chunkers_mod.chunk_document(_DOC, "s3://x", "fixed"))
    titan_client, titan_stub = _stub_titan(monkeypatch, n_calls=n_chunks)

    with titan_stub:
        summary = run_mod.ingest_strategy(
            "fixed", docs_dir=docs_dir,
            runtime_client=titan_client, s3vectors_client=s3vectors_backend,
            account=ACCOUNT,
        )
    assert summary["strategy"] == "fixed"
    assert summary["vector_bucket"] == VECTOR_BUCKET
    assert summary["chunks"] == n_chunks
    assert summary["vectors_upserted"] == n_chunks
    assert summary["embed_cost"] >= 0.0


def _query_response(hits: list[tuple[str, float, dict]]) -> dict:
    return {
        "vectors": [
            {"key": key, "distance": dist, "metadata": meta}
            for key, dist, meta in hits
        ],
        "distanceMetric": "cosine",
    }


def test_query_returns_topk_with_similarity():
    # moto has no query_vectors; stub it. The query embeds nothing here — we feed a
    # vector directly and assert the wiring (topK, filter, similarity math).
    client = boto3.client("s3vectors", region_name="us-east-1")
    stub = Stubber(client)
    stub.add_response(
        "query_vectors",
        _query_response([
            ("hierarchical#orders-export#1", 0.08,
             {"category": "orders", "source_uri": _SOURCE_URI, "strategy": "hierarchical"}),
            ("hierarchical#orders-export#0", 0.21,
             {"category": "orders", "source_uri": _SOURCE_URI, "strategy": "hierarchical"}),
        ]),
        {
            "vectorBucketName": VECTOR_BUCKET, "indexName": INDEX, "topK": 3,
            "queryVector": {"float32": ANY}, "returnMetadata": True,
            "returnDistance": True, "filter": {"strategy": "hierarchical"},
        },
    )
    with stub:
        hits = upsert_mod.query(
            VECTOR_BUCKET, INDEX, [0.03] * config.EMBED_DIMENSIONS,
            top_k=3, strategy="hierarchical", client=client,
        )
    assert len(hits) == 2
    assert hits[0].key == "hierarchical#orders-export#1"
    assert hits[0].similarity == pytest.approx(1.0 - 0.08)
    assert hits[0].metadata["category"] == "orders"


def test_query_combines_strategy_and_category_filter():
    # Metadata filtering (skill 1.4.2): strategy AND category -> $and filter.
    client = boto3.client("s3vectors", region_name="us-east-1")
    stub = Stubber(client)
    stub.add_response(
        "query_vectors",
        _query_response([("fixed#billing-duplicate-charge#0", 0.1,
                          {"category": "billing", "strategy": "fixed"})]),
        {
            "vectorBucketName": VECTOR_BUCKET, "indexName": INDEX, "topK": 2,
            "queryVector": {"float32": ANY}, "returnMetadata": True,
            "returnDistance": True,
            "filter": {"$and": [{"strategy": "fixed"}, {"category": "billing"}]},
        },
    )
    with stub:
        hits = upsert_mod.query(
            VECTOR_BUCKET, INDEX, [0.03] * config.EMBED_DIMENSIONS,
            top_k=2, strategy="fixed", category="billing", client=client,
        )
    assert hits[0].metadata["category"] == "billing"


def test_compare_scoring_marks_top1_hit_and_recall():
    # score_question is pure given the kNN client; stub the three per-strategy
    # queries and assert the hit/recall bookkeeping.
    client = boto3.client("s3vectors", region_name="us-east-1")
    stub = Stubber(client)
    question = {"question": "How do I export my order history?",
                "relevant_docs": ["orders-export"], "category": "orders"}
    # fixed: relevant doc at rank 1 -> top-1 hit. hierarchical: also a hit.
    # semantic: a miss (wrong doc) -> no top-1 hit, 0 recall.
    for strategy, stem in (("fixed", "orders-export"),
                           ("hierarchical", "orders-export"),
                           ("semantic", "shipping-tracking")):
        stub.add_response(
            "query_vectors",
            _query_response([(f"{strategy}#{stem}#0", 0.1,
                              {"category": "orders", "strategy": strategy})]),
            {
                "vectorBucketName": VECTOR_BUCKET, "indexName": INDEX, "topK": 3,
                "queryVector": {"float32": ANY}, "returnMetadata": True,
                "returnDistance": True,
                "filter": {"$and": [{"strategy": strategy}, {"category": "orders"}]},
            },
        )
    with stub:
        scores = compare_chunking.score_question(
            question, [0.03] * config.EMBED_DIMENSIONS,
            vector_bucket=VECTOR_BUCKET, index=INDEX, top_k=3,
            s3vectors_client=client,
        )
    assert scores["fixed"].top1_hit is True
    assert scores["hierarchical"].top1_hit is True
    assert scores["semantic"].top1_hit is False
    assert scores["fixed"].recall_hits == 1
    assert scores["semantic"].recall_hits == 0


# ===========================================================================
# Module 4 — setup/teardown idempotency on a moto backend (no creds)
# ===========================================================================
def test_setup_is_idempotent_on_moto(tmp_path):
    from moto import mock_aws
    import setup as setup_mod

    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3v = boto3.client("s3vectors", region_name="us-east-1")
        data_bucket = config.relay_bucket(ACCOUNT)

        # Run the three setup steps TWICE — must not error or duplicate.
        for _ in range(2):
            setup_mod.ensure_data_bucket(s3, data_bucket)
            setup_mod.upload_docs(s3, data_bucket, docs_dir=DOCS_DIR)
            setup_mod.ensure_vector_bucket(s3v, VECTOR_BUCKET)
            setup_mod.ensure_index(s3v, VECTOR_BUCKET, INDEX)

        # docs/ is populated; the index exists with the pinned dims.
        listed = s3.list_objects_v2(Bucket=data_bucket, Prefix="docs/")
        assert listed["KeyCount"] >= 6
        idx = s3v.get_index(vectorBucketName=VECTOR_BUCKET, indexName=INDEX)
        assert idx["index"]["dimension"] == config.EMBED_DIMENSIONS


def test_teardown_is_idempotent_on_moto():
    from moto import mock_aws
    import teardown as teardown_mod

    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3v = boto3.client("s3vectors", region_name="us-east-1")
        data_bucket = config.relay_bucket(ACCOUNT)

        # Stand up resources, then tear down TWICE (second run is a clean no-op).
        s3.create_bucket(Bucket=data_bucket)
        s3.put_object(Bucket=data_bucket, Key="docs/x.md", Body=b"hi")
        s3v.create_vector_bucket(vectorBucketName=VECTOR_BUCKET)
        s3v.create_index(
            vectorBucketName=VECTOR_BUCKET, indexName=INDEX,
            dataType="float32", dimension=config.EMBED_DIMENSIONS,
            distanceMetric=config.EMBED_DISTANCE_METRIC,
        )

        for _ in range(2):
            teardown_mod.delete_vector_store(s3v, VECTOR_BUCKET, INDEX)
            teardown_mod.empty_and_delete_bucket(s3, data_bucket)

        assert VECTOR_BUCKET not in [
            b["vectorBucketName"] for b in s3v.list_vector_buckets().get("vectorBuckets", [])
        ]


# ===========================================================================
# Module 4 — boundary grep gates (what the lab must NOT contain)
# ===========================================================================
def test_exactly_one_invoke_model_in_the_whole_lab():
    """The course's sole invoke_model is Titan embeddings in ingest/embed.py.

    Assemble the tokens from fragments so this guard does not itself trip the grep
    gate. Scan all lab .py files (relay/, ingest/, scripts) — exactly one file may
    contain the token, and it must be ingest/embed.py.
    """
    token = "invoke" + "_model"
    offenders = []
    for path in _ROOT.rglob("*.py"):
        if ".venv" in path.parts or path.name == "smoke_test.py":
            continue
        if token in path.read_text(encoding="utf-8"):
            offenders.append(path.relative_to(_ROOT).as_posix())
    assert offenders == ["ingest/embed.py"], offenders


def test_no_generation_invoke_only_embeddings():
    """The single invoke_model produces a VECTOR, never text: it must reference the
    Titan EMBED model, and embed.py must not import relay.llm (no generation here).
    """
    embed_src = (INGEST_DIR / "embed.py").read_text(encoding="utf-8")
    assert config.EMBED_MODEL_ID in embed_src
    assert "from relay import llm" not in embed_src and "relay.llm" not in embed_src


def test_no_opensearch_in_lab_code():
    """OpenSearch is theory-only in the article; the lab CODE must not touch it."""
    pattern = re.compile(r"opensearch|aoss_|create_collection", re.IGNORECASE)
    offenders = []
    for path in _ROOT.rglob("*.py"):
        if ".venv" in path.parts or path.name == "smoke_test.py":
            continue
        if pattern.search(path.read_text(encoding="utf-8")):
            offenders.append(path.relative_to(_ROOT).as_posix())
    assert offenders == [], offenders


def test_no_downstream_concepts_in_lab_code():
    """No managed-KB / agent tokens in M4 code (those are M5 / M7)."""
    pattern = re.compile(r"RetrieveAndGenerate|relay-kb|search_kb")
    offenders = []
    for path in _ROOT.rglob("*.py"):
        if ".venv" in path.parts or path.name == "smoke_test.py":
            continue
        if pattern.search(path.read_text(encoding="utf-8")):
            offenders.append(path.relative_to(_ROOT).as_posix())
    assert offenders == [], offenders


# ===========================================================================
# The LIVE tests (opt-in) — budget: up to 4 calls total (2 M2/M3 + 2 M4)
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


@pytest.mark.live
@pytest.mark.skipif(
    os.environ.get("RELAY_LIVE_TESTS") != "1",
    reason="set RELAY_LIVE_TESTS=1 to make real (sub-cent) Bedrock calls",
)
def test_live_titan_embeds_a_doc_chunk_and_a_query():
    """TWO real Titan Text Embeddings V2 calls (a doc chunk + a query). ~$0.

    Embeddings, not text — this is the course's sole invoke_model, exercised live.
    It does NOT create any S3 Vectors resource; it only proves the embedder returns
    a 1024-dim vector for both a passage and a query, with a real token count.
    """
    doc_vector, doc_tokens = embed_mod.embed_one(
        "To export your order history, open Settings -> Data & Privacy -> "
        "Export data, tick Orders, and click Start export."
    )
    query_vector, query_tokens = embed_mod.embed_one("How do I export my orders?")
    assert len(doc_vector) == config.EMBED_DIMENSIONS
    assert len(query_vector) == config.EMBED_DIMENSIONS
    assert doc_tokens > 0 and query_tokens > 0
