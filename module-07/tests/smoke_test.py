"""smoke_test.py — Module 7 lab tests (cumulative: Modules 2, 3, 4, 5, 6 and 7).

OFFLINE BY DEFAULT. Every test here except the ones marked `live` runs with NO
AWS credentials and makes NO network call:
  - schema tests are pure Pydantic (the M2 Ticket/Triage contract, the M5
    Citation/Answer contract, and the M6 Attachment + Ticket.attachments contract —
    all frozen field-for-field);
  - the router tests are pure functions on fixed inputs (no model call);
  - the converse() tests drive relay.llm with a botocore Stubber on the
    bedrock-runtime client (Converse, ConverseStream, and a ThrottlingException
    that proves the backoff path);
  - the triage tests drive relay.triage with Stubbers for the bedrock-agent
    (get_prompt) and the bedrock-runtime (converse) clients;
  - the Module 4 chunker tests are pure and deterministic on a fixed doc;
  - the Module 4 ingestion tests stub Titan embeddings and run the S3 Vectors
    bucket/index/PutVectors lifecycle on a moto backend; the kNN query (which moto
    does not implement) is driven by a botocore Stubber;
  - the Module 5 Knowledge Base tests drive relay.kb (Retrieve /
    RetrieveAndGenerate) and the setup/teardown control plane with botocore
    Stubbers on the bedrock-agent / bedrock-agent-runtime clients;
  - the Module 6 intake tests are pure for parse/normalize/validation gates, and
    drive the full intake() pipeline with Stubbers for Amazon Comprehend
    (detect_entities, which moto only cans), S3 (put_object), and bedrock-runtime
    (the Nova Lite vision Converse call);
  - the Module 7 agent tests run the FULL Strands ReAct loop with a SCRIPTED fake
    model (no Bedrock call) and the mcp_server.store data layer on a moto DynamoDB
    backend, asserting the agent produces a TicketRecord with >=1 AgentAction; the
    MCP server's tool registry + the Lambda Function-URL adapter are exercised
    in-process; the IAM resource-boundary policy and setup/teardown are checked on
    moto + Stubbers.
That is the course convention — anyone can `uv run pytest` on a fresh clone.

TESTS marked `live` make real calls:
    RELAY_LIVE_TESTS=1 uv run pytest -m live
LIVE-CALL BUDGET: at most SEVEN calls total —
  Modules 2/3 (2 calls): one ConverseStream on the FAST tier
    (us.amazon.nova-micro-v1:0) and one Converse on the SMART tier
    (us.amazon.nova-2-lite-v1:0), both maxTokens<=64.
  Module 4 (2 calls): two Amazon Titan Text Embeddings V2 embeddings calls.
  Module 5 (1 call): ONE RetrieveAndGenerate against the live KB `relay-kb` on the
    smart tier (maxTokens small). Skips cleanly if the KB is not set up.
  Module 6 (1 call): ONE real Nova Lite VISION Converse call reading the bundled
    payment_error.png screenshot (maxTokens<=220). It needs only credentials +
    us-east-1 (no KB), and asserts it read the visible error.
  Module 7 (1 capped agent RUN): ONE real Strands agent run on the SMART tier against
    the deployed MCP server + seeded relay-orders. A ReAct loop is a FEW model calls
    inside one run, capped by the max-iterations stop condition (< $0.02). Skips
    cleanly if the MCP server / tables are not set up.
Together that is well under $0.04 as of June 2026. They need AWS credentials and
us-east-1. NO live test creates or deletes a KB / Lambda / table, and NONE uploads to
S3 (the live vision call reads the screenshot from local bytes).
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

# Import the lab packages (module-06/ ships the cumulative relay/ package — now
# with relay/intake.py — the inherited ingest/ pipeline, and the lab scripts).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from relay import config  # noqa: E402
from relay import llm  # noqa: E402
from relay import kb as kb_mod  # noqa: E402
from relay import intake as intake_mod  # noqa: E402
from relay import triage as triage_mod  # noqa: E402
from relay.models import Answer, Attachment, Citation, Ticket, Triage  # noqa: E402
from ingest import chunkers as chunkers_mod  # noqa: E402
from ingest import embed as embed_mod  # noqa: E402
from ingest import run as run_mod  # noqa: E402
from ingest import upsert as upsert_mod  # noqa: E402
import compare_chunking  # noqa: E402
import compare_retrieval  # noqa: E402

_ROOT = Path(__file__).resolve().parent.parent
TICKETS_DIR = _ROOT / "data" / "tickets"
DOCS_DIR = _ROOT / "data" / "docs"
RAW_DIR = _ROOT / "data" / "raw"
RELAY_DIR = _ROOT / "relay"
INGEST_DIR = _ROOT / "ingest"


# ===========================================================================
# Module 2 contract — schemas and ticket fixtures (still LAW at M5)
# ===========================================================================
def test_ticket_is_m2_four_plus_m6_attachments_no_pii():
    # M2 froze 4 fields; M6 adds EXACTLY one — `attachments` — by addition. The
    # Module 10 `pii_redacted` field must STILL be absent at Module 6.
    assert set(Ticket.model_fields) == {
        "ticket_id", "channel", "customer_message", "attachments", "created_at"
    }
    assert "pii_redacted" not in Ticket.model_fields
    # The four M2 fields are present and untouched (none renamed/retyped/removed).
    assert {"ticket_id", "channel", "customer_message", "created_at"} <= set(
        Ticket.model_fields
    )


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
    assert [p.name for p in params] == ["messages", "tier", "stream", "params"]


def test_tiers_are_the_canonical_set():
    # "auto" is the router's request, never a profile key. M6 APPENDS the "vision"
    # tier (Nova Lite) by addition; fast/smart/frontier are unchanged.
    assert "auto" not in config.TIERS
    assert set(config.TIERS) == {"fast", "smart", "frontier", "vision"}


def test_tier_map_points_at_inference_profiles_not_bare_ids():
    for tier, profile in config.TIERS.items():
        assert profile.startswith(("us.", "global.")), (tier, profile)
        assert not profile.startswith(("amazon.", "anthropic.")), (tier, profile)


def test_fast_and_smart_map_to_nova_micro_and_nova2_lite():
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

    banned in the relay package, INCLUDING M6's multimodal image read (it goes
    through Converse content blocks, never a model-specific invoke payload). Titan
    embeddings in Module 4's ingest/ is the sole exception in the whole course, and
    lives outside relay/.
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
    client, stub = _stub_runtime(monkeypatch)
    stub.add_client_error(
        "converse", service_error_code="ThrottlingException",
        service_message="Too many requests", http_status_code=429,
    )
    stub.add_response(
        "converse",
        _converse_response("Recovered after backoff.", 25, 10),
        {"modelId": config.tier_profile("fast"), "messages": ANY},
    )
    sleeps: list[int] = []
    monkeypatch.setattr(llm, "_backoff_sleep", lambda attempt: sleeps.append(attempt))
    with stub:
        result = llm.converse(_user("hi"), tier="fast")
    assert result.text == "Recovered after backoff."
    assert sleeps == [1]


def test_converse_raises_llmerror_on_nonretryable(monkeypatch):
    client, stub = _stub_runtime(monkeypatch)
    stub.add_client_error(
        "converse", service_error_code="ValidationException",
        service_message="Invalid request", http_status_code=400,
    )
    with stub:
        with pytest.raises(llm.LLMError):
            llm.converse(_user("hi"), tier="fast")


def test_converse_exhausts_then_degrades_smart_to_fast(monkeypatch):
    client, stub = _stub_runtime(monkeypatch)
    monkeypatch.setattr(llm, "_backoff_sleep", lambda attempt: None)
    for _ in range(6):
        stub.add_client_error(
            "converse", service_error_code="ThrottlingException",
            service_message="Too many requests", http_status_code=429,
        )
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
    runtime_stub.add_response(
        "converse",
        _converse_response(
            '{"intent": "refund", "priority": "high", "sentiment": "negative"}',
            in_tok=210, out_tok=18,
        ),
        {"modelId": config.tier_profile("fast"), "messages": ANY,
         "inferenceConfig": ANY},
    )
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
    agent, agent_stub = _stubbed_agent_client()
    runtime, runtime_stub = _stub_runtime(monkeypatch)
    runtime_stub.add_response(
        "converse",
        _converse_response(
            '{"intent": "billing", "priority": "high", "sentiment": "negative"}',
            210, 18,
        ),
        {
            "modelId": config.tier_profile("fast"),
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
# Module 4 — config additions (resource names + the pinned embedder)
# ===========================================================================
def test_resource_names_are_field_for_field_frozen():
    assert config.relay_bucket("111122223333") == "relay-111122223333"
    assert config.relay_vector_bucket("111122223333") == "relay-vectors-111122223333"
    assert config.RELAY_INDEX == "relay-docs"
    assert config.RELAY_BUCKET_PREFIXES == ("docs/", "attachments/", "vectors/")


def test_embedder_is_titan_v2_pinned_at_1024_dims():
    assert config.EMBED_MODEL_ID == "amazon.titan-embed-text-v2:0"
    assert config.EMBED_DIMENSIONS == 1024
    assert config.EMBED_DISTANCE_METRIC == "cosine"
    assert "nova" not in config.EMBED_MODEL_ID.lower()


def test_tier_map_fast_smart_never_repointed():
    # M4/M5 add only constants; M6 APPENDS the "vision" tier — but the M3 entries
    # (fast/smart/frontier) are never re-pointed (bible §2.2 "by addition only").
    assert config.tier_profile("fast") == "us.amazon.nova-micro-v1:0"
    assert config.tier_profile("smart") == "us.amazon.nova-2-lite-v1:0"
    assert config.tier_profile("frontier") == "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    # The vision tier is Nova Lite — NOT Nova 2 Lite (the "smart" tier).
    assert config.tier_profile("vision") == "us.amazon.nova-lite-v1:0"
    assert config.tier_profile("vision") != config.tier_profile("smart")


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
    for strategy in ("fixed", "hierarchical", "semantic"):
        a = chunkers_mod.chunk_document(_DOC, _SOURCE_URI, strategy)
        b = chunkers_mod.chunk_document(_DOC, _SOURCE_URI, strategy)
        assert [c.text for c in a] == [c.text for c in b]
        assert a, f"{strategy} produced no chunks"


def test_every_chunk_carries_canonical_metadata():
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
    where = next(c for c in chunks if "Where the export lives" in c.heading)
    assert "Settings -> Data & Privacy" in where.text


def test_semantic_never_splits_mid_sentence():
    chunks = chunkers_mod.chunk_document(_DOC, _SOURCE_URI, "semantic")
    for chunk in chunks:
        assert chunk.text.rstrip()[-1] in ".!?", chunk.text[-40:]


def test_fixed_size_overlap_repeats_boundary_text():
    doc = chunkers_mod.parse_document(_DOC, _SOURCE_URI)
    chunks = chunkers_mod.fixed_size(doc, chunk_chars=200, overlap_chars=60)
    assert len(chunks) >= 2
    tail = chunks[0].text[-40:]
    assert any(seg and seg in chunks[1].text for seg in (tail[-20:], tail[:20]))


def test_unknown_strategy_raises():
    with pytest.raises(ValueError):
        chunkers_mod.chunk_document(_DOC, _SOURCE_URI, "windowed")


def test_shipped_docs_all_parse_and_chunk():
    docs = sorted(DOCS_DIR.glob("*.md"))
    # M5 adds billing-plans.md to the M4 corpus of six -> at least seven now.
    assert len(docs) >= 7
    for path in docs:
        text = path.read_text(encoding="utf-8")
        doc = chunkers_mod.parse_document(text, f"s3://b/docs/{path.name}")
        assert doc.category != "uncategorized", path.name
        for strategy in ("fixed", "hierarchical", "semantic"):
            assert chunkers_mod.chunk_document(text, "s3://b/docs/x", strategy)


def test_m4_questions_reference_real_doc_stems():
    # The inherited M4 questions file (drives compare_chunking.py) is unchanged.
    questions = json.loads((_ROOT / "data" / "questions.json").read_text("utf-8"))
    stems = {p.stem for p in DOCS_DIR.glob("*.md")}
    assert len(questions) >= 8
    for q in questions:
        for ref in q["relevant_docs"]:
            assert ref in stems, f"{ref} not in {stems}"
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
    assert result.input_tokens == 21
    assert all(len(v) == config.EMBED_DIMENSIONS for v in result.vectors)


def test_embed_rejects_dimension_mismatch(monkeypatch):
    client = boto3.client("bedrock-runtime", region_name="us-east-1")
    stub = Stubber(client)
    stub.add_response(
        "invoke_model",
        _titan_invoke_response(dims=512),
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
    chunks = chunkers_mod.hierarchical(doc)
    assert len(chunks) > 1
    with pytest.raises(ValueError):
        upsert_mod.upsert_chunks(
            VECTOR_BUCKET, INDEX, "hierarchical", "orders-export",
            chunks, [[0.0] * config.EMBED_DIMENSIONS], client=s3vectors_backend,
        )


def test_ingest_run_end_to_end_offline(monkeypatch, s3vectors_backend, tmp_path):
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


def test_compare_chunking_scoring_marks_top1_hit_and_recall():
    client = boto3.client("s3vectors", region_name="us-east-1")
    stub = Stubber(client)
    question = {"question": "How do I export my order history?",
                "relevant_docs": ["orders-export"], "category": "orders"}
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
# Module 4 — setup/teardown of the M4 STORAGE layer were owned by M4.
# Module 5's setup/teardown manage the KB; the M4 storage helpers are now
# imported by M5 setup (precheck) — covered in the M5 section below.
# ===========================================================================


# ===========================================================================
# Module 5 — the FROZEN Citation / Answer contract (06 §2 / bible §3.1)
# ===========================================================================
def test_module7_schema_set_is_m6_plus_agentaction_ticketrecord():
    """M7 adds EXACTLY AgentAction + TicketRecord to the M2–M6 schemas — nothing more.

    Guarding the full set keeps the "by addition only, in order" invariant honest: at
    Module 7 the schemas are precisely {Ticket, Triage, Citation, Answer, Attachment,
    AgentAction, TicketRecord}. (feedback_rating is a Module 13 field on TicketRecord;
    it must not appear yet.)
    """
    import relay.models as models_mod
    from pydantic import BaseModel

    schema_names = {
        name for name, obj in vars(models_mod).items()
        if isinstance(obj, type) and issubclass(obj, BaseModel) and obj is not BaseModel
    }
    assert schema_names == {
        "Ticket", "Triage", "Citation", "Answer", "Attachment",
        "AgentAction", "TicketRecord",
    }
    # Ticket is untouched by M7 (still M2's 4 + M6's attachments); still NO pii_redacted.
    assert set(Ticket.model_fields) == {
        "ticket_id", "channel", "customer_message", "attachments", "created_at"
    }
    assert "pii_redacted" not in Ticket.model_fields


def test_citation_is_exactly_two_fields_no_score():
    # Frozen M5: {source_uri, snippet}. NO score / confidence field, ever.
    assert set(Citation.model_fields) == {"source_uri", "snippet"}
    for forbidden in ("score", "confidence", "rank"):
        assert forbidden not in Citation.model_fields


def test_answer_is_exactly_three_fields_with_grounded_bool():
    # Frozen M5: {text, citations: list[Citation], grounded: bool}.
    assert set(Answer.model_fields) == {"text", "citations", "grounded"}
    assert Answer.model_fields["grounded"].annotation is bool
    a = Answer(text="x", citations=[Citation(source_uri="s3://b/docs/p.md",
                                             snippet="hi")], grounded=True)
    assert a.citations[0].source_uri == "s3://b/docs/p.md"
    assert a.grounded is True


def test_answer_accepts_zero_citations_grounded_false():
    a = Answer(text="I don't know.", citations=[], grounded=False)
    assert a.citations == []
    assert a.grounded is False


# ===========================================================================
# Module 5 — config additions (KB name, reranker, model_arn) — by addition
# ===========================================================================
def test_kb_resource_names_are_frozen():
    assert config.RELAY_KB_NAME == "relay-kb"
    assert config.RELAY_KB_INCLUSION_PREFIX == "docs/"
    assert config.KB_ANSWER_TIER == "smart"


def test_model_arn_is_built_from_the_tier_map_not_hardcoded():
    arn = config.model_arn("smart", account="111122223333")
    assert arn == (
        "arn:aws:bedrock:us-east-1:111122223333:inference-profile/"
        "us.amazon.nova-2-lite-v1:0"
    )
    # It uses the SAME profile the converse() layer uses (single source of IDs).
    assert config.tier_profile("smart") in arn


def test_rerank_model_arn_uses_pinned_cohere_rerank():
    # Live-verified June 2026: cohere.rerank-v3-5:0 is the ONLY reranker in the
    # us-east-1 catalogue (amazon.rerank-v1:0 is not available there). The default
    # is the available one; Amazon Rerank is kept as the documented alternative.
    arn = config.rerank_model_arn()
    assert arn == "arn:aws:bedrock:us-east-1::foundation-model/cohere.rerank-v3-5:0"
    assert config.RERANK_MODEL_ID == "cohere.rerank-v3-5:0"
    assert config.RERANK_ALT_MODEL_ID == "amazon.rerank-v1:0"


def test_kb_config_additions_did_not_touch_the_tier_map_or_embedder():
    # M5/M6 only append constants; the M4 embedder is untouched and M6 must NOT swap
    # it for the Nova multimodal embeddings (that would invalidate the index).
    assert config.EMBED_MODEL_ID == "amazon.titan-embed-text-v2:0"
    assert config.EMBED_DIMENSIONS == 1024
    assert "nova" not in config.EMBED_MODEL_ID.lower()
    # The M3 fast/smart entries are still pinned (M6 only APPENDED "vision").
    assert config.tier_profile("fast") == "us.amazon.nova-micro-v1:0"
    assert config.tier_profile("smart") == "us.amazon.nova-2-lite-v1:0"


# ===========================================================================
# Module 5 — relay.kb retrieve()/answer() over a stubbed agent-runtime client
# ===========================================================================
def _stub_kb_runtime() -> tuple[object, Stubber]:
    client = boto3.client("bedrock-agent-runtime", region_name="us-east-1")
    return client, Stubber(client)


def _retrieve_response(refs: list[tuple[str, str, float]]) -> dict:
    return {
        "retrievalResults": [
            {
                "content": {"text": text, "type": "TEXT"},
                "location": {"type": "S3", "s3Location": {"uri": uri}},
                "score": score,
                "metadata": {"category": "billing"},
            }
            for text, uri, score in refs
        ]
    }


def _rag_response(text: str, refs: list[tuple[str, str]]) -> dict:
    return {
        "output": {"text": text},
        "citations": [
            {
                "generatedResponsePart": {
                    "textResponsePart": {"text": text, "span": {"start": 0,
                                                                "end": len(text)}}
                },
                "retrievedReferences": [
                    {
                        "content": {"text": snippet, "type": "TEXT"},
                        "location": {"type": "S3", "s3Location": {"uri": uri}},
                        "metadata": {"category": "billing"},
                    }
                    for snippet, uri in refs
                ],
            }
        ],
        "sessionId": "sess-1",
    }


def test_retrieve_hybrid_with_rerank_builds_the_right_request():
    client, stub = _stub_kb_runtime()
    stub.add_response(
        "retrieve",
        _retrieve_response([
            ("The Growth plan is $79 per month.",
             "s3://relay-111122223333/docs/billing-plans.md", 0.93),
        ]),
        {
            "knowledgeBaseId": "KB123",
            "retrievalQuery": {"text": "How much is the Growth plan?"},
            "retrievalConfiguration": {
                "vectorSearchConfiguration": {
                    "numberOfResults": 5,
                    "overrideSearchType": "HYBRID",
                    "filter": {"equals": {"key": "category", "value": "billing"}},
                    "rerankingConfiguration": {
                        "type": "BEDROCK_RERANKING_MODEL",
                        "bedrockRerankingConfiguration": {
                            "modelConfiguration": {
                                "modelArn": config.rerank_model_arn(),
                            },
                            "numberOfRerankedResults": 5,
                        },
                    },
                }
            },
        },
    )
    with stub:
        hits = kb_mod.retrieve(
            "How much is the Growth plan?", top_k=5,
            search_type=kb_mod.SEARCH_HYBRID, rerank=True, category="billing",
            kb_id="KB123", client=client,
        )
    assert len(hits) == 1
    assert hits[0].source_uri.endswith("billing-plans.md")
    assert hits[0].score == pytest.approx(0.93)
    # The retrieval score is on Retrieved, NOT on the frozen Citation schema.
    assert "score" not in Citation.model_fields


def test_retrieve_semantic_without_rerank_omits_rerank_block():
    client, stub = _stub_kb_runtime()
    stub.add_response(
        "retrieve",
        _retrieve_response([("chunk", "s3://b/docs/x.md", 0.5)]),
        {
            "knowledgeBaseId": "KB123",
            "retrievalQuery": {"text": "q"},
            "retrievalConfiguration": {
                "vectorSearchConfiguration": {
                    "numberOfResults": 3,
                    "overrideSearchType": "SEMANTIC",
                }
            },
        },
    )
    with stub:
        hits = kb_mod.retrieve("q", top_k=3, search_type=kb_mod.SEARCH_SEMANTIC,
                               rerank=False, kb_id="KB123", client=client)
    assert len(hits) == 1


def test_retrieve_rejects_unknown_search_type():
    client, _ = _stub_kb_runtime()
    with pytest.raises(ValueError):
        kb_mod.retrieve("q", search_type="FUZZY", kb_id="KB1", client=client)


def test_answer_maps_rag_response_to_frozen_answer_grounded_true():
    client, stub = _stub_kb_runtime()
    stub.add_response(
        "retrieve_and_generate",
        _rag_response(
            "Open Billing -> Subscription and click Change plan.",
            [("Open Billing -> Subscription and click Change plan.",
              "s3://relay-111122223333/docs/billing-plans.md")],
        ),
        {
            "input": {"text": "How do I change my plan?"},
            "retrieveAndGenerateConfiguration": {
                "type": "KNOWLEDGE_BASE",
                "knowledgeBaseConfiguration": {
                    "knowledgeBaseId": "KB123",
                    "modelArn": config.model_arn("smart", account=ACCOUNT),
                    "retrievalConfiguration": {
                        "vectorSearchConfiguration": {
                            "numberOfResults": 5,
                            # answer() defaults to SEMANTIC (the only mode S3
                            # Vectors supports) with the reranker on for precision.
                            "overrideSearchType": "SEMANTIC",
                            "rerankingConfiguration": ANY,
                        }
                    },
                },
            },
        },
    )
    with stub:
        result = kb_mod.answer(
            "How do I change my plan?", kb_id="KB123", account=ACCOUNT,
            client=client,
        )
    assert isinstance(result, Answer)
    assert "Change plan" in result.text
    assert len(result.citations) == 1
    assert isinstance(result.citations[0], Citation)
    assert result.citations[0].source_uri.endswith("billing-plans.md")
    # M5 heuristic: cited at least one source -> grounded True.
    assert result.grounded is True


def test_answer_grounded_false_when_no_citations():
    client, stub = _stub_kb_runtime()
    stub.add_response(
        "retrieve_and_generate",
        {"output": {"text": "I could not find that in the docs."},
         "citations": [], "sessionId": "sess-empty"},
        {"input": ANY, "retrieveAndGenerateConfiguration": ANY},
    )
    with stub:
        result = kb_mod.answer("something off-topic", kb_id="KB123",
                               account=ACCOUNT, client=client)
    assert result.citations == []
    assert result.grounded is False  # bool(citations) heuristic


def test_answer_decompose_turns_on_query_decomposition():
    client, stub = _stub_kb_runtime()
    stub.add_response(
        "retrieve_and_generate",
        _rag_response("Downgrade keeps your order history.",
                      [("history", "s3://b/docs/billing-plans.md"),
                       ("export", "s3://b/docs/orders-export.md")]),
        {
            "input": {"text": "downgrade and keep history?"},
            "retrieveAndGenerateConfiguration": {
                "type": "KNOWLEDGE_BASE",
                "knowledgeBaseConfiguration": {
                    "knowledgeBaseId": "KB123",
                    "modelArn": config.model_arn("smart", account=ACCOUNT),
                    "retrievalConfiguration": ANY,
                    "orchestrationConfiguration": {
                        "queryTransformationConfiguration": {
                            "type": "QUERY_DECOMPOSITION"
                        }
                    },
                },
            },
        },
    )
    with stub:
        result = kb_mod.answer("downgrade and keep history?", decompose=True,
                               kb_id="KB123", account=ACCOUNT, client=client)
    # Two distinct sources cited (one per sub-query) -> two citations.
    assert len(result.citations) == 2
    assert result.grounded is True


def test_answer_dedupes_repeated_references():
    client, stub = _stub_kb_runtime()
    # Same passage cited for two spans -> one Citation after de-dup.
    stub.add_response(
        "retrieve_and_generate",
        {
            "output": {"text": "..."},
            "citations": [
                {"retrievedReferences": [
                    {"content": {"text": "same"},
                     "location": {"type": "S3",
                                  "s3Location": {"uri": "s3://b/docs/x.md"}}}]},
                {"retrievedReferences": [
                    {"content": {"text": "same"},
                     "location": {"type": "S3",
                                  "s3Location": {"uri": "s3://b/docs/x.md"}}}]},
            ],
            "sessionId": "sess-dedupe",
        },
        {"input": ANY, "retrieveAndGenerateConfiguration": ANY},
    )
    with stub:
        result = kb_mod.answer("q", kb_id="KB123", account=ACCOUNT, client=client)
    assert len(result.citations) == 1


def test_kb_call_raises_kberror_on_client_error():
    client, stub = _stub_kb_runtime()
    stub.add_client_error(
        "retrieve", service_error_code="ResourceNotFoundException",
        service_message="KB not found", http_status_code=404,
    )
    with stub:
        with pytest.raises(kb_mod.KBError):
            kb_mod.retrieve("q", kb_id="MISSING", client=client)


def test_resolve_kb_id_errors_without_setup(monkeypatch, tmp_path):
    monkeypatch.delenv("RELAY_KB_ID", raising=False)
    monkeypatch.setattr(kb_mod, "_KB_ID_FILE", tmp_path / "nope")
    with pytest.raises(kb_mod.KBError) as excinfo:
        kb_mod.resolve_kb_id()
    assert "setup.py" in str(excinfo.value)


# ===========================================================================
# Module 5 — compare_retrieval scoring over the four configurations (offline)
# ===========================================================================
def test_compare_retrieval_scores_four_configs_offline():
    """One question scored across M4 DIY + KB semantic / semantic+rerank / hybrid.

    The DIY column uses a stubbed S3 Vectors kNN; the three KB columns use a
    stubbed agent-runtime Retrieve, in the order score_question calls them
    (semantic, semantic+rerank, hybrid). We assert the bookkeeping: the reranker
    pulls the relevant doc to rank 1 where plain semantic missed it (the
    precision story, in miniature)."""
    s3v = boto3.client("s3vectors", region_name="us-east-1")
    s3v_stub = Stubber(s3v)
    # DIY hierarchical kNN: relevant doc NOT at rank 1 (a near-miss on the exact id).
    s3v_stub.add_response(
        "query_vectors",
        _query_response([("hierarchical#billing-duplicate-charge#0", 0.2,
                          {"category": "billing", "strategy": "hierarchical"})]),
        {
            "vectorBucketName": VECTOR_BUCKET, "indexName": INDEX, "topK": 5,
            "queryVector": {"float32": ANY}, "returnMetadata": True,
            "returnDistance": True,
            "filter": {"$and": [{"strategy": "hierarchical"},
                                {"category": "billing"}]},
        },
    )

    kb_client = boto3.client("bedrock-agent-runtime", region_name="us-east-1")
    kb_stub = Stubber(kb_client)
    # KB semantic (rerank off): still a miss on the exact plan name at rank 1.
    kb_stub.add_response(
        "retrieve",
        _retrieve_response([("wrong", "s3://b/docs/billing-duplicate-charge.md", 0.6)]),
        {"knowledgeBaseId": "KB1", "retrievalQuery": ANY,
         "retrievalConfiguration": ANY},
    )
    # KB semantic+rerank: reranker pulls the plan doc to rank 1.
    kb_stub.add_response(
        "retrieve",
        _retrieve_response([("Growth $79", "s3://b/docs/billing-plans.md", 0.99)]),
        {"knowledgeBaseId": "KB1", "retrievalQuery": ANY,
         "retrievalConfiguration": ANY},
    )
    # KB hybrid: in this offline stub it returns data (a hybrid-capable store); on
    # real S3 Vectors this call raises and the column is n/a (covered separately).
    kb_stub.add_response(
        "retrieve",
        _retrieve_response([("Growth $79", "s3://b/docs/billing-plans.md", 0.95)]),
        {"knowledgeBaseId": "KB1", "retrievalQuery": ANY,
         "retrievalConfiguration": ANY},
    )

    question = {"question": "How much does the Growth plan cost per month?",
                "relevant_docs": ["billing-plans"], "category": "billing"}
    with s3v_stub, kb_stub:
        scores = compare_retrieval.score_question(
            question, [0.03] * config.EMBED_DIMENSIONS,
            vector_bucket=VECTOR_BUCKET, index=INDEX, top_k=5, kb_id="KB1",
            s3vectors_client=s3v, kb_client=kb_client,
        )
    assert scores["m4_diy"].top1_hit is False
    assert scores["kb_semantic"].top1_hit is False
    assert scores["kb_sem_rerank"].top1_hit is True
    assert scores["kb_hybrid"].top1_hit is True
    assert scores["kb_hybrid"].na is False


def test_compare_retrieval_marks_hybrid_na_on_s3_vectors():
    """When HYBRID retrieval raises (S3 Vectors), the kb_hybrid cell is n/a — not
    a misleading zero. The semantic columns still score from their stubs."""
    s3v = boto3.client("s3vectors", region_name="us-east-1")
    s3v_stub = Stubber(s3v)
    s3v_stub.add_response(
        "query_vectors",
        _query_response([("hierarchical#billing-plans#0", 0.1,
                          {"category": "billing", "strategy": "hierarchical"})]),
        {"vectorBucketName": VECTOR_BUCKET, "indexName": INDEX, "topK": 5,
         "queryVector": {"float32": ANY}, "returnMetadata": True,
         "returnDistance": True,
         "filter": {"$and": [{"strategy": "hierarchical"}, {"category": "billing"}]}},
    )
    kb_client = boto3.client("bedrock-agent-runtime", region_name="us-east-1")
    kb_stub = Stubber(kb_client)
    # semantic + semantic-rerank succeed; the third (hybrid) raises like S3 Vectors.
    for _ in range(2):
        kb_stub.add_response(
            "retrieve",
            _retrieve_response([("Growth $79", "s3://b/docs/billing-plans.md", 0.95)]),
            {"knowledgeBaseId": "KB1", "retrievalQuery": ANY,
             "retrievalConfiguration": ANY},
        )
    kb_stub.add_client_error(
        "retrieve", service_error_code="ValidationException",
        service_message="HYBRID search type is not supported for search operation",
        http_status_code=400,
    )
    question = {"question": "How much does the Growth plan cost per month?",
                "relevant_docs": ["billing-plans"], "category": "billing"}
    with s3v_stub, kb_stub:
        scores = compare_retrieval.score_question(
            question, [0.03] * config.EMBED_DIMENSIONS,
            vector_bucket=VECTOR_BUCKET, index=INDEX, top_k=5, kb_id="KB1",
            s3vectors_client=s3v, kb_client=kb_client,
        )
    assert scores["kb_hybrid"].na is True
    assert scores["kb_semantic"].top1_hit is True
    assert scores["kb_sem_rerank"].top1_hit is True


def test_kb_questions_reference_real_doc_stems_and_have_exact_ids():
    questions = json.loads(
        (_ROOT / "data" / "kb_questions.json").read_text("utf-8")
    )
    stems = {p.stem for p in DOCS_DIR.glob("*.md")}
    assert len(questions) >= 8
    for q in questions:
        for ref in q["relevant_docs"]:
            assert ref in stems, f"{ref} not in {stems}"
    # At least two carry an EXACT identifier (skill 1.5.4 hybrid story).
    exact = [q for q in questions if q.get("exact_identifier")]
    assert len(exact) >= 2
    assert any("ERR-402" in q["question"] for q in questions)
    # And at least one COMPOUND question for query decomposition (skill 1.5.5).
    assert any(q.get("compound") for q in questions)
    # The freshness doc the lab edits must exist in the corpus.
    assert "billing-plans" in stems


# ===========================================================================
# Module 5 — setup/teardown over stubbed bedrock-agent + moto IAM (offline)
# ===========================================================================
def test_setup_kb_role_is_idempotent_on_moto():
    from moto import mock_aws
    import setup as setup_mod

    with mock_aws():
        iam = boto3.client("iam", region_name="us-east-1")
        data_bucket = config.relay_bucket(ACCOUNT)
        vector_bucket = config.relay_vector_bucket(ACCOUNT)
        # Run TWICE — second run must not error or duplicate the role.
        for _ in range(2):
            arn = setup_mod.ensure_kb_role(iam, ACCOUNT, data_bucket, vector_bucket)
        assert arn.endswith(f"role/{setup_mod.KB_ROLE_NAME}")
        policies = iam.list_role_policies(
            RoleName=setup_mod.KB_ROLE_NAME)["PolicyNames"]
        assert "relay-kb-permissions" in policies
        # The inline policy is least-privilege: real ARNs, no "*" Resource —
        # EXCEPT bedrock:Rerank, the one action AWS requires "*" on (it has no
        # resource-level scoping; the rerank MODEL is still pinned by the
        # InvokeModel statement). See the Bedrock "Permissions for reranking" doc.
        pol = iam.get_role_policy(
            RoleName=setup_mod.KB_ROLE_NAME, PolicyName="relay-kb-permissions"
        )["PolicyDocument"]
        for stmt in pol["Statement"]:
            actions = stmt["Action"]
            actions = actions if isinstance(actions, list) else [actions]
            resources = stmt["Resource"]
            resources = resources if isinstance(resources, list) else [resources]
            if actions == ["bedrock:Rerank"]:
                # The documented exception: Rerank requires Resource "*".
                assert resources == ["*"], stmt
                continue
            assert all(r != "*" for r in resources), stmt


def test_setup_finds_existing_kb_idempotently_via_stub():
    import setup as setup_mod

    agent = boto3.client("bedrock-agent", region_name="us-east-1")
    stub = Stubber(agent)
    stub.add_response(
        "list_knowledge_bases",
        {"knowledgeBaseSummaries": [
            {"knowledgeBaseId": "KBEXIST", "name": config.RELAY_KB_NAME,
             "status": "ACTIVE", "updatedAt": dt.datetime(2026, 6, 1)}
        ]},
        {},
    )
    with stub:
        kb_id = setup_mod._find_kb_by_name(agent, config.RELAY_KB_NAME)
    assert kb_id == "KBEXIST"


def test_setup_ingestion_waits_for_complete_via_stub():
    import setup as setup_mod

    agent = boto3.client("bedrock-agent", region_name="us-east-1")
    stub = Stubber(agent)
    stub.add_response(
        "start_ingestion_job",
        {"ingestionJob": {"knowledgeBaseId": "KB1", "dataSourceId": "DS1",
                          "ingestionJobId": "JOB1", "status": "STARTING",
                          "updatedAt": dt.datetime(2026, 6, 1),
                          "startedAt": dt.datetime(2026, 6, 1)}},
        {"knowledgeBaseId": "KB1", "dataSourceId": "DS1"},
    )
    stub.add_response(
        "get_ingestion_job",
        {"ingestionJob": {"knowledgeBaseId": "KB1", "dataSourceId": "DS1",
                          "ingestionJobId": "JOB1", "status": "COMPLETE",
                          "updatedAt": dt.datetime(2026, 6, 1),
                          "startedAt": dt.datetime(2026, 6, 1)}},
        {"knowledgeBaseId": "KB1", "dataSourceId": "DS1", "ingestionJobId": "JOB1"},
    )
    import relay  # noqa
    # Make the poll instant.
    setup_mod._INGESTION_POLL_S = 0
    with stub:
        status = setup_mod.start_ingestion(agent, "KB1", "DS1", wait=True)
    assert status == "COMPLETE"


def test_teardown_deletes_kb_and_role_idempotently():
    import setup as setup_mod
    import teardown as teardown_mod

    # KB deletion path: list data sources, delete each, delete KB, then it is gone.
    agent = boto3.client("bedrock-agent", region_name="us-east-1")
    stub = Stubber(agent)
    stub.add_response(
        "list_data_sources",
        {"dataSourceSummaries": [
            {"knowledgeBaseId": "KB1", "dataSourceId": "DS1",
             "name": config.RELAY_KB_DATA_SOURCE_NAME, "status": "AVAILABLE",
             "updatedAt": dt.datetime(2026, 6, 1)}
        ]},
        {"knowledgeBaseId": "KB1"},
    )
    stub.add_response(
        "delete_data_source",
        {"knowledgeBaseId": "KB1", "dataSourceId": "DS1", "status": "DELETING"},
        {"knowledgeBaseId": "KB1", "dataSourceId": "DS1"},
    )
    stub.add_response(
        "delete_knowledge_base",
        {"knowledgeBaseId": "KB1", "status": "DELETING"},
        {"knowledgeBaseId": "KB1"},
    )
    stub.add_client_error(
        "get_knowledge_base", service_error_code="ResourceNotFoundException",
        service_message="gone", http_status_code=404,
    )
    # Avoid touching the real .kb_id files on disk during the test.
    teardown_mod.KB_ID_FILE = Path("/tmp/_relay_kb_id_nope")
    teardown_mod.KB_DATA_SOURCE_ID_FILE = Path("/tmp/_relay_ds_id_nope")
    with stub:
        teardown_mod.delete_knowledge_base(agent, "KB1")

    # Role deletion on moto, idempotent.
    from moto import mock_aws

    with mock_aws():
        iam = boto3.client("iam", region_name="us-east-1")
        setup_mod.ensure_kb_role(iam, ACCOUNT, config.relay_bucket(ACCOUNT),
                                 config.relay_vector_bucket(ACCOUNT))
        for _ in range(2):  # delete twice — second is a clean no-op
            teardown_mod.delete_kb_role(iam)
        from botocore.exceptions import ClientError
        with pytest.raises(ClientError):
            iam.get_role(RoleName=setup_mod.KB_ROLE_NAME)


# ===========================================================================
# Module 6 — the FROZEN Attachment contract + Ticket.attachments (06 §2 / §3.1)
# ===========================================================================
def test_attachment_is_exactly_three_fields():
    # Frozen M6 field-for-field: {filename, media_type, s3_uri}. No bytes, no size.
    assert set(Attachment.model_fields) == {"filename", "media_type", "s3_uri"}
    att = Attachment(filename="payment_error.png", media_type="image/png",
                     s3_uri="s3://relay-111122223333/attachments/x-payment_error.png")
    assert att.media_type == "image/png"
    assert att.s3_uri.startswith("s3://")


def test_ticket_attachments_defaults_empty_and_back_compat():
    # The default [] is load-bearing: an M2–M5 ticket fixture (no attachments key)
    # still validates, AND a ticket can now carry attachments.
    legacy = Ticket(ticket_id="t1", channel="email",
                    customer_message="hi", created_at="2026-06-01T00:00:00Z")
    assert legacy.attachments == []
    withatt = Ticket(
        ticket_id="t2", channel="email", customer_message="see screenshot",
        attachments=[Attachment(filename="s.png", media_type="image/png",
                                s3_uri="s3://b/attachments/s.png")],
        created_at="2026-06-01T00:00:00Z",
    )
    assert len(withatt.attachments) == 1
    assert isinstance(withatt.attachments[0], Attachment)


def test_every_existing_ticket_fixture_still_validates_with_new_field():
    # All ten M2 fixtures (no attachments key) validate against the extended Ticket.
    for f in sorted(TICKETS_DIR.glob("ticket-*.json")):
        t = triage_mod.load_ticket(f)
        assert t.attachments == []


# ===========================================================================
# Module 6 — config additions (vision tier + intake policy) — by addition
# ===========================================================================
def test_vision_tier_is_nova_lite_via_config_only():
    assert config.VISION_TIER == "vision"
    assert config.tier_profile("vision") == "us.amazon.nova-lite-v1:0"
    # Nova Lite (vision) is NOT Nova 2 Lite (smart) — the easy confusion.
    assert config.tier_profile("vision") != config.tier_profile("smart")
    # The ID lives ONLY in config — never bare in intake.py / llm.py.
    for name in ("intake.py", "llm.py"):
        src = (RELAY_DIR / name).read_text(encoding="utf-8")
        assert not re.search(r"(us|global)\.(amazon|anthropic)\.", src), name


def test_intake_policy_constants_are_frozen():
    assert config.RELAY_ATTACHMENTS_PREFIX == "attachments/"
    assert config.RELAY_ATTACHMENTS_PREFIX in config.RELAY_BUCKET_PREFIXES
    assert config.MAX_MESSAGE_BYTES == 16 * 1024
    assert config.MESSAGE_ENCODING == "utf-8"
    assert config.COMPREHEND_LANGUAGE_CODE == "en"
    # Admitted attachment types are images only, and match the llm image map.
    assert set(config.ADMITTED_ATTACHMENT_MEDIA_TYPES) == set(
        llm.IMAGE_MEDIA_TYPE_TO_FORMAT
    )
    assert config.media_type_for_filename("Shot.PNG") == "image/png"
    assert config.media_type_for_filename("a.jpeg") == "image/jpeg"
    assert config.media_type_for_filename("a.pdf") is None


# ===========================================================================
# Module 6 — relay.llm.image_block (the Converse image content block)
# ===========================================================================
def test_image_block_builds_converse_native_shape():
    block = llm.image_block(b"\x89PNG\r\n", "image/png")
    assert block == {"image": {"format": "png",
                               "source": {"bytes": b"\x89PNG\r\n"}}}
    # Raw bytes, not base64 — boto3 encodes for the wire.
    assert block["image"]["source"]["bytes"] == b"\x89PNG\r\n"


def test_image_block_maps_each_admitted_type_to_a_converse_format():
    assert llm.image_block(b"x", "image/jpeg")["image"]["format"] == "jpeg"
    assert llm.image_block(b"x", "image/gif")["image"]["format"] == "gif"
    assert llm.image_block(b"x", "image/webp")["image"]["format"] == "webp"


def test_image_block_rejects_unsupported_media_type():
    with pytest.raises(ValueError):
        llm.image_block(b"x", "application/pdf")


# ===========================================================================
# Module 6 — parsing + normalization (pure, deterministic)
# ===========================================================================
def test_parse_raw_reads_header_and_body():
    raw = intake_mod.parse_raw(
        "channel: email\nticket_id: r-1\ncreated_at: 2026-06-01T00:00:00Z\n\nHello."
    )
    assert raw.channel == "email"
    assert raw.ticket_id == "r-1"
    assert raw.created_at == "2026-06-01T00:00:00Z"
    assert raw.body == "Hello."


def test_normalize_strips_quoted_thread_signature_and_html():
    body = (
        "<div><p>My plan renewal fails at checkout. Order #1042.</p></div>\n"
        "Thanks,\nDana\n--\nDana Whitfield\n+1 (555) 014-2231\n"
        "Confidential footer here.\n"
        "> On 2026-06-11, Support wrote:\n> earlier message\n"
    )
    norm = intake_mod.normalize(body)
    assert "#1042" in norm
    assert ">" not in norm                      # quoted thread gone
    assert "+1 (555)" not in norm               # signature gone
    assert "Confidential" not in norm           # footer gone
    assert "<div" not in norm and "<p>" not in norm  # HTML unwrapped


def test_normalize_on_the_shipped_billing_fixture():
    raw = intake_mod.parse_raw(
        (RAW_DIR / "email_billing_error.txt").read_text(encoding="utf-8")
    )
    norm = intake_mod.normalize(raw.body)
    for noise in (">", "confidential", "delaware", "+1 (555)", "operations lead"):
        assert noise.lower() not in norm.lower(), noise
    assert "#1042" in norm and "screenshot" in norm


# ===========================================================================
# Module 6 — validation gates (skill 1.3.1): explicit typed rejections
# ===========================================================================
def test_gate_rejects_oversized_message():
    raw = intake_mod.parse_raw(
        (RAW_DIR / "invalid_oversized.txt").read_text(encoding="utf-8")
    )
    with pytest.raises(intake_mod.IntakeRejected) as exc:
        intake_mod.validate_raw_bytes(raw.body.encode("utf-8"))
    assert exc.value.reason == "message_too_large"


def test_gate_rejects_empty_after_normalization():
    raw = intake_mod.parse_raw(
        (RAW_DIR / "invalid_empty.txt").read_text(encoding="utf-8")
    )
    normalized = intake_mod.normalize(raw.body)
    with pytest.raises(intake_mod.IntakeRejected) as exc:
        intake_mod.validate_nonempty(normalized)
    assert exc.value.reason == "empty_message"


def test_gate_rejects_non_image_attachment():
    with pytest.raises(intake_mod.IntakeRejected) as exc:
        intake_mod.validate_attachment("invoice.pdf", b"%PDF-1.4 ...")
    assert exc.value.reason == "bad_attachment_type"


def test_gate_rejects_oversized_attachment():
    big = b"\x00" * (config.MAX_ATTACHMENT_BYTES + 1)
    with pytest.raises(intake_mod.IntakeRejected) as exc:
        intake_mod.validate_attachment("shot.png", big)
    assert exc.value.reason == "attachment_too_large"


def test_gate_rejects_unknown_channel():
    with pytest.raises(intake_mod.IntakeRejected) as exc:
        intake_mod.validate_channel("phone")
    assert exc.value.reason == "bad_channel"


def test_gate_rejects_binary_message():
    with pytest.raises(intake_mod.IntakeRejected) as exc:
        intake_mod.validate_raw_bytes(b"\xff\xfe\x00\x01 not utf-8")
    assert exc.value.reason == "bad_encoding"


# ===========================================================================
# Module 6 — Amazon Comprehend entity extraction (skill 1.3.4), stubbed offline
# ===========================================================================
def _stub_comprehend(entities: list[tuple[str, str]]) -> tuple[object, Stubber]:
    client = boto3.client("comprehend", region_name="us-east-1")
    stub = Stubber(client)
    stub.add_response(
        "detect_entities",
        {"Entities": [
            {"Type": etype, "Text": text, "Score": 0.99,
             "BeginOffset": 0, "EndOffset": len(text)}
            for etype, text in entities
        ]},
        {"Text": ANY, "LanguageCode": "en"},
    )
    return client, stub


def test_detect_entities_groups_useful_types():
    client, stub = _stub_comprehend([
        ("QUANTITY", "#1042"), ("QUANTITY", "$49.00"),
        ("DATE", "2026-06-12"), ("PERSON", "Dana"),  # PERSON dropped (not useful)
    ])
    with stub:
        ents = intake_mod.detect_entities("order #1042 for $49.00 on 2026-06-12",
                                          client=client)
    assert not ents.is_empty()
    assert ents.by_type["QUANTITY"] == ["#1042", "$49.00"]
    assert ents.by_type["DATE"] == ["2026-06-12"]
    assert "PERSON" not in ents.by_type
    line = ents.as_line()
    assert "quantity: #1042, $49.00" in line and "date: 2026-06-12" in line


def test_detect_entities_raises_intakeerror_on_client_error():
    client = boto3.client("comprehend", region_name="us-east-1")
    stub = Stubber(client)
    stub.add_client_error(
        "detect_entities", service_error_code="TextSizeLimitExceededException",
        service_message="too big", http_status_code=400,
    )
    with stub:
        with pytest.raises(intake_mod.IntakeError):
            intake_mod.detect_entities("x", client=client)


# ===========================================================================
# Module 6 — attachment upload + the multimodal vision read (skills 1.3.2/1.3.3)
# ===========================================================================
def test_upload_attachment_lands_under_attachments_prefix():
    from moto import mock_aws

    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        bucket = config.relay_bucket(ACCOUNT)
        s3.create_bucket(Bucket=bucket)
        att = intake_mod.upload_attachment(
            b"\x89PNG fake", "payment_error.png", "image/png",
            account=ACCOUNT, s3_client=s3,
        )
        assert isinstance(att, Attachment)
        assert att.filename == "payment_error.png"
        assert att.media_type == "image/png"
        assert att.s3_uri.startswith(
            f"s3://{bucket}/{config.RELAY_ATTACHMENTS_PREFIX}"
        )
        # The object really exists under attachments/.
        listed = s3.list_objects_v2(Bucket=bucket,
                                    Prefix=config.RELAY_ATTACHMENTS_PREFIX)
        assert listed["KeyCount"] == 1


def _stub_vision_runtime(monkeypatch, reply: str) -> tuple[object, Stubber]:
    client = boto3.client("bedrock-runtime", region_name="us-east-1")
    stub = Stubber(client)
    stub.add_response(
        "converse",
        _converse_response(reply, in_tok=1100, out_tok=30),
        {"modelId": config.tier_profile("vision"), "messages": ANY,
         "inferenceConfig": ANY},
    )
    monkeypatch.setattr(llm, "_clients", {"runtime": client})
    return client, stub


def test_read_screenshot_sends_one_multimodal_vision_message(monkeypatch):
    # The Converse request must carry BOTH a text block and an image block on the
    # vision profile. We capture the request to assert the multimodal shape.
    client = boto3.client("bedrock-runtime", region_name="us-east-1")
    captured: dict = {}

    def fake_converse(**kwargs):
        captured.update(kwargs)
        return _converse_response(
            "Error: ERR-402 Payment declined\nScreen: checkout\n"
            "User action: use another card", 1100, 30,
        )

    monkeypatch.setattr(client, "converse", fake_converse)
    monkeypatch.setattr(llm, "_clients", {"runtime": client})

    png = (RAW_DIR / "payment_error.png").read_bytes()
    summary = intake_mod.read_screenshot(png, "image/png")

    assert captured["modelId"] == config.tier_profile("vision")
    content = captured["messages"][0]["content"]
    kinds = [next(iter(block)) for block in content]
    assert "text" in kinds and "image" in kinds        # multimodal: text + image
    image_block = next(b for b in content if "image" in b)
    assert image_block["image"]["format"] == "png"
    assert image_block["image"]["source"]["bytes"] == png
    assert "ERR-402" in summary


# ===========================================================================
# Module 6 — the full intake() pipeline, offline (Comprehend + S3 + vision stubbed)
# ===========================================================================
def test_intake_end_to_end_produces_clean_ticket_with_attachment(monkeypatch):
    comp, comp_stub = _stub_comprehend([("QUANTITY", "#1042"),
                                        ("DATE", "2026-06-12")])
    rt, rt_stub = _stub_vision_runtime(
        monkeypatch,
        "Error: ERR-402 Payment declined\nScreen: CloudCart checkout\n"
        "User action: use a different card and click Retry payment",
    )
    from moto import mock_aws

    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=config.relay_bucket(ACCOUNT))
        raw = intake_mod.parse_raw(
            (RAW_DIR / "email_billing_error.txt").read_text(encoding="utf-8")
        )
        png = (RAW_DIR / "payment_error.png").read_bytes()
        with comp_stub, rt_stub:
            result = intake_mod.intake(
                raw, attachment_bytes=png, attachment_filename="payment_error.png",
                account=ACCOUNT, comprehend_client=comp, s3_client=s3,
                run_vision=True,
            )

    t = result.ticket
    assert isinstance(t, Ticket)
    assert t.channel == "email"
    # Normalized: no quoted thread, no signature.
    assert ">" not in t.customer_message
    assert "+1 (555)" not in t.customer_message
    # Enriched: the [Entities] line and the [Attachment summary] from the vision read.
    assert intake_mod.ENTITIES_HEADER in t.customer_message
    assert intake_mod.ATTACHMENT_SUMMARY_HEADER in t.customer_message
    assert "ERR-402" in t.customer_message
    # The attachment is recorded (frozen schema) and points at attachments/.
    assert len(t.attachments) == 1
    assert t.attachments[0].media_type == "image/png"
    assert config.RELAY_ATTACHMENTS_PREFIX in t.attachments[0].s3_uri
    # M6 does NOT redact — there is no pii_redacted field at all.
    assert "pii_redacted" not in t.model_dump()


def test_intake_rejects_the_oversized_fixture():
    raw = intake_mod.parse_raw(
        (RAW_DIR / "invalid_oversized.txt").read_text(encoding="utf-8")
    )
    with pytest.raises(intake_mod.IntakeRejected) as exc:
        # No FM/AWS clients needed — the size gate fires before any call.
        intake_mod.intake(raw)
    assert exc.value.reason == "message_too_large"


def test_intake_without_attachment_skips_vision_and_upload(monkeypatch):
    comp, comp_stub = _stub_comprehend([("QUANTITY", "#2087"),
                                        ("DATE", "2026-06-09")])
    # No bedrock-runtime / S3 stub: a no-attachment intake must make NEITHER call.
    raw = intake_mod.parse_raw(
        (RAW_DIR / "chat_shipping.txt").read_text(encoding="utf-8")
    )
    with comp_stub:
        result = intake_mod.intake(raw, comprehend_client=comp)
    t = result.ticket
    assert t.channel == "chat"
    assert t.attachments == []
    assert intake_mod.ATTACHMENT_SUMMARY_HEADER not in t.customer_message
    assert "#2087" in t.customer_message


def test_intake_uploads_but_can_skip_the_fm_vision_read():
    # run_vision=False: still validate + upload + record the Attachment, but make no
    # Converse call (used when you only want the file recorded). No runtime stub here.
    comp, comp_stub = _stub_comprehend([])
    from moto import mock_aws

    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=config.relay_bucket(ACCOUNT))
        raw = intake_mod.parse_raw(
            (RAW_DIR / "email_billing_error.txt").read_text(encoding="utf-8")
        )
        png = (RAW_DIR / "payment_error.png").read_bytes()
        with comp_stub:
            result = intake_mod.intake(
                raw, attachment_bytes=png, attachment_filename="payment_error.png",
                account=ACCOUNT, comprehend_client=comp, s3_client=s3,
                run_vision=False,
            )
    assert len(result.ticket.attachments) == 1
    assert result.attachment_summary is None
    assert intake_mod.ATTACHMENT_SUMMARY_HEADER not in result.ticket.customer_message


# ===========================================================================
# Module 6 — the raw fixtures (3 valid incl. 1 with screenshot, 2 invalid)
# ===========================================================================
def test_raw_fixtures_present_and_shaped():
    txts = sorted(p.name for p in RAW_DIR.glob("*.txt"))
    assert len(txts) == 5, txts
    assert "payment_error.png" in {p.name for p in RAW_DIR.glob("*.png")}
    valid = [f for f in txts if not f.startswith("invalid_")]
    invalid = [f for f in txts if f.startswith("invalid_")]
    assert len(valid) == 3 and len(invalid) == 2
    # Each valid fixture parses to a known channel.
    for name in valid:
        raw = intake_mod.parse_raw((RAW_DIR / name).read_text(encoding="utf-8"))
        assert raw.channel in ("email", "chat"), name


def test_payment_error_png_is_a_real_png():
    data = (RAW_DIR / "payment_error.png").read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"           # PNG signature
    assert config.media_type_for_filename("payment_error.png") == "image/png"
    assert len(data) <= config.MAX_ATTACHMENT_BYTES   # within the gate


# ===========================================================================
# Module 6 — setup adds attachments/ prefix; teardown purges it (moto S3)
# ===========================================================================
def test_setup_creates_attachments_prefix_marker_idempotently():
    import setup as setup_mod
    from moto import mock_aws

    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        bucket = config.relay_bucket(ACCOUNT)
        s3.create_bucket(Bucket=bucket)
        for _ in range(2):  # idempotent
            setup_mod.ensure_attachments_prefix(s3, bucket)
        listed = s3.list_objects_v2(Bucket=bucket,
                                    Prefix=config.RELAY_ATTACHMENTS_PREFIX)
        keys = [o["Key"] for o in listed.get("Contents", [])]
        assert f"{config.RELAY_ATTACHMENTS_PREFIX}.keep" in keys


def test_teardown_purges_attachments_prefix_keeps_bucket():
    import teardown as teardown_mod
    from moto import mock_aws

    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        bucket = config.relay_bucket(ACCOUNT)
        s3.create_bucket(Bucket=bucket)
        # Seed some uploads + the docs/ corpus (which must be KEPT).
        s3.put_object(Bucket=bucket, Key="attachments/.keep", Body=b"")
        s3.put_object(Bucket=bucket, Key="attachments/a-shot.png", Body=b"img")
        s3.put_object(Bucket=bucket, Key="docs/billing-plans.md", Body=b"doc")
        deleted = teardown_mod.purge_attachments(s3, bucket)
        assert deleted == 2
        # attachments/ is empty; docs/ survives; bucket survives.
        assert s3.list_objects_v2(
            Bucket=bucket, Prefix="attachments/").get("KeyCount", 0) == 0
        assert s3.list_objects_v2(
            Bucket=bucket, Prefix="docs/")["KeyCount"] == 1
        # Idempotent: a second purge is a clean no-op.
        assert teardown_mod.purge_attachments(s3, bucket) == 0


# ===========================================================================
# Module 6 — boundary grep gates (what the lab must / must NOT contain)
# ===========================================================================
def test_exactly_one_invoke_model_in_the_whole_lab():
    """The course's sole invoke_model is Titan embeddings in ingest/embed.py."""
    token = "invoke" + "_model"
    offenders = []
    for path in _ROOT.rglob("*.py"):
        if ".venv" in path.parts or path.name == "smoke_test.py":
            continue
        if token in path.read_text(encoding="utf-8"):
            offenders.append(path.relative_to(_ROOT).as_posix())
    assert offenders == ["ingest/embed.py"], offenders


def test_no_opensearch_in_lab_code():
    """OpenSearch is theory-only in the article; the lab CODE must not touch it.

    (bible §2.2 / brief §10 grep gate: opensearch|aoss_|invoke_model|create_collection.)
    """
    pattern = re.compile(r"opensearch|aoss_|create_collection", re.IGNORECASE)
    offenders = []
    for path in _ROOT.rglob("*.py"):
        if ".venv" in path.parts or path.name == "smoke_test.py":
            continue
        if pattern.search(path.read_text(encoding="utf-8")):
            offenders.append(path.relative_to(_ROOT).as_posix())
    assert offenders == [], offenders


def test_no_M8_or_later_capabilities_in_lab_code():
    """M7 builds search_kb / lookup_order / create_ticket, AgentAction / TicketRecord,
    and relay-orders / relay-tickets (its own increment). But it must NOT USE a
    downstream CAPABILITY: no Billing-specialist handoff or specialists module (M8), no
    AgentCore deploy/Memory/runtime (M8), no human-approval flow (M8), no guardrail
    layer (M9), no public API (M11). Those are forbidden TOKENS in the lab CODE.

    Note this checks USAGE in code, not the one-line forward-reference teasers in
    docstrings (the agent docstring names Module 8 as the next step) — the pattern below
    targets API/identifier tokens those teasers do not contain.
    """
    pattern = re.compile(
        r"specialists|bedrock_agentcore|bedrock-agentcore|agentcore|"
        r"starter.toolkit|create_agent|invoke_agent|"
        r"\.approve\(|relay/approve|relay\.approve|langgraph|langchain",
        re.IGNORECASE,
    )
    offenders = []
    for path in _ROOT.rglob("*.py"):
        if ".venv" in path.parts or path.name == "smoke_test.py":
            continue
        if pattern.search(path.read_text(encoding="utf-8")):
            offenders.append(path.relative_to(_ROOT).as_posix())
    assert offenders == [], offenders
    # The M10 Ticket field must still not exist (a later-module boundary, the real check).
    assert "pii_redacted" not in Ticket.model_fields
    # `approved` is frozen on AgentAction but EFFECTIVE only at M8 — at M7 it is only
    # ever the default None (the lab assigns nothing else). The token may appear (it is a
    # schema field); we assert the SEMANTIC boundary in the M7 section below instead.


def test_kb_module_uses_no_bare_model_id_and_no_invoke():
    """relay/kb.py must hold NO us./global. profile ID and NO invoke_model — the
    answer model + reranker come from config; generation is RetrieveAndGenerate."""
    src = (RELAY_DIR / "kb.py").read_text(encoding="utf-8")
    assert not re.search(r"(us|global)\.(amazon|anthropic)\.", src)
    assert "invoke" + "_model" not in src
    # It references the model only through config helpers.
    assert "config.model_arn" in src
    assert "config.rerank_model_arn" in src


# ===========================================================================
# Module 7 — the FROZEN AgentAction / TicketRecord contract (06 §2 / bible §3.1)
# ===========================================================================
from relay.models import AgentAction, TicketRecord  # noqa: E402
from relay import tools as tools_mod  # noqa: E402
from relay import agent as agent_mod  # noqa: E402
from mcp_server import store as store_mod  # noqa: E402
from mcp_server import server as mcp_server_mod  # noqa: E402
from mcp_server import app as mcp_app_mod  # noqa: E402

ORDERS_SEED = json.loads((_ROOT / "data" / "orders.json").read_text("utf-8"))


def test_agentaction_is_exactly_four_fields_approved_defaults_none():
    # Frozen M7 field-for-field: {tool, tool_input, result, approved=None}.
    assert set(AgentAction.model_fields) == {"tool", "tool_input", "result", "approved"}
    a = AgentAction(tool="lookup_order", tool_input={"order_id": "1042"}, result="ok")
    # approved is EFFECTIVE only at M8 — it defaults to None and M7 assigns nothing else.
    assert a.approved is None
    assert AgentAction.model_fields["approved"].default is None
    # It still ACCEPTS True/False (the field is frozen now, used in M8) — the type is set.
    assert AgentAction(tool="t", tool_input={}, result="r", approved=True).approved is True


def test_ticketrecord_is_frozen_field_for_field_with_full_status_enum():
    # Frozen M7: the EXACT field list, in order, with the FULL 7-status enum present
    # though M7 only writes four of them. feedback_rating (M13) must be ABSENT.
    assert list(TicketRecord.model_fields) == [
        "ticket_id", "status", "triage", "answer", "actions", "escalated",
        "cost_cents", "updated_at",
    ]
    assert "feedback_rating" not in TicketRecord.model_fields
    status_ann = TicketRecord.model_fields["status"].annotation
    import typing
    assert set(typing.get_args(status_ann)) == {
        "received", "triaged", "awaiting_approval",
        "answered", "escalated", "closed", "failed",
    }
    # cost_cents is a float placeholder (0.0 at M7, populated at M12) — never re-typed.
    assert TicketRecord.model_fields["cost_cents"].annotation is float


def test_ticketrecord_round_trips_with_actions_and_zero_cost():
    rec = TicketRecord(
        ticket_id="ticket-1", status="answered",
        triage=Triage(intent="shipping", priority="high", sentiment="neutral"),
        answer=None,
        actions=[AgentAction(tool="lookup_order", tool_input={"order_id": "1042"},
                             result="in_transit")],
        escalated=False, cost_cents=0.0, updated_at="2026-06-13T00:00:00Z",
    )
    again = TicketRecord.model_validate_json(rec.model_dump_json())
    assert again.actions[0].tool == "lookup_order"
    assert again.actions[0].approved is None      # M7 invariant
    assert again.cost_cents == 0.0                # M7 placeholder
    assert again.escalated is False               # no escalation at M7


# ===========================================================================
# Module 7 — config additions (table names, MCP URL) — by addition
# ===========================================================================
def test_m7_table_names_and_keys_are_frozen():
    assert config.RELAY_ORDERS_TABLE == "relay-orders"
    assert config.RELAY_TICKETS_TABLE == "relay-tickets"
    assert config.ORDERS_KEY == "order_id"
    assert config.TICKETS_KEY == "ticket_id"
    assert config.MCP_SERVER_PATH == "/mcp"


def test_m7_config_did_not_touch_the_tier_map_or_embedder():
    # M7 only appends constants; the agent runs on the EXISTING smart tier (no new model).
    assert config.tier_profile("fast") == "us.amazon.nova-micro-v1:0"
    assert config.tier_profile("smart") == "us.amazon.nova-2-lite-v1:0"
    assert config.EMBED_MODEL_ID == "amazon.titan-embed-text-v2:0"
    assert config.EMBED_DIMENSIONS == 1024
    assert set(config.TIERS) == {"fast", "smart", "frontier", "vision"}  # unchanged


def test_resolve_mcp_url_prefers_env_then_errors_without_setup(monkeypatch, tmp_path):
    monkeypatch.setenv("RELAY_MCP_URL", "http://127.0.0.1:8000/mcp")
    assert config.resolve_mcp_url() == "http://127.0.0.1:8000/mcp"
    assert config.resolve_mcp_url("http://x/mcp") == "http://x/mcp"  # explicit wins
    monkeypatch.delenv("RELAY_MCP_URL", raising=False)
    monkeypatch.setattr(config, "MCP_URL_FILE_NAME", str(tmp_path / "nope"))
    with pytest.raises(ValueError) as exc:
        config.resolve_mcp_url()
    assert "setup.py" in str(exc.value)


# ===========================================================================
# Module 7 — the tool names are the canonical 06 §5.4 set (no synonyms)
# ===========================================================================
def test_canonical_tool_names_no_synonyms():
    assert tools_mod.LOCAL_TOOL_NAMES == ("search_kb",)
    assert tools_mod.MCP_TOOL_NAMES == ("lookup_order", "create_ticket")
    assert tools_mod.ALL_TOOL_NAMES == ("search_kb", "lookup_order", "create_ticket")
    # search_kb is a real Strands @tool whose name is exactly canonical.
    assert tools_mod.search_kb.tool_name == "search_kb"


# ===========================================================================
# Module 7 — mcp_server.store over moto DynamoDB (lookup_order + create_ticket)
# ===========================================================================
@pytest.fixture
def dynamodb_backend():
    from moto import mock_aws

    with mock_aws():
        resource = boto3.resource("dynamodb", region_name="us-east-1")
        resource.create_table(
            TableName=config.RELAY_ORDERS_TABLE,
            KeySchema=[{"AttributeName": config.ORDERS_KEY, "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": config.ORDERS_KEY,
                                   "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        ).wait_until_exists()
        resource.create_table(
            TableName=config.RELAY_TICKETS_TABLE,
            KeySchema=[{"AttributeName": config.TICKETS_KEY, "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": config.TICKETS_KEY,
                                   "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        ).wait_until_exists()
        yield resource


def test_seed_orders_loads_all_25(dynamodb_backend):
    n = store_mod.seed_orders(ORDERS_SEED, resource=dynamodb_backend)
    assert n == 25
    # idempotent: re-seeding upserts the same rows, no error, same count.
    assert store_mod.seed_orders(ORDERS_SEED, resource=dynamodb_backend) == 25


def test_lookup_order_returns_real_status_and_strips_hash(dynamodb_backend):
    store_mod.seed_orders(ORDERS_SEED, resource=dynamodb_backend)
    order = store_mod.lookup_order("#1042", resource=dynamodb_backend)  # leading # ok
    assert order["order_id"] == "1042"
    assert order["status"] == "in_transit"
    assert order["estimated_delivery"] == "2026-06-15"
    # Decimals were converted to native int/float (clean JSON for the model).
    assert isinstance(order["total"], (int, float))


def test_lookup_order_unknown_raises_model_facing_not_found(dynamodb_backend):
    store_mod.seed_orders(ORDERS_SEED, resource=dynamodb_backend)
    with pytest.raises(store_mod.OrderNotFound) as exc:
        store_mod.lookup_order("9999", resource=dynamodb_backend)
    assert "9999" in str(exc.value)


def test_lookup_order_blank_id_is_a_tool_input_error(dynamodb_backend):
    with pytest.raises(store_mod.ToolInputError):
        store_mod.lookup_order("   ", resource=dynamodb_backend)
    with pytest.raises(store_mod.ToolInputError):
        store_mod.lookup_order(None, resource=dynamodb_backend)


def test_create_ticket_persists_a_frozen_ticketrecord(dynamodb_backend):
    actions = [AgentAction(tool="lookup_order", tool_input={"order_id": "1042"},
                           result="in_transit").model_dump()]
    stored = store_mod.create_ticket(
        "ticket-xyz", status="answered", summary="shipping update",
        triage={"intent": "shipping", "priority": "high", "sentiment": "neutral"},
        actions=actions, resource=dynamodb_backend,
    )
    assert stored["ticket_id"] == "ticket-xyz"
    assert stored["status"] == "answered"
    assert stored["cost_cents"] == 0.0            # M7 placeholder
    # Read it back as a frozen TicketRecord — the round trip validates.
    rec = store_mod.get_ticket("ticket-xyz", resource=dynamodb_backend)
    assert isinstance(rec, TicketRecord)
    assert len(rec.actions) == 1
    assert rec.actions[0].tool == "lookup_order"
    assert rec.actions[0].approved is None        # M7 invariant
    assert rec.triage is not None and rec.triage.intent == "shipping"


def test_create_ticket_is_idempotent_on_ticket_id(dynamodb_backend):
    store_mod.create_ticket("ticket-dup", status="received",
                            resource=dynamodb_backend)
    store_mod.create_ticket("ticket-dup", status="answered",
                            resource=dynamodb_backend)  # overwrites same row
    table = dynamodb_backend.Table(config.RELAY_TICKETS_TABLE)
    scanned = table.scan()["Items"]
    rows = [r for r in scanned if r["ticket_id"] == "ticket-dup"]
    assert len(rows) == 1                          # one row, not two
    assert rows[0]["status"] == "answered"


def test_create_ticket_blank_id_is_a_tool_input_error(dynamodb_backend):
    with pytest.raises(store_mod.ToolInputError):
        store_mod.create_ticket("  ", resource=dynamodb_backend)


def test_create_ticket_rejects_invalid_status_with_tool_input_error(dynamodb_backend):
    # An out-of-enum status is caught by the frozen schema and reported to the model.
    with pytest.raises(store_mod.ToolInputError):
        store_mod.create_ticket("ticket-bad", status="not_a_status",
                                resource=dynamodb_backend)


# ===========================================================================
# Module 7 — the MCP server exposes exactly the two business tools (skill 2.1.7)
# ===========================================================================
def test_mcp_server_advertises_exactly_the_two_business_tools():
    import asyncio
    tool_objs = asyncio.run(mcp_server_mod.mcp.list_tools())
    names = sorted(t.name for t in tool_objs)
    assert names == ["create_ticket", "lookup_order"]   # search_kb stays LOCAL
    by_name = {t.name: t for t in tool_objs}
    # The DOCSTRING is the description, the TYPE HINTS are the schema (skill 2.1.6).
    assert by_name["lookup_order"].inputSchema["required"] == ["order_id"]
    assert by_name["create_ticket"].inputSchema["required"] == ["ticket_id"]
    assert by_name["lookup_order"].description.strip() != ""


def test_mcp_server_is_stateless():
    # Stateless HTTP is what makes it Lambda-shaped (skill 2.1.7).
    assert mcp_server_mod.mcp.settings.stateless_http is True


def test_lambda_handler_adapts_a_function_url_event_to_the_asgi_app():
    # The Function-URL -> ASGI adapter must produce a well-formed HTTP response for a
    # basic event (we hit an MCP GET without a session -> the server replies, not 500).
    event = {
        "requestContext": {"http": {"method": "GET"}},
        "rawPath": config.MCP_SERVER_PATH,
        "rawQueryString": "",
        "headers": {"accept": "text/event-stream"},
        "body": "",
        "isBase64Encoded": False,
    }
    response = mcp_app_mod.handler(event)
    assert isinstance(response, dict)
    assert "statusCode" in response and "body" in response
    assert isinstance(response["statusCode"], int)
    # The MCP server answered (any HTTP status is fine; the point is the adapter ran the
    # ASGI app and translated a real response, not a 500 from the adapter itself).
    assert response["statusCode"] != 500 or "headers" in response


# ===========================================================================
# Module 7 — search_kb: the LOCAL retrieval tool, over a stubbed KB (skill 1.5.6)
# ===========================================================================
def test_search_kb_formats_retrieved_passages_with_sources(monkeypatch):
    from relay.kb import Retrieved

    def fake_retrieve(query, *, top_k):
        assert query == "how do refunds work?"
        return [
            Retrieved(text="Refunds are issued within 5 business days.",
                      source_uri="s3://relay-x/docs/billing-plans.md", score=0.9),
            Retrieved(text="Open Billing -> Orders to request a refund.",
                      source_uri="s3://relay-x/docs/orders-export.md", score=0.8),
        ]

    monkeypatch.setattr(tools_mod.kb, "retrieve", fake_retrieve)
    out = tools_mod.search_kb("how do refunds work?")
    assert "Refunds are issued" in out
    assert "billing-plans.md" in out and "orders-export.md" in out


def test_search_kb_blank_query_returns_model_facing_message():
    out = tools_mod.search_kb("   ")
    assert "No query" in out


def test_search_kb_kberror_is_returned_to_the_model_not_raised(monkeypatch):
    def boom(query, *, top_k):
        raise tools_mod.kb.KBError("relay-kb not set up (run setup.py)")

    monkeypatch.setattr(tools_mod.kb, "retrieve", boom)
    out = tools_mod.search_kb("anything")
    # Clean, recoverable message — NOT a crash that breaks the agent loop.
    assert "could not be searched" in out


def test_search_kb_empty_result_is_a_clear_no_docs_message(monkeypatch):
    monkeypatch.setattr(tools_mod.kb, "retrieve", lambda q, *, top_k: [])
    out = tools_mod.search_kb("obscure question")
    assert "No documentation found" in out


# ===========================================================================
# Module 7 — the agent loop, driven by a SCRIPTED model (fully offline, no Bedrock)
# ===========================================================================
from strands.models.model import Model  # noqa: E402


class _ScriptedModel(Model):
    """A fake Strands model that yields pre-scripted stream events per turn.

    Drives the real ReAct event loop with NO Bedrock call, so the whole agent +
    AgentAction-journal + TicketRecord-persistence path runs offline.
    """

    def __init__(self, scripts):
        self._scripts = list(scripts)
        self._i = 0
        self._config = {"model_id": "scripted-test-model"}

    def get_config(self):
        return self._config

    def update_config(self, **kw):
        self._config.update(kw)

    async def structured_output(self, *a, **k):  # pragma: no cover - unused here
        if False:
            yield None

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        script = self._scripts[min(self._i, len(self._scripts) - 1)]
        self._i += 1
        for event in script:
            yield event


def _tool_turn(name, tool_use_id, tool_input):
    return [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockStart": {"start": {"toolUse": {"toolUseId": tool_use_id,
                                                     "name": name}}}},
        {"contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps(tool_input)}}}},
        {"contentBlockStop": {}},
        {"messageStop": {"stopReason": "tool_use"}},
        {"metadata": {"usage": {"inputTokens": 40, "outputTokens": 8, "totalTokens": 48},
                      "metrics": {"latencyMs": 1}}},
    ]


def _text_turn(text):
    return [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockStart": {"start": {}}},
        {"contentBlockDelta": {"delta": {"text": text}}},
        {"contentBlockStop": {}},
        {"messageStop": {"stopReason": "end_turn"}},
        {"metadata": {"usage": {"inputTokens": 50, "outputTokens": 12, "totalTokens": 62},
                      "metrics": {"latencyMs": 1}}},
    ]


def _store_backed_order_tool(dynamodb_backend):
    """A local lookup_order @tool that hits the store on moto (stands in for the MCP
    tool in offline tests — the wire is covered by the live test)."""
    @tool
    def lookup_order(order_id: str) -> str:
        """Look up an order. Args: order_id: the CloudCart order id."""
        try:
            return json.dumps(store_mod.lookup_order(order_id,
                                                    resource=dynamodb_backend))
        except store_mod.StoreError as err:
            return str(err)
    return lookup_order


def _store_persist(dynamodb_backend):
    def persist(ticket_id, *, status, summary, actions):
        return store_mod.create_ticket(ticket_id, status=status, summary=summary,
                                       actions=actions, resource=dynamodb_backend)
    return persist


from strands import tool  # noqa: E402


def test_agent_handles_an_order_ticket_persists_record_with_actions(dynamodb_backend):
    """The brief's headline result, OFFLINE: the agent calls lookup_order, then
    create_ticket, answers, and a TicketRecord with >=1 AgentAction is persisted."""
    store_mod.seed_orders(ORDERS_SEED, resource=dynamodb_backend)
    lookup = _store_backed_order_tool(dynamodb_backend)

    @tool
    def create_ticket(ticket_id: str, status: str = "answered",
                      summary: str | None = None) -> str:
        """Create a ticket. Args: ticket_id: id. status: outcome. summary: note."""
        store_mod.create_ticket(ticket_id, status=status, summary=summary,
                                resource=dynamodb_backend)
        return f"stored {ticket_id}"

    model = _ScriptedModel([
        _tool_turn("lookup_order", "t1", {"order_id": "1042"}),
        _tool_turn("create_ticket", "t2",
                   {"ticket_id": "ticket-order", "status": "answered",
                    "summary": "order in transit, ETA 2026-06-15"}),
        _text_turn("Your order 1042 is in transit and should arrive on 2026-06-15."),
    ])
    agent, journal = agent_mod.build_agent(model=model,
                                          extra_tools=[lookup, create_ticket])
    outcome = agent_mod.handle(
        "Where is order 1042? It was supposed to arrive Monday.",
        ticket_id="ticket-order", agent=agent, journal=journal,
        persist=_store_persist(dynamodb_backend),
    )
    assert outcome.record.status == "answered"
    assert outcome.stop_reason == "end_turn"
    assert "2026-06-15" in outcome.answer_text
    # >= 1 AgentAction, all approved=None (the M7 invariant), tool names canonical.
    assert len(outcome.record.actions) >= 1
    tool_names = [a.tool for a in outcome.record.actions]
    assert "lookup_order" in tool_names and "create_ticket" in tool_names
    assert all(a.approved is None for a in outcome.record.actions)
    # The record really landed in relay-tickets with its actions[] journal.
    persisted = store_mod.get_ticket("ticket-order", resource=dynamodb_backend)
    assert persisted is not None
    assert len(persisted.actions) >= 1
    assert persisted.cost_cents == 0.0


def test_agent_chooses_search_kb_for_a_documentation_question(monkeypatch,
                                                              dynamodb_backend):
    """A how-to question -> the agent calls search_kb, NOT lookup_order (the brief's
    'how do refunds work?' contrast)."""
    monkeypatch.setattr(
        tools_mod.kb, "retrieve",
        lambda q, *, top_k: [__import__("relay.kb", fromlist=["Retrieved"]).Retrieved(
            text="Refunds are issued within 5 business days to the original method.",
            source_uri="s3://relay-x/docs/billing-plans.md", score=0.9)],
    )
    model = _ScriptedModel([
        _tool_turn("search_kb", "s1", {"query": "how do refunds work?"}),
        _text_turn("Refunds are issued within 5 business days to your original "
                   "payment method."),
    ])
    agent, journal = agent_mod.build_agent(model=model)  # search_kb only (local tool)
    outcome = agent_mod.handle("How do refunds work?", ticket_id="ticket-refund",
                               agent=agent, journal=journal,
                               persist=_store_persist(dynamodb_backend))
    assert outcome.record.status == "answered"
    tool_names = [a.tool for a in outcome.record.actions]
    assert tool_names == ["search_kb"]          # chose the doc tool, not lookup_order
    assert "5 business days" in outcome.answer_text


def test_stop_condition_cuts_a_runaway_agent(dynamodb_backend):
    """GUARDRAIL DEMO (skill 2.1.3): a model that always asks for a tool would loop
    forever; the max-iterations stop condition cuts it, and the record is `failed`."""
    class _Looping(Model):
        def __init__(self):
            self._config = {"model_id": "looping"}
            self._n = 0

        def get_config(self):
            return self._config

        def update_config(self, **kw):
            self._config.update(kw)

        async def structured_output(self, *a, **k):  # pragma: no cover
            if False:
                yield None

        async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
            self._n += 1
            for ev in _tool_turn("poke", f"p{self._n}", {}):
                yield ev

    @tool
    def poke() -> str:
        """A tool that always says try again (to force a loop)."""
        return "still not done, call poke again"

    agent, journal = agent_mod.build_agent(model=_Looping(), extra_tools=[poke])
    outcome = agent_mod.handle("loop forever please", ticket_id="ticket-runaway",
                               agent=agent, journal=journal,
                               persist=_store_persist(dynamodb_backend),
                               max_iterations=3)
    assert outcome.stop_reason == "limit_turns"   # the stop condition fired
    assert outcome.record.status == "failed"      # cut before a clean answer
    # It made at most max_iterations tool calls, not an unbounded number.
    assert 0 < len(outcome.record.actions) <= 3


def test_agent_runs_on_the_smart_tier_via_config_only():
    # The agent's model ID comes from the SMART tier in config — no bare ID in agent.py.
    assert agent_mod.AGENT_TIER == "smart"
    assert config.tier_profile(agent_mod.AGENT_TIER) == "us.amazon.nova-2-lite-v1:0"
    src = (RELAY_DIR / "agent.py").read_text(encoding="utf-8")
    assert not re.search(r"(us|global)\.(amazon|anthropic)\.", src)


def test_bedrock_model_applies_the_wallclock_timeout_guardrail():
    """GUARDRAIL (skill 2.1.3, layer 2 of 3 — timeout): the agent's bedrock-runtime
    client carries the wall-clock read/connect timeout, so a stuck model call cannot hang
    the run past AGENT_TIMEOUT_S. This is the timeout sibling of the stop-condition
    (test_stop_condition_cuts_a_runaway_agent) and IAM-boundary
    (test_mcp_lambda_role_is_bounded...) tests — all three guardrail layers now asserted.
    Builds the real model offline (no Bedrock call) and reads back the boto client config."""
    model = agent_mod._bedrock_model()
    boto_cfg = model.client.meta.config
    assert boto_cfg.read_timeout == agent_mod.AGENT_TIMEOUT_S   # read timeout == the cap
    assert boto_cfg.connect_timeout == 10                       # bounded connect, too


# ===========================================================================
# Module 7 — the IAM resource boundary on the MCP Lambda role (skill 2.1.3)
# ===========================================================================
def test_mcp_lambda_role_is_bounded_to_orders_read_tickets_write():
    """The Lambda role policy reads ONLY relay-orders and writes ONLY relay-tickets —
    explicit table ARNs, no '*' on resources (the IAM resource boundary the lab demos)."""
    import setup as setup_mod

    policy = json.loads(setup_mod._mcp_lambda_policy("111122223333"))
    by_sid = {s["Sid"]: s for s in policy["Statement"]}

    orders = by_sid["ReadOrdersOnly"]
    assert all(a.startswith("dynamodb:") for a in orders["Action"])
    assert all("PutItem" not in a and "DeleteItem" not in a for a in orders["Action"])
    assert orders["Resource"] == [
        "arn:aws:dynamodb:us-east-1:111122223333:table/relay-orders"
    ]

    tickets = by_sid["WriteTicketsOnly"]
    assert "dynamodb:PutItem" in tickets["Action"]
    assert tickets["Resource"] == [
        "arn:aws:dynamodb:us-east-1:111122223333:table/relay-tickets"
    ]

    # No statement grants '*' on a DynamoDB resource — the boundary is real, not nominal.
    for stmt in policy["Statement"]:
        resources = stmt["Resource"]
        resources = resources if isinstance(resources, list) else [resources]
        if any("dynamodb" in r for r in resources):
            assert all(r != "*" for r in resources), stmt


# ===========================================================================
# Module 7 — setup/teardown of the tables + MCP Lambda (moto + stubs, offline)
# ===========================================================================
def test_setup_creates_tables_idempotently_and_seeds(monkeypatch):
    import setup as setup_mod
    from moto import mock_aws

    with mock_aws():
        ddb = boto3.client("dynamodb", region_name="us-east-1")
        resource = boto3.resource("dynamodb", region_name="us-east-1")
        # Run twice — idempotent, no error.
        for _ in range(2):
            setup_mod.ensure_table(ddb, name=config.RELAY_ORDERS_TABLE,
                                   key_attr=config.ORDERS_KEY)
            setup_mod.ensure_table(ddb, name=config.RELAY_TICKETS_TABLE,
                                   key_attr=config.TICKETS_KEY)
        # The seed file path points at the real data/orders.json (25 rows).
        n = setup_mod.seed_orders_table(resource)
        assert n == 25
        assert ddb.describe_table(
            TableName=config.RELAY_ORDERS_TABLE
        )["Table"]["BillingModeSummary"]["BillingMode"] == "PAY_PER_REQUEST"


def test_setup_mcp_lambda_role_is_idempotent_on_moto():
    import setup as setup_mod
    from moto import mock_aws

    with mock_aws():
        iam = boto3.client("iam", region_name="us-east-1")
        for _ in range(2):
            arn = setup_mod.ensure_mcp_lambda_role(iam, "111122223333")
        assert arn.endswith(f"role/{setup_mod.MCP_LAMBDA_ROLE_NAME}")
        policies = iam.list_role_policies(
            RoleName=setup_mod.MCP_LAMBDA_ROLE_NAME)["PolicyNames"]
        assert "relay-mcp-permissions" in policies


def test_teardown_deletes_mcp_lambda_and_role_idempotently(monkeypatch, tmp_path):
    import setup as setup_mod
    import teardown as teardown_mod
    from moto import mock_aws

    # Point the .mcp_url marker at a temp file so we do not touch the repo's.
    marker = tmp_path / ".mcp_url"
    marker.write_text("http://x/mcp\n", encoding="utf-8")
    monkeypatch.setattr(teardown_mod, "MCP_URL_FILE", marker)

    with mock_aws():
        # Lambda delete path: a Stubber (moto's lambda needs a real zip/role); we stub
        # the two delete calls + the not-found on a second pass.
        lmb = boto3.client("lambda", region_name="us-east-1")
        stub = Stubber(lmb)
        stub.add_response("delete_function_url_config", {},
                          {"FunctionName": setup_mod.MCP_LAMBDA_NAME})
        stub.add_response("delete_function", {},
                          {"FunctionName": setup_mod.MCP_LAMBDA_NAME})
        with stub:
            teardown_mod.delete_mcp_lambda(lmb)
        assert not marker.exists()                 # marker removed

        # Role delete on moto, idempotent (second call is a clean no-op).
        iam = boto3.client("iam", region_name="us-east-1")
        setup_mod.ensure_mcp_lambda_role(iam, "111122223333")
        for _ in range(2):
            teardown_mod.delete_mcp_lambda_role(iam)
        from botocore.exceptions import ClientError as _CE
        with pytest.raises(_CE):
            iam.get_role(RoleName=setup_mod.MCP_LAMBDA_ROLE_NAME)


def test_teardown_delete_tables_is_idempotent(monkeypatch):
    import teardown as teardown_mod
    from moto import mock_aws

    with mock_aws():
        ddb = boto3.client("dynamodb", region_name="us-east-1")
        for name, key in ((config.RELAY_ORDERS_TABLE, config.ORDERS_KEY),
                          (config.RELAY_TICKETS_TABLE, config.TICKETS_KEY)):
            ddb.create_table(
                TableName=name,
                KeySchema=[{"AttributeName": key, "KeyType": "HASH"}],
                AttributeDefinitions=[{"AttributeName": key, "AttributeType": "S"}],
                BillingMode="PAY_PER_REQUEST",
            )
        teardown_mod.delete_tables(ddb)
        # Idempotent: a second delete is a clean no-op (tables already gone).
        teardown_mod.delete_tables(ddb)
        remaining = ddb.list_tables()["TableNames"]
        assert config.RELAY_ORDERS_TABLE not in remaining
        assert config.RELAY_TICKETS_TABLE not in remaining


# ===========================================================================
# Module 7 — data/orders.json: 25 seeds, order 1042 present, valid shapes
# ===========================================================================
def test_orders_seed_has_25_with_order_1042_and_required_keys():
    assert len(ORDERS_SEED) == 25
    ids = {o["order_id"] for o in ORDERS_SEED}
    assert len(ids) == 25                          # unique ids
    assert "1042" in ids                           # the brief's demo order
    for order in ORDERS_SEED:
        assert config.ORDERS_KEY in order          # every row has the primary key
        assert order["status"]                     # and a status lookup_order returns


# ===========================================================================
# Module 7 — boundary: approved stays None, no M8 capabilities used
# ===========================================================================
def test_m7_never_assigns_approved_anything_but_none():
    """The HITL field is frozen now but EFFECTIVE at M8 — the M7 lab must only ever
    leave it None. Grep the lab code for an assignment of approved to True/False."""
    pattern = re.compile(r"approved\s*=\s*(True|False)")
    offenders = []
    for path in _ROOT.rglob("*.py"):
        if ".venv" in path.parts or path.name == "smoke_test.py":
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if pattern.search(line):
                offenders.append(f"{path.relative_to(_ROOT).as_posix()}:{lineno}")
    assert offenders == [], offenders


def test_agent_and_tools_use_no_bare_model_id_or_invoke():
    """relay/agent.py and relay/tools.py hold NO us./global. profile ID and NO
    invoke_model — the agent's model comes from config.tier_profile, the tools do I/O."""
    for name in ("agent.py", "tools.py"):
        src = (RELAY_DIR / name).read_text(encoding="utf-8")
        assert not re.search(r"(us|global)\.(amazon|anthropic)\.", src), name
        assert "invoke" + "_model" not in src, name


# ===========================================================================
# The LIVE tests (opt-in) — budget: up to 7 calls (2 M2/M3 + 2 M4 + 1 M5 + 1 M6 + 1 M7)
# ===========================================================================
@pytest.mark.live
@pytest.mark.skipif(
    os.environ.get("RELAY_LIVE_TESTS") != "1",
    reason="set RELAY_LIVE_TESTS=1 to make a real (sub-cent) Bedrock call",
)
def test_live_fast_tier_streaming():
    """ONE real ConverseStream on the FAST tier. < $0.0005 as of June 2026."""
    streaming = llm.converse(
        _user("In one short sentence, what is a support ticket?"),
        tier="fast", stream=True,
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
    reason="set RELAY_LIVE_TESTS=1 to make a real (sub-cent) Bedrock call",
)
def test_live_smart_tier_nonstreaming():
    """ONE real Converse on the SMART tier (auto-routed). < $0.0005 as of June 2026."""
    result = llm.converse(
        _user("Why was I charged twice for order #1042? Explain the dispute steps."),
        tier="auto",
        inferenceConfig={"maxTokens": 64, "temperature": 0.2},
    )
    assert result.tier == "smart"
    assert result.text.strip() != ""
    assert result.usage["inputTokens"] > 0


@pytest.mark.live
@pytest.mark.skipif(
    os.environ.get("RELAY_LIVE_TESTS") != "1",
    reason="set RELAY_LIVE_TESTS=1 to make real (sub-cent) Bedrock calls",
)
def test_live_titan_embeds_a_doc_chunk_and_a_query():
    """TWO real Titan Text Embeddings V2 calls (a doc chunk + a query). ~$0."""
    doc_vector, doc_tokens = embed_mod.embed_one(
        "To export your order history, open Settings -> Data & Privacy -> "
        "Export data, tick Orders, and click Start export."
    )
    query_vector, query_tokens = embed_mod.embed_one("How do I export my orders?")
    assert len(doc_vector) == config.EMBED_DIMENSIONS
    assert len(query_vector) == config.EMBED_DIMENSIONS
    assert doc_tokens > 0 and query_tokens > 0


@pytest.mark.live
@pytest.mark.skipif(
    os.environ.get("RELAY_LIVE_TESTS") != "1",
    reason="set RELAY_LIVE_TESTS=1 to make a real (sub-cent) Bedrock call",
)
def test_live_kb_answer_is_grounded_and_cited():
    """ONE real RetrieveAndGenerate against the live KB `relay-kb`. < $0.01.

    Skips cleanly (does NOT fail) when the KB id cannot be resolved — i.e. when
    setup.py has not built + synced the KB. It makes exactly one smart-tier
    RetrieveAndGenerate call and asserts the answer cites at least one source.
    """
    try:
        kb_mod.resolve_kb_id()
    except kb_mod.KBError:
        pytest.skip("relay-kb not set up (run setup.py) — skipping live KB call.")
    result = kb_mod.answer(
        "How do I change my CloudCart subscription plan?", top_k=4,
    )
    assert isinstance(result, Answer)
    assert result.text.strip() != ""
    assert len(result.citations) >= 1
    assert result.grounded is True
    assert result.citations[0].source_uri.startswith("s3://")


@pytest.mark.live
@pytest.mark.skipif(
    os.environ.get("RELAY_LIVE_TESTS") != "1",
    reason="set RELAY_LIVE_TESTS=1 to make a real (sub-cent) Bedrock call",
)
def test_live_vision_reads_the_payment_error_screenshot():
    """ONE real Amazon Nova Lite VISION Converse call on the bundled screenshot.

    Reads data/raw/payment_error.png from local bytes (no S3 upload, no KB) and
    asserts the model read the visible error. maxTokens<=220 → well under a cent as
    of June 2026. This is the one Module 6 live call (the brief's mandated real
    vision read).
    """
    png = (RAW_DIR / "payment_error.png").read_bytes()
    summary = intake_mod.read_screenshot(png, "image/png")
    assert summary.strip() != ""
    low = summary.lower()
    # The screenshot shows ERR-402 / "payment declined" — the model should surface
    # at least one of those visible cues (allow for paraphrase of "declined").
    assert ("err-402" in low) or ("declined" in low) or ("payment" in low)


@pytest.mark.live
@pytest.mark.skipif(
    os.environ.get("RELAY_LIVE_TESTS") != "1",
    reason="set RELAY_LIVE_TESTS=1 to make a real (capped) agent run",
)
def test_live_agent_handles_order_ticket_end_to_end():
    """ONE real Strands agent RUN on the SMART tier against the deployed MCP server.

    A ReAct loop is a few model calls inside ONE run, capped by the max-iterations stop
    condition (< $0.02 as of June 2026). Skips cleanly when the MCP server / tables are
    not set up (run setup.py). Asserts the agent produced a TicketRecord with >=1
    AgentAction and that order 1042's real status surfaced.
    """
    try:
        config.resolve_mcp_url()
    except ValueError:
        pytest.skip("MCP server not set up (run setup.py) — skipping live agent run.")

    with tools_mod.mcp_business_tools() as biz_tools:
        agent, journal = agent_mod.build_agent(extra_tools=biz_tools)
        outcome = agent_mod.handle(
            "Where is order 1042? It was supposed to arrive Monday.",
            agent=agent, journal=journal,
        )
    # The agent acted: at least one tool call journaled, the record persisted.
    assert isinstance(outcome.record, TicketRecord)
    assert len(outcome.record.actions) >= 1
    assert all(a.approved is None for a in outcome.record.actions)  # M7 invariant
    # It reached the order book (lookup_order) for an order-status question.
    assert any(a.tool == "lookup_order" for a in outcome.record.actions)
