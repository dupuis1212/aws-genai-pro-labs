"""smoke_test.py — Module 5 lab tests (cumulative: covers Modules 2, 3, 4 and 5).

OFFLINE BY DEFAULT. Every test here except the ones marked `live` runs with NO
AWS credentials and makes NO network call:
  - schema tests are pure Pydantic (the M2 Ticket/Triage contract, and the M5
    Citation/Answer contract — frozen field-for-field);
  - the router tests are pure functions on fixed inputs (no model call);
  - the converse() tests drive relay.llm with a botocore Stubber on the
    bedrock-runtime client (Converse, ConverseStream, and a ThrottlingException
    that proves the backoff path);
  - the triage tests drive relay.triage with Stubbers for the bedrock-agent
    (get_prompt) and the bedrock-runtime (converse) clients;
  - the Module 4 chunker tests are pure and deterministic on a fixed doc;
  - the Module 4 ingestion tests stub Titan embeddings (invoke_model) and run the
    S3 Vectors bucket/index/PutVectors lifecycle on a moto backend; the kNN query
    (which moto does not implement) is driven by a botocore Stubber;
  - the Module 5 Knowledge Base tests drive relay.kb (Retrieve /
    RetrieveAndGenerate) and the setup/teardown control plane with botocore
    Stubbers on the bedrock-agent / bedrock-agent-runtime clients (which moto does
    not fully implement), plus moto for IAM/S3 where it does.
That is the course convention — anyone can `uv run pytest` on a fresh clone.

TESTS marked `live` make real calls:
    RELAY_LIVE_TESTS=1 uv run pytest -m live
LIVE-CALL BUDGET: at most FIVE calls total —
  Modules 2/3 (2 calls): one ConverseStream on the FAST tier
    (us.amazon.nova-micro-v1:0) and one Converse on the SMART tier
    (us.amazon.nova-2-lite-v1:0), both maxTokens<=64.
  Module 4 (2 calls): two Amazon Titan Text Embeddings V2 invoke_model calls.
  Module 5 (1 call): ONE RetrieveAndGenerate against the live KB `relay-kb` on the
    smart tier (maxTokens small). It needs setup.py to have built + synced the KB;
    it SKIPS cleanly if the KB id cannot be resolved.
Together that is well under $0.01 (a cent) as of June 2026. They need AWS
credentials and us-east-1. NO live test creates or deletes a KB / ingestion job.
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

# Import the lab packages (module-05/ ships the cumulative relay/ package — now
# with relay/kb.py — the inherited ingest/ pipeline, and the lab scripts).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from relay import config  # noqa: E402
from relay import llm  # noqa: E402
from relay import kb as kb_mod  # noqa: E402
from relay import triage as triage_mod  # noqa: E402
from relay.models import Answer, Citation, Ticket, Triage  # noqa: E402
from ingest import chunkers as chunkers_mod  # noqa: E402
from ingest import embed as embed_mod  # noqa: E402
from ingest import run as run_mod  # noqa: E402
from ingest import upsert as upsert_mod  # noqa: E402
import compare_chunking  # noqa: E402
import compare_retrieval  # noqa: E402

_ROOT = Path(__file__).resolve().parent.parent
TICKETS_DIR = _ROOT / "data" / "tickets"
DOCS_DIR = _ROOT / "data" / "docs"
RELAY_DIR = _ROOT / "relay"
INGEST_DIR = _ROOT / "ingest"


# ===========================================================================
# Module 2 contract — schemas and ticket fixtures (still LAW at M5)
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
    assert [p.name for p in params] == ["messages", "tier", "stream", "params"]


def test_tiers_are_the_canonical_set():
    assert "auto" not in config.TIERS
    assert set(config.TIERS) == {"fast", "smart", "frontier"}


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

    banned in the relay package (Titan embeddings in Module 4's ingest/ is the sole
    exception in the whole course, and lives outside relay/).
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


def test_tier_map_unchanged_by_later_modules():
    # M4 and M5 only ADD constants; the M3 tier map is untouched (bible §2.2).
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
def test_module5_schema_set_is_exactly_the_m2_plus_citation_answer():
    """M5 adds EXACTLY Citation + Answer to the M2 schemas — nothing more.

    Attachment is M6; AgentAction/TicketRecord are M7. Guarding the full set keeps
    the "by addition only, in order" invariant honest.
    """
    import relay.models as models_mod
    from pydantic import BaseModel

    schema_names = {
        name for name, obj in vars(models_mod).items()
        if isinstance(obj, type) and issubclass(obj, BaseModel) and obj is not BaseModel
    }
    assert schema_names == {"Ticket", "Triage", "Citation", "Answer"}
    # Ticket is STILL exactly 4 fields (no early attachments/pii_redacted).
    assert set(Ticket.model_fields) == {
        "ticket_id", "channel", "customer_message", "created_at"
    }


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
    # M5 only appends constants; M3 tier map and M4 embedder are untouched.
    assert set(config.TIERS) == {"fast", "smart", "frontier"}
    assert config.EMBED_MODEL_ID == "amazon.titan-embed-text-v2:0"
    assert config.EMBED_DIMENSIONS == 1024


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
# Module 5 — boundary grep gates (what the lab must / must NOT contain)
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


def test_no_M6_or_later_capabilities_in_lab_code():
    """M5 may use RetrieveAndGenerate / relay-kb (its own increment), but must NOT
    USE a downstream CAPABILITY: the M7 tool token `search_kb`, the M6/M7/M10
    schema fields/classes (Attachment, AgentAction, TicketRecord, pii_redacted), or
    the M7 resource names (relay-orders / relay-tickets).

    (The inherited relay/__init__.py package docstring forward-REFERENCES
    relay.intake / relay.agent as future submodules — that is a one-line teaser, not
    a usage, and was already present byte-identical from Module 4; it is not flagged.)
    """
    pattern = re.compile(
        r"\bsearch_kb\b|\bAttachment\b|relay-orders|relay-tickets|"
        r"\bAgentAction\b|\bTicketRecord\b|\bpii_redacted\b"
    )
    offenders = []
    for path in _ROOT.rglob("*.py"):
        if ".venv" in path.parts or path.name == "smoke_test.py":
            continue
        if pattern.search(path.read_text(encoding="utf-8")):
            offenders.append(path.relative_to(_ROOT).as_posix())
    assert offenders == [], offenders


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
# The LIVE tests (opt-in) — budget: up to 5 calls total (2 M2/M3 + 2 M4 + 1 M5)
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
