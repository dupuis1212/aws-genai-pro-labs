"""smoke_test.py — Module 2 lab tests.

OFFLINE BY DEFAULT. Every test here except the one marked `live` runs with NO
AWS credentials and makes NO network call: schema tests are pure Pydantic, and
the triage tests drive relay.triage with botocore Stubbers for both the
bedrock-agent (get_prompt) and bedrock-runtime (converse) clients. That is the
course convention — anyone can `uv run pytest` on a fresh clone.

ONE test is marked `live` and makes a single real Converse call:
    RELAY_LIVE_TESTS=1 uv run pytest -m live
LIVE-CALL BUDGET: exactly ONE Nova Micro Converse call at temperature 0 with a
~300-token triage prompt and maxTokens=100. That is well under $0.0001 (one
hundredth of a cent) as of June 2026. It needs the prompt published
(uv run python setup.py) and AWS credentials.
"""

from __future__ import annotations

import datetime as dt
import os
import sys
from pathlib import Path

import boto3
import pytest
from botocore.stub import Stubber

# Import the lab package (module-02/ ships the relay/ package).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from relay.models import Ticket, Triage  # noqa: E402
from relay import triage as triage_mod  # noqa: E402

_ROOT = Path(__file__).resolve().parent.parent
TICKETS_DIR = _ROOT / "data" / "tickets"


# --- Schema tests (pure offline, no AWS at all) ----------------------------
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


def test_model_id_is_an_inference_profile_not_a_bare_id():
    # The real M1/M2 trap: a bare regional ID fails on-demand. Guard it.
    assert triage_mod._MODEL_ID.startswith(("us.", "global."))
    assert not triage_mod._MODEL_ID.startswith(("amazon.", "anthropic."))


def test_estimate_cost_from_usage_not_guessed():
    cost = triage_mod.estimate_cost(1000, 1000)
    expected = triage_mod._PRICE_PER_1K_INPUT + triage_mod._PRICE_PER_1K_OUTPUT
    assert cost == pytest.approx(expected)


# --- Stubbed triage helpers (no creds, no network) -------------------------
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


def _stubbed_agent_client() -> tuple[object, Stubber]:
    client = boto3.client("bedrock-agent", region_name="us-east-1")
    stubber = Stubber(client)
    stubber.add_response(
        "get_prompt",
        _get_prompt_response(),
        {"promptIdentifier": "PROMPTID01", "promptVersion": "1"},
    )
    return client, stubber


# --- Stubbed triage happy path ---------------------------------------------
def test_triage_returns_validated_triage_offline():
    agent, agent_stub = _stubbed_agent_client()
    runtime = boto3.client("bedrock-runtime", region_name="us-east-1")
    runtime_stub = Stubber(runtime)
    runtime_stub.add_response(
        "converse",
        _converse_response(
            '{"intent": "billing", "priority": "high", "sentiment": "negative"}',
            in_tok=210,
            out_tok=18,
        ),
    )

    ticket = triage_mod.load_ticket(TICKETS_DIR / "ticket-001.json")
    with agent_stub, runtime_stub:
        result, usage = triage_mod.triage(
            ticket,
            prompt_id="PROMPTID01",
            agent_client=agent,
            runtime_client=runtime,
        )

    assert isinstance(result, Triage)
    assert result.intent == "billing"
    assert result.priority == "high"
    assert result.sentiment == "negative"
    assert usage["inputTokens"] == 210
    assert usage["outputTokens"] == 18


def test_triage_strips_prose_around_json():
    """The model wraps JSON in prose; _extract_json + validation still succeed."""
    agent, agent_stub = _stubbed_agent_client()
    runtime = boto3.client("bedrock-runtime", region_name="us-east-1")
    runtime_stub = Stubber(runtime)
    runtime_stub.add_response(
        "converse",
        _converse_response(
            'Sure! Here is the JSON:\n'
            '{"intent": "technical", "priority": "urgent", "sentiment": "negative"}\n'
            'Hope that helps!',
            in_tok=205,
            out_tok=40,
        ),
    )

    ticket = triage_mod.load_ticket(TICKETS_DIR / "ticket-002.json")
    with agent_stub, runtime_stub:
        result, _ = triage_mod.triage(
            ticket,
            prompt_id="PROMPTID01",
            agent_client=agent,
            runtime_client=runtime,
        )
    assert result.intent == "technical"
    assert result.priority == "urgent"


def test_triage_retries_once_on_invalid_then_succeeds():
    """First reply has a bad enum; the single validation retry fixes it."""
    agent, agent_stub = _stubbed_agent_client()
    runtime = boto3.client("bedrock-runtime", region_name="us-east-1")
    runtime_stub = Stubber(runtime)
    # Attempt 1: invalid intent -> validation fails.
    runtime_stub.add_response(
        "converse",
        _converse_response(
            '{"intent": "refund", "priority": "high", "sentiment": "negative"}',
            in_tok=210,
            out_tok=18,
        ),
    )
    # Attempt 2 (retry with the error fed back): valid.
    runtime_stub.add_response(
        "converse",
        _converse_response(
            '{"intent": "billing", "priority": "high", "sentiment": "negative"}',
            in_tok=260,
            out_tok=18,
        ),
    )

    ticket = triage_mod.load_ticket(TICKETS_DIR / "ticket-001.json")
    with agent_stub, runtime_stub:
        result, usage = triage_mod.triage(
            ticket,
            prompt_id="PROMPTID01",
            agent_client=agent,
            runtime_client=runtime,
        )
    assert result.intent == "billing"
    # Usage is summed across BOTH calls so the cost line includes the retry.
    assert usage["inputTokens"] == 210 + 260
    assert usage["outputTokens"] == 18 + 18


def test_triage_raises_after_two_invalid_attempts_no_silent_pass():
    """Two invalid replies -> TriageError carrying the raw output. Not swallowed."""
    agent, agent_stub = _stubbed_agent_client()
    runtime = boto3.client("bedrock-runtime", region_name="us-east-1")
    runtime_stub = Stubber(runtime)
    for _ in range(2):
        runtime_stub.add_response(
            "converse",
            _converse_response("not json at all", in_tok=200, out_tok=5),
        )

    ticket = triage_mod.load_ticket(TICKETS_DIR / "ticket-001.json")
    with agent_stub, runtime_stub:
        with pytest.raises(triage_mod.TriageError) as excinfo:
            triage_mod.triage(
                ticket,
                prompt_id="PROMPTID01",
                agent_client=agent,
                runtime_client=runtime,
            )
    assert excinfo.value.raw_output == "not json at all"


def test_converse_request_uses_temperature_zero():
    """Triage must run at temperature 0 — guard the request shape via Stubber."""
    agent, agent_stub = _stubbed_agent_client()
    runtime = boto3.client("bedrock-runtime", region_name="us-east-1")
    runtime_stub = Stubber(runtime)
    rendered = _TEMPLATE_TEXT.replace(
        "{{ticket}}",
        triage_mod.load_ticket(TICKETS_DIR / "ticket-001.json").customer_message,
    )
    runtime_stub.add_response(
        "converse",
        _converse_response(
            '{"intent": "billing", "priority": "high", "sentiment": "negative"}',
            210,
            18,
        ),
        {
            "modelId": triage_mod._MODEL_ID,
            "messages": [{"role": "user", "content": [{"text": rendered}]}],
            "inferenceConfig": {"maxTokens": triage_mod._MAX_TOKENS, "temperature": 0.0},
        },
    )

    ticket = triage_mod.load_ticket(TICKETS_DIR / "ticket-001.json")
    with agent_stub, runtime_stub:
        result, _ = triage_mod.triage(
            ticket,
            prompt_id="PROMPTID01",
            agent_client=agent,
            runtime_client=runtime,
        )
    assert result.intent == "billing"


def test_resolve_prompt_id_errors_without_setup(monkeypatch, tmp_path):
    """No env var and no .prompt_id file -> a clear TriageError, not a crash."""
    monkeypatch.delenv("RELAY_TRIAGE_PROMPT_ID", raising=False)
    monkeypatch.setattr(triage_mod, "_PROMPT_ID_FILE", tmp_path / "nope")
    with pytest.raises(triage_mod.TriageError) as excinfo:
        triage_mod.resolve_prompt_id()
    assert "setup.py" in str(excinfo.value)


# --- The single LIVE test (opt-in) -----------------------------------------
@pytest.mark.live
@pytest.mark.skipif(
    os.environ.get("RELAY_LIVE_TESTS") != "1",
    reason="set RELAY_LIVE_TESTS=1 to make ONE real (sub-cent) Bedrock call",
)
def test_live_triage_real_call():
    """ONE real triage. Costs < $0.0001 as of June 2026. Needs creds + setup.py."""
    ticket = triage_mod.load_ticket(TICKETS_DIR / "ticket-001.json")
    result, usage = triage_mod.triage(ticket)
    assert isinstance(result, Triage)
    # ticket-001 is an unambiguous double-charge: billing, high priority.
    assert result.intent == "billing"
    assert result.priority == "high"
    assert usage["inputTokens"] > 0
    assert usage["outputTokens"] > 0
