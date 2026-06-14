"""smoke_test.py — Module 1 lab tests.

OFFLINE BY DEFAULT. Every test here except the one marked `live` uses a
botocore Stubber to fake the Converse response, so it runs with NO AWS
credentials and makes NO network call. That is the course convention: tests
pass offline so anyone can `uv run pytest` on a fresh clone.

ONE test is marked `live` and makes a real Bedrock call. It is skipped unless
you opt in:
    RELAY_LIVE_TESTS=1 uv run pytest -m live
LIVE-CALL BUDGET: a single Nova Lite Converse call with a ~20-token prompt and
maxTokens=50 costs well under $0.0001 (one hundredth of a cent) as of June 2026.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import boto3
import pytest
from botocore.stub import Stubber

# Import the lab module by path (module-01/ is a flat layout, not a package).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import hello_bedrock  # noqa: E402


# --- A realistic Converse response, byte-shaped like the real API ----------
def _fake_converse_response(reply: str, in_tok: int, out_tok: int) -> dict:
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": reply}],
            }
        },
        "stopReason": "end_turn",
        "usage": {
            "inputTokens": in_tok,
            "outputTokens": out_tok,
            "totalTokens": in_tok + out_tok,
        },
        "metrics": {"latencyMs": 412},
    }


# --- Pure-function tests (no client at all) --------------------------------
def test_cost_is_computed_from_usage_not_guessed():
    # 1000 in + 1000 out at the documented per-1k prices.
    cost = hello_bedrock.estimate_cost(1000, 1000)
    expected = hello_bedrock.PRICE_PER_1K_INPUT + hello_bedrock.PRICE_PER_1K_OUTPUT
    assert cost == pytest.approx(expected)


def test_cost_is_zero_for_zero_tokens():
    assert hello_bedrock.estimate_cost(0, 0) == 0.0


def test_model_id_is_an_inference_profile_not_a_bare_id():
    # The real M1 trap: a bare regional ID fails on-demand. Guard it in a test.
    assert hello_bedrock.MODEL_ID.startswith(("us.", "global."))
    assert not hello_bedrock.MODEL_ID.startswith(("amazon.", "anthropic."))


# --- Offline Converse test via Stubber (no creds, no network) --------------
def test_converse_call_offline(monkeypatch, capsys):
    client = boto3.client("bedrock-runtime", region_name="us-east-1")
    stubber = Stubber(client)

    response = _fake_converse_response(
        "CloudCart sells a hosted online-store platform for small merchants.",
        in_tok=23,
        out_tok=58,
    )
    # We assert the request shape too: modelId, system, messages, inferenceConfig.
    expected_params = {
        "modelId": hello_bedrock.MODEL_ID,
        "system": [{"text": hello_bedrock.SYSTEM_PROMPT}],
        "messages": [
            {"role": "user", "content": [{"text": "What does CloudCart sell?"}]}
        ],
        "inferenceConfig": {"maxTokens": 300, "temperature": 0.2},
    }
    stubber.add_response("converse", response, expected_params)

    # Make hello_bedrock use our stubbed client instead of a real one.
    monkeypatch.setattr(
        hello_bedrock.boto3, "client", lambda *a, **k: client
    )
    monkeypatch.setattr(sys, "argv", ["hello_bedrock.py"])  # default question

    with stubber:
        rc = hello_bedrock.main()

    assert rc == 0
    out = capsys.readouterr().out
    assert "CloudCart sells" in out
    assert "tokens: in=23 out=58" in out
    # 23/1000*0.00006 + 58/1000*0.00024 = 0.00001530 -> "$0.00002" at 5dp
    expected_cost = hello_bedrock.estimate_cost(23, 58)
    assert f"${expected_cost:.5f}" in out


def test_argv_question_is_used_offline(monkeypatch, capsys):
    client = boto3.client("bedrock-runtime", region_name="us-east-1")
    stubber = Stubber(client)
    stubber.add_response(
        "converse",
        _fake_converse_response("Yes, CloudCart supports refunds.", 30, 12),
        {
            "modelId": hello_bedrock.MODEL_ID,
            "system": [{"text": hello_bedrock.SYSTEM_PROMPT}],
            "messages": [
                {"role": "user", "content": [{"text": "Do you support refunds?"}]}
            ],
            "inferenceConfig": {"maxTokens": 300, "temperature": 0.2},
        },
    )
    monkeypatch.setattr(hello_bedrock.boto3, "client", lambda *a, **k: client)
    monkeypatch.setattr(sys, "argv", ["hello_bedrock.py", "Do you support refunds?"])

    with stubber:
        rc = hello_bedrock.main()

    assert rc == 0
    assert "refunds" in capsys.readouterr().out


def test_bare_id_error_is_taught_not_swallowed(monkeypatch, capsys):
    """A ValidationException about inference profiles prints the teaching fix."""
    client = boto3.client("bedrock-runtime", region_name="us-east-1")
    stubber = Stubber(client)
    stubber.add_client_error(
        "converse",
        service_error_code="ValidationException",
        service_message=(
            "Invocation of model ID amazon.nova-lite-v1:0 with on-demand "
            "throughput isn't supported. Retry with an inference profile."
        ),
    )
    monkeypatch.setattr(hello_bedrock.boto3, "client", lambda *a, **k: client)
    monkeypatch.setattr(sys, "argv", ["hello_bedrock.py", "hi"])

    with stubber:
        rc = hello_bedrock.main()

    assert rc == 1
    err = capsys.readouterr().err
    assert "inference profile" in err.lower()
    assert hello_bedrock.MODEL_ID in err  # shows the correct 'us.' fix


# --- The single LIVE test (opt-in) -----------------------------------------
@pytest.mark.live
@pytest.mark.skipif(
    os.environ.get("RELAY_LIVE_TESTS") != "1",
    reason="set RELAY_LIVE_TESTS=1 to make a real (sub-cent) Bedrock call",
)
def test_live_converse_real_call():
    """One real Converse call. Costs < $0.0001 as of June 2026. Needs creds."""
    client = boto3.client("bedrock-runtime", region_name="us-east-1")
    response = client.converse(
        modelId=hello_bedrock.MODEL_ID,
        system=[{"text": hello_bedrock.SYSTEM_PROMPT}],
        messages=[{"role": "user", "content": [{"text": "Reply with the word OK."}]}],
        inferenceConfig={"maxTokens": 50, "temperature": 0.0},
    )
    text = response["output"]["message"]["content"][0]["text"]
    assert isinstance(text, str) and text.strip()
    assert response["usage"]["inputTokens"] > 0
    assert response["usage"]["outputTokens"] > 0
