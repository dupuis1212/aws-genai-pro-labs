"""hello_bedrock.py — your first Amazon Bedrock Converse call.

Module 1 of AWS GenAI Pro Mastery. This is the PoC that proves the whole chain
works: credentials -> bedrock-runtime client -> Converse -> a foundation model
-> tokens and a real (tiny) bill. Everything else in the course builds on this.

Run it:
    uv run python hello_bedrock.py "What does CloudCart sell?"

It prints the model's reply, then a line like:
    tokens: in=38 out=43 | est. cost: $0.00001
"""

from __future__ import annotations

import os
import sys

import boto3
from botocore.exceptions import ClientError

# --- Model selection -------------------------------------------------------
# PROVISIONAL — relay/config.py owns this from M3 onward. In M1 there is no
# relay/ package yet, so the inference-profile ID lives here as a single
# top-of-file constant. From Module 3 the tier->profile map moves into
# relay/config.py and is the SOLE home of every model ID in the course.
#
# Why a "us." prefix and not the bare "amazon.nova-lite-v1:0"? Because this is
# an INFERENCE PROFILE, not a bare regional ID. Recent models on Amazon Bedrock
# can only be invoked on-demand through an inference profile; a bare regional ID
# fails with "Retry with an inference profile". See the error handler below.
MODEL_ID = "us.amazon.nova-lite-v1:0"

# --- Pricing ---------------------------------------------------------------
# Per-1,000-token prices for Amazon Nova Lite, us-east-1, on-demand.
# AS OF JUNE 2026 — re-verify on https://aws.amazon.com/bedrock/pricing/.
# Bedrock bills per token; we divide the published per-million price by 1000 so
# the cost line below is computed from the API's usage block, not guessed.
PRICE_PER_1K_INPUT = 0.00006   # $0.06 per million input tokens
PRICE_PER_1K_OUTPUT = 0.00024  # $0.24 per million output tokens

# A one-line system prompt. Real prompt engineering is Module 2 — here we only
# need enough grounding for the reply to make sense.
SYSTEM_PROMPT = (
    "You are a concise support assistant for CloudCart, a hosted e-commerce "
    "platform that lets small merchants run online stores. Answer in 2-3 sentences."
)

DEFAULT_QUESTION = "What does CloudCart sell?"

# Temperature is overridable from the environment so you can experiment in Step 5
# (`RELAY_TEMPERATURE=0.9 uv run python hello_bedrock.py`) WITHOUT editing this
# file — the offline test pins the default (0.2), so an in-place edit would make
# `uv run pytest` fail on the asserted request shape.
TEMPERATURE = float(os.environ.get("RELAY_TEMPERATURE", "0.2"))


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    """Cost in USD from the usage block the API returns — never a guess."""
    return (
        input_tokens / 1000 * PRICE_PER_1K_INPUT
        + output_tokens / 1000 * PRICE_PER_1K_OUTPUT
    )


def main() -> int:
    question = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUESTION

    # boto3 default session: respects AWS_PROFILE if set, else the default
    # credentials chain. No hardcoded keys, no .env for AWS (course rule).
    # Region is pinned to us-east-1 for the whole course.
    client = boto3.client("bedrock-runtime", region_name="us-east-1")

    try:
        response = client.converse(
            modelId=MODEL_ID,
            system=[{"text": SYSTEM_PROMPT}],
            messages=[{"role": "user", "content": [{"text": question}]}],
            inferenceConfig={"maxTokens": 300, "temperature": TEMPERATURE},
        )
    except ClientError as err:
        # No silent except. The single failure mode worth teaching in M1 is the
        # inference-profile trap: a bare regional ID, or a model the use-case
        # form has not unlocked, surfaces as ValidationException / AccessDenied.
        code = err.response["Error"]["Code"]
        message = err.response["Error"]["Message"]
        if "inference profile" in message.lower() or code == "ValidationException":
            print(
                "Converse failed with a validation error:\n"
                f"  {message}\n\n"
                "The usual M1 cause: a BARE regional model ID. Recent models on\n"
                "Amazon Bedrock must be called through an INFERENCE PROFILE.\n"
                f"  wrong: amazon.nova-lite-v1:0\n"
                f"  right: {MODEL_ID}   (note the 'us.' prefix)\n"
                "Fix MODEL_ID at the top of this file and re-run.",
                file=sys.stderr,
            )
        elif code == "AccessDeniedException":
            print(
                "Access denied for this model.\n"
                f"  {message}\n\n"
                "Nova models auto-activate, so this most likely means the model\n"
                "you switched to needs the one-time Anthropic use-case form.\n"
                "Run `uv run python setup.py` and read its model-access section.",
                file=sys.stderr,
            )
        else:
            # Re-raise anything we did not explicitly teach: a full traceback is
            # more honest than a swallowed error.
            raise
        return 1

    reply = response["output"]["message"]["content"][0]["text"]
    usage = response["usage"]
    in_tokens = usage["inputTokens"]
    out_tokens = usage["outputTokens"]
    cost = estimate_cost(in_tokens, out_tokens)

    print(reply.strip())
    print(
        f"\ntokens: in={in_tokens} out={out_tokens} "
        f"| est. cost: ${cost:.5f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
