"""demo_llm.py — see Relay's FM integration layer route, stream, and cost a call.

Module 3 of AWS GenAI Pro Mastery. This is the observable result of the lab: one
small CLI that exercises relay.llm.converse end to end and SHOWS you the three
things the layer adds over a raw boto3 call — the router's decision, streamed
delivery, and the tokens/cost line.

Usage:
    uv run python demo_llm.py "Why was I charged twice for order #1042?"
        -> router escalates to the smart tier (a billing dispute needs reasoning),
           the answer streams token by token, then tokens/cost.

    uv run python demo_llm.py "hi" --no-stream
        -> router stays on the fast tier (no complexity signal), one shot, no
           streaming — the fast path.

    uv run python demo_llm.py "..." --tier smart      # force a tier
    uv run python demo_llm.py "..." --tier frontier   # the Try-it-yourself path

It makes ONE real Converse/ConverseStream call. Cost: a fraction of a cent on the
fast tier; a bit more on smart/frontier. Needs AWS credentials (AWS_PROFILE) and
us-east-1. The model ID is never named here — only a tier; relay/config.py owns
the mapping.
"""

from __future__ import annotations

import argparse
import sys

from botocore.exceptions import ClientError, NoCredentialsError

from relay import config, llm

# A short system prompt so the demo answers like Relay would: a concise,
# customer-facing CloudCart support reply. This is illustrative — the grounded,
# cited answers come from the Knowledge Base in Module 5; here we just show the
# integration layer working.
_SYSTEM = [
    {
        "text": (
            "You are Relay, CloudCart's support assistant. Answer the customer "
            "concisely and politely. If you do not have order-specific data, say "
            "what you would check and what the customer should do next. Do not "
            "invent order details."
        )
    }
]

# A modest cap so a demo answer does not run away (and the bill stays a fraction
# of a cent). Passed through converse(**params) into Converse inferenceConfig.
_INFERENCE_CONFIG = {"maxTokens": 400, "temperature": 0.3}


def _build_messages(text: str) -> list[dict]:
    return [{"role": "user", "content": [{"text": text}]}]


def _print_route(decision: llm.RouteDecision | None, forced_tier: str | None) -> str:
    """Print the routing decision line and return the concrete tier label."""
    if forced_tier is not None:
        print(f"router: tier={forced_tier}, reason=forced via --tier {forced_tier}")
        return forced_tier
    if decision is not None:
        print(f"router: tier={decision.tier}, reason={decision.reason}")
        return decision.tier
    return "?"


def _cost_line(tier: str, usage: dict) -> str:
    cost = config.estimate_cost(
        tier, usage.get("inputTokens", 0), usage.get("outputTokens", 0)
    )
    return (
        f"tokens: in={usage.get('inputTokens', 0)} "
        f"out={usage.get('outputTokens', 0)} "
        f"| tier={tier} | est. cost: ${cost:.6f}"
    )


def run(text: str, *, tier: str, stream: bool) -> int:
    messages = _build_messages(text)
    params = {"system": _SYSTEM, "inferenceConfig": _INFERENCE_CONFIG}
    forced = None if tier == "auto" else tier

    try:
        if stream:
            streaming = llm.converse(messages, tier=tier, stream=True, **params)
            label = _print_route(streaming.result.route, forced)
            print("\nresponse (streaming):\n")
            for delta in streaming:
                sys.stdout.write(delta)
                sys.stdout.flush()
            print("\n")
            print(_cost_line(streaming.result.tier or label, streaming.result.usage))
        else:
            result = llm.converse(messages, tier=tier, stream=False, **params)
            _print_route(result.route, forced)
            print("\nresponse:\n")
            print(result.text)
            print()
            print(_cost_line(result.tier, result.usage))
    except NoCredentialsError:
        print(
            "No AWS credentials found. Set AWS_PROFILE (e.g. AWS_PROFILE="
            "aws-genai-pro) and ensure the profile has bedrock-runtime access.",
            file=sys.stderr,
        )
        return 2
    except (llm.LLMError, ClientError) as err:
        print(f"\nConverse call failed: {err}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Demo Relay's FM integration layer: routing, streaming, cost.",
    )
    parser.add_argument("message", help="the customer message to send")
    parser.add_argument(
        "--tier",
        choices=["auto", "fast", "smart", "frontier"],
        default="auto",
        help="force a tier; default 'auto' lets the complexity router pick",
    )
    parser.add_argument(
        "--no-stream",
        dest="stream",
        action="store_false",
        help="disable streaming (the fast/parser path)",
    )
    parser.set_defaults(stream=True)
    args = parser.parse_args(argv)

    return run(args.message, tier=args.tier, stream=args.stream)


if __name__ == "__main__":
    raise SystemExit(main())
