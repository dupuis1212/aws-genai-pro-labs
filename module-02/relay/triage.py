"""relay/triage.py — raw CloudCart ticket -> validated Triage JSON.

Module 2 of AWS GenAI Pro Mastery. This is Relay's first real job. The flow:

    Ticket
      -> load the triage template from Bedrock Prompt Management (by id+VERSION,
         not inline text — the repo's prompts/triage_prompt.md is the git mirror)
      -> render {{ticket}} with the customer's message
      -> Converse (Nova Micro, temperature 0)         [structured output by prompt]
      -> Triage.model_validate_json                    [validation, not trust]
      -> on a validation error: ONE retry, feeding the Pydantic error back into
         the prompt so the model can correct itself
      -> Triage

Run it on a single ticket:
    uv run python -m relay.triage data/tickets/ticket-001.json

It prints the validated Triage JSON, then a tokens/cost line.

IMPORTANT — the model ID lives here PROVISIONALLY. There is no relay/config.py
or relay/llm.py in Module 2; both arrive in Module 3, where this file is
refactored to drop its own constant and call `converse(tier="fast")` instead.
Until then, the single inference-profile ID for the whole module is the one
constant `_MODEL_ID` below. It is a `us.` inference profile, never a bare
regional ID (a bare ID fails on-demand with "Retry with an inference profile").

The retry here is a VALIDATION retry. Network retries, throttling backoff, and
cross-Region fallback are the FM integration layer's job — that is Module 3.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

from relay.models import Ticket, Triage

REGION = "us-east-1"

# --- Model selection (PROVISIONAL — relay/config.py owns this from M3) -------
# `fast` tier: Amazon Nova Micro, called through its us. INFERENCE PROFILE.
# Triage is a classifier, so this is the cheapest capable model. From Module 3
# this constant disappears and triage() calls converse(tier="fast").
_MODEL_ID = "us.amazon.nova-micro-v1:0"

# --- Pricing (AS OF JUNE 2026 — re-verify on the Bedrock pricing page) -------
# Amazon Nova Micro, us-east-1, on-demand. Per-1,000-token prices, derived from
# the published per-million figures ($0.035 in / $0.14 out) so the cost line is
# computed from the API usage block, never guessed.
_PRICE_PER_1K_INPUT = 0.000035   # $0.035 per million input tokens
_PRICE_PER_1K_OUTPUT = 0.00014   # $0.14 per million output tokens

# --- Inference config --------------------------------------------------------
# Temperature 0: triage is classification. Creativity in a classifier is a bug,
# not a feature. maxTokens is small — the output is a one-line JSON object.
_TEMPERATURE = 0.0
_MAX_TOKENS = 100

# The Prompt Management version this code consumes. Immutable versions are the
# whole point: the code pins a number, never the prompt text.
_PROMPT_VERSION = "1"

# The {{var}} placeholder name inside the Prompt Management template.
_TEMPLATE_VARIABLE = "ticket"

# Where setup.py records the created prompt's ID, so triage() can find it without
# an env var. An explicit RELAY_TRIAGE_PROMPT_ID env var wins if set.
_PROMPT_ID_FILE = Path(__file__).resolve().parent.parent / "prompts" / ".prompt_id"


class TriageError(RuntimeError):
    """Raised when triage cannot produce a valid Triage even after the retry.

    Carrying the raw model text makes the failure debuggable instead of silent.
    """

    def __init__(self, message: str, *, raw_output: str | None = None) -> None:
        super().__init__(message)
        self.raw_output = raw_output


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    """Cost in USD from the API usage block — never a guess."""
    return (
        input_tokens / 1000 * _PRICE_PER_1K_INPUT
        + output_tokens / 1000 * _PRICE_PER_1K_OUTPUT
    )


def resolve_prompt_id() -> str:
    """Find the Prompt Management prompt ID created by setup.py.

    Order: RELAY_TRIAGE_PROMPT_ID env var, then prompts/.prompt_id. If neither
    exists, raise with the exact fix — no silent fallback to an inline prompt.
    """
    env = os.environ.get("RELAY_TRIAGE_PROMPT_ID")
    if env:
        return env.strip()
    if _PROMPT_ID_FILE.exists():
        recorded = _PROMPT_ID_FILE.read_text(encoding="utf-8").strip()
        if recorded:
            return recorded
    raise TriageError(
        "No triage prompt ID found. Create the Prompt Management prompt first:\n"
        "    uv run python setup.py\n"
        "It publishes version 1 and records the ID in prompts/.prompt_id. "
        "Alternatively set RELAY_TRIAGE_PROMPT_ID to an existing prompt ID."
    )


def _agent_client():
    return boto3.client("bedrock-agent", region_name=REGION)


def _runtime_client():
    return boto3.client("bedrock-runtime", region_name=REGION)


def load_template(prompt_id: str, *, version: str = _PROMPT_VERSION,
                  client=None) -> str:
    """Fetch the governed triage template text from Prompt Management.

    The code consumes the prompt by IDENTIFIER + immutable VERSION. The returned
    text still contains the {{ticket}} placeholder; we render it ourselves so the
    single Converse call carries the final prompt.
    """
    client = client or _agent_client()
    prompt = client.get_prompt(promptIdentifier=prompt_id, promptVersion=version)
    # A prompt has one or more variants; we use the default (single) TEXT variant.
    variant = prompt["variants"][0]
    return variant["templateConfiguration"]["text"]["text"]


def _render(template: str, ticket_message: str) -> str:
    """Substitute the {{ticket}} placeholder. Prompt Management uses {{var}}."""
    return template.replace("{{" + _TEMPLATE_VARIABLE + "}}", ticket_message)


def _converse_once(prompt_text: str, *, client) -> tuple[str, dict]:
    """One Converse call at temperature 0. Returns (reply_text, usage)."""
    try:
        response = client.converse(
            modelId=_MODEL_ID,
            messages=[{"role": "user", "content": [{"text": prompt_text}]}],
            inferenceConfig={"maxTokens": _MAX_TOKENS, "temperature": _TEMPERATURE},
        )
    except ClientError as err:
        code = err.response["Error"]["Code"]
        message = err.response["Error"]["Message"]
        if "inference profile" in message.lower() or code == "ValidationException":
            # The M1 trap, surfaced honestly rather than swallowed.
            raise TriageError(
                "Converse failed with a validation error:\n"
                f"  {message}\n\n"
                "The usual cause is a BARE regional model ID. _MODEL_ID must be a\n"
                f"us. INFERENCE PROFILE — currently: {_MODEL_ID}"
            ) from err
        raise
    text = response["output"]["message"]["content"][0]["text"]
    return text, response["usage"]


def triage(ticket: Ticket, *, prompt_id: str | None = None,
           agent_client=None, runtime_client=None) -> tuple[Triage, dict]:
    """Classify one ticket into a validated Triage. Returns (triage, usage).

    Validation flow:
      1. Render the governed template with the ticket message and Converse once.
      2. Try Triage.model_validate_json on the model's output.
      3. If validation fails, retry ONCE, appending the Pydantic error to the
         prompt so the model can fix its own malformed/invalid output.
      4. If the second attempt still fails, raise TriageError (no silent pass).

    `usage` aggregates token counts across however many calls were made, so the
    cost line reflects the true spend including any retry.
    """
    prompt_id = prompt_id or resolve_prompt_id()
    agent_client = agent_client or _agent_client()
    runtime_client = runtime_client or _runtime_client()

    template = load_template(prompt_id, client=agent_client)
    prompt_text = _render(template, ticket.customer_message)

    total_in = total_out = 0
    last_output = ""
    last_error = ""

    for attempt in (1, 2):  # one initial call + one validation retry
        raw, usage = _converse_once(prompt_text, client=runtime_client)
        total_in += usage["inputTokens"]
        total_out += usage["outputTokens"]
        last_output = raw

        candidate = _extract_json(raw)
        try:
            result = Triage.model_validate_json(candidate)
        except ValueError as err:
            last_error = str(err)
            if attempt == 2:
                break
            # Feed the validation error back in for the single retry. This is the
            # structured-output correction loop: we tell the model exactly what
            # was wrong and ask only for the corrected JSON.
            prompt_text = (
                f"{prompt_text}\n\n"
                "Your previous reply was rejected by schema validation:\n"
                f"--- your reply ---\n{raw}\n--- end ---\n"
                f"Validation error:\n{last_error}\n\n"
                "Return ONLY the corrected JSON object — the three keys "
                '("intent", "priority", "sentiment") with allowed values, '
                "nothing else."
            )
            continue
        usage_total = {"inputTokens": total_in, "outputTokens": total_out,
                       "totalTokens": total_in + total_out}
        return result, usage_total

    raise TriageError(
        "Triage produced invalid output even after one validation retry.\n"
        f"Last validation error:\n{last_error}",
        raw_output=last_output,
    )


def _extract_json(text: str) -> str:
    """Best-effort: isolate the JSON object from the model's reply.

    The prompt asks for bare JSON, but a robust parser still strips a stray code
    fence or surrounding prose by taking the substring from the first '{' to the
    last '}'. If there is no brace pair, return the text unchanged so the
    validator produces a clear error (which then drives the retry).
    """
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    return text.strip()


def load_ticket(path: str | Path) -> Ticket:
    """Load and validate a ticket fixture from a JSON file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return Ticket.model_validate(data)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print(
            "Usage: uv run python -m relay.triage <path-to-ticket.json>\n"
            "Example: uv run python -m relay.triage data/tickets/ticket-001.json",
            file=sys.stderr,
        )
        return 1

    ticket = load_ticket(argv[0])
    try:
        result, usage = triage(ticket)
    except TriageError as err:
        print(f"Triage failed: {err}", file=sys.stderr)
        if err.raw_output is not None:
            print(f"\nRaw model output was:\n{err.raw_output}", file=sys.stderr)
        return 1

    cost = estimate_cost(usage["inputTokens"], usage["outputTokens"])
    print(result.model_dump_json())
    print(
        f"\ntokens: in={usage['inputTokens']} out={usage['outputTokens']} "
        f"| est. cost: ${cost:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
