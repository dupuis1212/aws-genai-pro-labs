"""relay/pii.py — mask PII at the edge with Amazon Comprehend, by offset.

Module 10 of AWS GenAI Pro Mastery. Module 9 made Relay resist hostile content, but a
real enterprise security review would still fail it: a CloudCart ticket carries a
customer's name, email, phone, and order number, and Modules 2-9 sent that text RAW to
the foundation model, into the agent's decision log, and into AgentCore Memory. The
auditor's question — "what customer data did your bot send the model last Tuesday?" —
has no good answer yet.

This file is the answer's first half: detect and MASK the **PII** in a piece of text
BEFORE it reaches any model, log, or store. We use **Amazon Comprehend**'s
`DetectPiiEntities` — a managed NLP classifier, NOT a Bedrock foundation model and NOT a
Bedrock guardrail. The exam (and the article's comparison table) draws the line three
ways and this module sits on the first:

  - Amazon Comprehend  : PII in TEXT IN FLIGHT — what `relay.intake` runs here, before
                         the FM call. Entity + character offset, per unit of text.
  - Amazon Macie       : PII in S3 AT REST — scheduled discovery jobs over a bucket.
                         Macie does NOT redact in-flight text; the exam loves to swap
                         the two. (Theory in this module — no Macie job is provisioned.)
  - Bedrock Guardrails : a PII filter at the MODEL-CALL moment (mask/block), the M9
                         layer. Useful, but it fires at the boundary of ONE call; it does
                         not stop PII from landing in YOUR decision log or memory first.

Why redact HERE, at intake, and not only at the guardrail: **redact at the edge, and
everything downstream inherits the protection**. Once `relay.intake` replaces the name
with `[NAME]` on the normalized text, the FM, the Comprehend entity pass, the agent's
decision log, the persisted `TicketRecord`, and AgentCore Memory ALL see the masked
version — there is no second place a raw email can leak. A guardrail-only design still
writes the raw prompt to your invocation logs.

What this module owns:

    detect_pii(text, *, client=None, min_score=...) -> list[PiiSpan]
        Run Comprehend `DetectPiiEntities` and return the detected spans (type +
        character offsets + confidence) ABOVE the confidence threshold. Comprehend
        returns offsets, not the matched substring — masking is done by SLICING the
        original text at those offsets, never by a home-grown regex. A Comprehend client
        error is raised as PiiError (no silent empty — a failed detection must not be
        mistaken for "no PII found").

    redact(text, *, client=None, min_score=...) -> RedactionResult
        Detect, then replace each detected span — from the end of the string backward, so
        earlier offsets stay valid — with its typed placeholder (`[NAME]`, `[EMAIL]`,
        `[PHONE]`, ...). Returns the masked text, whether anything was masked, and a
        type histogram (counts only — NO raw values, so the result itself is safe to
        log). This is what `relay.intake` calls before any FM call.

Boundary (bible §2.2 M10): the masking ACTION lives here; the Ticket gains exactly one
field (`pii_redacted: bool = False`) in relay.models; the entity types and the
confidence threshold live in relay.config (one place). This module holds NO model ID and
makes NO Bedrock call — Comprehend is a separate service.

Run it on one string:
    uv run python -m relay.pii "Hi, I'm Dana Lee (dana.lee@example.com, 555-0100), \\
        order #1042 is late."
    # -> Hi, I'm [NAME] ([EMAIL], [PHONE]), order #1042 is late.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

import boto3
from botocore.exceptions import (
    BotoCoreError,
    ClientError,
    NoCredentialsError,
    ProfileNotFound,
)

from relay import config

REGION = config.REGION


class PiiError(RuntimeError):
    """Raised when Comprehend PII detection cannot be completed.

    Carries the underlying AWS error so the failure is debuggable. We RAISE rather than
    return an empty result on a client error: silently treating a failed detection as
    "no PII found" would ship raw customer data to the model — exactly the failure this
    module exists to prevent (brief §6: no silent try/except).
    """

    def __init__(self, message: str, *, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.__cause__ = cause


@dataclass(frozen=True)
class PiiSpan:
    """One PII entity Comprehend found — a typed character span, no raw value stored.

    - entity_type : the Comprehend PII type ("NAME", "EMAIL", "PHONE", ...).
    - begin / end : the character offsets in the ORIGINAL text (end-exclusive), the way
                    Comprehend reports them. Masking slices the text at these offsets.
    - score       : the detection confidence in [0, 1].

    We deliberately do NOT keep the matched substring here — a PiiSpan is metadata about
    where PII is, not a copy of it, so a list of spans is safe to inspect/log.
    """

    entity_type: str
    begin: int
    end: int
    score: float

    @property
    def placeholder(self) -> str:
        """The typed placeholder this span is replaced with, e.g. `[NAME]`."""
        return placeholder_for(self.entity_type)


@dataclass
class RedactionResult:
    """The outcome of redact(): the masked text plus a SAFE summary of what was masked.

    - text     : the input with every detected PII span replaced by its placeholder.
    - redacted : True if at least one span was masked (this is what `relay.intake`
                 carries into Ticket.pii_redacted).
    - counts   : a {entity_type: n} histogram — COUNTS ONLY, never raw values, so the
                 result is safe to print/log (an audit line can say "masked 1 NAME, 1
                 EMAIL" without leaking the name or the email).
    """

    text: str
    redacted: bool
    counts: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        """A one-line, PII-free description of what was masked (for logs/CLI)."""
        if not self.redacted:
            return "no PII detected"
        parts = [f"{n} {etype}" for etype, n in sorted(self.counts.items())]
        return "masked " + ", ".join(parts)


def placeholder_for(entity_type: str) -> str:
    """The typed placeholder for a Comprehend PII type — `[NAME]`, `[EMAIL]`, ...

    Uppercased and bracketed so the masked text is human-readable and the type survives
    into the FM prompt ("the customer [NAME] reports [PHONE] is wrong") — the model still
    knows a name/phone was there, just not its value. A single, deterministic mapping so
    intake, the audit log, and the tests agree by construction.
    """
    return f"[{entity_type.upper()}]"


def _comprehend_client():
    """An Amazon Comprehend client (the in-flight PII detector — not a Bedrock service)."""
    return boto3.client("comprehend", region_name=REGION)


def detect_pii(
    text: str,
    *,
    client=None,
    min_score: float | None = None,
) -> list[PiiSpan]:
    """Detect PII in `text` with Comprehend DetectPiiEntities (skill 3.2.2/3.2.3).

    Returns the detected spans ABOVE the confidence threshold, sorted by start offset.
    Comprehend reports each entity as a {Type, BeginOffset, EndOffset, Score} — character
    OFFSETS, not the substring — so callers mask by slicing the original text, never by a
    regex guess. An empty input short-circuits (no call). A Comprehend client error is
    raised as PiiError; we never return [] on failure (that would leak raw PII).

    Args:
        text: the text to scan (the normalized customer message, at intake).
        client: an injected Comprehend client (tests pass a Stubber-wrapped one).
        min_score: the confidence floor; defaults to config.PII_MIN_CONFIDENCE. A span
            below it is dropped — we would rather miss a low-confidence borderline than
            mask a real order number that Comprehend half-suspects is a phone.
    """
    if not text.strip():
        return []
    floor = config.PII_MIN_CONFIDENCE if min_score is None else min_score
    client = client or _comprehend_client()
    try:
        response = client.detect_pii_entities(
            Text=text, LanguageCode=config.COMPREHEND_LANGUAGE_CODE
        )
    except ClientError as err:
        raise PiiError(
            "Amazon Comprehend detect_pii_entities failed: "
            f"{err.response['Error']['Code']} — {err.response['Error']['Message']}. "
            "Refusing to continue: a failed PII scan must not be treated as 'no PII'."
        ) from err

    spans: list[PiiSpan] = []
    for ent in response.get("Entities", []):
        etype = ent.get("Type", "")
        score = float(ent.get("Score", 0.0))
        begin = ent.get("BeginOffset")
        end = ent.get("EndOffset")
        if not etype or begin is None or end is None:
            continue
        if etype not in config.PII_ENTITY_TYPES:
            # A type we deliberately do NOT mask (e.g. DATE_TIME — a delivery date is
            # operational signal, not identity; config.PII_ENTITY_TYPES is the allowlist).
            continue
        if score < floor:
            continue
        spans.append(PiiSpan(entity_type=etype, begin=int(begin), end=int(end),
                             score=score))
    spans.sort(key=lambda s: s.begin)
    return spans


def mask_spans(text: str, spans: list[PiiSpan]) -> str:
    """Replace each span in `text` with its typed placeholder, by offset.

    Applies the replacements from the LAST span to the FIRST so that replacing one span
    (which changes the string length) never shifts the offsets of spans not yet applied.
    Overlapping spans are coalesced (the widest wins) so a NAME inside an ADDRESS is not
    double-bracketed. Pure function: deterministic, no AWS call — the masking step the
    smoke test can prove without a network."""
    if not spans:
        return text
    # Sort by start, then drop spans contained in / overlapping an already-kept one.
    ordered = sorted(spans, key=lambda s: (s.begin, -(s.end - s.begin)))
    kept: list[PiiSpan] = []
    last_end = -1
    for span in ordered:
        if span.begin >= last_end:
            kept.append(span)
            last_end = span.end
        elif span.end > last_end:
            # Overlap that extends further — widen the previous kept span's reach.
            last_end = span.end
    result = text
    for span in sorted(kept, key=lambda s: s.begin, reverse=True):
        result = result[:span.begin] + span.placeholder + result[span.end:]
    return result


def redact(
    text: str,
    *,
    client=None,
    min_score: float | None = None,
) -> RedactionResult:
    """Detect + mask PII in `text`; return the masked text and a SAFE summary.

    The function `relay.intake` calls before ANY foundation-model call (the FM, the
    Comprehend entity pass, the vision read, the decision log, and AgentCore Memory all
    see the masked text — redact at the edge, everything downstream inherits it). The
    returned `counts` histogram carries no raw values, so logging the result cannot leak
    PII. Raises PiiError if detection fails (no silent fallthrough to the raw text).
    """
    spans = detect_pii(text, client=client, min_score=min_score)
    if not spans:
        return RedactionResult(text=text, redacted=False, counts={})
    counts: dict[str, int] = {}
    for span in spans:
        counts[span.entity_type] = counts.get(span.entity_type, 0) + 1
    masked = mask_spans(text, spans)
    return RedactionResult(text=masked, redacted=True, counts=counts)


# =============================================================================
# CLI — redact one string; print the masked text + a PII-free summary to stderr.
# =============================================================================
def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print(
            'Usage: uv run python -m relay.pii "<text with PII>"\n'
            'Example: uv run python -m relay.pii "Hi, I\'m Dana Lee '
            '(dana.lee@example.com, 555-0100)."',
            file=sys.stderr,
        )
        return 1
    try:
        result = redact(argv[0])
    except PiiError as err:
        print(f"PII detection failed: {err}", file=sys.stderr)
        return 1
    except (NoCredentialsError, ProfileNotFound, BotoCoreError) as err:
        print(f"AWS credentials/config problem: {err}\n"
              "Set AWS_PROFILE=aws-genai-pro and run from us-east-1.",
              file=sys.stderr)
        return 1
    print(result.text)
    print(f"[pii] {result.summary()}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
