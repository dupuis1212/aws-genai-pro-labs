"""relay/intake.py — raw email/chat (± screenshot) -> a validated, normalized Ticket.

Module 6 of AWS GenAI Pro Mastery. Modules 2–5 always fed Relay CLEAN tickets. The
real world does not: a support email arrives with a 40-line quoted thread, a
signature block, broken HTML, "see attached", and a screenshot of the actual error —
which exists ONLY in the image. Pass that raw to the Module 2 triage and the
classifier drowns in the signature and never sees the error. Garbage in, garbage out:
the best prompt in the world does not repair a polluted input. The fix is a PIPELINE
that runs BEFORE any foundation-model call.

This file is that pipeline. `intake()` takes a raw payload (+ optional attachment
bytes) and runs, IN ORDER:

  1. VALIDATE (skill 1.3.1 — validate before you generate). Gates that cost nothing
     and protect everything downstream: the body must decode as UTF-8, be non-empty
     after normalization, and stay under the size ceiling; an attachment must be an
     admitted image type under the byte ceiling. A failure raises a structured
     IntakeRejected — an EXPLICIT business error the CLI prints and exits 1 on. There
     is NO silent try/except: an invalid input is rejected loudly, not coerced.
  2. NORMALIZE (skill 1.3.4 — enhance input quality). Strip the signature block and
     the quoted reply thread, unwrap simple HTML, collapse whitespace. Fewer tokens,
     more signal: the classifier sees the customer's actual problem, not 40 lines of
     legal footer. (This step is where Module 10's PII redaction will slot in — it
     runs here, on the normalized text, BEFORE Comprehend and BEFORE the vision call.
     Module 6 does NOT redact anything: there is no PII redaction here, and the
     Ticket has no `pii_redacted` field yet.)
  3. ENTITIES (skill 1.3.4 — Amazon Comprehend). detect_entities pulls order numbers,
     product/commercial items, and dates out of the normalized text. They are logged
     and appended as a short, structured `[Entities]` line so triage and the agent get
     the salient facts up front. Comprehend is a managed NLP service — NOT a Bedrock
     foundation model, and NOT a Bedrock guardrail.
  4. ATTACHMENT (skills 1.3.2 / 1.3.3 — multimodal input via Converse). An accepted
     screenshot is uploaded to s3://relay-<account_id>/attachments/ and recorded as an
     Attachment (filename, media_type, s3_uri). Then Amazon Nova Lite (the "vision"
     tier in relay.config) READS it through relay.llm.converse — a Converse message
     carrying a text block AND an image content block together — and a short extraction
     ("what error, what screen, what user action") is appended under an `[Attachment
     summary]` separator. The Ticket SCHEMA is not changed for the summary: it rides
     inside customer_message, while the file metadata rides in Ticket.attachments.

The result is a validated, normalized `Ticket` (06 §2 schema, +Attachment/+attachments
frozen here) ready for the Module 2 triage — which `demo()` runs end to end.

This is Converse-only multimodal: the image goes through `relay.llm.image_block` and
`converse(tier="vision")`, never the legacy single-prompt invoke path with base64 in
a model-specific body (07 §3.3). The vision model ID lives only in relay.config.

Run it:
    uv run python -m relay.intake data/raw/email_billing_error.txt \\
        --attachment data/raw/payment_error.png
    uv run python -m relay.intake data/raw/invalid_oversized.txt   # -> exit 1
    uv run python -m relay.intake data/raw/email_billing_error.txt --triage  # + triage
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import boto3
from botocore.exceptions import (
    BotoCoreError,
    ClientError,
    NoCredentialsError,
    ProfileNotFound,
)

from relay import config, llm
from relay.models import Attachment, Ticket

REGION = config.REGION

# The separator the screenshot extraction is appended under inside customer_message.
# A literal the article and the smoke test both pin (the Ticket schema is unchanged).
ATTACHMENT_SUMMARY_HEADER = "[Attachment summary]"
ENTITIES_HEADER = "[Entities]"

# Inference config for the vision read: a few tokens of structured description, low
# temperature (we want the error text read back, not invented). Passed through
# converse(**params) into the Converse inferenceConfig.
_VISION_INFERENCE = {"maxTokens": 220, "temperature": 0.0}

# The extraction prompt for the screenshot. Structured, narrow, and grounded in the
# image — "read what is visible", not "imagine what might be wrong".
_VISION_PROMPT = (
    "This is a screenshot a customer attached to a support ticket. Read it and "
    "report ONLY what is visible, in three short labelled lines:\n"
    "Error: <the error message or code shown, verbatim if present>\n"
    "Screen: <which screen/page this is>\n"
    "User action: <the action the screen tells the user to take, if any>\n"
    "If the image shows no error, say so. Do not speculate beyond the image."
)


# =============================================================================
# Errors — explicit, structured, never a silent pass (style guide / brief §10).
# =============================================================================
class IntakeError(RuntimeError):
    """Base for intake failures. Carries a clear, actionable message."""


class IntakeRejected(IntakeError):
    """A validation gate rejected the input (skill 1.3.1).

    `reason` is a short machine-readable code (e.g. "message_too_large"); the
    message is the human-readable explanation the CLI prints before exiting 1. This
    is a BUSINESS rejection — a malformed/oversized/unsupported input — not a bug and
    not a transient AWS error. It is raised, never swallowed.
    """

    def __init__(self, message: str, *, reason: str) -> None:
        super().__init__(message)
        self.reason = reason


# =============================================================================
# The raw payload — a tiny header block + the raw body (fixtures in data/raw/).
# =============================================================================
@dataclass
class RawIntake:
    """A raw intake payload before validation/normalization.

    `channel` / `ticket_id` / `created_at` come from the header lines; `body` is the
    raw, messy customer text (signature, quoted thread, HTML — all still present).
    """

    channel: str
    body: str
    ticket_id: str | None = None
    created_at: str | None = None


_HEADER_LINE = re.compile(r"^([a-z_]+):\s*(.*)$")


def parse_raw(text: str) -> RawIntake:
    """Parse a raw fixture: `key: value` header lines, a blank line, then the body.

    Lenient on the header (only `channel` is required to know how to read the rest);
    everything after the first blank line is the raw body, untouched. A missing or
    unknown channel is a validation rejection — we never guess the channel.
    """
    lines = text.splitlines()
    headers: dict[str, str] = {}
    body_start = 0
    for i, line in enumerate(lines):
        if line.strip() == "":
            body_start = i + 1
            break
        m = _HEADER_LINE.match(line)
        if m:
            headers[m.group(1)] = m.group(2).strip()
        else:
            # First non-header, non-blank line: the body starts here (no header block).
            body_start = i
            break
    else:
        body_start = len(lines)

    body = "\n".join(lines[body_start:])
    channel = headers.get("channel", "")
    return RawIntake(
        channel=channel,
        body=body,
        ticket_id=headers.get("ticket_id") or None,
        created_at=headers.get("created_at") or None,
    )


# =============================================================================
# Step 1 — validation gates (skill 1.3.1). Cheap, before any FM call.
# =============================================================================
def validate_channel(channel: str) -> str:
    """The channel must be one of the Ticket literals. Reject anything else."""
    if channel not in ("email", "chat"):
        raise IntakeRejected(
            f"Unknown or missing channel {channel!r}. Expected 'email' or 'chat' "
            "in the header (e.g. 'channel: email').",
            reason="bad_channel",
        )
    return channel


def validate_raw_bytes(raw: bytes) -> str:
    """Decode the raw body bytes as UTF-8 and enforce the size ceiling.

    Two gates in one: a binary blob mislabelled as text fails to decode (rejected),
    and an oversized paste is rejected BEFORE we spend a token on it. We check the
    byte length (not character count) because that is what bills and what storage
    sees.
    """
    if len(raw) > config.MAX_MESSAGE_BYTES:
        raise IntakeRejected(
            f"Message is {len(raw)} bytes, over the "
            f"{config.MAX_MESSAGE_BYTES}-byte limit. A support message this large is "
            "almost always a pasted log or runaway thread — trim it or attach the "
            "log as a file.",
            reason="message_too_large",
        )
    try:
        return raw.decode(config.MESSAGE_ENCODING)
    except UnicodeDecodeError as err:
        raise IntakeRejected(
            f"Message is not valid {config.MESSAGE_ENCODING} text "
            f"({err}). A binary file is not a ticket body — attach it instead.",
            reason="bad_encoding",
        ) from err


def validate_nonempty(message: str) -> str:
    """After normalization the message must carry actual content. Reject if empty.

    A ticket that is nothing but a signature/quoted thread (everything stripped) has
    no question to answer — rejecting it here keeps an empty FM call from happening.
    """
    if not message.strip():
        raise IntakeRejected(
            "Message is empty after normalization (only a signature or quoted "
            "thread, no actual content). Nothing to triage.",
            reason="empty_message",
        )
    return message


def validate_attachment(filename: str, data: bytes) -> str:
    """Validate one attachment's type + size. Returns the admitted media_type.

    Skills 1.3.2/1.3.3: Relay reads error SCREENSHOTS, so only the image formats
    Converse can read are admitted (config.ADMITTED_ATTACHMENT_MEDIA_TYPES). A PDF,
    a zip, or an over-size image is rejected at the gate — before any upload or FM
    call. (PDF/document OCR is the Textract/BDA path the article covers, not built
    here.)
    """
    media_type = config.media_type_for_filename(filename)
    if media_type is None or media_type not in config.ADMITTED_ATTACHMENT_MEDIA_TYPES:
        raise IntakeRejected(
            f"Attachment {filename!r} is not an admitted image type. Relay reads "
            f"screenshots; admitted types are "
            f"{', '.join(config.ADMITTED_ATTACHMENT_MEDIA_TYPES)}. (A PDF or document "
            "goes through Textract / Bedrock Data Automation, not this intake.)",
            reason="bad_attachment_type",
        )
    if len(data) > config.MAX_ATTACHMENT_BYTES:
        raise IntakeRejected(
            f"Attachment {filename!r} is {len(data)} bytes, over the "
            f"{config.MAX_ATTACHMENT_BYTES}-byte limit.",
            reason="attachment_too_large",
        )
    if len(data) == 0:
        raise IntakeRejected(
            f"Attachment {filename!r} is empty.",
            reason="attachment_empty",
        )
    return media_type


# =============================================================================
# Step 2 — normalization (skill 1.3.4). Strip the noise; keep the signal.
# =============================================================================
# Lines at or below which everything is a signature/footer. Simple, explicit rules
# (the brief provides simple rules; we keep them readable, not a magic regex zoo).
_SIGNATURE_DELIMITERS = ("-- ", "--", "__")
_SIGNATURE_CUES = re.compile(
    r"^(thanks|thank you|regards|best|cheers|sincerely|sent from my )",
    re.IGNORECASE,
)
# A quoted-reply line (email thread): "> ...", or an "On <date>, <who> wrote:" lead-in.
_QUOTED_LINE = re.compile(r"^\s*>")
_REPLY_LEADIN = re.compile(r"^\s*On .+ wrote:\s*$", re.IGNORECASE)
# Minimal HTML unwrap: drop tags, decode the few entities a support email carries.
_HTML_TAG = re.compile(r"<[^>]+>")
_HTML_ENTITIES = {"&amp;": "&", "&lt;": "<", "&gt;": ">", "&nbsp;": " ", "&quot;": '"'}


def _strip_html(text: str) -> str:
    """Unwrap simple HTML: drop tags, decode the common entities. Not a full parser
    (a support email is not a web page) — just enough to recover the readable text."""
    if "<" not in text and "&" not in text:
        return text
    text = _HTML_TAG.sub("", text)
    for entity, char in _HTML_ENTITIES.items():
        text = text.replace(entity, char)
    return text


def _drop_quoted_thread(lines: list[str]) -> list[str]:
    """Cut the email reply thread: from the first quoted block / reply lead-in on.

    Everything from the first ">"-quoted line or "On ... wrote:" lead-in down is the
    previous conversation, not this message. We keep only what precedes it.
    """
    for i, line in enumerate(lines):
        if _QUOTED_LINE.match(line) or _REPLY_LEADIN.match(line):
            return lines[:i]
    return lines


def _drop_signature(lines: list[str]) -> list[str]:
    """Cut the closing + signature/footer block off the end.

    By convention the closing sign-off ("Thanks,", "Regards," "Sent from my phone")
    and an explicit delimiter line ("-- ") both mark where the message ends and the
    signature begins. We cut at the EARLIEST of the two so the name, title, phone,
    address and legal footer below it all go — keeping only the customer's message.
    A sign-off in the FIRST third of the message is ignored (it is mid-sentence, not
    a closing) so we never amputate a real short message.
    """
    cut = None
    floor = max(1, len(lines) // 3)
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped in _SIGNATURE_DELIMITERS or (
            i >= floor and _SIGNATURE_CUES.match(stripped)
        ):
            cut = i
            break
    return lines[:cut] if cut is not None else lines


def normalize(body: str) -> str:
    """Strip signature + quoted thread + HTML, collapse whitespace. Skill 1.3.4.

    The order matters: unwrap HTML first (so a quoted thread inside <div>s is
    visible), then drop the quoted thread, then the signature, then squeeze blank
    runs. The result is the customer's actual message — far fewer tokens, far more
    signal for the classifier. (Module 10 will redact PII on THIS normalized text,
    before Comprehend and the vision call. Module 6 does not.)
    """
    text = _strip_html(body)
    lines = text.splitlines()
    lines = _drop_quoted_thread(lines)
    lines = _drop_signature(lines)
    cleaned = "\n".join(lines)
    # Collapse 3+ blank lines to one, strip trailing spaces, trim ends.
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


# =============================================================================
# Step 3 — entity extraction with Amazon Comprehend (skill 1.3.4).
# =============================================================================
# The Comprehend entity types we surface for a support ticket. Comprehend returns
# many types; these are the ones that matter for CloudCart triage/lookup.
_USEFUL_ENTITY_TYPES = ("QUANTITY", "DATE", "COMMERCIAL_ITEM", "ORGANIZATION", "OTHER")


@dataclass
class Entities:
    """The salient entities Comprehend found, grouped by type (for logging + enrich)."""

    by_type: dict[str, list[str]] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not any(self.by_type.values())

    def as_line(self) -> str:
        """One compact line for the [Entities] enrichment block."""
        parts = []
        for etype in _USEFUL_ENTITY_TYPES:
            values = self.by_type.get(etype)
            if values:
                parts.append(f"{etype.lower()}: {', '.join(values)}")
        return "; ".join(parts)


def _comprehend_client():
    return boto3.client("comprehend", region_name=REGION)


def detect_entities(text: str, *, client=None) -> Entities:
    """Run Comprehend detect_entities and group the useful types. Skill 1.3.4.

    Order numbers and amounts surface as QUANTITY/OTHER, products as COMMERCIAL_ITEM,
    dates as DATE — the facts a human agent scans for first. We de-duplicate values
    and keep types stable so the [Entities] line is deterministic. On a Comprehend
    client error we raise IntakeError with the real cause (never a silent empty).
    """
    client = client or _comprehend_client()
    try:
        response = client.detect_entities(
            Text=text, LanguageCode=config.COMPREHEND_LANGUAGE_CODE
        )
    except ClientError as err:
        raise IntakeError(
            "Amazon Comprehend detect_entities failed: "
            f"{err.response['Error']['Code']} — {err.response['Error']['Message']}"
        ) from err

    grouped: dict[str, list[str]] = {}
    for ent in response.get("Entities", []):
        etype = ent.get("Type", "")
        if etype not in _USEFUL_ENTITY_TYPES:
            continue
        value = (ent.get("Text") or "").strip()
        if not value:
            continue
        bucket = grouped.setdefault(etype, [])
        if value not in bucket:
            bucket.append(value)
    return Entities(by_type=grouped)


# =============================================================================
# Step 4 — the attachment: upload to attachments/, then read it with Nova Lite.
# =============================================================================
def _s3_client():
    return boto3.client("s3", region_name=REGION)


def upload_attachment(
    data: bytes,
    filename: str,
    media_type: str,
    *,
    account: str | None = None,
    s3_client=None,
) -> Attachment:
    """Upload one validated attachment to attachments/ and return an Attachment.

    The key is attachments/<uuid>-<filename> so two customers' "screenshot.png" do
    not collide. The bucket name is the frozen relay-<account_id> (config), the
    account resolved from STS when not supplied — never hard-coded. Returns the
    frozen Attachment(filename, media_type, s3_uri).
    """
    s3_client = s3_client or _s3_client()
    acct = account or config.account_id()
    bucket = config.relay_bucket(acct)
    key = f"{config.RELAY_ATTACHMENTS_PREFIX}{uuid.uuid4().hex}-{filename}"
    try:
        s3_client.put_object(
            Bucket=bucket, Key=key, Body=data, ContentType=media_type
        )
    except ClientError as err:
        raise IntakeError(
            f"Failed to upload attachment to s3://{bucket}/{key}: "
            f"{err.response['Error']['Code']} — {err.response['Error']['Message']}"
        ) from err
    return Attachment(
        filename=filename,
        media_type=media_type,
        s3_uri=f"s3://{bucket}/{key}",
    )


def read_screenshot(data: bytes, media_type: str) -> str:
    """Read an error screenshot with Amazon Nova Lite (vision tier). Skill 1.3.2.

    Builds ONE multimodal Converse message — a text instruction block AND an image
    content block (via relay.llm.image_block) — and sends it through the single
    Bedrock call site relay.llm.converse(tier="vision"). Returns a short, structured
    description of the visible error. The image is passed as RAW bytes the Converse
    way; there is no single-prompt invoke path and no model-specific base64 payload here.
    """
    message = {
        "role": "user",
        "content": [
            {"text": _VISION_PROMPT},
            llm.image_block(data, media_type),
        ],
    }
    result = llm.converse(
        [message], tier=config.VISION_TIER, inferenceConfig=_VISION_INFERENCE
    )
    return result.text.strip()


# =============================================================================
# The pipeline — validate -> normalize -> entities -> attachment -> Ticket.
# =============================================================================
@dataclass
class IntakeResult:
    """The output of intake(): the clean Ticket plus what each step found (for logs)."""

    ticket: Ticket
    entities: Entities
    attachment_summary: str | None = None


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def intake(
    raw: RawIntake,
    *,
    attachment_bytes: bytes | None = None,
    attachment_filename: str | None = None,
    account: str | None = None,
    comprehend_client=None,
    s3_client=None,
    run_vision: bool = True,
) -> IntakeResult:
    """Turn a raw payload into a validated, normalized Ticket. The whole pipeline.

    Steps run in the order PII redaction (Module 10) will need: validate -> normalize
    -> [Module 10 redaction slots in here] -> Comprehend entities -> screenshot read.
    Each FM/AWS-touching step happens only AFTER the cheap validation gates pass.

    Args:
        raw: the parsed raw payload (channel + body + optional header ids).
        attachment_bytes / attachment_filename: an optional screenshot to validate,
            upload, and read. Both or neither.
        account: AWS account id for the bucket name (STS-resolved when omitted).
        run_vision: read the screenshot with Nova Lite (True) or skip the FM call and
            still upload + record the Attachment (False — used by offline tests).

    Returns an IntakeResult(ticket, entities, attachment_summary). Raises
    IntakeRejected on a validation failure (CLI -> exit 1) and IntakeError on an AWS
    failure — never a silent pass.
    """
    # --- 1. Validate the channel + the raw body (cheap gates, no FM/AWS yet) ---
    channel = validate_channel(raw.channel)
    raw_text = validate_raw_bytes(raw.body.encode(config.MESSAGE_ENCODING))

    # --- 2. Normalize (strip signature/thread/HTML), then the non-empty gate ----
    normalized = normalize(raw_text)
    normalized = validate_nonempty(normalized)
    # (Module 10 redaction would run HERE, on `normalized`, before anything below.)

    # --- 3. Comprehend entities, appended as a structured [Entities] line --------
    entities = detect_entities(normalized, client=comprehend_client)
    message = normalized
    if not entities.is_empty():
        message = f"{normalized}\n\n{ENTITIES_HEADER}\n{entities.as_line()}"

    # --- 4. Attachment: validate -> upload -> vision read -> [Attachment summary]
    attachments: list[Attachment] = []
    summary: str | None = None
    if attachment_bytes is not None:
        if not attachment_filename:
            raise IntakeRejected(
                "An attachment was provided without a filename.",
                reason="attachment_no_name",
            )
        media_type = validate_attachment(attachment_filename, attachment_bytes)
        att = upload_attachment(
            attachment_bytes, attachment_filename, media_type,
            account=account, s3_client=s3_client,
        )
        attachments.append(att)
        if run_vision:
            summary = read_screenshot(attachment_bytes, media_type)
            if summary:
                message = (
                    f"{message}\n\n{ATTACHMENT_SUMMARY_HEADER}\n{summary}"
                )

    ticket = Ticket(
        ticket_id=raw.ticket_id or f"intake-{uuid.uuid4().hex[:8]}",
        channel=channel,
        customer_message=message,
        attachments=attachments,
        created_at=raw.created_at or _now_iso(),
    )
    return IntakeResult(ticket=ticket, entities=entities, attachment_summary=summary)


def intake_file(
    raw_path: str | Path,
    *,
    attachment_path: str | Path | None = None,
    **kwargs,
) -> IntakeResult:
    """Convenience wrapper: read a raw fixture (+ optional attachment) from disk."""
    raw = parse_raw(Path(raw_path).read_text(encoding="utf-8"))
    attachment_bytes = attachment_filename = None
    if attachment_path is not None:
        p = Path(attachment_path)
        attachment_bytes = p.read_bytes()
        attachment_filename = p.name
    return intake(
        raw,
        attachment_bytes=attachment_bytes,
        attachment_filename=attachment_filename,
        **kwargs,
    )


# =============================================================================
# CLI — print the validated Ticket JSON; on rejection, print + exit 1.
# =============================================================================
def _print_ticket(result: IntakeResult) -> None:
    print(result.ticket.model_dump_json(indent=2))
    if not result.entities.is_empty():
        print(f"\n{ENTITIES_HEADER} {result.entities.as_line()}", file=sys.stderr)
    for att in result.ticket.attachments:
        print(f"attachment -> {att.s3_uri} ({att.media_type})", file=sys.stderr)


def _run_triage(result: IntakeResult) -> int:
    """Optionally run the Module 2 triage on the validated Ticket (end-to-end demo)."""
    from relay import triage as triage_mod

    try:
        classification, usage = triage_mod.triage(result.ticket)
    except (triage_mod.TriageError, llm.LLMError, ClientError) as err:
        print(f"Triage failed: {err}", file=sys.stderr)
        return 1
    print("\n--- triage on the validated ticket ---", file=sys.stderr)
    print(classification.model_dump_json(), file=sys.stderr)
    cost = config.estimate_cost("fast", usage["inputTokens"], usage["outputTokens"])
    print(f"triage tokens: in={usage['inputTokens']} out={usage['outputTokens']} "
          f"| est. cost: ${cost:.6f}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m relay.intake",
        description="Validate + normalize a raw email/chat (± screenshot) into a "
                    "clean Ticket.",
    )
    parser.add_argument("raw", help="path to a raw intake .txt (data/raw/...)")
    parser.add_argument("--attachment", help="path to a screenshot to read", default=None)
    parser.add_argument("--triage", action="store_true",
                        help="also run the Module 2 triage on the validated ticket")
    parser.add_argument("--no-vision", action="store_true",
                        help="upload + record the attachment but skip the FM vision read")
    args = parser.parse_args(argv)

    try:
        result = intake_file(
            args.raw,
            attachment_path=args.attachment,
            run_vision=not args.no_vision,
        )
    except IntakeRejected as err:
        print(f"REJECTED ({err.reason}): {err}", file=sys.stderr)
        return 1
    except IntakeError as err:
        print(f"Intake failed: {err}", file=sys.stderr)
        return 1
    except (NoCredentialsError, ProfileNotFound, BotoCoreError) as err:
        print(f"AWS credentials/config problem: {err}\n"
              "Set AWS_PROFILE=aws-genai-pro and run from us-east-1.",
              file=sys.stderr)
        return 1

    _print_ticket(result)
    if args.triage:
        return _run_triage(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
