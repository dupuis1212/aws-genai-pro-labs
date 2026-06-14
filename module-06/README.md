# Module 6 — Data Pipelines for FMs: Validation, Multimodal Input, and Document Processing

**What:** Modules 2–5 always fed Relay **clean** tickets. The real world does not:
a CloudCart support email arrives with a 40-line quoted thread, a signature block,
broken HTML, "see attached", and a **screenshot** of the actual error — which exists
only in the image. Pass that raw to the Module 2 triage and the classifier drowns in
the signature and never sees the error. Garbage in, garbage out.

Module 6 gives Relay an **intake pipeline** — `relay/intake.py`, upstream of
everything — that turns a raw email/chat (± screenshot) into a **validated,
normalized `Ticket`**, with **clean rejections** for bad input:

1. **Validate before you generate** (skill 1.3.1) — gates that cost nothing run
   *before* any FM call: UTF-8 decoding, a size ceiling, a non-empty body, and an
   admitted-image-type check for attachments. A failure raises a structured
   `IntakeRejected` the CLI prints and exits 1 on. No silent `try/except`.
2. **Normalize** (skill 1.3.4) — strip the signature, the quoted thread, and simple
   HTML. Fewer tokens, more signal.
3. **Entity extraction** (skill 1.3.4) — **Amazon Comprehend** `detect_entities`
   pulls order numbers, amounts, and dates and appends them under `[Entities]`.
4. **Multimodal input** (skills 1.3.2 / 1.3.3) — an accepted screenshot is uploaded
   to `s3://relay-<account_id>/attachments/` and **read by Amazon Nova Lite (vision)**
   through a `Converse` message that carries a **text block and an image content
   block together**. The short extraction is appended under `[Attachment summary]`.

This module **freezes** one schema and extends `Ticket` by **addition only**
(06 §2 / bible §3.1), reproduced field-for-field:

```python
class Attachment(BaseModel):     # frozen M6
    filename: str
    media_type: str
    s3_uri: str

class Ticket(BaseModel):         # M2's 4 fields + ONE added in M6
    ticket_id: str
    channel: Literal["email", "chat"]
    customer_message: str
    attachments: list[Attachment] = []   # ADDED M6 (default [] — back-compat)
    created_at: str
```

`Triage`/`Citation`/`Answer` are **untouched**. There is **no `pii_redacted`** field
(that is Module 10 — the intake normalizes but **redacts nothing** here), **no
guardrail / injection check** (Module 9), **no agent or tools** (Module 7), **no
SQS/API** (Module 11), and **no real Bedrock Data Automation / Textract job** (theory
only, in the article). The image goes through the **Converse content-block** path in
`relay/llm.py`, never a legacy single-prompt invoke payload.

**How to run** (region us-east-1, profile `AWS_PROFILE=aws-genai-pro`; no AWS key in
code or `.env`):

```bash
uv sync

# 0. Module 6 reuses Module 4's data bucket relay-<account_id>. setup.py ensures the
#    attachments/ prefix exists (and keeps the inherited Module 5 Knowledge Base).
uv run python setup.py

# 1. Intake a raw email WITH a screenshot -> a validated, normalized Ticket JSON:
#    normalized message, [Entities], [Attachment summary] from the vision read, and
#    attachments[] with an s3_uri.
uv run python -m relay.intake data/raw/email_billing_error.txt \
    --attachment data/raw/payment_error.png

# 2. Intake an INVALID raw input -> structured rejection, exit 1:
uv run python -m relay.intake data/raw/invalid_oversized.txt   # message_too_large
uv run python -m relay.intake data/raw/invalid_empty.txt       # empty_message

# 3. End to end: intake THEN run the Module 2 triage on the clean ticket:
uv run python -m relay.intake data/raw/email_billing_error.txt \
    --attachment data/raw/payment_error.png --triage

# 4. Offline tests (no credentials, no network) — the Attachment/Ticket contract,
#    the validation gates, normalization, Comprehend/S3/vision over stubs, the full
#    intake() pipeline, and setup/teardown idempotency. Cumulative over Modules 2–6.
uv run pytest

# 5. Up to six sub-cent real calls (budgeted), incl. ONE real Nova Lite vision read:
RELAY_LIVE_TESTS=1 uv run pytest -m live

# 6. Purge the intake's attachments/ uploads (keeps the bucket, docs/, and the KB).
uv run python teardown.py
```

Full step-by-step walkthrough — the validation gates, the content-block anatomy, the
multimodal screenshot read, the "Which service for which step?" comparison
(Comprehend / Textract / FM multimodal / Bedrock Data Automation), and the two
"Try it yourself" exercises — is in [`lab.md`](lab.md).

Files (NEW or MODIFIED in Module 6):

- `relay/intake.py` — **NEW.** The intake pipeline: `validate_*` gates → `normalize`
  → Comprehend `detect_entities` → `upload_attachment` → `read_screenshot` (Nova Lite
  vision via `converse(tier="vision")`) → a validated `Ticket`. Structured
  `IntakeRejected` / `IntakeError`, no silent passes. The steps are ordered so Module
  10's PII redaction slots in *before* Comprehend and the vision call. CLI:
  `python -m relay.intake <raw.txt> [--attachment <png>] [--triage] [--no-vision]`.
- `data/raw/` — **NEW.** Five raw fixtures (3 valid incl. one with a screenshot, 2
  invalid) + `payment_error.png` (a real PNG of a CloudCart `ERR-402` checkout error).
- `relay/models.py` — **MODIFIED (additive).** Adds `Attachment` and
  `Ticket.attachments: list[Attachment] = []`. The four M2 fields are untouched; no
  `pii_redacted`.
- `relay/llm.py` — **MODIFIED (additive).** Adds `image_block(data, media_type)` (the
  Converse image content block) and the admitted-format map. The frozen
  `converse(messages, *, tier="auto", stream=False, **params)` **signature is
  unchanged** — image blocks ride inside `messages`.
- `relay/config.py` — **MODIFIED (additive).** **Appends** the `vision` tier
  (`us.amazon.nova-lite-v1:0` — Nova Lite, *not* Nova 2 Lite) and the intake policy
  (attachments prefix, size/encoding/type gates, Comprehend language). The M3
  fast/smart/frontier entries and the M4 embedder are untouched.
- `relay/__init__.py` — **MODIFIED (additive).** Tracks the new `intake` submodule.
- `setup.py` / `teardown.py` — **MODIFIED (additive).** setup ensures the
  `attachments/` prefix; teardown purges it (keeps the bucket, docs/, and the KB).
- `relay/kb.py`, `relay/triage.py`, `ingest/`, `prompts/`, `data/tickets/`,
  `data/docs/`, `compare_chunking.py`, `compare_retrieval.py`, `freshness_test.py` —
  **inherited from Module 5, byte-identical.**
- `tests/smoke_test.py` — offline by default (cumulative Modules 2–6); live calls
  opt-in (`RELAY_LIVE_TESTS=1`) with a documented budget (≤6 calls; one is a real
  Nova Lite vision read of the bundled screenshot).
