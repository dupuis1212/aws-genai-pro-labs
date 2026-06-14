# Module 6 lab — data pipelines for FMs: validation, multimodal input, document processing

> **This lab cost me $0.01 on June 2026 prices** (measured end to end on 13 June
> 2026, account in us-east-1 — well under the syllabus budget of < $1 for Module 6).
> Every token figure below is read from the API response, never guessed. The spend
> is a handful of **Amazon Nova Lite** vision reads + **Amazon Comprehend** entity
> calls + one Module 5 KB answer + a few **Amazon S3** uploads. Measured breakdown:
>
> | Item | Real usage observed | Cost |
> |---|---|---|
> | Nova Lite **vision** reads (×4) | 808 in / 38 out tokens each | $0.00023 |
> | **triage** (fast tier, ×3 fixtures) | ~1,360 in / ~22 out each | $0.00016 |
> | live smoke `converse` (fast + smart) | ≤64 out tokens each | $0.00018 |
> | live smoke Titan embeddings (×2) | ~40 tokens | ~$0 |
> | live smoke **KB** RetrieveAndGenerate | ~1,500 in / ~200 out | $0.00095 |
> | **Comprehend** `detect_entities` (~7 calls) | ~5 units (100 chars) each | $0.0035 |
> | setup KB ingestion (×2, Titan embeddings) | 7 small docs | $0.0003 |
> | S3 uploads + storage | 2 small PNGs | <$0.001 |
> | **Total** | | **≈ $0.006 → $0.01** |
>
> The single biggest line is Comprehend (its 3-unit / 300-character minimum per
> request dominates), then the one KB answer. The vision reads — the headline
> Module 6 increment — are the cheapest part of the run.
>
> - **Vision** — each screenshot read is one **Converse** call on **Amazon Nova
>   Lite** (`us.amazon.nova-lite-v1:0`, ~$0.06 in / ~$0.24 out per million tokens, AS
>   OF JUNE 2026). The measured `payment_error.png` read was **808 input + 38 output
>   tokens** — $0.000058 per read, a small fraction of a cent.
> - **Comprehend** — `detect_entities` is billed per **unit of 100 characters**
>   (minimum 3 units / 300 characters per request), ~$0.0001 per unit (AS OF JUNE
>   2026). A normalized support message is a few units: **fractions of a cent** each.
> - **S3** — uploading a few small PNGs to `attachments/` and storing them is
>   **fractions of a cent**, and the storage bills **~$0 idle**.
> - **No idle billing.** Comprehend and Converse are **per-call** services — they
>   create nothing that bills while idle. The only standing resource Module 6 adds is
>   the `attachments/` prefix (S3 storage, ~$0 idle), and teardown purges the uploads.
>
> Prices are **as of June 2026** — re-verify on the
> [Bedrock pricing page](https://aws.amazon.com/bedrock/pricing/), the
> [Comprehend pricing page](https://aws.amazon.com/comprehend/pricing/), and the
> [S3 pricing page](https://aws.amazon.com/s3/pricing/) before you run it.
>
> **Teardown reminder:** run `uv run python teardown.py` when you're done. It
> **purges every object the intake uploaded under `attachments/`** and **keeps** the
> data bucket, the `docs/` corpus, and the Module 5 Knowledge Base (downstream
> modules reuse them). Comprehend and the Nova Lite vision read are per-call, so
> there is nothing else to delete. The M1 $5 budget stays; it backstops the course.

**Goal:** build `relay/intake.py` — raw email/chat (± screenshot) → a **validated,
normalized `Ticket`** (the multimodal model reads the screenshot), with **clean
rejections** for invalid input. Because a foundation model fed dirty input produces
dirty output, no matter the prompt.

Region for the whole course: **us-east-1**. Profile: `AWS_PROFILE=aws-genai-pro` (any
profile that resolves to your course account works). No AWS key in code or `.env`.

---

## Step 1 — Carry the cumulative state forward

Module 6 starts from Module 5's `relay/` package byte-for-byte (`models.py`,
`config.py`, `llm.py`, `triage.py`, `kb.py`) plus the inherited `ingest/` pipeline,
the data bucket `relay-<account_id>` (with `docs/` populated), and the Module 5
Knowledge Base. The intake pipeline inserts **upstream** of all of it.

```bash
uv sync   # boto3 + pydantic; no new runtime dependency for Module 6
aws sts get-caller-identity   # the account the bucket name is suffixed with
```

---

## Step 2 — The frozen contract: `Attachment` + `Ticket.attachments`

`relay/models.py` gains, **by addition only** (06 §2 / bible §3.1):

```python
class Attachment(BaseModel):     # frozen M6 — exactly 3 fields
    filename: str
    media_type: str
    s3_uri: str

class Ticket(BaseModel):
    ticket_id: str
    channel: Literal["email", "chat"]
    customer_message: str
    attachments: list[Attachment] = []   # ADDED M6 (default [] — back-compat)
    created_at: str
```

The default `[]` is load-bearing: **every Module 2–5 ticket fixture still validates**
(none has an `attachments` key). There is **no `pii_redacted`** field — that is Module
10. The intake **normalizes** the message but **redacts nothing**.

---

## Step 3 — Validate before you generate (skill 1.3.1)

Validation lives **before** the FM, for two reasons the exam tests:

- **Cost.** Every Converse / Comprehend call on garbage is billed. A 16 KiB pasted
  log triaged "just in case" is real money at scale.
- **Integrity.** An invalid input that is *not* rejected becomes a silent error
  downstream. A foundation model is **not** an input validator; the pipeline before
  it is.

`relay/intake.py` gates the input and raises a **structured** `IntakeRejected` (a
business error with a machine-readable `reason`), which the CLI prints and exits 1
on — **no silent `try/except`**:

| Gate | Rejects | `reason` |
|---|---|---|
| UTF-8 decode | a binary blob mislabelled as text | `bad_encoding` |
| size ceiling (16 KiB) | a runaway log/thread paste | `message_too_large` |
| non-empty after normalize | a body that is all signature/thread | `empty_message` |
| admitted image type | a PDF / zip / .exe attachment | `bad_attachment_type` |
| attachment size (4 MB) | an over-size image | `attachment_too_large` |
| channel | anything but `email` / `chat` | `bad_channel` |

```bash
uv run python -m relay.intake data/raw/invalid_oversized.txt
# REJECTED (message_too_large): Message is 43889 bytes, over the 16384-byte limit...
echo $?   # 1

uv run python -m relay.intake data/raw/invalid_empty.txt
# REJECTED (empty_message): Message is empty after normalization...
echo $?   # 1
```

> **Validation ≠ security.** The integrity gates here (Module 6) and the **guardrails**
> for prompt injection / unsafe content (Domain 3, Module 9) are two distinct layers.
> A well-formed input can still be an attack; a malformed one is rejected before
> either runs. The two layers stack — this lab builds the first.
>
> The exam also places **Glue Data Quality** and **SageMaker Data Wrangler** here —
> but those validate **datasets** (a training/feature pipeline), not a **per-request**
> support ticket. Per-request integrity is custom code (Lambda-style), which is
> exactly what `relay/intake.py` is.

---

## Step 4 — Normalize, then extract entities (skill 1.3.4)

`normalize()` strips the **signature**, the **quoted reply thread**, and simple
**HTML**, then collapses whitespace — fewer tokens, more signal for the classifier.
The raw billing fixture is 40+ lines; after normalization it is the customer's actual
problem.

Then **Amazon Comprehend** `detect_entities` pulls the salient facts — order numbers,
amounts (`QUANTITY`), dates (`DATE`), products (`COMMERCIAL_ITEM`) — and the intake
appends them under an `[Entities]` line so triage and the agent get them up front.
Comprehend is a **managed NLP service**, not a Bedrock foundation model, and not a
guardrail.

The pipeline runs in the order **Module 10's PII redaction will need**:
`validate → normalize → [redaction slots in here] → Comprehend → vision`. Module 6
leaves that slot empty (it redacts nothing).

---

## Step 5 — Multimodal input: make Relay read the screenshot (skills 1.3.2 / 1.3.3)

The error in the billing ticket exists **only in the image** (`payment_error.png`: a
CloudCart checkout showing `ERR-402 — payment declined`). Relay reads it with **Amazon
Nova Lite (vision)** through the **Converse content-block** path:

- the attachment is validated (admitted image type, size), uploaded to
  `s3://relay-<account_id>/attachments/`, and recorded as an `Attachment`;
- `relay/llm.image_block(data, media_type)` builds the Converse image block
  `{"image": {"format": "png", "source": {"bytes": ...}}}` — **raw bytes**, the SDK
  base64-encodes for the wire (never a legacy single-prompt invoke payload, 07 §3.3);
- one **Converse message carries a `text` block AND an `image` block together**, sent
  through `relay.llm.converse(tier="vision")` — the model ID lives only in
  `relay/config.py` (the `vision` tier → `us.amazon.nova-lite-v1:0`, **Nova Lite**,
  which is **not** the `smart` tier's Nova 2 Lite);
- the short, structured extraction ("Error / Screen / User action") is appended under
  an `[Attachment summary]` separator. The `Ticket` **schema** is not changed for the
  summary — it rides inside `customer_message`, while the file metadata rides in
  `Ticket.attachments`.

```bash
uv run python -m relay.intake data/raw/email_billing_error.txt \
    --attachment data/raw/payment_error.png
# {
#   "ticket_id": "raw-email-001",
#   "channel": "email",
#   "customer_message": "Hi CloudCart support, ... \n\n[Entities]\nquantity: #1042 ...\n
#                        \n[Attachment summary]\nError: ERR-402 Payment declined\n
#                        Screen: CloudCart checkout\nUser action: use a different card...",
#   "attachments": [
#     { "filename": "payment_error.png", "media_type": "image/png",
#       "s3_uri": "s3://relay-<account_id>/attachments/<uuid>-payment_error.png" }
#   ],
#   "created_at": "2026-06-12T14:03:00Z"
# }
```

**Image-format constraints (skill 1.3.3):** Converse accepts `png / jpeg / gif / webp`
images via a content block; the intake admits exactly those. Image size and per-request
count are model-side limits — re-verify the current Nova Lite limits on the
[Amazon Nova docs](https://docs.aws.amazon.com/nova/latest/userguide/) before you push
large or many images. Relay reads **one screenshot per ticket** here.

**Audio is a survey, not a lab.** The exam path for audio is **Amazon Transcribe → text
→ FM**, not "audio straight to the model." Module 6 builds the image path only.

End to end — intake feeds the **Module 2 triage** on the clean ticket:

```bash
uv run python -m relay.intake data/raw/email_billing_error.txt \
    --attachment data/raw/payment_error.png --triage
# ... the validated Ticket JSON, then:
# --- triage on the validated ticket ---
# {"intent": "billing", "priority": "high", "sentiment": "negative"}
```

Compare that to triaging the **raw** email: the signature and quoted thread bury the
signal and the error is invisible (it was only in the image). The pipeline is the fix,
not a bigger prompt.

---

## Step 6 — Which service for which step?

The article's comparison table (read it for the full matrix). In one line each:

- **Amazon Comprehend** — targeted **entity extraction** from text (order #, dates,
  products). Use it to enrich a known-shape message.
- **Amazon Textract** — **OCR** of text and structure out of **documents** (scanned
  PDFs, forms, tables). It **extracts text**; it does **not** *understand* a screenshot.
- **FM multimodal (Nova Lite)** — **contextual understanding** of an image (what the
  error means, what the screen is). This is what reads `payment_error.png`.
- **Bedrock Data Automation (BDA)** — a **managed end-to-end document pipeline**
  (parse → extract → transform) for documents at scale. **Theory only** in this lab —
  no real BDA or Textract job runs (budget; re-verify BDA status/pricing as of June
  2026 before any chiffred claim).

The exam's trap: **Textract reads text in a document; it does not interpret a
screenshot** — that is the multimodal FM. Orchestrating these steps (Step Functions /
EventBridge) is Module 11.

---

## Step 7 — Offline tests, then teardown

```bash
uv run pytest                          # offline: no creds, no network (Modules 2–6)
RELAY_LIVE_TESTS=1 uv run pytest -m live  # up to 6 sub-cent real calls (budgeted)
uv run python teardown.py              # purge attachments/ uploads; KEEP bucket + KB
```

The offline tests cover the frozen `Attachment` / `Ticket.attachments` contract, the
validation gates (each typed rejection), normalization on the shipped fixtures, the
Comprehend / S3 / vision steps over stubs, the **full `intake()` pipeline** producing a
clean ticket with an `[Attachment summary]`, and `setup.py`/`teardown.py` idempotency.
The live marker makes at most **six** real calls total — the two Module 2/3 `converse`,
two Module 4 Titan embeddings, one Module 5 `RetrieveAndGenerate` (skips if the KB is
not set up), and **one Module 6 Nova Lite vision read** of the bundled screenshot.

Teardown purges the intake's `attachments/` uploads and **keeps** the data bucket, the
`docs/` corpus, and the Knowledge Base — downstream modules reuse them, and they bill
~$0 idle. Comprehend and Converse are per-call, so nothing else needs deleting.

---

## Try it yourself

1. **Add a language gate.** Use Comprehend `detect_dominant_language` on the normalized
   message and reject (or flag) a ticket that is not English with a clear
   `IntakeRejected`. Where in `intake()` does it belong — before or after
   `detect_entities`? (Hint: `detect_entities` takes a `LanguageCode`.)
2. **Stress the vision read.** Feed a busier screenshot — multiple windows, several
   error dialogs at once — and watch where the structured extraction degrades. Does the
   model still read the right error code? This is the precision/limit boundary of
   multimodal input: a clean, single-error screenshot is the reliable case.
