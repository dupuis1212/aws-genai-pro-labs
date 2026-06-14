"""relay/config.py — the SOLE home of model-ID literals in the whole repo.

Module 3 of AWS GenAI Pro Mastery introduces this file and FREEZES the model-ID
containment law: every Amazon Bedrock model ID Relay uses lives here, mapped from
a Relay *tier* to an **inference profile** ID. Nowhere else in `relay/` may a
`us.`/`global.` profile ID appear — `relay/llm.py` and every downstream caller go
through `tier_profile()` / `TIERS`, never a hard-coded string. The grep gate proves
it:

    grep -rE '(us|global|eu)\\.(amazon|anthropic)\\.' relay/ | grep -v config.py
    # -> empty

Why inference profiles and not bare regional IDs? A recent model invoked on-demand
with a bare regional ID (e.g. `amazon.nova-micro-v1:0`) fails with "Retry with an
inference profile". The `us.`/`global.` prefix IS the **cross-Region inference**
profile: it routes requests across the Regions in the geography for capacity. That
is the NOMINAL mode for these models, not a disaster-recovery toggle.

This file grows ONLY BY ADDITION in later modules (bucket/index names at M4, the
Knowledge Base ID at M5, table names at M7, the guardrail ID at M9, a per-tier
price map and Flex profile at M12, ...). **The tier map itself is never edited
after Module 3** — new tiers may be appended, existing ones never re-pointed.

All IDs below are the live-verified ACTIVE inference profiles for us-east-1
(account checked 13 June 2026). Prices are "as of June 2026" — re-verify on the
Bedrock pricing page; they drive the cost line in `demo_llm.py`, never a decision
the code makes silently.
"""

from __future__ import annotations

from typing import Literal

# Region for every Bedrock call (course decision B8: us-east-1 everywhere).
REGION = "us-east-1"

# --- Tier -> inference profile map (THE single source of model IDs) ----------
# Canonical tiers, no synonyms (06 §2 / bible §3.2):
#   "fast"  : triage, the router's own classifier, tests       -> Nova Micro
#   "smart" : complex answers, agent reasoning                  -> Nova 2 Lite
#   "frontier" : reference-grade only (Module 3 Table 1 + the
#                "Try it yourself" cost-delta exercise)         -> Claude Sonnet
#   "vision" : multimodal screenshot read (Module 6 intake)     -> Nova Lite
#
# "auto" is NOT a profile — it is the router's request to PICK a tier at call
# time (see relay/llm.py route()); it never appears as a key here.
#
# Nova Micro / Nova 2 Lite via their `us.` profiles; both are reachable. Nova 2
# Lite also has a `global.` profile (wider Region pool) — recorded but the `us.`
# profile is the default so the cost line and Region story stay predictable.
#
# MODULE 6 ADDITION (by addition, never a re-point): the "vision" tier is APPENDED
# below. The M3 entries (fast/smart/frontier) are byte-identical and untouched —
# the model-ID containment law requires the multimodal model ID to live HERE and
# nowhere else, so relay.intake resolves vision by TIER, never by a bare ID. Note
# the ID: `us.amazon.nova-lite-v1:0` is **Amazon Nova Lite** — the multimodal
# (vision) model — which is NOT the same as the "smart" tier's `nova-2-lite`
# (Nova 2 Lite). Easy to confuse; they are different models.
#
# MODULE 13 ADDITION (by addition, never a re-point): the "judge" tier is APPENDED
# below — the LLM-as-a-judge runs through the SAME single converse() call site (the
# model-ID containment law: the judge's Claude Haiku 4.5 ID lives HERE and nowhere
# else, so evals/judge.py calls converse(tier="judge"), never a bare ID). The judge
# is NOT a Relay CANDIDATE: it never answers a customer; it only SCORES answers the
# candidate tiers produced. Its ID is a different model FAMILY from every candidate
# (Anthropic Claude vs Amazon Nova) — self-preference bias designed out (brief §9).
# JUDGE_PROFILE (below) is the canonical accessor; this map entry keeps it on the one
# call site. The M3 entries (fast/smart/frontier) and the M6 vision entry are
# byte-identical and untouched — nothing is re-pointed.
Tier = Literal["fast", "smart", "frontier", "vision", "judge"]

TIERS: dict[str, str] = {
    "fast": "us.amazon.nova-micro-v1:0",
    "smart": "us.amazon.nova-2-lite-v1:0",
    # Reference / Try-it-yourself only. A frontier model is overkill for support
    # triage and answers; it lives here so the cost-delta exercise has a real ID,
    # not so production traffic routes to it.
    "frontier": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    # Module 6: the multimodal/vision tier. Amazon Nova Lite reads the screenshot a
    # customer attaches; relay.intake calls converse(tier="vision", ...). Nova Lite
    # (vision) != Nova 2 Lite (the "smart" tier above).
    "vision": "us.amazon.nova-lite-v1:0",
    # Module 13: the LLM-as-a-judge tier. Anthropic Claude Haiku 4.5 — a DIFFERENT
    # model family from every candidate (self-preference bias designed out). The judge
    # never serves a customer; evals/judge.py calls converse(tier="judge", service_tier=
    # config.JUDGE_SERVICE_TIER). Kept in sync with JUDGE_PROFILE by the assert below.
    "judge": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
}

# Alternate (wider-pool) profiles, documented but not the default. Cross-Region
# inference already routes the `us.` profiles across us-east/us-west; the
# `global.` profile widens the pool further for the smart tier.
ALT_PROFILES: dict[str, str] = {
    "smart": "global.amazon.nova-2-lite-v1:0",
    # Module 13: the judge's wider-pool alternate (cross-Region inference for capacity
    # on the eval/backfill path).
    "judge": "global.anthropic.claude-haiku-4-5-20251001-v1:0",
}

# Default tier for converse(tier="auto") when the router cannot find a reason to
# escalate. Fast is the floor: ~80% of CloudCart tickets are fast-tier work.
DEFAULT_TIER: str = "fast"


def tier_profile(tier: str) -> str:
    """Resolve a Relay tier to its inference profile ID — the only lookup allowed.

    Raises a clear error (never a silent fallback) for an unknown tier, so a typo
    surfaces immediately instead of routing to the wrong model.
    """
    try:
        return TIERS[tier]
    except KeyError:
        raise ValueError(
            f"Unknown tier {tier!r}. Known tiers: {', '.join(sorted(TIERS))}. "
            "Use tier='auto' to let the router choose between 'fast' and 'smart'."
        ) from None


# --- Pricing per tier (AS OF JUNE 2026 — re-verify; drives the cost line) -----
# Per-1,000-token prices, derived from the published per-million figures so the
# cost line in demo_llm.py is computed from the API usage block, never guessed.
# These are for reporting only — no routing decision reads them.
PRICE_PER_1K: dict[str, dict[str, float]] = {
    # Nova Micro: $0.035 in / $0.14 out per million tokens.
    "fast": {"input": 0.000035, "output": 0.00014},
    # Nova 2 Lite: ~$0.30 in / ~$2.50 out per million tokens (verify).
    "smart": {"input": 0.00030, "output": 0.00250},
    # Claude Sonnet 4.5: published per-million pricing — re-verify the day you run
    # the frontier "Try it yourself"; figures here are placeholders for the delta.
    "frontier": {"input": 0.0030, "output": 0.0150},
    # Module 6: Amazon Nova Lite (vision). Image tokens are billed as input tokens
    # in the Converse usage block; ~$0.06 in / ~$0.24 out per million (AS OF JUNE
    # 2026 — re-verify on the Bedrock pricing page). Reporting only.
    "vision": {"input": 0.00006, "output": 0.00024},
}


def estimate_cost(tier: str, input_tokens: int, output_tokens: int) -> float:
    """Cost in USD for a call, from the API usage block — never a guess."""
    price = PRICE_PER_1K.get(tier, PRICE_PER_1K["fast"])
    return (
        input_tokens / 1000 * price["input"]
        + output_tokens / 1000 * price["output"]
    )


# =============================================================================
# Module 4 ADDITIONS — RAG resource names + the pinned embedder.
# =============================================================================
# Added BY ADDITION in Module 4 (RAG foundations). NOTHING above this line is
# touched: the tier -> inference-profile map is frozen at Module 3 and is never
# re-pointed. Module 4 only appends the resource names and the embeddings model
# the ingestion pipeline needs.
#
# Why here? The same containment law applies to resource identifiers as to model
# IDs: there is ONE place a bucket/index name or an embeddings model ID is
# written, so `setup.py`, `teardown.py`, `ingest/`, and `compare_chunking.py` all
# agree by construction. Downstream modules (the managed Knowledge Base at M5,
# the agent at M7) REFERENCE these names — they never recreate them.

# --- AWS resource names (canonical, frozen at Module 4 — 06 §2 / bible §3.3) --
# Both buckets are suffixed with the 12-digit AWS account ID so the names are
# globally unique without you inventing one. We resolve the account ID at call
# time from STS (never hard-coded) so the same code works in any account.
#
# Data bucket:   relay-<account_id>           (prefixes docs/ attachments/ vectors/)
# Vector bucket: relay-vectors-<account_id>   (S3 Vectors)
# Index:         relay-docs                   (the one index, 1024-dim, cosine)
RELAY_BUCKET_PREFIX = "relay-"
RELAY_VECTOR_BUCKET_PREFIX = "relay-vectors-"
RELAY_INDEX = "relay-docs"

# The three key prefixes inside the data bucket. `attachments/` is created empty
# now and filled by the multimodal intake at Module 6; `vectors/` is reserved for
# any raw vector export. Module 4 fills `docs/`.
RELAY_BUCKET_PREFIXES = ("docs/", "attachments/", "vectors/")


def relay_bucket(account_id: str) -> str:
    """The course data bucket name for an account: relay-<account_id>."""
    return f"{RELAY_BUCKET_PREFIX}{account_id}"


def relay_vector_bucket(account_id: str) -> str:
    """The S3 Vectors bucket name for an account: relay-vectors-<account_id>."""
    return f"{RELAY_VECTOR_BUCKET_PREFIX}{account_id}"


def account_id(sts_client=None) -> str:
    """Resolve the caller's 12-digit AWS account ID via STS (never hard-coded).

    Kept here so every script derives the bucket names the same way. Pass a
    client in tests; in normal use it builds one in REGION.
    """
    import boto3  # local import: keep module import side-effect free for offline tests

    sts_client = sts_client or boto3.client("sts", region_name=REGION)
    return sts_client.get_caller_identity()["Account"]


# --- Embeddings (the course's SOLE non-Converse Bedrock call) -----------------
# Amazon Titan Text Embeddings V2, PINNED at 1024 dimensions. The vector contract
# the index `relay-docs` is built on is exactly these 1024 dims — M5's managed
# Knowledge Base and M12's semantic cache reuse this very index, so the embedder
# is NEVER silently swapped. (`amazon.nova-2-multimodal-embeddings-v1:0` exists as
# of June 2026 and is evaluated in the article as a comparison row only; it stays
# OUT of this index because changing dimensions would invalidate it.)
#
# The embeddings call goes through the bedrock-runtime embeddings path (the Converse
# API cannot embed). That is the ONE such call the course tolerates — it lives in
# ingest/embed.py, returns a vector, never text. All GENERATION stays on converse()
# in llm.py. Note: Titan embeddings are invoked by their bare model ID (not an
# inference profile) — embeddings models are not on the inference-profile path.
EMBED_MODEL_ID = "amazon.titan-embed-text-v2:0"
EMBED_DIMENSIONS = 1024  # pinned — never change without rebuilding the index
EMBED_DISTANCE_METRIC = "cosine"  # cosine similarity over normalized Titan vectors

# Titan V2 batch pricing for the cost line (AS OF JUNE 2026 — re-verify on the
# Bedrock pricing page). $0.02 per million input tokens. Reporting only.
EMBED_PRICE_PER_1K_INPUT = 0.00002


def estimate_embed_cost(input_tokens: int) -> float:
    """Embeddings cost in USD from the Titan token count — never a guess."""
    return input_tokens / 1000 * EMBED_PRICE_PER_1K_INPUT


# =============================================================================
# Module 5 ADDITIONS — the managed Knowledge Base + the reranker.
# =============================================================================
# Added BY ADDITION in Module 5 (managed RAG). NOTHING above this line changes:
# the tier -> inference-profile map is frozen at Module 3, the resource names and
# the embedder are frozen at Module 4. Module 5 only appends the Knowledge Base
# name and the reranker model the retrieval pipeline needs — same containment law:
# there is ONE place a KB or reranker ID is written, so setup.py, teardown.py, and
# relay/kb.py all agree by construction.

# --- The managed Knowledge Base (canonical, 06 §2 / bible §3.3) ---------------
# Bedrock Knowledge Base name. Frozen at Module 5; the agent's KB-search tool
# (Module 7) is built on this very KB. It uses an S3 Vectors index in Module 4's
# vector bucket `relay-vectors-<account_id>` — NOT a new always-on serverless
# search collection (that would bill ~$174/month idle; S3 Vectors bills ~$0 idle).
RELAY_KB_NAME = "relay-kb"

# The S3 Vectors index the Knowledge Base OWNS and queries, in Module 4's vector
# bucket. It is a DEDICATED, KB-managed index — NOT Module 4's `relay-docs`
# (RELAY_INDEX) DIY index, on purpose:
#
#   A Bedrock Knowledge Base over S3 Vectors writes its OWN Bedrock-schema vectors
#   (the chunk text under the `AMAZON_BEDROCK_TEXT` metadata key, the source under
#   `x-amz-bedrock-kb-*`). Module 4's hand-built ingestion writes RAW vectors with
#   a different metadata schema (text under a `snippet` key). The KB cannot read a
#   foreign vector's text, so if both populations share one index the KB returns
#   half-empty results and the reranker (which needs non-empty candidate text)
#   fails. So the KB gets its own clean index; Module 4's `relay-docs` stays the
#   DIY baseline `compare_retrieval.py` benchmarks against. Both are S3 Vectors in
#   the SAME frozen vector bucket — idle ~$0, no always-on cluster either way.
#
# (The Module 5 brief predates this live finding — it assumed one shared index.
# The course choice S3 Vectors and the bucket name are unchanged; only the KB's
# index is its own. This is the kind of API/behaviour drift the brief flags to
# re-verify at generation.)
RELAY_KB_INDEX = "relay-kb-docs"

# The data source name attached to the KB (one S3 data source over docs/).
RELAY_KB_DATA_SOURCE_NAME = "relay-docs-source"

# The inclusion prefix the data source crawls inside the data bucket. The KB only
# ingests objects under docs/ (the CloudCart corpus Module 4 uploaded) — never
# attachments/ or vectors/.
RELAY_KB_INCLUSION_PREFIX = "docs/"

# The Relay tier that generates the cited answer in answer() (RetrieveAndGenerate).
# A name, never a model ID — answer() resolves it through tier_profile()/model_arn().
# Answers are the "complex" workload (grounded synthesis over retrieved context),
# so the smart tier, exactly as the spec pins it.
KB_ANSWER_TIER = "smart"

# Default retrieval depth for retrieve()/answer(). top_k is configurable per call;
# this is the floor the lab and the smoke test use.
KB_DEFAULT_TOP_K = 5

# --- The Bedrock reranker (verified us-east-1 catalogue, June 2026) -----------
# Cohere Rerank 3.5 is the Bedrock-managed reranker the lab uses. A reranker is a
# cross-encoder that RE-SCORES the retriever's candidates against the query and
# reorders them — it improves PRECISION of the top-k, not recall (it never
# surfaces a document the retriever did not already return). It is addressed by a
# Bedrock foundation-model ARN built from the Region at call time, so no
# account-specific ARN is hard-coded here; only the model ID is pinned.
#
# LIVE-VERIFIED (June 2026, us-east-1, account 901353600690): the ONLY reranker in
# this Region's catalogue is `cohere.rerank-v3-5:0`. Amazon Rerank
# (`amazon.rerank-v1:0`) is NOT available here (RetrieveAndGenerate returns "The
# provided model identifier is invalid"), so Cohere Rerank is the lab default. The
# brief flagged the reranker catalogue to re-verify at generation — this is that
# correction. (Amazon Rerank is kept below as the documented alternative for
# Regions that carry it / for the article's comparison row.)
RERANK_MODEL_ID = "cohere.rerank-v3-5:0"
RERANK_ALT_MODEL_ID = "amazon.rerank-v1:0"

# How many results the reranker returns after re-scoring the retriever's
# candidates. Kept <= the retrieval depth so reranking is a re-ordering, not a
# silent widening of k.
RERANK_NUMBER_OF_RESULTS = 5

# Reranker pricing for the cost line (AS OF JUNE 2026 — re-verify on the Bedrock
# pricing page). Cohere Rerank 3.5 on Bedrock: ~$2.00 per 1,000 queries (a "query"
# = one reranked request of up to 100 documents). Reporting only — no routing
# decision reads it.
RERANK_PRICE_PER_1K_QUERIES = 2.00


def model_arn(tier: str, *, region: str = REGION, account: str | None = None) -> str:
    """Build the inference-profile ARN a Knowledge Base needs for generation.

    RetrieveAndGenerate wants a model ARN, not the short profile ID converse()
    uses. We assemble it from the SAME tier -> inference-profile mapping (the sole
    model-ID home) so kb.py never hard-codes a model ID either. The account is
    resolved from STS at call time when not supplied (cross-Region inference
    profiles are account-scoped ARNs).

        tier="smart" -> arn:aws:bedrock:us-east-1:<acct>:inference-profile/us.amazon.nova-2-lite-v1:0
    """
    profile_id = tier_profile(tier)
    acct = account or account_id()
    return f"arn:aws:bedrock:{region}:{acct}:inference-profile/{profile_id}"


def rerank_model_arn(*, region: str = REGION, model_id: str = RERANK_MODEL_ID) -> str:
    """Build the reranker model ARN (a foundation-model ARN, Region-scoped).

    Rerankers are addressed by a bedrock foundation-model ARN. Only the model ID
    is pinned in config; the ARN is assembled here so no account/Region literal
    leaks into kb.py.

        arn:aws:bedrock:us-east-1::foundation-model/amazon.rerank-v1:0
    """
    return f"arn:aws:bedrock:{region}::foundation-model/{model_id}"


def estimate_rerank_cost(n_queries: int) -> float:
    """Reranker cost in USD from the query count — never a guess (reporting only)."""
    return n_queries / 1000 * RERANK_PRICE_PER_1K_QUERIES


# =============================================================================
# Module 6 ADDITIONS — intake pipeline policy (validation gates + attachments).
# =============================================================================
# Added BY ADDITION in Module 6 (data pipelines / intake). NOTHING above this line
# changes: the tier map gained only the appended "vision" tier; the resource names,
# the embedder, and the KB/reranker constants are untouched. Module 6 appends the
# POLICY the intake pipeline enforces — kept here for the same reason model IDs are:
# one place to read "what Relay accepts at the door", so relay/intake.py, setup.py,
# and the tests agree by construction.

# --- The S3 prefix attachments land under, inside the data bucket -------------
# Module 4 created the data bucket relay-<account_id> with the three prefixes
# (docs/ attachments/ vectors/). Module 6 is the first writer of attachments/:
# every accepted attachment is uploaded to s3://relay-<account_id>/attachments/.
RELAY_ATTACHMENTS_PREFIX = "attachments/"

# --- The multimodal tier the intake screenshot read runs on -------------------
# A tier NAME (resolved through tier_profile()/model_arn()), never a model ID — the
# vision model lives in the TIERS map above (Amazon Nova Lite). relay/intake.py
# calls converse(tier=VISION_TIER, ...) to read a screenshot.
VISION_TIER = "vision"

# --- Validation gates: the admitted input (skill 1.3.1) -----------------------
# A validation workflow runs BEFORE any FM call (cost + integrity). These are the
# limits the gate enforces; an input outside them is REJECTED explicitly, never
# silently truncated or coerced.
#
# Max customer-message size after we decode the raw bytes. 16 KiB is generous for a
# support email and small enough that a runaway log-dump paste is caught at the door
# instead of burning tokens on noise. (A real attachment goes to S3, not the body.)
MAX_MESSAGE_BYTES = 16 * 1024

# Text must decode as UTF-8 — a binary blob mislabelled as an email is rejected.
MESSAGE_ENCODING = "utf-8"

# Admitted attachment MIME types: the image formats Converse can read (skill 1.3.3
# image constraints). These map 1:1 onto relay.llm.IMAGE_MEDIA_TYPE_TO_FORMAT, so
# "what intake admits" and "what the vision call can send" are the SAME set. An
# attachment of any other type (PDF, zip, .exe) is rejected — Relay reads error
# SCREENSHOTS, not arbitrary files (PDF document processing is the Textract/BDA
# path discussed in the article, not built here).
ADMITTED_ATTACHMENT_MEDIA_TYPES = (
    "image/png",
    "image/jpeg",
    "image/gif",
    "image/webp",
)

# Max attachment size accepted for upload + a vision read. 4 MB keeps a screenshot
# well inside Converse's image limits and the lab's sub-$1 budget; a larger file is
# rejected at the gate (it never reaches S3 or the FM).
MAX_ATTACHMENT_BYTES = 4 * 1024 * 1024

# The language code Amazon Comprehend runs detect_entities with (skill 1.3.4). The
# CloudCart corpus is English; the "Try it yourself" adds a detect_dominant_language
# gate. A name, not a model — Comprehend is a managed NLP service, not a Bedrock FM.
COMPREHEND_LANGUAGE_CODE = "en"


def media_type_for_filename(filename: str) -> str | None:
    """Map a file name's extension to its admitted image MIME type, or None.

    Used by the gate to decide an attachment's media_type from its name before any
    upload. Returns None for an extension that is not an admitted image type, so the
    caller raises an explicit rejection (never guesses a type)."""
    lower = filename.lower()
    for media_type, ext in (
        ("image/png", ".png"),
        ("image/jpeg", ".jpg"),
        ("image/jpeg", ".jpeg"),
        ("image/gif", ".gif"),
        ("image/webp", ".webp"),
    ):
        if lower.endswith(ext):
            return media_type
    return None


# =============================================================================
# Module 7 ADDITIONS — agent resources: the DynamoDB tables + the MCP server URL.
# =============================================================================
# Added BY ADDITION in Module 7 (agentic AI). NOTHING above this line changes: the
# tier -> inference-profile map is frozen at Module 3 (the agent runs on the existing
# "smart" tier — no new model), the resource names + embedder are frozen at M4, the
# KB/reranker at M5, the intake policy at M6. Module 7 appends only the names of the
# two CloudCart business systems the agent acts on, and how to find the MCP server —
# same containment law: ONE place a table name or the MCP URL is written, so setup.py,
# teardown.py, relay/tools.py, relay/agent.py, and mcp_server/ all agree.

# --- The two CloudCart business tables (canonical, 06 §2 / bible §3.3) ----------
# DynamoDB ON-DEMAND tables (PAY_PER_REQUEST -> ~$0 idle, no provisioned capacity to
# tear down). Frozen names; no suffix — unlike the buckets, a table name is already
# account- and Region-scoped, and the spec pins these EXACT names:
#
#   relay-orders   : the CloudCart order book. SEEDED with 25 orders (data/orders.json)
#                    so `lookup_order` returns a real status. Primary key: order_id (S).
#   relay-tickets  : where the agent PERSISTS a TicketRecord (status, triage, answer,
#                    the actions[] journal). Primary key: ticket_id (S).
#
# These are the ONLY two resources the agent's IAM boundary (skill 2.1.3) lets the
# MCP server's Lambda touch: read relay-orders, write relay-tickets — nothing else.
RELAY_ORDERS_TABLE = "relay-orders"
RELAY_TICKETS_TABLE = "relay-tickets"

# The primary-key attribute name on each table (one place, so seeding, the tools, and
# setup/teardown agree). Both are simple string hash keys.
ORDERS_KEY = "order_id"
TICKETS_KEY = "ticket_id"


# --- The CloudCart MCP server URL (resolved, never hard-coded) -----------------
# The stateless MCP server (mcp_server/) is deployed on AWS Lambda by setup.py, which
# records its invoke URL (a Lambda Function URL) in the .mcp_url file so relay/tools.py
# can build an MCP client without an env var. An explicit RELAY_MCP_URL env var wins —
# handy for pointing the agent at a LOCAL `uv run python -m mcp_server` during dev.
#
# Containment: the URL is account/deploy-specific (a Function URL like
# https://<id>.lambda-url.us-east-1.on.aws/mcp), so it is NOT a literal here — it is
# RESOLVED at call time from the env var or the recorded file, the same pattern kb.py
# uses for the KB id. The path the server mounts its streamable-HTTP transport on is a
# stable constant, though, so the client and server agree by construction.
MCP_SERVER_PATH = "/mcp"

# The on-disk marker file setup.py writes the deployed Function URL to (git-ignored,
# account-specific) — resolve_mcp_url() reads it. Kept as a name so setup.py/teardown.py
# and resolve_mcp_url() agree on the filename.
MCP_URL_FILE_NAME = ".mcp_url"


def resolve_mcp_url(url: str | None = None) -> str:
    """Find the CloudCart MCP server URL the agent's MCP client should connect to.

    Order: explicit argument, the RELAY_MCP_URL env var (set this to a local
    `http://127.0.0.1:8000/mcp` during dev), then the `.mcp_url` file setup.py writes
    after deploying the Lambda. Raises a clear, actionable error if none is set — no
    silent fallback to a wrong endpoint.
    """
    import os
    from pathlib import Path

    if url:
        return url
    env = os.environ.get("RELAY_MCP_URL")
    if env:
        return env.strip()
    marker = Path(__file__).resolve().parent.parent / MCP_URL_FILE_NAME
    if marker.exists():
        recorded = marker.read_text(encoding="utf-8").strip()
        if recorded:
            return recorded
    raise ValueError(
        "No CloudCart MCP server URL found. Deploy the MCP server first:\n"
        "    uv run python setup.py\n"
        "It deploys mcp_server/ to AWS Lambda and records the Function URL in "
        ".mcp_url.\nOr run the server locally and point the agent at it:\n"
        "    uv run python -m mcp_server            # serves http://127.0.0.1:8000/mcp\n"
        "    RELAY_MCP_URL=http://127.0.0.1:8000/mcp uv run python -m relay.agent \"...\""
    )


# =============================================================================
# Module 8 ADDITIONS — multi-agent (Billing specialist), AgentCore Memory + Runtime,
# and the HITL refund gate.
# =============================================================================
# Added BY ADDITION in Module 8 (multi-agent systems + Bedrock AgentCore). NOTHING
# above this line changes: the tier -> inference-profile map is frozen at Module 3
# (the Billing specialist runs on the existing "smart" tier — no new model), the
# resource names + embedder are frozen at M4, the KB/reranker at M5, the intake
# policy at M6, the tables + MCP URL at M7. Module 8 appends only:
#   - the AgentCore Memory id (resolved at call time, like the MCP URL — it is
#     account/deploy-specific, so it is NEVER a literal here);
#   - the AgentCore Runtime / Memory canonical names setup.py creates and
#     teardown.py purges (the long-term Memory store is the SOLE idle-billed item);
#   - the HITL gate policy: which tool name is SENSITIVE (refund) and the canonical
#     "Billing specialist" handoff signal.
# Same containment law: one place each name lives, so setup.py, teardown.py,
# relay/specialists.py, relay/agent.py, relay/approve.py, relay/run.py, and the
# agentcore/ deploy config all agree by construction.

# --- The Billing specialist (canonical name, 06 §5.4 — NO synonym) -------------
# Relay (the generalist) HANDS OFF a billing/refund ticket to the "Billing
# specialist": a second Strands agent with its OWN system prompt (refund tone +
# rules). It is NOT a new model — it reasons on the SAME smart tier, resolved
# through tier_profile(); only its prompt and its tool set differ. The exact string
# is reproduced everywhere (article, code, lab); a synonym is a rejection cause.
BILLING_SPECIALIST_NAME = "Billing specialist"

# The Relay tier the Billing specialist reasons on — a NAME, resolved through
# tier_profile()/model_arn(), never a model ID. Refund reasoning is the "complex"
# workload (read the order, weigh the refund rules), so the smart tier — exactly
# like the generalist agent.
BILLING_SPECIALIST_TIER = "smart"

# The Triage intent (M2 frozen enum) that routes a ticket toward the Billing
# specialist. A NAME from the frozen Triage.intent literals — no new value invented.
# relay.agent hands off when triage.intent == this AND the request is refund-shaped.
BILLING_HANDOFF_INTENT = "billing"


# --- The HITL gate: the SENSITIVE tool (06 §2 — AgentAction.approved becomes
#     effective at M8, TicketRecord status awaiting_approval is exercised) ---------
# The human-in-the-loop principle (skill 2.1.5): you do NOT gate every action —
# only the SENSITIVE ones, or the gate is pure friction. For CloudCart the one
# financially sensitive action is a REFUND. When the Billing specialist proposes
# `refund`, the agent records an AgentAction(approved=None) and DOES NOT execute it;
# the TicketRecord goes to `awaiting_approval` and waits for a human decision
# (relay.approve). Reading an order, citing a doc, or creating a ticket are NOT
# gated. The tool name is canonical and lives here so agent.py, approve.py, and the
# specialist's tool agree by construction.
REFUND_TOOL_NAME = "refund"

# The set of tool names whose proposal is GATED behind human approval. Exactly one
# at Module 8 (refund). Kept as a frozenset so a future sensitive action is added by
# addition, and `is_sensitive_tool()` is the single decision point.
SENSITIVE_TOOLS: frozenset[str] = frozenset({REFUND_TOOL_NAME})


def is_sensitive_tool(tool_name: str) -> bool:
    """True if proposing `tool_name` must be gated behind human approval (skill 2.1.5).

    The SOLE decision point for the HITL gate: agent.py asks this before executing a
    proposed action. Only the refund tool is sensitive at Module 8; every other tool
    (search_kb, lookup_order, create_ticket) runs without a gate."""
    return tool_name in SENSITIVE_TOOLS


# --- AgentCore Runtime + Memory (canonical names, 06 §2 / bible §3.3) ----------
# Bedrock AgentCore is GA (since 13 Oct 2025). The lab uses TWO GA components:
#   - Runtime : the managed microVM where the deployed agent runs (sessions up to
#               8 h, idle FREE — billed only per second of active consumption).
#   - Memory  : short-term (session events) + long-term (cross-session records).
# Gateway and Identity are GA too but not built here. Agent Registry / Payments are
# PREVIEW (as of June 2026) and are deliberately NOT used.
#
# The Runtime agent's logical name (the `agentcore configure --name` value, and the
# name the deployed agent is invoked by). The agentcore CLI owns the actual runtime
# ARN; this is the stable handle setup/teardown and the deploy config agree on.
AGENTCORE_RUNTIME_NAME = "relay-agent"

# The AgentCore Memory store name setup.py creates over the bedrock-agentcore-control
# plane. It holds Relay's short-term session events AND its long-term records. The
# long-term store is the ONLY idle-billed item in the whole lab (~$0.75 / 1K records
# / month as of June 2026), so teardown.py PURGES it (deletes the Memory) — B5.
AGENTCORE_MEMORY_NAME = "relay-memory"

# The AgentCore CreateMemory API constrains `name` (and a strategy's `name`) to the
# pattern [a-zA-Z][a-zA-Z0-9_]{0,47} — letters, digits, underscores only, NO hyphens
# (live-verified June 2026). Our canonical handle stays `relay-memory` (the logical
# name in the lab + this config); the AWS resource name is the same handle with the
# hyphen mapped to an underscore, derived here so setup.py / teardown.py never type a
# second literal. The created store's id is what relay.run resolves and teardown
# purges — the name is just the create-time label.
def agentcore_memory_api_name() -> str:
    """The AWS-API-valid AgentCore Memory name: the canonical handle, hyphens -> '_'.

    CreateMemory rejects a hyphen in `name`; this maps `relay-memory` -> `relay_memory`
    so the one canonical handle still drives the resource name (no second literal)."""
    return AGENTCORE_MEMORY_NAME.replace("-", "_")


# The long-term semantic strategy's name. Same API constraint (no hyphens), so it is
# an underscore identifier — the long-term store that distils durable NON-PII facts.
AGENTCORE_MEMORY_STRATEGY_NAME = "relay_customer_facts"

# How long AgentCore Memory retains long-term records (days). A short window keeps
# the cross-session store small and the recurring cost near zero; the lab does not
# need months of history. A retention/cost decision (D2 + D4), set in ONE place.
AGENTCORE_MEMORY_EXPIRY_DAYS = 30

# The on-disk marker file setup.py writes the created Memory id to (git-ignored,
# account-specific) — resolve_memory_id() reads it. The Memory id is account/deploy-
# specific, so it is RESOLVED at call time (env var or marker file), NEVER a literal
# here — the same pattern as the MCP URL and the KB id.
MEMORY_ID_FILE_NAME = ".memory_id"

# The AgentCore Runtime arn marker file (written by setup.py after the agentcore CLI
# launch records it, or by the deploy step). Resolved, never a literal — account-
# and deploy-specific.
RUNTIME_ARN_FILE_NAME = ".runtime_arn"

# The namespace template AgentCore Memory long-term records are written under, keyed
# by CUSTOMER so each customer's cross-session facts are isolated. `{actor_id}` is
# the AgentCore actor (we use the customer id). One place, so run.py and the lab
# agree on the namespace shape.
MEMORY_LONG_TERM_NAMESPACE = "support/customer/{actor_id}/facts"


def resolve_memory_id(memory_id: str | None = None) -> str:
    """Find the AgentCore Memory id the agent should read/write its memory in.

    Order: explicit argument, the RELAY_MEMORY_ID env var, then the `.memory_id`
    file setup.py writes after creating the Memory store. Raises a clear, actionable
    error if none is set — no silent fallback to a wrong store. Same resolution
    pattern as resolve_mcp_url()/kb.resolve_kb_id().
    """
    import os
    from pathlib import Path

    if memory_id:
        return memory_id
    env = os.environ.get("RELAY_MEMORY_ID")
    if env:
        return env.strip()
    marker = Path(__file__).resolve().parent.parent / MEMORY_ID_FILE_NAME
    if marker.exists():
        recorded = marker.read_text(encoding="utf-8").strip()
        if recorded:
            return recorded
    raise ValueError(
        "No AgentCore Memory id found. Create the Memory store first:\n"
        "    uv run python setup.py\n"
        "It creates the AgentCore Memory '" + AGENTCORE_MEMORY_NAME + "' and records "
        "its id in ." + "memory_id" + ".\nOr set it explicitly for a dev run:\n"
        "    RELAY_MEMORY_ID=<memory-id> uv run python -m relay.run \"...\""
    )


# =============================================================================
# Module 9 ADDITIONS — the Bedrock Guardrail (`relay-guardrail`) + grounding gate.
# =============================================================================
# Added BY ADDITION in Module 9 (safety engineering). NOTHING above this line
# changes: the tier -> inference-profile map is frozen at Module 3 (the guardrail is
# model-INDEPENDENT — it filters text on either side of any model, so there is no new
# tier and no re-point), the resource names + embedder are frozen at M4, the
# KB/reranker at M5, the intake policy at M6, the tables + MCP URL at M7, the
# AgentCore names at M8. Module 9 appends only the guardrail's canonical NAME, the
# resolution of its account/deploy-specific id + version, the policy thresholds
# setup.py creates it with, and the ONE grounding threshold the whole course reuses —
# same containment law: ONE place each value lives, so setup.py, teardown.py,
# relay/safety.py, relay/llm.py, and relay/kb.py all agree by construction.

# --- The guardrail (canonical name, 06 §2 / bible §3.3 — NO variation) ----------
# Bedrock Guardrails is the MANAGED safety layer. `relay-guardrail` is attached to
# Relay's model calls (through converse()) AND used standalone via ApplyGuardrail
# (relay/safety.py) to filter any text — including the answer's grounding check. The
# exact name is reproduced everywhere (article, setup, teardown, config); a synonym
# (or "safety filter" / "content moderation") is a rejection cause (06 §5 terminology).
RELAY_GUARDRAIL_NAME = "relay-guardrail"

# Guardrails tier. The course uses STANDARD (06 §4): broader language coverage and the
# prompt-attack + contextual-grounding capabilities the lab relies on. CLASSIC is the
# older, narrower tier — taught as the comparison, never used here.
GUARDRAIL_TIER = "STANDARD"

# The guardrail cross-Region (inference) profile. As of June 2026 the Bedrock API
# REQUIRES a guardrail profile (cross-Region inference) to use the STANDARD policy tier:
# CreateGuardrail rejects `tierConfig.tierName=STANDARD` without a `crossRegionConfig`
# ("Enable cross-Region inference for your guardrail to use Standard tier"). The profile
# is the guardrail analogue of the model inference profiles (us./global.) the course
# pins everywhere — it lives HERE in config.py, never as a literal in setup.py. `us.`
# routes the guardrail evaluation across the US commercial Regions for capacity.
GUARDRAIL_CROSS_REGION_PROFILE = "us.guardrail.v1:0"

# The guardrail VERSION relay.safety / relay.llm apply. A published guardrail has
# numbered versions (1, 2, ...) plus the mutable "DRAFT". The lab applies the first
# PUBLISHED version (recorded by setup.py); DRAFT is for editing, a numbered version is
# what you attach to traffic (the article's draft-vs-version promotion story). This is
# the DEFAULT — resolve_guardrail_version() lets an env var / marker override it.
GUARDRAIL_DEFAULT_VERSION = "1"

# --- The ONE grounding threshold (bible §4 M9 coherence LAW) -------------------
# The contextual grounding check returns a `grounding` score (is the answer supported
# by the retrieved context?) and a `relevance` score (does it answer the query?), each
# in [0, 1]. Below the threshold, the answer is treated as UNGROUNDED: kb.answer() sets
# Answer.grounded = False and Relay ESCALATES (it does not ship a possibly-hallucinated
# promise to a customer). 0.8 is deliberately strict for a support agent that can make
# financial promises.
#
# DEFINE IT ONCE, HERE. The SAME 0.8 constant is reused downstream as the Module 13
# eval regression gate (aggregate.grounding < 0.8) and the Module 14 `relay-ops`
# grounding alarm threshold — gate <-> alarm <-> escalation coherence (bible §3.4 / §4
# M9). A divergent literal anywhere is a coherence break.
GROUNDING_THRESHOLD = 0.8

# The relevance threshold the contextual grounding check uses alongside grounding.
# Kept equal to the grounding threshold so "is it supported AND on-topic" use one bar;
# tuned in the lab's "Try it yourself" (raise it, watch false positives on legitimate
# tickets).
RELEVANCE_THRESHOLD = 0.8

# --- The content-filter strength the guardrail is created with -----------------
# A name, not a magic string scattered across setup.py. The four standard content-filter
# categories (HATE / INSULTS / SEXUAL / VIOLENCE) plus MISCONDUCT run at HIGH on both
# input and output; PROMPT_ATTACK (the prompt-injection/jailbreak classifier) runs at
# HIGH on INPUT only — AWS requires PROMPT_ATTACK's output strength to be NONE (the
# attack is in the user/content side, not the model's reply). These thresholds drive
# the block rate the lab MEASURES; raising them is a "Try it yourself" lever.
CONTENT_FILTER_STRENGTH = "HIGH"

# The PII action the guardrail's sensitive-information filter applies. MASK (the Bedrock
# API enum is ANONYMIZE) replaces a detected entity with a typed placeholder ([NAME],
# [EMAIL], ...) rather than BLOCKING the whole request — so a legitimate ticket that
# merely MENTIONS an email still flows, with the email masked. The FULL PII redaction
# PIPELINE at intake (Comprehend, by offset, before any FM call) is Module 10; here it
# is only the guardrail's own PII filter (one line, 06 §2 boundary).
PII_GUARDRAIL_ACTION = "ANONYMIZE"

# The on-disk markers setup.py writes the created guardrail's id + published version to
# (git-ignored, account/deploy-specific) — resolve_guardrail_id/_version read them. The
# guardrail id is account-specific, so it is RESOLVED at call time (env var or marker
# file), NEVER a literal here — the same pattern as the KB id, MCP URL, and Memory id.
GUARDRAIL_ID_FILE_NAME = ".guardrail_id"
GUARDRAIL_VERSION_FILE_NAME = ".guardrail_version"


def resolve_guardrail_id(guardrail_id: str | None = None) -> str:
    """Find the `relay-guardrail` id created by setup.py.

    Order: explicit argument, the RELAY_GUARDRAIL_ID env var, then the `.guardrail_id`
    file setup.py writes after CreateGuardrail. Raises a clear, actionable error if none
    is set — no silent fallback to a wrong (or no) guardrail. Same resolution pattern as
    resolve_mcp_url() / resolve_memory_id() / kb.resolve_kb_id().
    """
    import os
    from pathlib import Path

    if guardrail_id:
        return guardrail_id
    env = os.environ.get("RELAY_GUARDRAIL_ID")
    if env:
        return env.strip()
    marker = Path(__file__).resolve().parent.parent / GUARDRAIL_ID_FILE_NAME
    if marker.exists():
        recorded = marker.read_text(encoding="utf-8").strip()
        if recorded:
            return recorded
    raise ValueError(
        "No guardrail id found. Create '" + RELAY_GUARDRAIL_NAME + "' first:\n"
        "    uv run python setup.py\n"
        "It creates the Bedrock Guardrail, publishes a version, and records the id in ."
        + "guardrail_id.\nOr set it explicitly for a dev run:\n"
        "    RELAY_GUARDRAIL_ID=<guardrail-id> uv run python run_attacks.py --guarded"
    )


def resolve_guardrail_version(version: str | None = None) -> str:
    """Find the published guardrail version to apply.

    Order: explicit argument, the RELAY_GUARDRAIL_VERSION env var, the `.guardrail_version`
    file setup.py writes, then GUARDRAIL_DEFAULT_VERSION ("1"). Unlike the id, a version
    has a sensible default (the first published version), so this never raises — a guarded
    call with the id but no recorded version still works against version "1".
    """
    import os
    from pathlib import Path

    if version:
        return version
    env = os.environ.get("RELAY_GUARDRAIL_VERSION")
    if env:
        return env.strip()
    marker = Path(__file__).resolve().parent.parent / GUARDRAIL_VERSION_FILE_NAME
    if marker.exists():
        recorded = marker.read_text(encoding="utf-8").strip()
        if recorded:
            return recorded
    return GUARDRAIL_DEFAULT_VERSION


# =============================================================================
# Module 10 ADDITIONS — PII redaction policy + least-privilege IAM role names.
# =============================================================================
# Added BY ADDITION in Module 10 (security, privacy, governance). NOTHING above this
# line changes: the tier -> inference-profile map is frozen at Module 3, the resource
# names + embedder at M4, the KB/reranker at M5, the intake policy at M6, the tables +
# MCP URL at M7, the AgentCore names at M8, the guardrail at M9. Module 10 appends only:
#   - the PII redaction POLICY (which Comprehend entity types are masked, the
#     confidence floor) — read by relay/pii.py and relay/intake.py;
#   - the least-privilege IAM role NAMES setup.py creates and teardown.py deletes
#     (one role per Relay component) + the decision-log file name.
# Same containment law: ONE place each value lives, so relay/pii.py, relay/intake.py,
# relay/agent.py, setup.py, teardown.py, iam/policies/, and audit_report.py all agree.
#
# NO new FM call and NO new model ID here — PII detection is Amazon Comprehend (a
# separate managed service), and the model card lists only the ACTIVE inference-profile
# IDs already in the tier map above.

# --- PII redaction policy (skills 3.2.2 / 3.2.3) -------------------------------
# The Amazon Comprehend PII entity types relay/pii.py MASKS at intake, before any FM
# call. This is an ALLOWLIST: a type not here is left in place. We mask the identity +
# contact + financial entities a CloudCart ticket can carry, and deliberately DO NOT
# mask DATE_TIME (a delivery/billing date is operational signal the agent needs) or an
# order number (Comprehend does not class "#1042" as PII — it is a business key, kept so
# lookup_order still works). The placeholders are the typed `[NAME]`/`[EMAIL]`/`[PHONE]`
# forms relay/pii.placeholder_for() builds. Comprehend's full type list is much larger
# (live-verified June 2026: NAME, EMAIL, PHONE, ADDRESS, SSN, CREDIT_DEBIT_NUMBER,
# BANK_ACCOUNT_NUMBER, USERNAME, PASSWORD, IP_ADDRESS, ... — re-verify on the Comprehend
# docs); this is the subset that matters for support tickets.
PII_ENTITY_TYPES: frozenset[str] = frozenset({
    "NAME",
    "EMAIL",
    "PHONE",
    "ADDRESS",
    "SSN",
    "CREDIT_DEBIT_NUMBER",
    "BANK_ACCOUNT_NUMBER",
    "BANK_ROUTING",
    "PIN",
    "USERNAME",
    "PASSWORD",
    "IP_ADDRESS",
    "AWS_ACCESS_KEY",
    "AWS_SECRET_KEY",
    "PASSPORT_NUMBER",
    "DRIVER_ID",
})

# The confidence floor a Comprehend PII detection must clear to be masked. Below it we
# leave the text alone: we would rather miss a borderline low-confidence guess than mask
# a real order number Comprehend half-suspects is a phone. 0.5 is Comprehend's own
# practical floor for PII; raising it is a "Try it yourself" lever (fewer false masks,
# more misses). A retention/privacy decision, set in ONE place so pii.py and the tests
# agree. (Comprehend uses the same COMPREHEND_LANGUAGE_CODE the M6 entity pass uses.)
PII_MIN_CONFIDENCE = 0.5

# --- Least-privilege IAM role names, one per Relay component (skill 3.2.1) ------
# Module 10 gives EACH component its own minimal IAM role instead of one broad lab role:
# the intake pipeline, the agent/tools, the Knowledge Base reader, and the future public
# API. Each role's policy (iam/policies/*.json) names EXPLICIT actions + resource ARNs
# (the canonical relay-orders / relay-tickets / relay-<account_id> / relay-guardrail
# names) with ZERO wildcards — `Action: "*"` / `Resource: "*"` never appear (the grep
# gate proves it, brief §10). setup.py creates these roles from the JSON files;
# teardown.py deletes them. The names are frozen here so every consumer agrees.
#
# These reuse the existing canonical resource names in their ARNs — they create NO new
# AWS resource of their own, only a scoped identity for a component that already exists.
IAM_INTAKE_ROLE_NAME = "relay-intake-role"
IAM_AGENT_ROLE_NAME = "relay-agent-role"
IAM_KB_READER_ROLE_NAME = "relay-kb-reader-role"
IAM_API_ROLE_NAME = "relay-api-role"

# The ordered list of (role name, policy file stem) pairs setup.py iterates to create
# the per-component roles, and teardown.py iterates to delete them. The stem maps to
# iam/policies/<stem>.json. ONE list so create + delete + the smoke test stay in lockstep
# and a fifth component is added by appending one tuple.
IAM_COMPONENT_ROLES: tuple[tuple[str, str], ...] = (
    (IAM_INTAKE_ROLE_NAME, "intake"),
    (IAM_AGENT_ROLE_NAME, "agent"),
    (IAM_KB_READER_ROLE_NAME, "kb-reader"),
    (IAM_API_ROLE_NAME, "api"),
)

# The inline-policy name attached to each component role (one inline policy per role,
# loaded from its JSON file). A stable name so put_role_policy / delete_role_policy in
# setup.py / teardown.py target the same policy by construction.
IAM_COMPONENT_POLICY_NAME = "relay-least-privilege"

# --- The agent decision log (governance: prove what the agent DECIDED) ----------
# CloudTrail proves who called which AWS API; the decision log proves what RELAY decided
# and WHY — the distinction the exam draws (3.3.1/3.3.2). relay/agent.py appends one
# structured JSON-Lines record per agent run (the tool calls, their REDACTED inputs, the
# result, the outcome status, and a timestamp). audit_report.py reads it back. The file
# name is git-ignored (it is runtime output, and even redacted it is operational data) —
# kept here so agent.py and audit_report.py agree on the path.
DECISION_LOG_FILE_NAME = "decision_log.jsonl"


# =============================================================================
# Module 11 ADDITIONS — the serverless front door: SQS queue + the relay-events bus.
# =============================================================================
# Added BY ADDITION in Module 11 (serverless deployment / enterprise integration / CI-CD).
# NOTHING above this line changes: the tier -> inference-profile map is frozen at Module 3
# (the API exposes the SAME converse()/agent — no new model and no re-point), the resource
# names + embedder at M4, the KB/reranker at M5, the intake policy at M6, the tables + MCP
# URL at M7, the AgentCore names at M8, the guardrail at M9, the PII/IAM names at M10.
# Module 11 appends only the NAMES of the new standing resources its CDK stack creates —
# the work queue and the event bus — plus the API request limits and the EventBridge
# detail-types. Same containment law: ONE place each name lives, so the CDK stack, the
# four relay/api handlers, the smoke test, and teardown all agree by construction.
#
# These names are reproduced FIELD-FOR-FIELD from 06 §2 / bible §3.3. The API itself is
# the API Gateway REST API the CDK stack stands up (`POST /tickets` -> {ticket_id};
# `GET /tickets/{id}` -> TicketRecord; `POST /tickets/{id}/approve` body {approved: bool})
# — the routes are owned by the stack, not literals here. No model ID, no FM call here.

# --- The async work queue (Amazon SQS) ----------------------------------------
# POST /tickets returns 202 IMMEDIATELY and pushes the ticket onto this queue; the
# worker Lambda consumes it and runs the agent (several seconds — too long to hold an
# HTTP connection). A standard (not FIFO) queue is enough: ordering does not matter for
# independent tickets, and the worker is idempotent on ticket_id (a redelivered message
# overwrites the same relay-tickets row, never a duplicate). On-demand pricing, ~$0 idle.
# The DLQ catches a ticket that fails repeatedly (poison message) so it stops being
# retried forever — the "In production" dead-letter-queue note, wired in the CDK stack.
RELAY_QUEUE_NAME = "relay-tickets-queue"
RELAY_DLQ_NAME = "relay-tickets-dlq"

# How many times SQS redelivers a failing message before routing it to the DLQ (the
# redrive maxReceiveCount). Low, so a poison ticket lands in the DLQ fast instead of
# burning the worker on every redelivery. A reliability decision, set in ONE place so
# the CDK stack reads it.
RELAY_QUEUE_MAX_RECEIVE = 3

# The SQS visibility timeout (seconds) the worker runs under. It MUST exceed the worker
# Lambda's own timeout so a long agent run is not redelivered to a SECOND worker while
# the first is still working it (which would double-process the ticket). Kept here, read
# by the CDK stack, so the two never drift.
RELAY_WORKER_TIMEOUT_S = 120
RELAY_QUEUE_VISIBILITY_TIMEOUT_S = 180  # > RELAY_WORKER_TIMEOUT_S (no double-processing)

# The env var the deployed Lambdas read the queue URL from (the CDK stack injects it; a
# local dev/test run sets it explicitly). RESOLVED at call time, never a literal — the
# queue URL is account/Region/deploy-specific, the same pattern as the MCP URL / KB id.
RELAY_QUEUE_URL_ENV = "RELAY_QUEUE_URL"


def resolve_queue_url(queue_url: str | None = None, *, sqs_client=None) -> str:
    """Find the SQS work-queue URL the API/worker should use.

    Order: explicit argument, the RELAY_QUEUE_URL env var (the CDK stack sets it on the
    deployed Lambdas), then a live GetQueueUrl by the frozen RELAY_QUEUE_NAME. Raises a
    clear, actionable error if none resolves — no silent fallback to a wrong queue. Same
    resolution pattern as resolve_mcp_url() / resolve_memory_id().
    """
    import os

    if queue_url:
        return queue_url
    env = os.environ.get(RELAY_QUEUE_URL_ENV)
    if env:
        return env.strip()
    try:
        import boto3

        client = sqs_client or boto3.client("sqs", region_name=REGION)
        return client.get_queue_url(QueueName=RELAY_QUEUE_NAME)["QueueUrl"]
    except Exception as err:  # noqa: BLE001 — turn any failure into an actionable message.
        raise ValueError(
            "No Relay work-queue URL found. Deploy the stack first:\n"
            "    uv run cdk deploy\n"
            "It creates the SQS queue '" + RELAY_QUEUE_NAME + "' and injects "
            + RELAY_QUEUE_URL_ENV + " on the Lambdas.\nOr set it explicitly for a dev "
            "run:\n    " + RELAY_QUEUE_URL_ENV + "=<queue-url> uv run python ...\n"
            f"(underlying: {type(err).__name__})"
        ) from err


# --- The event bus (Amazon EventBridge) — loose coupling (06 §2 / bible §3.3) --
# Relay does NOT know CloudCart's human escalation queue or approval inbox. When the
# agent ESCALATES or parks a refund AWAITING APPROVAL, the worker PUBLISHES an event to
# this bus and moves on; EventBridge RULES route it to whatever targets CloudCart wires
# (a human queue, an approval inbox, a webhook). That is the loose-coupling pattern: a
# new consumer is a new rule, not a code change in Relay. A custom bus (not the default
# bus) so Relay's events are isolated and easy to rule on. Frozen NAME, no variation.
RELAY_EVENT_BUS_NAME = "relay-events"

# The event SOURCE field every Relay event carries (the EventBridge "source"). A custom
# reverse-DNS-ish source so a rule can match all Relay events with one prefix.
RELAY_EVENT_SOURCE = "relay.support"

# The two detail-types Relay emits (06 §2 / bible §3.3 — reproduced FIELD-FOR-FIELD):
#   relay.escalation        : the agent handed the ticket to a human (escalated=True).
#   relay.approval_required : a refund is parked awaiting human approval (the M8 HITL
#                             gate, now surfaced as an EVENT so an approval inbox can pick
#                             it up — the public POST /approve endpoint resolves it).
# Kept as named constants so the worker, the CDK rules, and the smoke test agree.
RELAY_DETAIL_ESCALATION = "relay.escalation"
RELAY_DETAIL_APPROVAL_REQUIRED = "relay.approval_required"

# The env var the worker reads the event-bus name from (the CDK stack injects it; defaults
# to the frozen RELAY_EVENT_BUS_NAME otherwise, since the name — unlike a URL or id — is a
# stable literal, not account-specific). Resolution has a sensible default, so it never
# raises: a worker with the name but no env var still publishes to `relay-events`.
RELAY_EVENT_BUS_ENV = "RELAY_EVENT_BUS_NAME"


def resolve_event_bus_name(bus_name: str | None = None) -> str:
    """Find the EventBridge bus name the worker publishes to.

    Order: explicit argument, the RELAY_EVENT_BUS_NAME env var, then the frozen default
    RELAY_EVENT_BUS_NAME ("relay-events"). Unlike the queue URL, a bus name is a stable
    literal (account/Region-scoped already), so this has a sensible default and never
    raises — the worker always knows where to publish."""
    import os

    if bus_name:
        return bus_name
    env = os.environ.get(RELAY_EVENT_BUS_ENV)
    if env:
        return env.strip()
    return RELAY_EVENT_BUS_NAME

# --- The CDK stack + CodePipeline canonical names -----------------------------
# The CloudFormation stack name the CDK app synthesizes (the front-door infra) and the
# CodePipeline name the pipeline stack creates. Kept here so `cdk deploy`, the smoke test,
# and teardown.py (which deletes the pipeline so it is not idle-billed ~$1/month, B5) all
# agree on the exact names. The pipeline is the ONLY M11 resource with a real idle cost —
# it is DESTROYED at teardown.
RELAY_STACK_NAME = "RelayApiStack"
RELAY_PIPELINE_STACK_NAME = "RelayPipelineStack"
RELAY_PIPELINE_NAME = "relay-pipeline"

# The API Gateway stage name the REST API is deployed under (the URL is
# https://<api-id>.execute-api.<region>.amazonaws.com/<stage>/tickets). One place so the
# stack, the smoke test, and the lab's curl examples agree.
RELAY_API_STAGE = "prod"


# =============================================================================
# Module 12 ADDITIONS — the token economy: prompt caching, the semantic-cache table,
# the Flex service tier, and the batch-inference discount (cost instrumentation).
# =============================================================================
# Added BY ADDITION in Module 12 (cost & performance optimization). NOTHING above this
# line changes: the tier -> inference-profile map is FROZEN at Module 3 (Module 12 picks
# the SAME fast/smart models, only cheaper — via a cache, a cached prefix, a batch job,
# or the Flex tier; no new tier and NO re-point), the resource names + embedder at M4
# (the semantic cache REUSES the pinned Titan V2 1024-dim embedder — a swap would
# invalidate the index, bible §5.2), the KB/reranker at M5, the intake policy at M6, the
# tables + MCP URL at M7, the AgentCore names at M8, the guardrail at M9, the PII/IAM at
# M10, the SQS queue + relay-events bus at M11. The per-tier PRICE MAP (PRICE_PER_1K /
# estimate_cost) ALREADY EXISTS since Module 3 — Module 12 does NOT re-introduce it; it
# CONSUMES it to populate TicketRecord.cost_cents (bible §2.2 M12). This block appends
# only: the semantic-cache table name + threshold + TTL, the prompt-caching wiring
# constants, the Flex service-tier name, and the batch/Flex/prompt-caching discount
# factors used in the before/after report. Same containment law: ONE place each value
# lives, so relay/cache.py, relay/llm.py, relay/api/worker_handler.py, cost_report.py,
# setup.py, teardown.py, and the cdk stack all agree by construction.
#
# Boundary (brief §6 / §9): Flex and batch are -50% but LATENCY-TOLERANT — they ride the
# EVAL/BACKFILL path ONLY, NEVER Relay's interactive traffic. Prompt caching is the only
# lever wired onto the interactive path (it caches an INPUT prefix provider-side, so it
# cannot stale an answer). The semantic cache is opt-in, in FRONT of the answer path, and
# guarded by a similarity threshold + a TTL (never a blind cache).

# --- The semantic-cache table (DynamoDB ON-DEMAND — 06 §2 / bible §3.3) ---------
# The semantic cache stores {question_hash, embedding, answer, created_at, expires_at}
# for frequent questions so a near-duplicate question is served from DynamoDB instead of
# a fresh converse() call (cost ~ 0). PAY_PER_REQUEST (on-demand): ~$0 idle, no
# provisioned capacity to plan or tear down — the same billing model as relay-orders /
# relay-tickets. The frozen NAME (06 §2 names the table in config.py). teardown.py drops
# it by hygiene (B5) even though on-demand idle is ~$0.
RELAY_CACHE_TABLE = "relay-cache"

# The primary-key attribute on the cache table: the DETERMINISTIC request hash (the exact
# question, normalized + SHA-256). An EXACT repeat is an O(1) GetItem hit — the
# "deterministic request hashing / result fingerprinting" lever (skill 4.1.4), distinct
# from the SEMANTIC (embedding-similarity) lookup. One place, so cache.py + setup.py +
# teardown.py agree on the key.
CACHE_KEY = "question_hash"

# The DynamoDB attribute the cache stores the row's TTL epoch under. DynamoDB's native TTL
# deletes an item once this attribute's epoch passes — the cache INVALIDATION mechanism
# (skill 4.1.4): a stale answer ages out on its own, no sweeper to run. setup.py enables
# TTL on this attribute; cache.py writes it = created_at + CACHE_TTL_SECONDS.
CACHE_TTL_ATTRIBUTE = "expires_at"

# How long a cached answer stays servable (seconds). 24 h: long enough that a busy day of
# repeated "where is my order?" questions hits the cache, short enough that a CloudCart
# doc change (the M5 freshness/re-sync story) flushes within a day. A correctness/cost
# decision (D4), set in ONE place; the lab's "Try it yourself" tunes it. INVALIDATION is
# this TTL (passive) PLUS an explicit cache.invalidate() (active, on a known doc change) —
# never a blind, never-expiring cache (brief §9).
CACHE_TTL_SECONDS = 24 * 60 * 60

# The cosine-similarity threshold a candidate must clear to be a SEMANTIC hit. Titan V2
# vectors are L2-normalized (ingest/embed.py asks for normalize=True), so a dot product IS
# cosine similarity in [-1, 1]. 0.95 is deliberately STRICT: a semantic cache's real risk
# is a FALSE HIT (serving a stored answer to a question that only LOOKS similar — a stale
# or wrong answer to the customer, brief §9), so the bar is high. The product owner, not
# only the engineer, owns this number (a false hit is a bad customer answer) — the lab's
# headline "Try it yourself" sweeps it and measures hit rate vs false hits. ONE place.
CACHE_SIMILARITY_THRESHOLD = 0.95

# Bedrock prompt caching saves up to ~90% on the CACHED input tokens (a reused prefix —
# a long system prompt or reused context — kept warm provider-side). AS OF JUNE 2026 —
# re-verify on the Bedrock pricing page. Used ONLY for the before/after cost line (a
# cached input token bills at ~10% of a normal input token); no routing decision reads it.
PROMPT_CACHE_INPUT_DISCOUNT = 0.90

# The minimum reused-prefix size (tokens) prompt caching is worth turning on for. Below a
# model's minimum cacheable length Bedrock ignores the cache point and you pay full price,
# so the cache point only earns its keep on a SUBSTANTIAL reused prefix (Relay's long
# system prompt + reused KB context). A guidance figure for the lab; re-verify per model.
PROMPT_CACHE_MIN_TOKENS = 1024

# --- The Flex service tier + the batch discount (LATENCY-TOLERANT JOBS ONLY) ----
# Bedrock service tiers (re:Invent 2025): PRIORITY (lowest latency, premium), STANDARD
# (the default, on-demand price), FLEX (-50%, higher/variable latency — for jobs that can
# WAIT). Relay's interactive ticket path runs on STANDARD; the EVAL/BACKFILL path (the
# batch re-scoring Module 13 builds on) can run on FLEX. The tier is selected PER CALL via
# converse(..., service_tier=...) (relay/llm.py, by addition through **params) — never a
# parallel client. NAMES live here so llm.py + cost_report.py + the batch job agree.
SERVICE_TIER_STANDARD = "standard"
SERVICE_TIER_FLEX = "flex"
SERVICE_TIER_PRIORITY = "priority"

# Relay's interactive default. The INTERACTIVE path is STANDARD — a customer is waiting,
# so latency matters; Flex's -50% never justifies a slower answer (brief §9, a hard
# boundary enforced in code: the worker/agent never pass service_tier=flex).
DEFAULT_SERVICE_TIER = SERVICE_TIER_STANDARD

# The Flex / batch discount applied to the per-tier price for a LATENCY-TOLERANT job. Both
# the Flex service tier AND asynchronous batch inference (CreateModelInvocationJob) price
# at ~-50% vs on-demand (AS OF JUNE 2026 — re-verify on the Bedrock pricing page). Used by
# estimate_cost(..., discount=) and the before/after report; no routing decision reads it.
FLEX_DISCOUNT = 0.50
BATCH_DISCOUNT = 0.50

# --- Batch inference (the eval-backfill path — NEVER interactive) --------------
# A batch job (Bedrock CreateModelInvocationJob) reads a JSONL of records from S3, runs
# them asynchronously at -50%, and writes the outputs back to S3. Module 12 demonstrates
# it for a small EVAL BACKFILL (re-scoring a set of tickets offline) — the exact path
# Module 13's eval harness backfills through (a one-sentence teaser, not built here). The
# S3 prefixes live under the existing data bucket (no new bucket): input under
# batch/input/, output under batch/output/. teardown.py purges both prefixes (B5).
RELAY_BATCH_INPUT_PREFIX = "batch/input/"
RELAY_BATCH_OUTPUT_PREFIX = "batch/output/"

# The IAM role name the batch job assumes to read the input / write the output in the data
# bucket (Bedrock service-role pattern). Created by setup.py from a least-privilege policy,
# deleted by teardown.py — same one-place-per-name law as the M10 component roles.
RELAY_BATCH_ROLE_NAME = "relay-batch-role"

# The minimum record count a Bedrock batch job requires (live-verified June 2026: batch
# jobs need at least 100 records). The lab's DEMO backfill is small, so setup.py pads the
# JSONL to this floor with repeated golden records — documented in lab.md so the figure is
# not a surprise. Re-verify the floor on the batch-inference docs at generation.
BATCH_MIN_RECORDS = 100


def estimate_cost_discounted(
    tier: str, input_tokens: int, output_tokens: int, *,
    discount: float = 0.0, cached_input_tokens: int = 0,
) -> float:
    """Cost in USD for ONE call, from the API usage block — with the M12 cost levers.

    Extends the M3 estimate_cost() BY ADDITION (the M3 function is untouched and still the
    baseline). Two optional levers, both from the API usage numbers, never a guess:

      - `discount` in [0, 1): a Flex/batch reduction on the WHOLE call (use
        config.FLEX_DISCOUNT / BATCH_DISCOUNT). 0.0 = on-demand Standard (the default, so
        the interactive path is unchanged).
      - `cached_input_tokens`: of the input tokens, how many were served from the Bedrock
        prompt cache (the Converse usage block reports cacheReadInputTokens). Those bill at
        (1 - PROMPT_CACHE_INPUT_DISCOUNT) of a normal input token; the rest bill in full.

    With both levers at their defaults this returns EXACTLY estimate_cost(tier, in, out) —
    so an interactive call's cost line is identical to Module 3's. Reporting only; no
    routing decision reads it.
    """
    price = PRICE_PER_1K.get(tier, PRICE_PER_1K["fast"])
    cached = max(0, min(int(cached_input_tokens), int(input_tokens)))
    uncached_input = int(input_tokens) - cached
    raw = (
        uncached_input / 1000 * price["input"]
        + cached / 1000 * price["input"] * (1 - PROMPT_CACHE_INPUT_DISCOUNT)
        + int(output_tokens) / 1000 * price["output"]
    )
    return raw * (1 - max(0.0, min(1.0, discount)))


# =============================================================================
# Module 13 ADDITIONS — the evaluation harness: the LLM-as-a-judge model, the
# Bedrock RAG-evaluation job, and the regression-gate constants.
# =============================================================================
# Added BY ADDITION in Module 13 (evaluating GenAI applications). NOTHING above this
# line changes: the tier -> inference-profile map is FROZEN at Module 3 (the CANDIDATES
# Relay evals — triage, the agent, kb.answer — run on the SAME fast/smart/vision tiers
# already in TIERS; Module 13 adds NO new tier and re-points NOTHING), the resource
# names + embedder at M4 (the Bedrock RAG-eval job scores the SAME `relay-kb` on the
# SAME pinned Titan V2 index — a swap would change what is measured), the KB/reranker at
# M5, the intake policy at M6, the tables + MCP URL at M7, the AgentCore names at M8, the
# guardrail + the ONE grounding threshold (0.8) at M9, the PII/IAM at M10, the SQS queue
# + relay-events bus + pipeline at M11, the cache table + Flex tier + batch discount at
# M12. The per-tier PRICE MAP already exists since M3 — Module 13 only APPENDS the judge's
# own price row (the judge is NOT a Relay tier; it never serves a customer).
#
# This block appends only: the judge model ID + its price, the judge-runs-on-Flex pin, the
# judge != candidate constraint (a hard invariant, asserted), the eval S3 artifact prefix
# + the eval IAM role name + the Bedrock model-evaluation job-name prefix, and the
# regression-gate thresholds (which REUSE the M9 GROUNDING_THRESHOLD — one constant, no
# divergent literal). Same containment law: ONE place each value lives, so evals/judge.py,
# evals/run_evals.py, setup.py, teardown.py, and the cdk eval-gate stage all agree.
#
# Boundary (brief §9): the judge model ID lives ONLY here (the grep gate covers it like
# every other `us.anthropic.`/`us.amazon.` ID). The JUDGE must NEVER equal a CANDIDATE
# model (self-preference bias) — Relay answers with Nova (fast/smart), so the judge is
# Anthropic Claude Haiku 4.5; crossing vendors kills the bias argument at zero cost. The
# judge rides the FLEX service tier (-50%, latency-tolerant eval/backfill ONLY — NEVER
# Relay's interactive traffic). A Bedrock model-evaluation job has NO job surcharge — you
# pay only the tokens it consumes (brief §9 / 07 §2); never invent a job price.

# --- The LLM-as-a-judge model (M13 — judge != candidate, the hard invariant) ----
# The Relay TIER the judge runs on (it is the "judge" entry appended to TIERS above — the
# judge flows through the SAME converse() call site, with its Claude Haiku 4.5 ID living in
# the one model-ID home). evals/judge.py calls converse(tier=JUDGE_TIER, service_tier=
# JUDGE_SERVICE_TIER); never a bare ID, never a parallel client.
JUDGE_TIER = "judge"

# The judge inference-profile ID, read from the ONE model-ID home (TIERS["judge"]). Anthropic
# Claude Haiku 4.5 — a DIFFERENT model family from every Relay CANDIDATE (Amazon Nova
# fast/smart/vision), so the judge cannot prefer "its own" answers (self-preference bias,
# brief §9 / arXiv 2306.05685). A `us.` cross-Region inference profile, NEVER a bare regional
# ID (the M1 trap). JUDGE_PROFILE mirrors TIERS["judge"] so a reader has a named accessor; the
# assert below proves they never drift.
JUDGE_PROFILE = TIERS["judge"]
JUDGE_ALT_PROFILE = ALT_PROFILES["judge"]

# The Relay tiers a judged ANSWER can be produced by — the CANDIDATE models. The judge
# must differ from all of them (self-preference). Kept as a frozen set so the invariant is
# checkable in code and in the smoke test, not just asserted in prose. "judge" is NOT a
# candidate — it scores, it never answers a customer.
JUDGE_CANDIDATE_TIERS: frozenset[str] = frozenset({"fast", "smart", "vision"})


def judge_profile() -> str:
    """Resolve the LLM-as-a-judge inference profile — the only lookup allowed for it.

    Mirrors tier_profile("judge"): ONE place the judge ID is read, so a grep over relay/
    finds every model ID in config.py. It also ENFORCES the judge != candidate invariant at
    resolve time — if the judge profile ever collided with a CANDIDATE tier profile (a bad
    edit), this raises instead of silently scoring with a self-preferring judge.
    """
    profile = tier_profile(JUDGE_TIER)
    candidate_profiles = {TIERS[t] for t in JUDGE_CANDIDATE_TIERS if t in TIERS}
    candidate_profiles |= {ALT_PROFILES[t] for t in JUDGE_CANDIDATE_TIERS
                           if t in ALT_PROFILES}
    if profile in candidate_profiles:
        raise ValueError(
            f"Judge profile {profile!r} equals a CANDIDATE tier profile. "
            "The judge must never be the model that produced the answer "
            "(self-preference bias). Pin the judge to a different model family."
        )
    return profile


# Claude Haiku 4.5 per-1,000-token price for the judge's cost line (AS OF JUNE 2026 —
# re-verify on the Bedrock pricing page; ~$1.00 in / ~$5.00 out per million). The judge is
# NOT a Relay tier (it never answers a customer), so its price lives in its OWN row rather
# than in PRICE_PER_1K — evals/run_evals.py reads it to add the judge's spend to the run's
# cost_cents. Reporting only; no routing decision reads it.
JUDGE_PRICE_PER_1K: dict[str, float] = {"input": 0.0010, "output": 0.0050}

# The judge runs on the FLEX service tier (-50%, re:Invent 2025) — an eval job tolerates
# latency, so it never pays the interactive premium (brief §9). A NAME (resolved against
# the M12 SERVICE_TIER_* constants), never a parallel client; evals/judge.py passes
# converse(..., service_tier=JUDGE_SERVICE_TIER). This is the ONE place the judge's tier is
# set, so judge.py, run_evals.py, and the cost line agree.
JUDGE_SERVICE_TIER = SERVICE_TIER_FLEX


def estimate_judge_cost(input_tokens: int, output_tokens: int, *,
                        discount: float = FLEX_DISCOUNT) -> float:
    """Cost in USD for ONE judge call, from its API usage block — never a guess.

    Mirrors estimate_cost_discounted() but reads the judge's OWN price row (the judge is
    not a Relay tier). `discount` defaults to FLEX_DISCOUNT because the judge runs on Flex
    (-50%); pass 0.0 to price it at on-demand. Reporting only.
    """
    raw = (
        int(input_tokens) / 1000 * JUDGE_PRICE_PER_1K["input"]
        + int(output_tokens) / 1000 * JUDGE_PRICE_PER_1K["output"]
    )
    return raw * (1 - max(0.0, min(1.0, discount)))


# --- The regression gate thresholds (REUSE the ONE M9 grounding constant) -------
# The gate (evals/run_evals.py --gate) FAILS a run when aggregate grounding falls below the
# floor OR regresses more than the allowed drop vs the committed baseline (contract 06 §2 /
# bible §3.4). The FLOOR is the SAME 0.8 GROUNDING_THRESHOLD the M9 escalation and the M14
# `relay-ops` alarm use — DEFINED ONCE at M9, REUSED here, never a divergent literal
# (gate <-> alarm <-> escalation coherence, bible §4 M9). EVAL_REGRESSION_MAX_DROP is the
# >5-pt allowed slip (0.05 on the [0,1] grounding scale) before a run counts as a
# regression even when it still clears the floor.
EVAL_GROUNDING_FLOOR = GROUNDING_THRESHOLD          # = 0.8 (the ONE M9 constant)
EVAL_REGRESSION_MAX_DROP = 0.05                     # > 5 pts vs baseline -> gate fails

# --- The fairness rubric tolerance (skill 3.4.2 — the judge-borne fairness eval) -
# The fairness eval (evals/judge.py fairness rubric) runs the SAME judge over PAIRS of twin
# tickets (same problem, different irrelevant customer attributes) and FAILS the pair when
# the two answers' scores diverge by more than this many points. 1 point on the judge's
# 1-5 scale: a tone/quality gap larger than one point across an irrelevant attribute is a
# fairness signal, not noise. ONE place, so judge.py + run_evals.py + the article agree.
FAIRNESS_MAX_SCORE_DIVERGENCE = 1

# --- Eval artifacts: the S3 prefix, the eval IAM role, the Bedrock job-name prefix --
# The Bedrock model-evaluation (RAG) job reads its dataset from / writes its report to the
# data bucket under this prefix (no new bucket — same one-bucket law as the M12 batch
# prefixes). teardown.py PURGES this prefix (B5: the eval artifacts are the only idle-billed
# thing M13 adds — small S3 objects, but the course rule is leave nothing behind).
RELAY_EVAL_PREFIX = "evals/"
RELAY_EVAL_INPUT_PREFIX = "evals/input/"
RELAY_EVAL_OUTPUT_PREFIX = "evals/output/"

# The IAM service role the Bedrock model-evaluation job assumes to read the dataset / write
# the report in the data bucket + invoke the KB and the generation model (Bedrock
# evaluation-job service-role pattern). Created by setup.py from a least-privilege policy,
# deleted by teardown.py — same one-place-per-name law as the M10 component roles and the
# M12 batch role.
RELAY_EVAL_ROLE_NAME = "relay-eval-role"

# The name PREFIX setup.py gives the Bedrock RAG-evaluation job (a unique suffix is appended
# per run; job names must be unique per account). ONE place so setup.py creates and
# teardown.py finds + cancels/deletes the job by the same prefix.
RELAY_EVAL_JOB_PREFIX = "relay-rag-eval"

# The RAG-evaluation metrics the job is configured to compute on `relay-kb` (skills 5.1.2/
# 5.1.5/5.1.6). These are the Bedrock Builtin metric IDs for a RETRIEVE-AND-GENERATE job — the
# managed analogue of the home judge's rubric:
#   Correctness     -> answer accuracy (the home judge's coverage/task_completion view)
#   Faithfulness    -> grounding (the managed analogue of the M9 contextual-grounding check)
#   Completeness    -> how fully the answer covers the question
#   CitationPrecision -> were the cited passages correctly cited (the must_cite view)
# (Live-verified June 2026 against the Bedrock RAG-evaluation metrics doc:
# Builtin.ContextRelevance / Builtin.ContextCoverage are RETRIEVE-ONLY metrics and are INVALID
# in a retrieve-and-generate job — CreateEvaluationJob rejects a mixed set. The two grounding
# views the article compares are the home judge's `grounding` and this job's Builtin.Faithfulness.)
RELAY_EVAL_RAG_METRICS: tuple[str, ...] = (
    "Builtin.Correctness",
    "Builtin.Faithfulness",
    "Builtin.Completeness",
    "Builtin.CitationPrecision",
)

# The on-disk marker setup.py writes the created RAG-eval job ARN to (git-ignored,
# account/run-specific) — teardown.py reads it to find the job to clean up. The job ARN is
# account-specific, so it is RESOLVED from this marker, NEVER a literal — the same pattern as
# the KB id, guardrail id, MCP URL, and Memory id.
RELAY_EVAL_JOB_ARN_FILE_NAME = ".eval_job_arn"
