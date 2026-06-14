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
#
# "auto" is NOT a profile — it is the router's request to PICK a tier at call
# time (see relay/llm.py route()); it never appears as a key here.
#
# Nova Micro / Nova 2 Lite via their `us.` profiles; both are reachable. Nova 2
# Lite also has a `global.` profile (wider Region pool) — recorded but the `us.`
# profile is the default so the cost line and Region story stay predictable.
Tier = Literal["fast", "smart", "frontier"]

TIERS: dict[str, str] = {
    "fast": "us.amazon.nova-micro-v1:0",
    "smart": "us.amazon.nova-2-lite-v1:0",
    # Reference / Try-it-yourself only. A frontier model is overkill for support
    # triage and answers; it lives here so the cost-delta exercise has a real ID,
    # not so production traffic routes to it.
    "frontier": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
}

# Alternate (wider-pool) profiles, documented but not the default. Cross-Region
# inference already routes the `us.` profiles across us-east/us-west; the
# `global.` profile widens the pool further for the smart tier.
ALT_PROFILES: dict[str, str] = {
    "smart": "global.amazon.nova-2-lite-v1:0",
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
