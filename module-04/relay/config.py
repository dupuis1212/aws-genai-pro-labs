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
