"""relay/cache.py — Relay's semantic cache (Module 12).

Module 12 of AWS GenAI Pro Mastery is the token economy: it makes Relay's $/ticket
MEASURABLE and then attacks it. This file is one of the four levers — the **semantic
cache**, a DynamoDB store that answers a FREQUENT question (a near-duplicate of one Relay
already answered) WITHOUT a fresh foundation-model call, so the second-time cost is ~0.

Three caches the exam keeps distinct, and which one this is:

  - **prompt caching** = a reused INPUT PREFIX kept warm provider-side (≈ -90% on the
    cached input). It can NEVER serve a stale answer — it caches input, not output. That
    lever lives in relay/llm.py (a cache point on the system prompt, via converse(...,
    cache_prompt=True)). It is NOT this file.
  - **deterministic request hashing / result fingerprinting** = an EXACT-match key. The
    same normalized question hashes to the same key; an identical repeat is an O(1) GetItem
    hit. This file does that FIRST (lookup_exact) — it is cheap, embedding-free, and
    zero-risk (a byte-identical question must get the byte-identical stored answer).
  - **semantic caching** = match a question that is CLOSE in meaning (not identical) by
    EMBEDDING SIMILARITY + a threshold, and serve the stored answer. This file does that
    SECOND (lookup_semantic), reusing the pinned Titan V2 (1024-dim) embedder from Module 4
    — the same embedder/index contract the Knowledge Base uses, never swapped (bible §5.2).

The real RISK of a semantic cache is a FALSE HIT — serving a stored answer to a question
that only LOOKS similar, i.e. a stale or wrong answer to a customer (brief §9). So this is
never a blind cache:

  - a strict cosine THRESHOLD (config.CACHE_SIMILARITY_THRESHOLD = 0.95) gates every
    semantic hit — below it, MISS and call the model;
  - a TTL (config.CACHE_TTL_SECONDS) ages every entry out (passive invalidation — DynamoDB
    native TTL on the `expires_at` attribute), so a CloudCart doc change flushes within a
    day (the M5 freshness story);
  - invalidate() drops entries on a KNOWN change (active invalidation), so you do not wait
    for the TTL when the docs actually moved.

Storage (DynamoDB ON-DEMAND, ~$0 idle — config.RELAY_CACHE_TABLE):

    { question_hash (PK), question, embedding [1024 floats], answer (JSON), created_at,
      expires_at (epoch — DynamoDB TTL) }

For a course-scale cache the semantic lookup SCANS the (small) table and ranks by cosine
similarity in-process — honest and dependency-free. The article is explicit that a
production cache of millions of entries keys the semantic lookup off a vector store (S3
Vectors / the KB index), not a DynamoDB scan; here the point is the PATTERN and the
correctness guards, not the index engine.

No model ID and no generation here: an embedding goes through the Module 4 Titan path
(ingest.embed, the course's sole non-Converse embeddings call), the ANSWER comes from the
caller's converse()/answer() on a miss — this file only stores and matches.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Callable

import boto3

from relay import config
from relay.models import Answer


# =============================================================================
# Deterministic request hashing (skill 4.1.4) — the exact-match key.
# =============================================================================
# Normalize a question to a canonical form (lowercase, collapsed whitespace, stripped
# trailing punctuation) so trivial variations of the SAME question ("Where is my order?"
# vs "where is my order") fingerprint to the SAME key — then SHA-256 it. This is the
# "result fingerprinting / deterministic request hashing" lever: an identical (post-
# normalization) question is a zero-cost, zero-risk hit, no embedding needed.
def normalize_question(question: str) -> str:
    """Canonicalize a question for the deterministic hash: lower, collapse ws, strip punct.

    Deliberately conservative: it folds case + whitespace + trailing ?/./! only. It does
    NOT stem or drop words — two questions that differ in wording are NOT meant to collide
    here (that is the SEMANTIC lookup's job, with its similarity threshold). Over-normalizing
    would turn the exact cache into a sloppy semantic one without the threshold guard.
    """
    lowered = question.strip().lower()
    collapsed = re.sub(r"\s+", " ", lowered)
    return collapsed.rstrip("?.! ")


def question_hash(question: str) -> str:
    """The deterministic cache key: SHA-256 of the normalized question (hex)."""
    canonical = normalize_question(question)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# =============================================================================
# Cosine similarity (Titan V2 vectors are L2-normalized → dot product == cosine).
# =============================================================================
def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity in [-1, 1] between two equal-length vectors.

    Titan V2 returns L2-normalized vectors (ingest/embed.py requests normalize=True), so a
    plain dot product already IS the cosine. We still divide by the norms defensively so a
    non-normalized vector (a test fixture) scores correctly. Mismatched lengths are a
    contract break (a swapped embedder/dimension) — raise, never silently truncate.
    """
    if len(a) != len(b):
        raise ValueError(
            f"Embedding length mismatch ({len(a)} vs {len(b)}). The cache embedder must be "
            f"the pinned Titan V2 at {config.EMBED_DIMENSIONS} dims — never swap it."
        )
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


# =============================================================================
# The cache entry + the lookup result.
# =============================================================================
@dataclass
class CacheEntry:
    """One stored cache row: the question, its embedding, the answer, and its lifetime."""

    question_hash: str
    question: str
    embedding: list[float]
    answer: Answer
    created_at: str
    expires_at: int  # epoch seconds — DynamoDB native TTL ages it out (invalidation)


@dataclass
class CacheLookup:
    """The result of a cache lookup: hit/miss, how it matched, and the served answer.

    `hit` is True when an exact or semantic match cleared the bar. `match_type` is
    "exact" (deterministic hash), "semantic" (embedding similarity ≥ threshold), or "" on a
    miss. `similarity` is the cosine score for a semantic hit (1.0 for an exact hit, 0.0 on
    a miss). A hit carries the stored `answer`; a miss carries None and the caller calls the
    model, then store()s the fresh answer.
    """

    hit: bool
    match_type: str
    similarity: float
    answer: Answer | None
    question_hash: str


# =============================================================================
# The semantic cache.
# =============================================================================
class SemanticCache:
    """Relay's semantic cache over DynamoDB (on-demand) + Titan V2 embeddings.

    Wire it in FRONT of the answer path for frequent questions:

        cache = SemanticCache()
        hit = cache.lookup("where is my order 1042?")
        if hit.hit:
            return hit.answer                 # cost ≈ 0, cache_hit=True
        answer = kb.answer(question)          # a real converse() — the miss path
        cache.store(question, answer)
        return answer

    Every dependency is injectable so the smoke test runs offline (a moto DynamoDB table +
    a stubbed embedder): pass `table` (a boto3 DynamoDB Table) and `embed` (a callable
    question -> (vector, tokens)). In normal use both resolve from the boto3 session and the
    Module 4 Titan path.
    """

    def __init__(
        self,
        *,
        table=None,
        embed: Callable[[str], tuple[list[float], int]] | None = None,
        threshold: float | None = None,
        ttl_seconds: int | None = None,
    ) -> None:
        self._table = table
        self._embed = embed
        # The similarity bar and the TTL default to the config values (one place), but are
        # overridable per instance so the lab's "Try it yourself" can sweep the threshold.
        self.threshold = (
            config.CACHE_SIMILARITY_THRESHOLD if threshold is None else float(threshold)
        )
        self.ttl_seconds = (
            config.CACHE_TTL_SECONDS if ttl_seconds is None else int(ttl_seconds)
        )

    # --- lazily-resolved collaborators (kept out of __init__ so import stays offline) ---
    def _dynamo_table(self):
        if self._table is None:
            resource = boto3.resource("dynamodb", region_name=config.REGION)
            self._table = resource.Table(config.RELAY_CACHE_TABLE)
        return self._table

    def _embed_question(self, question: str) -> tuple[list[float], int]:
        """Embed a question with the pinned Titan V2 embedder (Module 4 path)."""
        if self._embed is not None:
            return self._embed(question)
        from ingest.embed import embed_one

        return embed_one(question)

    # --- reads -----------------------------------------------------------------------
    def lookup(self, question: str) -> CacheLookup:
        """Look a question up: exact hash first, then semantic similarity. Embeds once.

        1. EXACT (deterministic hashing): GetItem by the normalized-question hash. A live,
           unexpired exact match is a zero-risk hit (similarity 1.0) — no embedding needed,
           so we try it first and skip the embed call entirely on a hit.
        2. SEMANTIC: embed the question, scan the (small) cache for the most-similar live
           entry, and HIT only if cosine ≥ threshold. Below the bar -> MISS (call the
           model). Expired entries are ignored (belt-and-braces with DynamoDB TTL, which is
           eventually-consistent — an entry can linger minutes past expiry).
        """
        qhash = question_hash(question)

        exact = self._get_live_entry(qhash)
        if exact is not None:
            return CacheLookup(
                hit=True, match_type="exact", similarity=1.0,
                answer=exact.answer, question_hash=qhash,
            )

        # Semantic: embed and rank the live entries by cosine similarity.
        query_vec, _tokens = self._embed_question(question)
        best: CacheEntry | None = None
        best_score = -1.0
        for entry in self._iter_live_entries():
            score = cosine_similarity(query_vec, entry.embedding)
            if score > best_score:
                best_score, best = score, entry

        if best is not None and best_score >= self.threshold:
            return CacheLookup(
                hit=True, match_type="semantic", similarity=best_score,
                answer=best.answer, question_hash=qhash,
            )
        return CacheLookup(
            hit=False, match_type="", similarity=max(best_score, 0.0),
            answer=None, question_hash=qhash,
        )

    def _get_live_entry(self, qhash: str) -> CacheEntry | None:
        """GetItem by hash; return the entry only if it exists AND has not expired."""
        table = self._dynamo_table()
        item = table.get_item(Key={config.CACHE_KEY: qhash}).get("Item")
        if not item:
            return None
        entry = _entry_from_item(item)
        return entry if not _is_expired(entry) else None

    def _iter_live_entries(self):
        """Yield every unexpired cache entry (a Scan — fine for a course-scale cache).

        The article is explicit that a production cache keys the semantic lookup off a vector
        store, not a Scan; for the lab's handful of frequent questions a Scan is honest and
        dependency-free. Expired rows are skipped (DynamoDB TTL is eventually consistent).
        """
        table = self._dynamo_table()
        scan_kwargs: dict[str, Any] = {}
        while True:
            page = table.scan(**scan_kwargs)
            for item in page.get("Items", []):
                entry = _entry_from_item(item)
                if entry is not None and not _is_expired(entry):
                    yield entry
            start_key = page.get("LastEvaluatedKey")
            if not start_key:
                break
            scan_kwargs["ExclusiveStartKey"] = start_key

    # --- writes ----------------------------------------------------------------------
    def store(self, question: str, answer: Answer, *, embedding: list[float] | None = None,
              now: dt.datetime | None = None) -> CacheEntry:
        """Store {question_hash, question, embedding, answer, created_at, expires_at}.

        Called on a MISS, after the model produced a fresh answer, so the NEXT near-duplicate
        question hits the cache. PutItem on the hash key is idempotent (a re-store overwrites
        the same row). `embedding` is computed via Titan when not supplied (so an exact
        re-store can reuse the vector). The row's `expires_at` = now + ttl_seconds drives
        DynamoDB's native TTL — passive invalidation.
        """
        now = now or dt.datetime.now(dt.timezone.utc)
        if embedding is None:
            embedding, _tokens = self._embed_question(question)
        qhash = question_hash(question)
        expires_at = int(now.timestamp()) + self.ttl_seconds
        entry = CacheEntry(
            question_hash=qhash,
            question=question,
            embedding=list(embedding),
            answer=answer,
            created_at=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            expires_at=expires_at,
        )
        self._dynamo_table().put_item(Item=_item_from_entry(entry))
        return entry

    def invalidate(self, question: str) -> bool:
        """ACTIVE invalidation: drop a cached question NOW (a known doc change made it stale).

        Returns True if a row was deleted. The lab calls this when a CloudCart doc the answer
        cited has changed — you do not wait for the TTL when you KNOW the answer is stale.
        Complements the passive TTL; together they are the cache's invalidation story (never
        a blind, never-expiring cache, brief §9).
        """
        qhash = question_hash(question)
        table = self._dynamo_table()
        existed = "Item" in table.get_item(Key={config.CACHE_KEY: qhash})
        table.delete_item(Key={config.CACHE_KEY: qhash})
        return existed


# =============================================================================
# Item <-> entry serialization (DynamoDB wants Decimal, not float).
# =============================================================================
def _item_from_entry(entry: CacheEntry) -> dict[str, Any]:
    """Serialize a CacheEntry to a DynamoDB item. Floats -> Decimal (DynamoDB's number type).

    The embedding (1024 floats) is stored as a list of Decimal; the answer is stored as its
    JSON string (so the nested Citation list round-trips through the frozen Answer schema on
    read, exactly like relay-tickets stores a TicketRecord). expires_at stays an int so
    DynamoDB's TTL reads it as an epoch.
    """
    return {
        config.CACHE_KEY: entry.question_hash,
        "question": entry.question,
        "embedding": [Decimal(str(x)) for x in entry.embedding],
        "answer": entry.answer.model_dump_json(),
        "created_at": entry.created_at,
        config.CACHE_TTL_ATTRIBUTE: entry.expires_at,
    }


def _entry_from_item(item: dict[str, Any]) -> CacheEntry | None:
    """Rebuild a CacheEntry from a DynamoDB item, or None if the item is malformed.

    The answer round-trips through the frozen Answer schema (Answer.model_validate_json), so a
    corrupt row is dropped (None) rather than crashing a lookup — the cache degrades to a miss,
    never a 500. Decimals come back as Decimal; we cast the embedding to float for the cosine.
    """
    try:
        return CacheEntry(
            question_hash=str(item[config.CACHE_KEY]),
            question=str(item.get("question", "")),
            embedding=[float(x) for x in item.get("embedding", [])],
            answer=Answer.model_validate_json(item["answer"]),
            created_at=str(item.get("created_at", "")),
            expires_at=int(item.get(config.CACHE_TTL_ATTRIBUTE, 0)),
        )
    except (KeyError, ValueError, TypeError):
        return None


def _is_expired(entry: CacheEntry, *, now: dt.datetime | None = None) -> bool:
    """True if the entry's TTL has passed (we never serve an expired entry, even if DynamoDB
    has not swept it yet — TTL deletion is eventually consistent)."""
    now = now or dt.datetime.now(dt.timezone.utc)
    return entry.expires_at <= int(now.timestamp())
