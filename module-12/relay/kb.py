"""relay/kb.py — Relay's managed-RAG layer over the Bedrock Knowledge Base.

Module 5 of AWS GenAI Pro Mastery. Module 4 built retrieval BY HAND — chunk,
embed, upsert, kNN — and you know every bolt. This module hands that pipeline to a
**Bedrock Knowledge Base** (`relay-kb`) that ingests the CloudCart docs, keeps the
index in sync, reranks, and returns CITED answers. The KB writes into its OWN
dedicated S3 Vectors index `relay-kb-docs` (config.RELAY_KB_INDEX), NOT Module 4's
`relay-docs` DIY index: a Bedrock KB writes its own Bedrock-schema vector metadata
and cannot read Module 4's raw vectors, so the two stay separate (see
config.RELAY_KB_INDEX for the full rationale). Both are S3 Vectors in the same
vector bucket — the same Titan V2 (1024-dim) contract, never a new always-on
search cluster.

This file exposes the TWO standardized access patterns the exam tests (skill
1.5.6, access-pattern slice):

    retrieve(query, ...)  -> list[Retrieved]   via the Retrieve API
        You get the raw passages back and control the prompt + model yourself.
        Hybrid search and the reranker are toggled here.

    answer(query, ...)    -> Answer            via the RetrieveAndGenerate API
        The KB retrieves AND generates a grounded answer with turnkey citations,
        on the SMART tier (resolved through relay.config — never a hard-coded ID).
        We map the SDK response into the frozen Answer / Citation schemas.

Two more exam levers live here, both as config on the retrieval request, not as
extra services:

  - HYBRID SEARCH (skill 1.5.4): `search_type="HYBRID"` combines keyword (BM25-style)
    and vector matching. Pure semantic search misses EXACT identifiers — a SKU, a
    CloudCart error code like ERR-402 — because "402" is just a token with no
    semantic neighborhood. Hybrid nails it. `overrideSearchType` is the field.
  - RERANKER (skill 1.5.4): a Bedrock reranker (Cohere Rerank 3.5,
    `cohere.rerank-v3-5:0`, in relay.config; Amazon Rerank is the unavailable
    alternative in this Region) re-scores the retriever's candidates and reorders
    them for PRECISION. It does
    NOT improve recall — it cannot surface a doc the retriever never returned.
    Coverage first (hybrid, sync), order second.

And QUERY DECOMPOSITION (skill 1.5.5): answer(decompose=True) turns on the KB's
`QUERY_DECOMPOSITION` orchestration, which breaks a compound question ("how do I
downgrade my plan AND keep my order history?") into sub-queries, retrieves for
each, and synthesizes one answer. It is a flag on the orchestration config.

On `grounded`: at Module 5 it is the heuristic `bool(answer.citations)` — an answer
that cited at least one retrieved source. The REAL contextual grounding check (a
Bedrock guardrail that verifies the answer is supported by the retrieved context,
and escalates when it is not) arrives in Module 9; it writes the SAME `Answer.grounded`
field, recomputed. The field name and type do not change.

No generation here bypasses the model-ID containment law: the answer model is the
SMART tier resolved via relay.config.model_arn(), and the reranker ARN is built
from relay.config.RERANK_MODEL_ID. There is no direct single-prompt model call and
no hard-coded us./global. profile ID in this file (the grep gate proves it) — all
generation is RetrieveAndGenerate, all retrieval is Retrieve.

Run it on one question (after setup.py has built and synced the KB):
    uv run python -m relay.kb "How do I change my CloudCart subscription plan?"

It prints the generated answer, a numbered list of citations (source_uri + snippet),
and `grounded: True`.
"""

from __future__ import annotations

import random
import sys
import time
from dataclasses import dataclass, field

import boto3
from botocore.exceptions import (
    BotoCoreError,
    ClientError,
    NoCredentialsError,
    ProfileNotFound,
)

from relay import config
from relay.models import Answer, Citation

REGION = config.REGION

# --- Bounded retry for transient KB-plane errors -----------------------------
# Mirrors relay/llm.py's policy: the data plane (Retrieve / RetrieveAndGenerate)
# throttles and, right after a fresh ingestion job or KB create, occasionally
# returns a transient "model identifier is invalid" / 5xx blip that clears on
# retry (a Bedrock-side propagation lag, not a real validation error). We retry
# those with exponential backoff + jitter; everything else (a genuine bad KB id,
# HYBRID-on-S3-Vectors) raises immediately so the real cause surfaces — no silent
# swallow. The offline stubs hit the happy path, so retry never fires in tests.
_KB_MAX_RETRIES = 3
_KB_BACKOFF_BASE = 0.6
_KB_BACKOFF_CAP = 6.0

# Codes worth retrying on the agent-runtime plane. The intermittent post-ingestion
# "The provided model identifier is invalid" surfaces as a ValidationException, so
# we retry that code ONLY when its message carries the transient signature; a real
# validation error (e.g. HYBRID unsupported) is matched out below and raised.
_KB_RETRYABLE_CODES = frozenset({
    "ThrottlingException",
    "TooManyRequestsException",
    "ServiceUnavailableException",
    "InternalServerException",
    "ModelNotReadyException",
})
_KB_TRANSIENT_VALIDATION = "model identifier is invalid"


def _kb_should_retry(err: ClientError) -> bool:
    code = err.response["Error"]["Code"]
    if code in _KB_RETRYABLE_CODES:
        return True
    # The only ValidationException we retry is the transient post-ingestion model
    # propagation blip; a real validation error (HYBRID on S3 Vectors, bad KB id)
    # is NOT retried — it raises immediately with its true message.
    if code == "ValidationException":
        msg = err.response["Error"].get("Message", "")
        return _KB_TRANSIENT_VALIDATION in msg
    return False


def _kb_backoff_sleep(attempt: int) -> None:
    """Sleep exp(attempt) with full jitter before retry `attempt` (1-indexed)."""
    delay = min(_KB_BACKOFF_CAP, _KB_BACKOFF_BASE * (2 ** (attempt - 1)))
    time.sleep(random.uniform(0, delay))


def _call_with_retry(fn, *, op: str, kb: str):
    """Run a KB-plane call with bounded retry on transient errors. Returns its result.

    `fn` is a zero-arg callable making one API call. On a retryable transient it
    backs off and retries up to _KB_MAX_RETRIES; on a non-retryable ClientError it
    wraps the real cause in KBError (never silent).
    """
    last: ClientError | None = None
    for attempt in range(_KB_MAX_RETRIES + 1):
        try:
            return fn()
        except ClientError as err:
            last = err
            if attempt < _KB_MAX_RETRIES and _kb_should_retry(err):
                _kb_backoff_sleep(attempt + 1)
                continue
            raise KBError(
                f"{op} failed on KB {kb}: "
                f"{err.response['Error']['Code']} — "
                f"{err.response['Error']['Message']}",
                cause=err,
            ) from err
    # Unreachable (loop either returns or raises), but keeps type checkers happy.
    raise KBError(f"{op} failed on KB {kb}", cause=last)


# --- Search type: the hybrid lever -------------------------------------------
# The two values the Retrieve / RetrieveAndGenerate API accepts for
# `overrideSearchType`. SEMANTIC is pure vector similarity (Module 4's world);
# HYBRID adds keyword matching for exact tokens.
#
# IMPORTANT (live-verified June 2026): Bedrock Knowledge Bases support HYBRID
# search ONLY on vector stores with a filterable text field — the managed-search /
# relational vector stores (Aurora PostgreSQL, MongoDB Atlas, and the always-on
# serverless search cluster the article discusses as the costly alternative). On
# **Amazon S3 Vectors** (the course's vector store — idle ~$0, no always-on
# cluster) HYBRID is NOT supported; the API returns "HYBRID search type is not
# supported for search operation on index ...". So the lab's DEFAULT here is
# SEMANTIC. HYBRID stays a first-class option (you flip it on a hybrid-capable
# store) and a teaching point: hybrid is the remedy for exact-identifier misses
# (SKUs, error codes), but it is a property of the VECTOR STORE, not a free switch
# — choosing S3 Vectors trades hybrid for ~$0 idle. The reranker DOES work over S3
# Vectors' semantic candidates and is the precision lever the lab uses instead.
# (Article T5.3 / Exam corner make the vector-store ↔ hybrid trade-off explicit.)
SEARCH_SEMANTIC = "SEMANTIC"
SEARCH_HYBRID = "HYBRID"
_SEARCH_TYPES = frozenset({SEARCH_SEMANTIC, SEARCH_HYBRID})


class KBError(RuntimeError):
    """Raised when a Knowledge Base call cannot be completed.

    Carries the underlying AWS error so the failure is debuggable, never silent.
    A common first-run cause is "KB not found / not yet synced" — run setup.py.
    """

    def __init__(self, message: str, *, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.__cause__ = cause


@dataclass
class Retrieved:
    """One passage the Retrieve API returned: text, where it came from, its score.

    `score` is a RETRIEVAL internal — it lives here so retrieve() callers can rank
    and inspect, and it is what the reranker reorders by. It is deliberately NOT
    on the frozen Citation schema (Citation has only source_uri + snippet): a
    score is not part of the answer contract.
    """

    text: str
    source_uri: str
    score: float
    metadata: dict = field(default_factory=dict)


# --- Knowledge Base ID resolution --------------------------------------------
# setup.py records the created KB id (and its data-source id) so kb.py can find
# them without an env var. An explicit RELAY_KB_ID env var wins if set.
from pathlib import Path  # noqa: E402  (kept near its use)

_KB_ID_FILE = Path(__file__).resolve().parent.parent / ".kb_id"


def resolve_kb_id(kb_id: str | None = None) -> str:
    """Find the Knowledge Base id created by setup.py.

    Order: explicit argument, RELAY_KB_ID env var, then the .kb_id file setup.py
    writes. If none exists, raise with the exact fix — no silent fallback.
    """
    import os

    if kb_id:
        return kb_id
    env = os.environ.get("RELAY_KB_ID")
    if env:
        return env.strip()
    if _KB_ID_FILE.exists():
        recorded = _KB_ID_FILE.read_text(encoding="utf-8").strip()
        if recorded:
            return recorded
    raise KBError(
        f"No Knowledge Base id found. Create and sync '{config.RELAY_KB_NAME}' first:\n"
        "    uv run python setup.py\n"
        "It creates the KB, attaches the S3 data source, runs the first ingestion\n"
        "job to COMPLETE, and records the id in .kb_id. Or set RELAY_KB_ID."
    )


def _agent_runtime_client():
    """A bedrock-agent-runtime client — the Retrieve / RetrieveAndGenerate plane."""
    return boto3.client("bedrock-agent-runtime", region_name=REGION)


# --- Shared retrieval-config builder -----------------------------------------
def _vector_search_config(
    *,
    top_k: int,
    search_type: str,
    rerank: bool,
    category: str | None,
    account: str | None,
) -> dict:
    """Assemble the vectorSearchConfiguration shared by Retrieve and RAG.

    This one helper builds: numberOfResults (top-k), overrideSearchType (the
    hybrid lever), an optional metadata filter on `category` (multi-tenant /
    scoped retrieval, skill 1.4.2 recall), and an optional Bedrock reranker block.
    Field names are the live API names (verified against the botocore model).
    """
    if search_type not in _SEARCH_TYPES:
        raise ValueError(
            f"Unknown search_type {search_type!r}. Use {SEARCH_SEMANTIC!r} or "
            f"{SEARCH_HYBRID!r}."
        )

    vsc: dict = {
        "numberOfResults": top_k,
        "overrideSearchType": search_type,
    }

    if category is not None:
        # KB metadata filter (different shape from S3 Vectors' raw filter): the
        # KB indexes doc front-matter as filterable metadata, so we can scope
        # retrieval to one CloudCart category — the multi-tenant lever.
        vsc["filter"] = {"equals": {"key": "category", "value": category}}

    if rerank:
        # Re-score the retriever's candidates with a Bedrock reranker and reorder
        # for precision. numberOfRerankedResults <= top_k so this REORDERS, never
        # silently widens k. The reranker ARN comes from relay.config (only the
        # model ID is pinned; the ARN is assembled there).
        vsc["rerankingConfiguration"] = {
            "type": "BEDROCK_RERANKING_MODEL",
            "bedrockRerankingConfiguration": {
                "modelConfiguration": {
                    "modelArn": config.rerank_model_arn(),
                },
                "numberOfRerankedResults": min(
                    config.RERANK_NUMBER_OF_RESULTS, top_k
                ),
            },
        }
    return vsc


def _reference_to_retrieved(ref: dict) -> Retrieved:
    """Map one API retrieval result / retrievedReference into a Retrieved."""
    content = ref.get("content", {})
    text = content.get("text", "")
    location = ref.get("location", {})
    source_uri = location.get("s3Location", {}).get("uri", "")
    score = float(ref.get("score", 0.0)) if "score" in ref else 0.0
    return Retrieved(
        text=text,
        source_uri=source_uri,
        score=score,
        metadata=ref.get("metadata", {}),
    )


def retrieve(
    query: str,
    *,
    top_k: int = config.KB_DEFAULT_TOP_K,
    search_type: str = SEARCH_SEMANTIC,
    rerank: bool = False,
    category: str | None = None,
    kb_id: str | None = None,
    account: str | None = None,
    client=None,
) -> list[Retrieved]:
    """Retrieve passages from the Knowledge Base (the Retrieve access pattern).

    You get the raw chunks back and own the prompt + model yourself — this is the
    pattern Relay switches to when the agent arrives (Module 7's KB-search tool is
    built on exactly this). Module 5 uses it for the benchmark (compare hybrid
    vs semantic vs hybrid+rerank) and answer() uses RetrieveAndGenerate instead
    when it wants turnkey citations.

    Args:
        query: the user's question.
        top_k: how many passages to return.
        search_type: SEARCH_SEMANTIC (default — the only mode S3 Vectors supports;
            see the SEARCH_* note above) or SEARCH_HYBRID (hybrid-capable stores).
        rerank: re-score and reorder the candidates with a Bedrock reranker.
        category: scope retrieval to one CloudCart category (metadata filter).
        kb_id: the KB id (resolved from .kb_id / RELAY_KB_ID when omitted).

    Returns the top-k passages as Retrieved (text + source_uri + score).
    """
    client = client or _agent_runtime_client()
    kb = resolve_kb_id(kb_id)
    vsc = _vector_search_config(
        top_k=top_k, search_type=search_type, rerank=rerank,
        category=category, account=account,
    )
    response = _call_with_retry(
        lambda: client.retrieve(
            knowledgeBaseId=kb,
            retrievalQuery={"text": query},
            retrievalConfiguration={"vectorSearchConfiguration": vsc},
        ),
        op="Retrieve", kb=kb,
    )

    return [_reference_to_retrieved(r) for r in response.get("retrievalResults", [])]


def _citations_from_rag(response: dict) -> list[Citation]:
    """Flatten a RetrieveAndGenerate `citations` block into Citation objects.

    The SDK nests citations -> retrievedReferences -> {content.text, location.
    s3Location.uri}. We map each reference to a Citation(source_uri, snippet),
    de-duplicating on (source_uri, snippet) so the same passage cited for two
    spans is listed once. The frozen Citation schema has NO score — we drop it.
    """
    seen: set[tuple[str, str]] = set()
    citations: list[Citation] = []
    for citation in response.get("citations", []):
        for ref in citation.get("retrievedReferences", []):
            snippet = ref.get("content", {}).get("text", "")
            source_uri = ref.get("location", {}).get("s3Location", {}).get("uri", "")
            key = (source_uri, snippet)
            if key in seen:
                continue
            seen.add(key)
            citations.append(Citation(source_uri=source_uri, snippet=snippet))
    return citations


def answer(
    query: str,
    *,
    top_k: int = config.KB_DEFAULT_TOP_K,
    search_type: str = SEARCH_SEMANTIC,
    rerank: bool = True,
    decompose: bool = False,
    category: str | None = None,
    kb_id: str | None = None,
    account: str | None = None,
    client=None,
    grounding_check: bool = False,
    safety_client=None,
) -> Answer:
    """Answer a question from the docs, with citations (the RetrieveAndGenerate pattern).

    The KB retrieves AND generates in one call, returning turnkey citations — the
    reason Relay uses this pattern here rather than Retrieve + its own prompt. The
    answer model is the SMART tier, resolved through relay.config.model_arn()
    (never a hard-coded ID). The result is mapped into the frozen Answer schema.

    Args:
        query: the user's question.
        top_k: passages to retrieve before generating.
        search_type: SEARCH_SEMANTIC (default — the only mode S3 Vectors supports)
            or SEARCH_HYBRID (hybrid-capable vector stores).
        rerank: re-score + reorder retrieved passages before generation.
        decompose: turn on QUERY_DECOMPOSITION orchestration — break a compound
            question into sub-queries, retrieve per sub-query, synthesize one
            answer (skill 1.5.5). Use it for "X and also Y" tickets.
        category: scope retrieval to one CloudCart category (metadata filter).
        grounding_check: Module 9 — run the contextual grounding check (a Bedrock
            guardrail, via relay.safety) over the generated answer against its
            retrieved context. When True, `grounded` is RECOMPUTED from the real
            grounding/relevance scores (below relay.config.GROUNDING_THRESHOLD ->
            grounded=False, and Relay escalates). When False (the default, so M5
            behaviour and tests are unchanged) `grounded` stays the bool(citations)
            heuristic. Same Answer field, different computation — no new field.
        safety_client: optional bedrock-runtime client for the grounding check (for
            tests / dependency injection); built on demand otherwise.

    Returns an Answer(text, citations, grounded).
    """
    client = client or _agent_runtime_client()
    kb = resolve_kb_id(kb_id)

    vsc = _vector_search_config(
        top_k=top_k, search_type=search_type, rerank=rerank,
        category=category, account=account,
    )

    kb_config: dict = {
        "knowledgeBaseId": kb,
        # The generation model ARN — the SMART tier, from the sole model-ID home.
        "modelArn": config.model_arn(config.KB_ANSWER_TIER, account=account),
        "retrievalConfiguration": {"vectorSearchConfiguration": vsc},
    }
    if decompose:
        # Query decomposition is an ORCHESTRATION setting: the KB rewrites the
        # compound query into sub-queries before retrieval. QUERY_DECOMPOSITION is
        # the only value the API exposes for this transformation.
        kb_config["orchestrationConfiguration"] = {
            "queryTransformationConfiguration": {"type": "QUERY_DECOMPOSITION"}
        }

    response = _call_with_retry(
        lambda: client.retrieve_and_generate(
            input={"text": query},
            retrieveAndGenerateConfiguration={
                "type": "KNOWLEDGE_BASE",
                "knowledgeBaseConfiguration": kb_config,
            },
        ),
        op="RetrieveAndGenerate", kb=kb,
    )

    text = response.get("output", {}).get("text", "")
    citations = _citations_from_rag(response)
    # Module 5 grounding heuristic: an answer that cited >=1 retrieved source is
    # treated as grounded. Module 9 (grounding_check=True) RECOMPUTES this SAME field
    # from a real contextual grounding check — name and type unchanged.
    grounded = bool(citations)
    if grounding_check:
        grounded = _grounding_check(query, text, citations, client=safety_client)
    return Answer(text=text, citations=citations, grounded=grounded)


def _grounding_check(query: str, text: str, citations: list[Citation], *,
                     client=None) -> bool:
    """Recompute `grounded` with the Module 9 contextual grounding check (skill 3.1.3).

    Builds the grounding CONTEXT from the citations' snippets (the retrieved passages the
    answer is supposed to be supported by), then asks relay.safety.grounding_check whether
    the answer stays supported (grounding) and on-topic (relevance) above the configured
    threshold. Below it, the answer is treated as UNGROUNDED and the caller (the agent /
    intake flow) escalates instead of shipping a possibly hallucinated promise.

    Imported locally so relay.kb stays importable with no guardrail set up (the grounding
    check is opt-in); a missing guardrail surfaces a clear SafetyError only when used.
    """
    from relay import safety

    context = "\n\n".join(c.snippet for c in citations if c.snippet)
    result = safety.grounding_check(text, context, query, client=client)
    return result.grounded


def _print_answer(result: Answer) -> None:
    print(result.text.strip())
    print()
    if result.citations:
        print(f"Citations ({len(result.citations)}):")
        for i, citation in enumerate(result.citations, 1):
            snippet = citation.snippet.strip().replace("\n", " ")
            if len(snippet) > 200:
                snippet = snippet[:197] + "..."
            print(f"  [{i}] {citation.source_uri}")
            print(f"      {snippet}")
    else:
        print("Citations: none.")
    print(f"\ngrounded: {result.grounded}")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print(
            'Usage: uv run python -m relay.kb "<your question>"\n'
            'Example: uv run python -m relay.kb '
            '"How do I change my CloudCart subscription plan?"',
            file=sys.stderr,
        )
        return 1

    try:
        result = answer(argv[0])
    except KBError as err:
        print(f"Knowledge Base call failed: {err}", file=sys.stderr)
        return 1
    except (NoCredentialsError, ProfileNotFound, BotoCoreError) as err:
        print(f"AWS credentials/config problem: {err}\n"
              "Set AWS_PROFILE=aws-genai-pro and run from us-east-1.",
              file=sys.stderr)
        return 1

    _print_answer(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
