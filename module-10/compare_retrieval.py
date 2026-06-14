"""compare_retrieval.py — benchmark managed retrieval against Module 4's DIY baseline.

The observable result of Module 5's relevance half. For each question in
data/kb_questions.json it runs FOUR retrieval configurations and prints, per
question and in aggregate, which put a relevant doc at rank 1 (top-1 hit) and how
many relevant docs it surfaced in the top-k (recall):

  1. M4 DIY        — raw S3 Vectors kNN over Module 4's hand-built index
                     (pure semantic similarity, the hierarchical namespace).
  2. KB semantic   — the managed Knowledge Base, Retrieve, overrideSearchType
                     SEMANTIC. Same vector store, now managed.
  3. KB sem+rerank — KB semantic PLUS a Bedrock reranker re-scoring the candidates.
                     Precision of the ordering, not recall.
  4. KB hybrid     — the managed KB, Retrieve, overrideSearchType HYBRID
                     (keyword + vector). NOT supported on Amazon S3 Vectors, so it
                     reports n/a here (see the NOTE on CONFIGS below); kept as the
                     exam teaching point for exact-identifier questions.

This is RAW retrieval inspected BY HAND. There is NO LLM-as-a-judge and NO RAG
evaluation harness here — that is Module 13. "Relevance" is the hand label in the
questions file (`relevant_docs`); the score is whether a relevant doc stem appears,
and at what rank. The reranker reorders what the retriever already returned — it
NEVER surfaces a doc the retriever missed, which is exactly what you should watch
for in the numbers: hybrid moves recall, the reranker moves precision.

Two questions carry an EXACT identifier on purpose (the error code `ERR-402` and
the plan name `Growth`): pure semantic similarity blurs them; hybrid search pins
them. One question is COMPOUND ("downgrade my plan AND keep my order history") — see
freshness_test.py and `relay.kb.answer(decompose=True)` for query decomposition.

Run it after setup.py has built and synced the KB (and Module 4 ingested the
DIY index):
    uv run python compare_retrieval.py
    uv run python compare_retrieval.py --top-k 5
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from ingest import embed, upsert
from relay import config, kb

_ROOT = Path(__file__).resolve().parent
QUESTIONS_FILE = _ROOT / "data" / "kb_questions.json"

# The four configurations, in display order. Keys are stable; labels are headers.
#
# NOTE (live-verified June 2026): Bedrock Knowledge Bases run HYBRID search only on
# hybrid-capable vector stores (Aurora PostgreSQL, MongoDB Atlas, the always-on
# serverless search cluster). On **Amazon S3 Vectors** — the course's ~$0-idle
# store — HYBRID is NOT supported; the API rejects it. So the live benchmark
# compares M4 DIY vs KB semantic vs KB semantic+rerank, and the `kb_hybrid` column
# reports "n/a" on S3 Vectors (it stays in the table as the exam teaching point:
# hybrid is the exact-identifier remedy, but it is a property of the VECTOR STORE,
# not a free switch — you would pick an always-on store to get it, trading ~$0 idle
# for hybrid). The reranker is the precision lever S3 Vectors DOES support.
CONFIGS = ("m4_diy", "kb_semantic", "kb_sem_rerank", "kb_hybrid")
CONFIG_LABELS = {
    "m4_diy": "M4 DIY",
    "kb_semantic": "KB sem",
    "kb_sem_rerank": "KB sem+rr",
    "kb_hybrid": "KB hyb*",
}
# Configs that need a hybrid-capable (Aurora/Mongo/serverless-search) store and
# are therefore N/A on S3 Vectors.
_HYBRID_NA_ON_S3_VECTORS = "kb_hybrid"

# Which Module 4 namespace stands in for "the DIY baseline". M4 compared three
# chunkers; hierarchical won on CloudCart's short, sectioned help docs, so it is
# the fair DIY baseline to beat here.
DIY_STRATEGY = "hierarchical"


@dataclass
class ConfigScore:
    """One configuration's result for one question.

    `na` marks a configuration that is not runnable on the current vector store
    (HYBRID on S3 Vectors) — it is shown as "n/a", not a zero, so the table never
    pretends an unsupported mode scored badly.
    """

    top1_hit: bool
    recall_hits: int
    recall_total: int
    top_stems: list[str]
    na: bool = False


def _doc_stem(source_uri_or_key: str) -> str:
    """Extract a doc stem from an s3:// URI ('.../docs/billing-plans.md') or a
    namespaced S3 Vectors key ('hierarchical#billing-plans#0')."""
    if "#" in source_uri_or_key:
        parts = source_uri_or_key.split("#")
        return parts[1] if len(parts) >= 2 else source_uri_or_key
    name = source_uri_or_key.rsplit("/", 1)[-1]
    return name[:-3] if name.endswith(".md") else name


def _score(stems: list[str], relevant: set[str]) -> ConfigScore:
    top1 = bool(stems) and stems[0] in relevant
    recall = len(relevant & set(stems))
    return ConfigScore(top1, recall, len(relevant), stems)


def _diy_stems(
    question: dict, query_vector, *, vector_bucket: str, index: str,
    top_k: int, s3vectors_client,
) -> list[str]:
    """Module 4 DIY retrieval: raw S3 Vectors kNN over the hierarchical namespace."""
    hits = upsert.query(
        vector_bucket, index, query_vector,
        top_k=top_k, strategy=DIY_STRATEGY, category=question.get("category"),
        client=s3vectors_client,
    )
    return [_doc_stem(h.key) for h in hits]


def _kb_stems(
    question: dict, *, search_type: str, rerank: bool, top_k: int, kb_id, client,
) -> list[str]:
    """Managed Knowledge Base retrieval (Retrieve), one configuration."""
    hits = kb.retrieve(
        question["question"], top_k=top_k, search_type=search_type,
        rerank=rerank, category=question.get("category"),
        kb_id=kb_id, client=client,
    )
    return [_doc_stem(h.source_uri) for h in hits]


def _na_score(relevant: set[str]) -> ConfigScore:
    """A 'not applicable' cell — the mode is unsupported on this vector store."""
    return ConfigScore(False, 0, len(relevant), [], na=True)


def score_question(
    question: dict,
    query_vector,
    *,
    vector_bucket: str,
    index: str,
    top_k: int,
    kb_id=None,
    s3vectors_client=None,
    kb_client=None,
) -> dict[str, ConfigScore]:
    """Score the four configurations for one question. Clients are injectable so
    the offline smoke test drives this with stubs (no creds, no network).

    The `kb_hybrid` column is HYBRID search, which S3 Vectors does not support;
    if the API rejects it we record an 'n/a' cell rather than a misleading zero
    (the teaching point: hybrid is a vector-store property, not a free switch)."""
    relevant = set(question["relevant_docs"])

    diy = _diy_stems(
        question, query_vector, vector_bucket=vector_bucket, index=index,
        top_k=top_k, s3vectors_client=s3vectors_client,
    )
    sem = _kb_stems(question, search_type=kb.SEARCH_SEMANTIC, rerank=False,
                    top_k=top_k, kb_id=kb_id, client=kb_client)
    semrr = _kb_stems(question, search_type=kb.SEARCH_SEMANTIC, rerank=True,
                      top_k=top_k, kb_id=kb_id, client=kb_client)

    # HYBRID: try it; on S3 Vectors the API rejects it -> n/a, not a fake zero.
    try:
        hyb_stems = _kb_stems(question, search_type=kb.SEARCH_HYBRID, rerank=False,
                              top_k=top_k, kb_id=kb_id, client=kb_client)
        hyb = _score(hyb_stems, relevant)
    except kb.KBError:
        hyb = _na_score(relevant)

    return {
        "m4_diy": _score(diy, relevant),
        "kb_semantic": _score(sem, relevant),
        "kb_sem_rerank": _score(semrr, relevant),
        "kb_hybrid": hyb,
    }


def _print_question_block(question: dict, scores: dict[str, ConfigScore]) -> None:
    print(f'\nQuestion: "{question["question"]}"')
    flags = []
    if question.get("exact_identifier"):
        flags.append(f"exact id: {question['exact_identifier']}")
    if question.get("compound"):
        flags.append("compound")
    if flags:
        print(f"  ({'; '.join(flags)})")
    header = "                 " + "".join(
        f"{CONFIG_LABELS[c]:<12}" for c in CONFIGS
    )
    print(header)

    def row(label, cell):
        return f"  {label:<15}" + "".join(f"{cell(c):<12}" for c in CONFIGS)

    print(row("top-1 hit",
              lambda c: "n/a" if scores[c].na else ("Y" if scores[c].top1_hit else ".")))
    print(row("recall",
              lambda c: "n/a" if scores[c].na
              else f"{scores[c].recall_hits}/{scores[c].recall_total}"))


def _print_aggregate(per_question: list[dict[str, ConfigScore]]) -> None:
    n = len(per_question)
    print("\n" + "=" * 64)
    print(f"Summary over {n} questions (top-1 hit rate, mean recall):")
    print("             " + "".join(f"{CONFIG_LABELS[c]:<12}" for c in CONFIGS))

    def cell(c: str, metric) -> str:
        # Average over the questions where this config is applicable (not n/a).
        live = [q[c] for q in per_question if not q[c].na]
        if not live:
            return "n/a"
        return f"{sum(metric(s) for s in live) / len(live):.2f}"

    rows = (
        ("top-1 rate", lambda s: 1.0 if s.top1_hit else 0.0),
        ("mean recall", lambda s: s.recall_hits / s.recall_total),
    )
    for label, metric in rows:
        print(f"  {label:<11}" + "".join(f"{cell(c, metric):<12}" for c in CONFIGS))

    print("\nKB hyb* = HYBRID search, which Amazon S3 Vectors does NOT support, so")
    print("it shows n/a here (you would need a hybrid-capable store — the always-on")
    print("kind the article prices out — to run it; S3 Vectors trades hybrid for ~$0")
    print("idle). Read the rest by hand: the reranker (KB sem+rr) reorders the")
    print("semantic candidates for precision but never surfaces a doc the retriever")
    print("missed. Relevance is the HAND label in data/kb_questions.json — NO")
    print("LLM-as-a-judge, NO RAG-eval harness (that is Module 13).")


def run_comparison(
    *,
    top_k: int = config.KB_DEFAULT_TOP_K,
    questions_file: Path = QUESTIONS_FILE,
    runtime_client=None,
    s3vectors_client=None,
    kb_client=None,
    kb_id=None,
    account: str | None = None,
) -> list[dict[str, ConfigScore]]:
    """Run the four-way comparison over every question. Returns per-question scores."""
    acct = account or config.account_id()
    vector_bucket = config.relay_vector_bucket(acct)
    index = config.RELAY_INDEX

    questions = json.loads(questions_file.read_text(encoding="utf-8"))
    per_question: list[dict[str, ConfigScore]] = []
    for question in questions:
        # The M4 DIY column needs the query embedded with the SAME Titan embedder
        # the index was built with. The KB columns embed server-side.
        query_vector, _ = embed.embed_one(question["question"], client=runtime_client)
        scores = score_question(
            question, query_vector,
            vector_bucket=vector_bucket, index=index, top_k=top_k,
            kb_id=kb_id, s3vectors_client=s3vectors_client, kb_client=kb_client,
        )
        _print_question_block(question, scores)
        per_question.append(scores)

    _print_aggregate(per_question)
    return per_question


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare M4 DIY vs KB semantic/hybrid/hybrid+rerank retrieval."
    )
    parser.add_argument("--top-k", type=int, default=config.KB_DEFAULT_TOP_K,
                        help=f"results per configuration (default "
                             f"{config.KB_DEFAULT_TOP_K}).")
    args = parser.parse_args(argv)

    try:
        run_comparison(top_k=args.top_k)
    except Exception as err:  # surface the cause; never a silent pass
        print(f"Comparison failed: {type(err).__name__}: {err}", file=sys.stderr)
        print("Did you run setup.py (KB synced) and Module 4's ingestion "
              "(DIY index)?", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
