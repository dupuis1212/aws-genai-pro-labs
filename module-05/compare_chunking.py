"""compare_chunking.py — rank the three chunking strategies on real questions.

The observable result of Module 4's RETRIEVAL half. For each question in
data/questions.json it:

  1. embeds the question with Titan Text Embeddings V2 (the SAME embedder the docs
     were embedded with — you must embed the query and the corpus with one model);
  2. runs a top-k kNN query against the S3 Vectors index `relay-docs`, ONCE PER
     STRATEGY (filtered to that strategy's namespace);
  3. checks each hit against the question's `relevant_docs` — the HAND-LABELLED
     ground truth — and prints which strategy put a relevant chunk at rank 1
     (top-1 hit) and how many of the relevant docs it surfaced in the top-k
     (top-k recall), with the cosine similarity of the best hit.

This is RAW retrieval inspected BY HAND. There is NO LLM-as-a-judge and NO RAG
evaluation harness here — that is Module 13. The "relevance" is the human label in
the questions file; the score is plain cosine similarity. Module 5 will add a
managed Knowledge Base, hybrid search, and a reranker and benchmark them against
exactly this DIY baseline.

One question carries an exact identifier (the error code `ERR-402`) on purpose:
watch pure semantic similarity struggle to pin an exact token a keyword search
would nail instantly — the motivation for the hybrid search Module 5 adds.

Run it after ingesting all three strategies:
    uv run python -m ingest.run --strategy fixed
    uv run python -m ingest.run --strategy hierarchical
    uv run python -m ingest.run --strategy semantic
    uv run python compare_chunking.py
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from ingest import embed, upsert
from ingest.chunkers import CHUNKERS
from relay import config

_ROOT = Path(__file__).resolve().parent
QUESTIONS_FILE = _ROOT / "data" / "questions.json"
STRATEGIES = ("fixed", "hierarchical", "semantic")


@dataclass
class StrategyScore:
    """One strategy's result for one question."""

    top1_hit: bool
    recall_hits: int
    recall_total: int
    best_similarity: float
    top_keys: list[str]


def _doc_stem_of(hit: upsert.Retrieved) -> str:
    """Extract the source doc stem from a namespaced key (strategy#stem#index)."""
    parts = hit.key.split("#")
    return parts[1] if len(parts) >= 2 else hit.key


def score_question(
    question: dict,
    query_embedding: list[float],
    *,
    vector_bucket: str,
    index: str,
    top_k: int,
    s3vectors_client=None,
) -> dict[str, StrategyScore]:
    """kNN each strategy for one question; score against the hand-labelled docs."""
    relevant = set(question["relevant_docs"])
    category = question.get("category")  # optional metadata filter
    scores: dict[str, StrategyScore] = {}

    for strategy in STRATEGIES:
        hits = upsert.query(
            vector_bucket, index, query_embedding,
            top_k=top_k, strategy=strategy, category=category,
            client=s3vectors_client,
        )
        hit_stems = [_doc_stem_of(h) for h in hits]
        top1_hit = bool(hit_stems) and hit_stems[0] in relevant
        recall_hits = len(relevant & set(hit_stems))
        best_sim = hits[0].similarity if hits else 0.0
        scores[strategy] = StrategyScore(
            top1_hit=top1_hit,
            recall_hits=recall_hits,
            recall_total=len(relevant),
            best_similarity=best_sim,
            top_keys=hit_stems,
        )
    return scores


def _print_question_block(question: dict, scores: dict[str, StrategyScore]) -> None:
    print(f'\nQuestion: "{question["question"]}"')
    if question.get("note"):
        print(f"  ({question['note']})")
    header = "                 " + "".join(f"{s:<14}" for s in STRATEGIES)
    print(header)

    def row(label: str, cell) -> str:
        return f"  {label:<15}" + "".join(f"{cell(s):<14}" for s in STRATEGIES)

    print(row("top-1 hit", lambda s: "Y" if scores[s].top1_hit else "."))
    print(row("top-k recall",
              lambda s: f"{scores[s].recall_hits}/{scores[s].recall_total}"))
    print(row("best sim", lambda s: f"{scores[s].best_similarity:.3f}"))


def _print_aggregate(per_question: list[dict[str, StrategyScore]]) -> None:
    n = len(per_question)
    print("\n" + "=" * 60)
    print(f"Summary over {n} questions (top-1 hit rate, mean top-k recall):")
    print("             " + "".join(f"{s:<14}" for s in STRATEGIES))
    for label, fn in (
        ("top-1 rate", lambda s: sum(q[s].top1_hit for q in per_question) / n),
        ("mean recall", lambda s: sum(
            q[s].recall_hits / q[s].recall_total for q in per_question) / n),
    ):
        print(f"  {label:<11}" + "".join(f"{fn(s):<14.2f}" for s in STRATEGIES))
    print("\nNote: relevance is the HAND-LABELLED ground truth in "
          "data/questions.json;")
    print("scores are plain cosine similarity. No LLM-as-a-judge, no RAG eval "
          "harness")
    print("(that is Module 13). Inspect the chunks yourself — that is the point.")


def run_comparison(
    *,
    top_k: int = 3,
    questions_file: Path = QUESTIONS_FILE,
    runtime_client=None,
    s3vectors_client=None,
    account: str | None = None,
) -> list[dict[str, StrategyScore]]:
    """Embed every question and score the three strategies. Returns per-question."""
    acct = account or config.account_id()
    vector_bucket = config.relay_vector_bucket(acct)
    index = config.RELAY_INDEX

    questions = json.loads(questions_file.read_text(encoding="utf-8"))
    per_question: list[dict[str, StrategyScore]] = []
    for question in questions:
        query_vector, _ = embed.embed_one(question["question"], client=runtime_client)
        scores = score_question(
            question, query_vector,
            vector_bucket=vector_bucket, index=index, top_k=top_k,
            s3vectors_client=s3vectors_client,
        )
        _print_question_block(question, scores)
        per_question.append(scores)

    _print_aggregate(per_question)
    return per_question


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare fixed/hierarchical/semantic retrieval on test questions."
    )
    parser.add_argument("--top-k", type=int, default=3,
                        help="kNN neighbours to retrieve per strategy (default 3).")
    args = parser.parse_args(argv)

    # Sanity: the three namespaces must exist. We do not check the index here
    # (the query will surface a clear error); we just remind on failure.
    assert set(STRATEGIES) == set(CHUNKERS), "strategy set drifted from chunkers"

    try:
        run_comparison(top_k=args.top_k)
    except Exception as err:  # surface the cause; never a silent pass
        print(f"Comparison failed: {type(err).__name__}: {err}", file=sys.stderr)
        print("Did you ingest all three strategies first?\n"
              "  uv run python -m ingest.run --strategy fixed\n"
              "  uv run python -m ingest.run --strategy hierarchical\n"
              "  uv run python -m ingest.run --strategy semantic",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
