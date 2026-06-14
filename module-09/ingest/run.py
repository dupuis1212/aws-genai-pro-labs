"""ingest/run.py — ingest CloudCart docs under one chunking strategy.

The observable result of Module 4's ingestion half:

    uv run python -m ingest.run --strategy hierarchical

reads every Markdown doc in data/docs/, chunks it with the chosen strategy, embeds
the chunks with Amazon Titan Text Embeddings V2 (1024 dims), and upserts the
vectors (+ metadata) into the S3 Vectors index `relay-docs` under a namespace for
that strategy. It prints the chunk count, the embeddings cost (from the Titan
token count — never guessed), and confirms the upsert.

Run it three times — `--strategy fixed`, `--strategy hierarchical`,
`--strategy semantic` — to populate all three namespaces, then compare them with
`uv run python compare_chunking.py`.

Resource names and the embedder come from relay.config. The bucket name is
relay-vectors-<account_id>, resolved from STS at run time. `setup.py` must have
created the bucket + index first; this script only writes vectors.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ingest import embed, upsert
from ingest.chunkers import CHUNKERS, chunk_document
from relay import config

_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = _ROOT / "data" / "docs"


def _source_uri(bucket: str, doc_path: Path) -> str:
    """The s3:// URI a doc lives at under the data bucket's docs/ prefix."""
    return f"s3://{bucket}/docs/{doc_path.name}"


def ingest_strategy(
    strategy: str,
    *,
    docs_dir: Path = DOCS_DIR,
    runtime_client=None,
    s3vectors_client=None,
    account: str | None = None,
) -> dict:
    """Chunk + embed + upsert every doc under one strategy. Returns a summary.

    Clients are injectable so the offline smoke test can drive this with a moto
    s3vectors backend and a stubbed Titan embeddings client — no credentials, no
    network. In normal use they are built from relay.config.
    """
    if strategy not in CHUNKERS:
        raise ValueError(
            f"Unknown strategy {strategy!r}. Known: {', '.join(sorted(CHUNKERS))}."
        )

    acct = account or config.account_id()
    data_bucket = config.relay_bucket(acct)
    vector_bucket = config.relay_vector_bucket(acct)
    index = config.RELAY_INDEX

    doc_paths = sorted(docs_dir.glob("*.md"))
    if not doc_paths:
        raise SystemExit(f"No Markdown docs found in {docs_dir}.")

    total_chunks = 0
    total_tokens = 0
    total_vectors = 0
    per_doc: list[tuple[str, int]] = []

    for doc_path in doc_paths:
        text = doc_path.read_text(encoding="utf-8")
        source_uri = _source_uri(data_bucket, doc_path)
        chunks = chunk_document(text, source_uri, strategy)
        if not chunks:
            continue

        result = embed.embed_texts([c.text for c in chunks], client=runtime_client)
        written = upsert.upsert_chunks(
            vector_bucket, index, strategy, doc_path.stem, chunks, result.vectors,
            client=s3vectors_client,
        )

        total_chunks += len(chunks)
        total_tokens += result.input_tokens
        total_vectors += written
        per_doc.append((doc_path.name, len(chunks)))

    return {
        "strategy": strategy,
        "index": index,
        "vector_bucket": vector_bucket,
        "documents": len(doc_paths),
        "chunks": total_chunks,
        "vectors_upserted": total_vectors,
        "embed_tokens": total_tokens,
        "embed_cost": config.estimate_embed_cost(total_tokens),
        "per_doc": per_doc,
    }


def _print_summary(summary: dict) -> None:
    print(f"\nIngested {summary['documents']} docs with the "
          f"'{summary['strategy']}' chunker:")
    for name, count in summary["per_doc"]:
        print(f"  {name:<32} {count:>3} chunks")
    print(f"\n  chunks total      : {summary['chunks']}")
    print(f"  vectors upserted  : {summary['vectors_upserted']} "
          f"-> index '{summary['index']}' "
          f"(bucket {summary['vector_bucket']})")
    print(f"  embeddings        : Titan Text Embeddings V2, "
          f"{config.EMBED_DIMENSIONS} dims")
    print(f"  embed tokens      : {summary['embed_tokens']}")
    print(f"  embed cost        : ${summary['embed_cost']:.6f} "
          f"(as of June 2026 — re-verify pricing)")
    print(f"\nUpsert confirmed under namespace '{summary['strategy']}#...'. "
          "Run the other two strategies, then compare_chunking.py.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ingest CloudCart docs into S3 Vectors under one chunking strategy."
    )
    parser.add_argument(
        "--strategy", required=True, choices=sorted(CHUNKERS),
        help="Which chunking strategy to ingest under (its own namespace).",
    )
    args = parser.parse_args(argv)

    try:
        summary = ingest_strategy(args.strategy)
    except SystemExit:
        raise
    except Exception as err:  # surface the real cause; never a silent pass
        print(f"Ingestion failed: {type(err).__name__}: {err}", file=sys.stderr)
        print("Did you run setup.py first (it creates the bucket + index)?",
              file=sys.stderr)
        return 1

    _print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
