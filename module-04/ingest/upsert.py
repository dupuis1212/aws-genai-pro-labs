"""ingest/upsert.py — write vectors into the S3 Vectors index, and query them.

Amazon S3 Vectors (GA December 2025) is Relay's vector store: it bills ~$0 idle
(storage + per-query, no 24/7 cluster), which is why the course chooses it over a
provisioned, always-on search cluster (~$174/month billed around the clock — the
#1 cost trap in pre-2026 RAG tutorials; see the article for the full comparison).
This module is the thin boto3 layer over the two S3 Vectors calls the lab needs:

  - upsert : PutVectors — write {key, embedding, metadata} into index `relay-docs`.
  - query  : QueryVectors — k-nearest-neighbour search for a query embedding,
             returning the top-k chunks with their distance and metadata.

Every vector key is NAMESPACED by chunking strategy:
    <strategy>#<doc_stem>#<chunk_index>      e.g. "hierarchical#orders-export#2"
so all three strategies coexist in ONE index and compare_chunking.py can query
each namespace independently (filtered on the `strategy` metadata key). That is
how `run.py --strategy X` re-ingests under a distinct namespace without a second
index.

Resource names and the distance metric come from relay.config — never typed here.
We address the index by (vectorBucketName, indexName); both are accepted by every
S3 Vectors call, so no ARN bookkeeping is needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import boto3

from relay import config
from ingest.chunkers import Chunk


@dataclass
class Retrieved:
    """One kNN hit: the chunk key, its cosine distance, and its metadata."""

    key: str
    distance: float
    metadata: dict

    @property
    def similarity(self) -> float:
        """Cosine SIMILARITY (1 - distance) — the intuitive 'closeness' score.

        S3 Vectors returns cosine DISTANCE (0 = identical, 2 = opposite). Most
        people reason in similarity, so we expose both: similarity = 1 - distance.
        """
        return 1.0 - self.distance


def _client():
    """An s3vectors client in the course Region (us-east-1)."""
    return boto3.client("s3vectors", region_name=config.REGION)


def vector_key(strategy: str, doc_stem: str, chunk_index: int) -> str:
    """The namespaced key for one chunk's vector (strategy#doc#index)."""
    return f"{strategy}#{doc_stem}#{chunk_index}"


def upsert_chunks(
    vector_bucket: str,
    index_name: str,
    strategy: str,
    doc_stem: str,
    chunks: list[Chunk],
    embeddings: list[list[float]],
    *,
    client=None,
) -> int:
    """Write one document's chunks (+ embeddings + metadata) into the index.

    Each vector carries the canonical metadata {category, source_uri, chunk_index}
    plus the `strategy` it was produced by (so a query can filter to one strategy)
    and a short `snippet` for human inspection in compare_chunking.py. Returns the
    number of vectors written. Idempotent on key: re-upserting the same key
    overwrites it, so re-running ingestion never duplicates.
    """
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings must be the same length")

    client = client or _client()
    vectors = []
    for chunk, embedding in zip(chunks, embeddings):
        metadata = dict(chunk.metadata())
        metadata["strategy"] = strategy
        # A short, human-readable preview for the comparison table. Kept off the
        # filterable set (see setup.py's nonFilterableMetadataKeys) so it does not
        # bloat the index's filter structures.
        metadata["snippet"] = chunk.text[:240]
        vectors.append({
            "key": vector_key(strategy, doc_stem, chunk.chunk_index),
            "data": {"float32": [float(x) for x in embedding]},
            "metadata": metadata,
        })

    # S3 Vectors PutVectors accepts up to 500 vectors per call; the lab's
    # per-document batches are far smaller, but we chunk to be safe.
    written = 0
    for start in range(0, len(vectors), 500):
        client.put_vectors(
            vectorBucketName=vector_bucket,
            indexName=index_name,
            vectors=vectors[start:start + 500],
        )
        written += len(vectors[start:start + 500])
    return written


def query(
    vector_bucket: str,
    index_name: str,
    query_embedding: list[float],
    *,
    top_k: int = 3,
    strategy: str | None = None,
    category: str | None = None,
    client=None,
) -> list[Retrieved]:
    """k-nearest-neighbour search for a query embedding. Returns top-k Retrieved.

    `strategy` filters to one chunking namespace (so the three can be compared in
    one index); `category` filters by document category (the metadata-filtering
    skill — e.g. only `billing` docs). Both use S3 Vectors metadata filters; an
    AND of the two when both are given.
    """
    client = client or _client()

    filters = []
    if strategy is not None:
        filters.append({"strategy": strategy})
    if category is not None:
        filters.append({"category": category})
    metadata_filter = None
    if len(filters) == 1:
        metadata_filter = filters[0]
    elif len(filters) > 1:
        metadata_filter = {"$and": filters}

    request: dict = {
        "vectorBucketName": vector_bucket,
        "indexName": index_name,
        "topK": top_k,
        "queryVector": {"float32": [float(x) for x in query_embedding]},
        "returnMetadata": True,
        "returnDistance": True,
    }
    if metadata_filter is not None:
        request["filter"] = metadata_filter

    response = client.query_vectors(**request)
    return [
        Retrieved(
            key=hit["key"],
            distance=float(hit.get("distance", 0.0)),
            metadata=hit.get("metadata", {}),
        )
        for hit in response.get("vectors", [])
    ]
