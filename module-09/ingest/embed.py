"""ingest/embed.py — embed chunks with Amazon Titan Text Embeddings V2.

This file holds the ONE `invoke_model` call the whole course tolerates. Amazon
Bedrock's Converse API cannot produce embeddings — it returns text — so the
embeddings path uses bedrock-runtime `invoke_model` directly. The contract is
narrow and load-bearing:

  - it calls EXACTLY ONE model: Amazon Titan Text Embeddings V2
    (`amazon.titan-embed-text-v2:0`), pinned in relay.config (the sole home of
    model IDs). The dimension is pinned at 1024 — the vector contract the index
    `relay-docs` is built on. Never silently swap the embedder or the dimension.
  - it returns a VECTOR, never text. There is no generation here. Every generation
    call in Relay still goes through the converse() layer (see relay/llm.py).

Titan V2 embeds ONE text per request (it has no server-side batch endpoint on the
synchronous path), so "batch" here means we loop, amortizing client setup and
summing the token count for one honest cost line. For a course-scale corpus
(dozens of docs x three strategies) that is cents; the article explains the
Lambda/async batch pattern for a corpus of millions.

The Nova successor `amazon.nova-2-multimodal-embeddings-v1:0` EXISTS as of June
2026, but it stays OUT of this index: re-embedding the corpus would produce vectors the
existing 1024-dim relay-docs index cannot compare against, and the managed Knowledge
Base at Module 5 and the semantic cache at Module 12 reuse THAT index. Changing the
embedder would invalidate all three. So Titan V2 is
the pinned default and the only model this file knows.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import boto3

from relay import config


@dataclass
class EmbedResult:
    """A batch of embeddings plus the token count for the cost line."""

    vectors: list[list[float]]
    input_tokens: int

    @property
    def count(self) -> int:
        return len(self.vectors)


def _runtime_client():
    """A bedrock-runtime client in the course Region (us-east-1)."""
    return boto3.client("bedrock-runtime", region_name=config.REGION)


def embed_one(text: str, *, client=None) -> tuple[list[float], int]:
    """Embed a single text with Titan V2. Returns (vector, input_token_count).

    The request body pins the dimension and asks Titan to L2-normalize the vector,
    which is what makes cosine similarity well-behaved on the S3 Vectors side.
    """
    client = client or _runtime_client()
    body = json.dumps({
        "inputText": text,
        "dimensions": config.EMBED_DIMENSIONS,
        "normalize": True,
    })
    response = client.invoke_model(
        modelId=config.EMBED_MODEL_ID,
        body=body,
        accept="application/json",
        contentType="application/json",
    )
    payload = json.loads(response["body"].read())
    vector = payload["embedding"]
    if len(vector) != config.EMBED_DIMENSIONS:
        # Fail loudly: a dimension drift silently corrupts the index. (06 §2 pins
        # 1024; the index, the KB at M5, and the cache at M12 all depend on it.)
        raise ValueError(
            f"Titan returned {len(vector)} dims, expected "
            f"{config.EMBED_DIMENSIONS}. Refusing to write a mismatched vector."
        )
    return vector, int(payload.get("inputTextTokenCount", 0))


def embed_texts(texts: list[str], *, client=None) -> EmbedResult:
    """Embed a list of texts (the chunking pass), summing the token count.

    "Batch" on the synchronous Titan path is a loop: Titan V2 embeds one input per
    call. We reuse one client and sum tokens so the pipeline reports a single
    honest embeddings cost. (For a corpus of millions, the article shows the
    asynchronous Lambda-fan-out pattern; the lab corpus does not need it.)
    """
    client = client or _runtime_client()
    vectors: list[list[float]] = []
    total_tokens = 0
    for text in texts:
        vector, tokens = embed_one(text, client=client)
        vectors.append(vector)
        total_tokens += tokens
    return EmbedResult(vectors=vectors, input_tokens=total_tokens)
