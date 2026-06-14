"""ingest — Relay's hand-built RAG ingestion pipeline (Module 4).

Module 4 of AWS GenAI Pro Mastery. Relay can talk (the Module 3 LLM layer) but it
knows NOTHING about CloudCart: ask it "how do I export my order history?" and it
invents a plausible, wrong answer. This package builds retrieval BY HAND — every
bolt — before Module 5 hands it to a managed Bedrock Knowledge Base:

  - ingest.chunkers : split CloudCart Markdown docs into chunks three ways —
                      fixed_size, hierarchical (Markdown headings), semantic
                      (sentence-boundary grouping). Each chunk carries metadata
                      {category, source_uri, chunk_index} for filtering.
  - ingest.embed    : embed chunks with Amazon Titan Text Embeddings V2 (1024
                      dims) in batch. This is the course's SOLE embeddings call —
                      it returns a vector, never text. Generation stays on the
                      converse() layer.
  - ingest.upsert   : upsert vectors (+ metadata) into the S3 Vectors index
                      `relay-docs`, namespaced by chunking strategy so the three
                      can be compared.
  - ingest.run      : the CLI — `python -m ingest.run --strategy hierarchical`.

This package lives ALONGSIDE relay/ (it is tooling, not part of the runtime
agent) and reads every resource name and the embedder ID from relay.config — the
sole home of those literals. Nothing here generates an answer or cites a source;
that is Module 5. Module 4 stops at raw retrieval inspected by hand.
"""

from __future__ import annotations

from ingest.chunkers import CHUNKERS, Chunk, chunk_document

__all__ = ["CHUNKERS", "Chunk", "chunk_document"]
