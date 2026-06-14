"""ingest/chunkers.py — the three chunking strategies, compared in the lab.

Chunking is the lever nobody tunes. A foundation model does not digest a 40-page
doc in one shot, and retrieval returns a CHUNK, not a file — so HOW you cut the
docs decides what retrieval can ever find. Module 4 builds three strategies by
hand and the lab measures which wins on CloudCart's docs:

  - fixed_size : cut every N characters with an overlap window. Dead simple, works
                 anywhere, ignores document structure (can split mid-sentence,
                 mid-heading). The baseline.
  - hierarchical : split on Markdown headings (#, ##, ###). Each section becomes a
                 chunk whose text keeps its heading trail, so a chunk is a
                 self-contained answer unit. Best for short, well-sectioned help
                 articles like CloudCart's.
  - semantic   : group consecutive SENTENCES until a size budget, breaking only on
                 sentence boundaries (never mid-sentence). A cheap, dependency-free
                 stand-in for embedding-similarity grouping: it respects meaning
                 units without an extra model call. Best for continuous prose.

EVERY chunker is DETERMINISTIC and pure: same input doc -> same chunks, byte for
byte. That is what makes the offline smoke test possible and the comparison fair
(only the strategy changes, never the randomness).

Each chunk carries the canonical metadata the rest of the pipeline filters on:
    {category, source_uri, chunk_index}
(06 §2 / bible §3.3 vector-metadata canon). `category` comes from the doc's front
matter; `source_uri` is the doc's s3:// location; `chunk_index` is the chunk's
position within its document under that strategy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

# --- Tunable defaults (the lab's "Try it yourself" changes these) -------------
# Fixed-size: characters per chunk and the overlap window. ~1,200 chars ~= 300
# tokens — a comfortable retrieval unit. Overlap keeps a sentence that straddles a
# cut from being lost to both chunks.
FIXED_CHUNK_CHARS = 1200
FIXED_OVERLAP_CHARS = 200

# Semantic: the soft character budget per chunk. We add whole sentences until the
# next one would exceed this, then start a new chunk — so chunks land near the
# budget without ever splitting a sentence.
SEMANTIC_TARGET_CHARS = 900


@dataclass
class Chunk:
    """One unit of retrievable text plus the metadata retrieval filters on.

    `text` is what gets embedded. `metadata` is the canonical vector metadata
    {category, source_uri, chunk_index}; `heading` is kept for human inspection in
    compare_chunking.py (it is folded into metadata at upsert time).
    """

    text: str
    category: str
    source_uri: str
    chunk_index: int
    heading: str = ""

    def metadata(self) -> dict[str, str | int]:
        """The vector metadata stored alongside the embedding (canonical keys)."""
        meta: dict[str, str | int] = {
            "category": self.category,
            "source_uri": self.source_uri,
            "chunk_index": self.chunk_index,
        }
        if self.heading:
            meta["heading"] = self.heading
        return meta


@dataclass
class Document:
    """A parsed CloudCart help-center doc: front matter + Markdown body."""

    category: str
    title: str
    source_uri: str
    body: str


# --- Front-matter parsing (tiny, dependency-free) -----------------------------
# CloudCart docs start with a minimal YAML-ish front-matter block:
#   ---
#   title: Exporting your order history
#   category: orders
#   ---
# We parse only `title` and `category` (the metadata the lab filters on); no YAML
# dependency is pulled in for two keys.
_FRONT_MATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def parse_document(text: str, source_uri: str) -> Document:
    """Split a raw doc into front matter (title/category) and Markdown body."""
    match = _FRONT_MATTER_RE.match(text)
    category = "uncategorized"
    title = ""
    body = text
    if match:
        for line in match.group(1).splitlines():
            if ":" in line:
                key, _, value = line.partition(":")
                key, value = key.strip().lower(), value.strip()
                if key == "category":
                    category = value or category
                elif key == "title":
                    title = value
        body = text[match.end():]
    return Document(category=category, title=title, source_uri=source_uri,
                    body=body.strip())


# --- Strategy 1: fixed-size with overlap --------------------------------------
def fixed_size(
    doc: Document,
    *,
    chunk_chars: int = FIXED_CHUNK_CHARS,
    overlap_chars: int = FIXED_OVERLAP_CHARS,
) -> list[Chunk]:
    """Cut the body every `chunk_chars`, stepping back `overlap_chars` each time.

    The baseline. It ignores structure entirely — a heading or a sentence can land
    mid-chunk. Overlap is the one knob that matters: it keeps a fact that straddles
    a boundary retrievable from at least one chunk. The lab's first "Try it
    yourself" sweeps this overlap and re-measures recall.
    """
    if overlap_chars >= chunk_chars:
        raise ValueError("overlap_chars must be smaller than chunk_chars")

    text = _normalize_ws(doc.body)
    step = chunk_chars - overlap_chars
    chunks: list[Chunk] = []
    start = 0
    index = 0
    while start < len(text):
        piece = text[start:start + chunk_chars].strip()
        if piece:
            chunks.append(Chunk(
                text=piece, category=doc.category, source_uri=doc.source_uri,
                chunk_index=index,
            ))
            index += 1
        start += step
    return chunks


# --- Strategy 2: hierarchical on Markdown headings ----------------------------
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def hierarchical(doc: Document) -> list[Chunk]:
    """Split on Markdown headings; each section is one chunk that keeps its trail.

    Each chunk's text is prefixed with the heading path (e.g. "Orders > Exporting
    your order history") so the chunk is a self-contained answer unit even out of
    context. For short, well-sectioned help articles, this beats fixed-size: a
    section is exactly the unit a customer's question maps to. The heading path is
    also stored in metadata for inspection.
    """
    lines = doc.body.splitlines()
    chunks: list[Chunk] = []
    index = 0
    heading_stack: list[tuple[int, str]] = []
    current_heading = doc.title or "(intro)"
    buffer: list[str] = []

    def flush() -> None:
        nonlocal index
        body_text = _normalize_ws("\n".join(buffer))
        if not body_text:
            return
        trail = " > ".join(h for _, h in heading_stack) or current_heading
        text = f"{trail}\n\n{body_text}" if trail else body_text
        chunks.append(Chunk(
            text=text, category=doc.category, source_uri=doc.source_uri,
            chunk_index=index, heading=trail,
        ))
        index += 1

    for line in lines:
        m = _HEADING_RE.match(line.strip())
        if m:
            # New section: flush the previous one, then update the heading trail.
            flush()
            buffer = []
            level = len(m.group(1))
            title = m.group(2).strip()
            # Pop deeper-or-equal headings, then push this one (build the path).
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            current_heading = title
        else:
            buffer.append(line)
    flush()
    return chunks


# --- Strategy 3: semantic (sentence-boundary grouping) ------------------------
# A pragmatic, deterministic stand-in for embedding-similarity grouping: pack
# whole sentences up to a budget, breaking only on sentence boundaries. It respects
# meaning units (never splits a sentence) without an extra model call — which keeps
# the chunker pure and the offline test exact.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def semantic(doc: Document, *, target_chars: int = SEMANTIC_TARGET_CHARS) -> list[Chunk]:
    """Group consecutive sentences up to a budget; never split mid-sentence.

    Best for continuous prose where headings are sparse: a fixed cut would slice
    through an explanation, while this keeps each explanation whole. It is the
    "respect the meaning boundary" strategy without the cost/nondeterminism of a
    live embedding pass.
    """
    text = _normalize_ws(doc.body)
    # Strip Markdown heading markers so a heading line joins its following prose
    # instead of becoming its own micro-sentence.
    text = _strip_heading_markers(text)
    sentences = [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]

    chunks: list[Chunk] = []
    index = 0
    buffer: list[str] = []
    size = 0
    for sentence in sentences:
        if buffer and size + len(sentence) + 1 > target_chars:
            chunks.append(Chunk(
                text=" ".join(buffer), category=doc.category,
                source_uri=doc.source_uri, chunk_index=index,
            ))
            index += 1
            buffer, size = [], 0
        buffer.append(sentence)
        size += len(sentence) + 1
    if buffer:
        chunks.append(Chunk(
            text=" ".join(buffer), category=doc.category,
            source_uri=doc.source_uri, chunk_index=index,
        ))
    return chunks


# --- Helpers ------------------------------------------------------------------
def _normalize_ws(text: str) -> str:
    """Collapse 3+ blank lines to a paragraph break and strip trailing spaces."""
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _strip_heading_markers(text: str) -> str:
    """Remove leading '#' markers from heading lines (keep the heading words)."""
    return re.sub(r"(?m)^#{1,6}\s+", "", text)


# --- Registry + the one entry point the pipeline uses -------------------------
# The three strategies by canonical name. `run.py --strategy {fixed|hierarchical|
# semantic}` indexes by these exact keys; they also become the namespace prefix on
# each vector key so the three can coexist in one index and be compared.
CHUNKERS: dict[str, Callable[[Document], list[Chunk]]] = {
    "fixed": fixed_size,
    "hierarchical": hierarchical,
    "semantic": semantic,
}


def chunk_document(text: str, source_uri: str, strategy: str) -> list[Chunk]:
    """Parse a raw doc and chunk it with the named strategy. The pipeline's seam.

    Raises a clear error for an unknown strategy (never a silent default) so a typo
    in `--strategy` surfaces immediately.
    """
    try:
        chunker = CHUNKERS[strategy]
    except KeyError:
        raise ValueError(
            f"Unknown strategy {strategy!r}. Known: {', '.join(sorted(CHUNKERS))}."
        ) from None
    doc = parse_document(text, source_uri)
    return chunker(doc)
