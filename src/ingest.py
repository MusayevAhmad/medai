#!/usr/bin/env python3
"""
PDF Ingestion Pipeline

Extracts text from medical PDFs, splits into entity-aware chunks,
and optionally annotates each chunk with NER predictions.

Pipeline stages:
    1. extract_text_from_pdf  -- PDF -> raw page texts (PyMuPDF)
    2. chunk_pages            -- page texts -> sized chunks (semantic + token)
    3. annotate_chunks        -- chunks -> chunks with NER entities
    4. process_pdf            -- orchestrator combining all three

Usage:
    from pathlib import Path
    from src.ingest import process_pdf

    chunks = process_pdf(Path("data/raw_pdfs/guideline.pdf"))
"""

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import fitz  # PyMuPDF

from src.inference import Entity, MedicalNER


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class PageText:
    """Raw text extracted from a single PDF page.

    Attributes:
        page_number: 1-indexed page number inside the PDF.
        text: Full text content of the page.
    """

    page_number: int
    text: str


@dataclass
class Chunk:
    """A text chunk ready for embedding / vector storage.

    Attributes:
        text: The chunk body text.
        chunk_id: Deterministic identifier derived from source + position.
        source_file: Original PDF filename (not the full path).
        page_number: 1-indexed page the chunk originates from.
        start_char: Character offset of the chunk start within the page.
        end_char: Character offset of the chunk end within the page.
        entities: NER entities found in this chunk (empty until annotation).
        metadata: Arbitrary metadata dict; always contains
                  ``extracted_entities`` after annotation.
    """

    text: str
    chunk_id: str
    source_file: str
    page_number: int
    start_char: int
    end_char: int
    entities: List[Entity] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialise to a plain dictionary."""
        return {
            "text": self.text,
            "chunk_id": self.chunk_id,
            "source_file": self.source_file,
            "page_number": self.page_number,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "entities": [e.to_dict() for e in self.entities],
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Regex patterns for detecting section headers in medical documents
# ---------------------------------------------------------------------------

_HEADER_PATTERNS: List[re.Pattern] = [
    re.compile(r"^#{1,3}\s+.+", re.MULTILINE),             # Markdown headers
    re.compile(r"^\d+\.[\d.]*\s+[A-Z].*$", re.MULTILINE),  # Numbered sections (e.g. "1.2 Introduction")
    re.compile(r"^[A-Z][A-Z\s]{4,}$", re.MULTILINE),       # ALL-CAPS lines (≥5 chars)
]


# ---------------------------------------------------------------------------
# Stage 1 – PDF text extraction
# ---------------------------------------------------------------------------

def extract_text_from_pdf(pdf_path: Path) -> List[PageText]:
    """Extract raw text from every page of a PDF file.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of ``PageText`` objects (one per page, 1-indexed).

    Raises:
        FileNotFoundError: If *pdf_path* does not exist.
        RuntimeError: If PyMuPDF cannot open the file.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages: List[PageText] = []
    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text")
            pages.append(PageText(page_number=page_num, text=text))

    return pages


# ---------------------------------------------------------------------------
# Stage 2 – Chunking helpers
# ---------------------------------------------------------------------------

def _approximate_token_count(text: str) -> int:
    """Rough token count using whitespace splitting.

    This is intentionally simple; a proper tokeniser would give exact
    counts but would add a heavy dependency to the chunking layer.
    """
    return len(text.split())


def _generate_chunk_id(source_file: str, page_number: int, start_char: int) -> str:
    """Create a deterministic, short hash-based chunk identifier."""
    raw = f"{source_file}::{page_number}::{start_char}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def _find_sentence_boundary(text: str, target_pos: int) -> int:
    """Find the nearest sentence-ending boundary at or before *target_pos*.

    Falls back to *target_pos* if no sentence boundary is found nearby.
    """
    # Look backwards from target_pos for a sentence-ending character
    search_window = text[max(0, target_pos - 200):target_pos]
    # Prefer ". ", ".\n", "? ", "! " as boundaries
    for pattern in [". ", ".\n", "? ", "! ", "?\n", "!\n"]:
        idx = search_window.rfind(pattern)
        if idx != -1:
            # Return absolute position right after the boundary character
            return max(0, target_pos - 200) + idx + len(pattern)
    return target_pos


def chunk_text(
    text: str,
    max_tokens: int = 500,
    overlap: int = 50,
) -> List[str]:
    """Split *text* into overlapping chunks respecting approximate token limits.

    The splitter tries to break at sentence boundaries so that chunks
    remain semantically coherent.

    Args:
        text: Input text to chunk.
        max_tokens: Maximum approximate tokens per chunk.
        overlap: Number of approximate tokens to overlap between chunks.

    Returns:
        List of text chunk strings.
    """
    if not text or not text.strip():
        return []

    words = text.split()
    if len(words) <= max_tokens:
        return [text]

    chunks: List[str] = []
    start_word = 0

    while start_word < len(words):
        end_word = min(start_word + max_tokens, len(words))

        # Reconstruct the candidate chunk
        chunk_str = " ".join(words[start_word:end_word])

        # Try to snap to a sentence boundary unless we're at the very end
        if end_word < len(words):
            boundary = _find_sentence_boundary(chunk_str, len(chunk_str))
            if boundary < len(chunk_str) * 0.5:
                # Boundary too far back – just use the full window
                boundary = len(chunk_str)
            chunk_str = chunk_str[:boundary].rstrip()
            # Recount how many words we actually consumed
            consumed = len(chunk_str.split())
        else:
            consumed = end_word - start_word

        if chunk_str.strip():
            chunks.append(chunk_str.strip())

        # Advance with overlap
        start_word += max(consumed - overlap, 1)

    return chunks


def _split_on_headers(text: str) -> List[str]:
    """Split *text* into sections using header-pattern detection.

    Returns the list of section strings.  If no headers are found the
    original text is returned as a single-element list.
    """
    # Collect all header match positions
    split_positions: List[int] = []
    for pattern in _HEADER_PATTERNS:
        for match in pattern.finditer(text):
            split_positions.append(match.start())

    if not split_positions:
        return [text]

    split_positions = sorted(set(split_positions))

    sections: List[str] = []
    # Add leading text before the first header (if any)
    if split_positions[0] > 0:
        leading = text[: split_positions[0]].strip()
        if leading:
            sections.append(leading)

    for i, pos in enumerate(split_positions):
        end = split_positions[i + 1] if i + 1 < len(split_positions) else len(text)
        section = text[pos:end].strip()
        if section:
            sections.append(section)

    return sections


def chunk_pages(
    pages: List[PageText],
    source_file: str = "",
    max_tokens: int = 500,
    overlap: int = 50,
) -> List[Chunk]:
    """Convert extracted pages into sized ``Chunk`` objects.

    Strategy (two passes):
        1. Split each page on section headers.
        2. If a section exceeds *max_tokens*, subdivide it with
           ``chunk_text()``.

    Args:
        pages: Output of ``extract_text_from_pdf``.
        source_file: Filename to embed in every chunk.
        max_tokens: Maximum approximate token count per chunk.
        overlap: Token overlap between consecutive sub-chunks.

    Returns:
        Ordered list of ``Chunk`` objects.
    """
    chunks: List[Chunk] = []

    for page in pages:
        if not page.text.strip():
            continue

        # Pass 1 – section-level splits
        sections = _split_on_headers(page.text)

        for section in sections:
            # Track character offsets relative to the page
            start_char = page.text.find(section)
            if start_char == -1:
                start_char = 0

            # Pass 2 – token-level splits if section is too long
            if _approximate_token_count(section) > max_tokens:
                sub_chunks = chunk_text(section, max_tokens=max_tokens, overlap=overlap)
            else:
                sub_chunks = [section]

            running_offset = start_char
            for sub in sub_chunks:
                # Find exact position inside page text
                sub_start = page.text.find(sub, running_offset)
                if sub_start == -1:
                    sub_start = running_offset
                sub_end = sub_start + len(sub)

                chunk = Chunk(
                    text=sub,
                    chunk_id=_generate_chunk_id(source_file, page.page_number, sub_start),
                    source_file=source_file,
                    page_number=page.page_number,
                    start_char=sub_start,
                    end_char=sub_end,
                    metadata={
                        "extracted_entities": [],
                    },
                )
                chunks.append(chunk)
                running_offset = sub_start + 1  # move forward for next find

    return chunks


# ---------------------------------------------------------------------------
# Stage 3 – NER annotation
# ---------------------------------------------------------------------------

def annotate_chunks(
    chunks: List[Chunk],
    ner: MedicalNER,
    threshold: float = 0.0,
) -> List[Chunk]:
    """Run NER over each chunk and populate entity metadata.

    Args:
        chunks: Chunks to annotate (modified in-place **and** returned).
        ner: An initialised ``MedicalNER`` instance.
        threshold: Confidence threshold forwarded to ``predict_entities``.

    Returns:
        The same list of chunks, now with ``entities`` and
        ``metadata["extracted_entities"]`` populated.
    """
    texts = [c.text for c in chunks]
    batch_entities = ner.predict_batch(texts, threshold=threshold)

    for chunk, entities in zip(chunks, batch_entities):
        chunk.entities = entities
        # Store unique entity labels in metadata for downstream filtering
        chunk.metadata["extracted_entities"] = list(
            {f"{e.label}:{e.text}" for e in entities}
        )

    return chunks


# ---------------------------------------------------------------------------
# Stage 4 – Orchestrator
# ---------------------------------------------------------------------------

def process_pdf(
    pdf_path: Path,
    ner: Optional[MedicalNER] = None,
    max_tokens: int = 500,
    overlap: int = 50,
    threshold: float = 0.0,
) -> List[Chunk]:
    """End-to-end pipeline: PDF -> extracted, chunked, annotated chunks.

    Args:
        pdf_path: Path to the PDF file to ingest.
        ner: Optional ``MedicalNER`` instance.  If *None*, chunks are
             returned without entity annotations.
        max_tokens: Maximum approximate tokens per chunk.
        overlap: Token overlap between chunks.
        threshold: NER confidence threshold.

    Returns:
        List of ``Chunk`` objects ready for vector-store insertion.
    """
    pdf_path = Path(pdf_path)
    source_file = pdf_path.name

    # 1. Extract
    pages = extract_text_from_pdf(pdf_path)

    # 2. Chunk
    chunks = chunk_pages(
        pages,
        source_file=source_file,
        max_tokens=max_tokens,
        overlap=overlap,
    )

    # 3. Annotate (optional)
    if ner is not None:
        chunks = annotate_chunks(chunks, ner, threshold=threshold)

    return chunks
