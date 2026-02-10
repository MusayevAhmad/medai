#!/usr/bin/env python3
"""
Multimodal PDF Ingestion Pipeline (Phase 4)

Extends the base text ingestion with:
    - Table extraction: PyMuPDF find_tables() → Markdown → chunk_type="table"
    - Figure extraction: PyMuPDF get_images() → saved PNGs → chunk_type="figure"
    - Unified pipeline: text + tables + figures all stored in Qdrant

Usage:
    from src.multimodal_ingest import process_pdf_multimodal

    chunks = process_pdf_multimodal(
        Path("data/raw_pdfs/guideline.pdf"),
        ner=ner,
        figures_dir=Path("data/figures"),
    )
"""

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF

from src.inference import Entity, MedicalNER
from src.ingest import Chunk, PageText, annotate_chunks, chunk_pages, extract_text_from_pdf

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Table extraction
# ---------------------------------------------------------------------------

def _table_to_markdown(table: "fitz.table.Table") -> str:
    """Convert a PyMuPDF Table object to a Markdown string.

    Args:
        table: A table object from ``page.find_tables()``.

    Returns:
        Markdown-formatted table string. Returns empty string if table
        has fewer than 2 rows.
    """
    try:
        data = table.extract()
    except Exception:
        return ""

    if not data or len(data) < 2:
        return ""

    # Clean cell values
    def _clean(cell: Optional[str]) -> str:
        if cell is None:
            return ""
        return str(cell).replace("\n", " ").strip()

    # Build Markdown
    headers = [_clean(c) for c in data[0]]
    rows = [[_clean(c) for c in row] for row in data[1:]]

    md_lines = []
    md_lines.append("| " + " | ".join(headers) + " |")
    md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        # Pad row to match header length
        padded = row + [""] * max(0, len(headers) - len(row))
        md_lines.append("| " + " | ".join(padded[:len(headers)]) + " |")

    return "\n".join(md_lines)


def extract_tables_from_pdf(
    pdf_path: Path,
    min_rows: int = 2,
    min_cols: int = 2,
) -> List[Chunk]:
    """Extract tables from a PDF and return them as Markdown chunks.

    Args:
        pdf_path: Path to the PDF file.
        min_rows: Minimum rows for a table to be included.
        min_cols: Minimum columns for a table to be included.

    Returns:
        List of Chunk objects with ``chunk_type: "table"`` in metadata.
    """
    chunks: List[Chunk] = []
    source_file = pdf_path.name

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as e:
        logger.warning("Failed to open %s for table extraction: %s", pdf_path, e)
        return []

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        page_number = page_idx + 1

        try:
            tables = page.find_tables()
        except Exception:
            continue

        for table_idx, table in enumerate(tables):
            try:
                data = table.extract()
            except Exception:
                continue

            if not data or len(data) < min_rows:
                continue
            if data[0] and len(data[0]) < min_cols:
                continue

            markdown = _table_to_markdown(table)
            if not markdown or len(markdown.strip()) < 20:
                continue

            # Build chunk ID
            chunk_id = hashlib.md5(
                f"{source_file}:table:p{page_number}:t{table_idx}".encode()
            ).hexdigest()

            # Try to find a caption near the table (look for "Table N" text)
            caption = _find_table_caption(page, table, table_idx)

            text = ""
            if caption:
                text = f"{caption}\n\n{markdown}"
            else:
                text = f"Table (page {page_number}):\n\n{markdown}"

            chunks.append(Chunk(
                text=text,
                chunk_id=chunk_id,
                source_file=source_file,
                page_number=page_number,
                start_char=0,
                end_char=len(text),
                metadata={
                    "chunk_type": "table",
                    "table_index": table_idx,
                    "num_rows": len(data),
                    "num_cols": len(data[0]) if data[0] else 0,
                    "caption": caption or "",
                },
            ))

    doc.close()
    return chunks


def _find_table_caption(
    page: fitz.Page,
    table: "fitz.table.Table",
    table_idx: int,
) -> str:
    """Try to find a 'Table N: ...' caption near a table on the page.

    Searches for text matching 'Table [number]' patterns in the page text,
    returns the first matching line as the caption.
    """
    try:
        page_text = page.get_text("text")
    except Exception:
        return ""

    # Find "Table N" patterns
    pattern = re.compile(r"(Table\s+\d+[\.:]\s*.{0,120})", re.IGNORECASE)
    matches = pattern.findall(page_text)

    if matches and table_idx < len(matches):
        return matches[table_idx].strip()
    elif matches:
        return matches[0].strip()

    return ""


# ---------------------------------------------------------------------------
# Figure extraction
# ---------------------------------------------------------------------------

def extract_figures_from_pdf(
    pdf_path: Path,
    figures_dir: Path,
    min_width: int = 100,
    min_height: int = 100,
    min_size_bytes: int = 5000,
) -> List[Chunk]:
    """Extract images/figures from a PDF and save them as PNG files.

    Args:
        pdf_path: Path to the PDF file.
        figures_dir: Directory to save extracted figure images.
        min_width: Minimum image width in pixels to include.
        min_height: Minimum image height in pixels to include.
        min_size_bytes: Minimum raw image size in bytes.

    Returns:
        List of Chunk objects with ``chunk_type: "figure"`` in metadata,
        including the saved image path.
    """
    chunks: List[Chunk] = []
    source_file = pdf_path.name
    figures_dir.mkdir(parents=True, exist_ok=True)

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as e:
        logger.warning("Failed to open %s for figure extraction: %s", pdf_path, e)
        return []

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        page_number = page_idx + 1

        image_list = page.get_images(full=True)

        for img_idx, img_info in enumerate(image_list):
            xref = img_info[0]

            try:
                base_image = doc.extract_image(xref)
            except Exception:
                continue

            if not base_image:
                continue

            width = base_image.get("width", 0)
            height = base_image.get("height", 0)
            image_bytes = base_image.get("image", b"")

            # Filter out tiny icons, decorations, etc.
            if width < min_width or height < min_height:
                continue
            if len(image_bytes) < min_size_bytes:
                continue

            # Save the image
            ext = base_image.get("ext", "png")
            stem = pdf_path.stem
            img_filename = f"{stem}_p{page_number}_img{img_idx}.{ext}"
            img_path = figures_dir / img_filename

            try:
                img_path.write_bytes(image_bytes)
            except Exception as e:
                logger.warning("Failed to save image %s: %s", img_path, e)
                continue

            # Find a caption near this image
            caption = _find_figure_caption(page, img_idx)

            chunk_id = hashlib.md5(
                f"{source_file}:figure:p{page_number}:i{img_idx}".encode()
            ).hexdigest()

            text = caption if caption else f"Figure from {source_file}, page {page_number}"

            chunks.append(Chunk(
                text=text,
                chunk_id=chunk_id,
                source_file=source_file,
                page_number=page_number,
                start_char=0,
                end_char=len(text),
                metadata={
                    "chunk_type": "figure",
                    "image_path": str(img_path),
                    "image_width": width,
                    "image_height": height,
                    "caption": caption or "",
                },
            ))

    doc.close()
    return chunks


def _find_figure_caption(page: fitz.Page, img_idx: int) -> str:
    """Try to find a 'Figure N: ...' caption on the page."""
    try:
        page_text = page.get_text("text")
    except Exception:
        return ""

    pattern = re.compile(r"(Fig(?:ure|\.)\s*\d+[\.:]\s*.{0,150})", re.IGNORECASE)
    matches = pattern.findall(page_text)

    if matches and img_idx < len(matches):
        return matches[img_idx].strip()
    elif matches:
        return matches[0].strip()

    return ""


# ---------------------------------------------------------------------------
# Unified multimodal pipeline
# ---------------------------------------------------------------------------

def process_pdf_multimodal(
    pdf_path: Path,
    ner: Optional[MedicalNER] = None,
    max_tokens: int = 500,
    overlap: int = 50,
    threshold: float = 0.0,
    figures_dir: Optional[Path] = None,
    extract_tables: bool = True,
    extract_figures: bool = True,
) -> List[Chunk]:
    """Process a PDF extracting text, tables, and figures.

    This is the multimodal replacement for ``ingest.process_pdf()``.
    It produces three types of chunks:
        - ``chunk_type: "text"`` — standard text chunks (same as Phase 1)
        - ``chunk_type: "table"`` — tables converted to Markdown
        - ``chunk_type: "figure"`` — figure captions with image paths

    Args:
        pdf_path: Path to the PDF file.
        ner: NER model for entity annotation (optional).
        max_tokens: Max tokens per text chunk.
        overlap: Token overlap between text chunks.
        threshold: NER confidence threshold.
        figures_dir: Directory to save figure images.
        extract_tables: Whether to extract tables.
        extract_figures: Whether to extract figures.

    Returns:
        Combined list of text, table, and figure chunks.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    all_chunks: List[Chunk] = []

    # 1. Text chunks (existing pipeline)
    pages = extract_text_from_pdf(pdf_path)
    text_chunks = chunk_pages(pages, source_file=pdf_path.name,
                              max_tokens=max_tokens, overlap=overlap)
    # Tag as text
    for chunk in text_chunks:
        chunk.metadata["chunk_type"] = "text"
    all_chunks.extend(text_chunks)

    # 2. Table chunks
    if extract_tables:
        table_chunks = extract_tables_from_pdf(pdf_path)
        all_chunks.extend(table_chunks)
        if table_chunks:
            logger.info("Extracted %d tables from %s", len(table_chunks), pdf_path.name)

    # 3. Figure chunks
    if extract_figures and figures_dir is not None:
        figure_chunks = extract_figures_from_pdf(pdf_path, figures_dir=figures_dir)
        all_chunks.extend(figure_chunks)
        if figure_chunks:
            logger.info("Extracted %d figures from %s", len(figure_chunks), pdf_path.name)

    # 4. NER annotation on all chunks
    if ner is not None:
        all_chunks = annotate_chunks(all_chunks, ner=ner, threshold=threshold)

    logger.info(
        "Processed %s: %d text, %d tables, %d figures",
        pdf_path.name,
        sum(1 for c in all_chunks if c.metadata.get("chunk_type") == "text"),
        sum(1 for c in all_chunks if c.metadata.get("chunk_type") == "table"),
        sum(1 for c in all_chunks if c.metadata.get("chunk_type") == "figure"),
    )

    return all_chunks
