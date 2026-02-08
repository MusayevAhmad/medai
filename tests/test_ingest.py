#!/usr/bin/env python3
"""
Unit and integration tests for the PDF ingestion pipeline.

All tests use programmatically generated PDFs so no external files are needed.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

import fitz  # PyMuPDF

from src.inference import Entity, MedicalNER
from src.ingest import (
    Chunk,
    PageText,
    annotate_chunks,
    chunk_pages,
    chunk_text,
    extract_text_from_pdf,
    process_pdf,
    _approximate_token_count,
    _split_on_headers,
)


# ---------------------------------------------------------------------------
# Fixtures – programmatic PDF generation
# ---------------------------------------------------------------------------

SAMPLE_MEDICAL_TEXT_PAGE1 = (
    "1. Introduction\n\n"
    "This clinical practice guideline provides evidence-based recommendations "
    "for the management of Type 2 Diabetes Mellitus in adult patients. "
    "The guideline covers diagnosis, pharmacological treatment, lifestyle "
    "modifications, and monitoring strategies.\n\n"
    "2. Diagnosis\n\n"
    "Type 2 Diabetes is diagnosed when fasting plasma glucose is greater "
    "than or equal to 126 mg/dL on two separate occasions. An HbA1c level "
    "of 6.5% or higher also confirms the diagnosis. Patients presenting "
    "with classic symptoms of hyperglycemia including polyuria, polydipsia, "
    "and unexplained weight loss may be diagnosed with a single random "
    "plasma glucose of 200 mg/dL or higher."
)

SAMPLE_MEDICAL_TEXT_PAGE2 = (
    "3. Pharmacological Treatment\n\n"
    "Metformin remains the first-line pharmacological therapy for Type 2 "
    "Diabetes. Starting dose is typically 500 mg once or twice daily, "
    "titrated up to a maximum of 2000 mg per day based on tolerability "
    "and glycemic control. Common side effects include gastrointestinal "
    "disturbances such as nausea, diarrhea, and abdominal discomfort.\n\n"
    "For patients who do not achieve adequate glycemic control with "
    "Metformin monotherapy, second-line agents include Sulfonylureas, "
    "DPP-4 inhibitors (e.g. Sitagliptin), SGLT2 inhibitors (e.g. "
    "Empagliflozin), and GLP-1 receptor agonists (e.g. Semaglutide)."
)

SAMPLE_MEDICAL_TEXT_PAGE3 = (
    "4. Monitoring and Follow-Up\n\n"
    "Patients on Metformin should have renal function monitored at least "
    "annually. HbA1c should be measured every three months until stable, "
    "then every six months. Blood pressure and lipid profiles should be "
    "assessed at each visit.\n\n"
    "5. Lifestyle Modifications\n\n"
    "Regular physical activity of at least 150 minutes per week of "
    "moderate-intensity aerobic exercise is recommended. Dietary counseling "
    "should emphasize reduction of refined carbohydrates and increased "
    "fiber intake. Weight loss of 5-10% of body weight can significantly "
    "improve glycemic control in overweight patients with Type 2 Diabetes."
)


@pytest.fixture
def dummy_pdf_path(tmp_path: Path) -> Path:
    """Create a multi-page PDF containing realistic medical text."""
    pdf_path = tmp_path / "sample_guideline.pdf"
    doc = fitz.open()

    for text in [
        SAMPLE_MEDICAL_TEXT_PAGE1,
        SAMPLE_MEDICAL_TEXT_PAGE2,
        SAMPLE_MEDICAL_TEXT_PAGE3,
    ]:
        page = doc.new_page(width=612, height=792)  # US Letter
        page.insert_text(
            fitz.Point(50, 72),
            text,
            fontsize=11,
            fontname="helv",
        )

    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def single_page_pdf(tmp_path: Path) -> Path:
    """Create a minimal 1-page PDF for edge-case testing."""
    pdf_path = tmp_path / "single_page.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        fitz.Point(50, 72),
        "Patient presents with fever and headache. Prescribed aspirin 100 mg.",
        fontsize=11,
    )
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def empty_pdf(tmp_path: Path) -> Path:
    """Create a PDF with a single blank page."""
    pdf_path = tmp_path / "empty.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


# ---------------------------------------------------------------------------
# Tests – extract_text_from_pdf
# ---------------------------------------------------------------------------

class TestExtractTextFromPdf:
    """Tests for the PDF text extraction stage."""

    def test_extracts_correct_page_count(self, dummy_pdf_path: Path):
        """Extracted page list length must match the PDF page count."""
        pages = extract_text_from_pdf(dummy_pdf_path)
        assert len(pages) == 3

    def test_page_numbers_are_one_indexed(self, dummy_pdf_path: Path):
        """Page numbers should start at 1."""
        pages = extract_text_from_pdf(dummy_pdf_path)
        assert pages[0].page_number == 1
        assert pages[2].page_number == 3

    def test_text_content_preserved(self, dummy_pdf_path: Path):
        """Key phrases from the source text must appear in extracted text."""
        pages = extract_text_from_pdf(dummy_pdf_path)
        assert "Diabetes" in pages[0].text
        assert "Metformin" in pages[1].text

    def test_single_page_extraction(self, single_page_pdf: Path):
        """A 1-page PDF should yield exactly one PageText."""
        pages = extract_text_from_pdf(single_page_pdf)
        assert len(pages) == 1
        assert "fever" in pages[0].text

    def test_empty_pdf_returns_pages(self, empty_pdf: Path):
        """An empty page should still produce a PageText (with blank text)."""
        pages = extract_text_from_pdf(empty_pdf)
        assert len(pages) == 1

    def test_missing_file_raises(self, tmp_path: Path):
        """A non-existent path must raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            extract_text_from_pdf(tmp_path / "nonexistent.pdf")


# ---------------------------------------------------------------------------
# Tests – chunk_text
# ---------------------------------------------------------------------------

class TestChunkText:
    """Tests for the low-level token-based chunking function."""

    def test_short_text_is_single_chunk(self):
        """Text shorter than max_tokens should not be split."""
        text = "Short medical note about fever."
        chunks = chunk_text(text, max_tokens=500)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_empty_text_returns_empty(self):
        """Empty or whitespace-only input returns an empty list."""
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_chunks_respect_max_tokens(self):
        """Every chunk must stay at or below max_tokens (approx)."""
        long_text = " ".join(["word"] * 1200)
        chunks = chunk_text(long_text, max_tokens=500, overlap=50)
        for chunk in chunks:
            assert _approximate_token_count(chunk) <= 550  # small tolerance

    def test_overlap_present(self):
        """Consecutive chunks should share some trailing/leading words."""
        long_text = " ".join([f"word{i}" for i in range(1000)])
        chunks = chunk_text(long_text, max_tokens=200, overlap=50)
        assert len(chunks) >= 2
        # Check that the end of chunk 0 overlaps with the start of chunk 1
        tail_words = set(chunks[0].split()[-50:])
        head_words = set(chunks[1].split()[:50])
        assert len(tail_words & head_words) > 0

    def test_produces_multiple_chunks_for_long_text(self):
        """Text much longer than max_tokens should yield multiple chunks."""
        long_text = " ".join(["word"] * 2000)
        chunks = chunk_text(long_text, max_tokens=500, overlap=50)
        assert len(chunks) >= 3


# ---------------------------------------------------------------------------
# Tests – _split_on_headers
# ---------------------------------------------------------------------------

class TestSplitOnHeaders:
    """Tests for the header-based section splitter."""

    def test_numbered_headers(self):
        """Text with '1. Title' style headers should be split."""
        text = "1. Introduction\nSome intro text.\n2. Methods\nSome methods."
        sections = _split_on_headers(text)
        assert len(sections) >= 2

    def test_all_caps_headers(self):
        """ALL-CAPS lines should trigger section splits."""
        text = "INTRODUCTION\nSome text here.\nMETHODS\nMore text."
        sections = _split_on_headers(text)
        assert len(sections) >= 2

    def test_no_headers_returns_single(self):
        """Text without any header patterns returns as a single section."""
        text = "This is plain text with no headers at all."
        sections = _split_on_headers(text)
        assert len(sections) == 1
        assert sections[0] == text


# ---------------------------------------------------------------------------
# Tests – chunk_pages
# ---------------------------------------------------------------------------

class TestChunkPages:
    """Tests for the two-pass page chunking function."""

    def test_produces_chunks_from_pages(self, dummy_pdf_path: Path):
        """chunk_pages should produce at least one chunk per page."""
        pages = extract_text_from_pdf(dummy_pdf_path)
        chunks = chunk_pages(pages, source_file="test.pdf")
        assert len(chunks) >= 3  # at least one per page

    def test_chunks_carry_source_file(self, dummy_pdf_path: Path):
        """Every chunk should have the correct source_file."""
        pages = extract_text_from_pdf(dummy_pdf_path)
        chunks = chunk_pages(pages, source_file="guideline.pdf")
        for chunk in chunks:
            assert chunk.source_file == "guideline.pdf"

    def test_chunks_have_page_numbers(self, dummy_pdf_path: Path):
        """Every chunk must reference a valid page number."""
        pages = extract_text_from_pdf(dummy_pdf_path)
        chunks = chunk_pages(pages, source_file="test.pdf")
        for chunk in chunks:
            assert 1 <= chunk.page_number <= 3

    def test_chunk_ids_are_unique(self, dummy_pdf_path: Path):
        """All chunk IDs should be distinct."""
        pages = extract_text_from_pdf(dummy_pdf_path)
        chunks = chunk_pages(pages, source_file="test.pdf")
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_splits_on_section_headers(self):
        """Sections delimited by numbered headers should yield separate chunks."""
        pages = [
            PageText(
                page_number=1,
                text=(
                    "1. Introduction\nThis is the intro section with enough words.\n"
                    "2. Methods\nThis is the methods section with different content."
                ),
            )
        ]
        chunks = chunk_pages(pages, source_file="test.pdf")
        # There should be at least 2 chunks (one per header section)
        assert len(chunks) >= 2

    def test_long_section_is_subdivided(self):
        """A section exceeding max_tokens must be split further."""
        long_section = "1. Big Section\n" + " ".join(["word"] * 1200)
        pages = [PageText(page_number=1, text=long_section)]
        chunks = chunk_pages(pages, source_file="test.pdf", max_tokens=500)
        assert len(chunks) >= 2

    def test_metadata_has_extracted_entities_key(self, dummy_pdf_path: Path):
        """Every chunk's metadata dict must contain 'extracted_entities'."""
        pages = extract_text_from_pdf(dummy_pdf_path)
        chunks = chunk_pages(pages, source_file="test.pdf")
        for chunk in chunks:
            assert "extracted_entities" in chunk.metadata

    def test_empty_page_produces_no_chunks(self):
        """A page with only whitespace should not create any chunks."""
        pages = [PageText(page_number=1, text="   \n  \n  ")]
        chunks = chunk_pages(pages, source_file="test.pdf")
        assert len(chunks) == 0


# ---------------------------------------------------------------------------
# Tests – annotate_chunks
# ---------------------------------------------------------------------------

class TestAnnotateChunks:
    """Tests for NER annotation of chunks."""

    def test_entities_attached_to_chunks(self):
        """Mock NER should populate chunk.entities and metadata."""
        mock_ner = Mock(spec=MedicalNER)
        mock_ner.predict_batch.return_value = [
            [Entity(text="fever", label="Symptom", confidence=0.95, span=(0, 5))],
            [],
        ]

        chunks = [
            Chunk(
                text="Patient has fever.",
                chunk_id="aaa",
                source_file="f.pdf",
                page_number=1,
                start_char=0,
                end_char=18,
                metadata={"extracted_entities": []},
            ),
            Chunk(
                text="No entities here.",
                chunk_id="bbb",
                source_file="f.pdf",
                page_number=1,
                start_char=18,
                end_char=35,
                metadata={"extracted_entities": []},
            ),
        ]

        result = annotate_chunks(chunks, mock_ner)

        assert len(result[0].entities) == 1
        assert result[0].entities[0].label == "Symptom"
        assert len(result[0].metadata["extracted_entities"]) == 1

        assert len(result[1].entities) == 0
        assert len(result[1].metadata["extracted_entities"]) == 0

    def test_predict_batch_called_once(self):
        """annotate_chunks should call predict_batch exactly once."""
        mock_ner = Mock(spec=MedicalNER)
        mock_ner.predict_batch.return_value = [[]]

        chunks = [
            Chunk(
                text="text",
                chunk_id="x",
                source_file="f.pdf",
                page_number=1,
                start_char=0,
                end_char=4,
                metadata={"extracted_entities": []},
            ),
        ]

        annotate_chunks(chunks, mock_ner)
        mock_ner.predict_batch.assert_called_once()


# ---------------------------------------------------------------------------
# Tests – process_pdf (full pipeline)
# ---------------------------------------------------------------------------

class TestProcessPdf:
    """Integration tests for the end-to-end pipeline."""

    def test_without_ner(self, dummy_pdf_path: Path):
        """Running process_pdf without NER should return unannotated chunks."""
        chunks = process_pdf(dummy_pdf_path, ner=None)

        assert len(chunks) >= 3
        for chunk in chunks:
            assert chunk.source_file == "sample_guideline.pdf"
            assert chunk.page_number >= 1
            assert chunk.text.strip() != ""
            assert chunk.entities == []
            assert "extracted_entities" in chunk.metadata

    def test_with_mock_ner(self, dummy_pdf_path: Path):
        """Running process_pdf with a mock NER should annotate every chunk."""
        mock_ner = Mock(spec=MedicalNER)
        # Return one dummy entity per chunk
        def fake_batch(texts, **kwargs):
            return [
                [Entity(text="Diabetes", label="Disease", confidence=0.9, span=(0, 8))]
                for _ in texts
            ]
        mock_ner.predict_batch.side_effect = fake_batch

        chunks = process_pdf(dummy_pdf_path, ner=mock_ner)

        assert len(chunks) >= 3
        for chunk in chunks:
            assert len(chunk.entities) == 1
            assert chunk.entities[0].label == "Disease"
            assert len(chunk.metadata["extracted_entities"]) == 1

    def test_chunk_metadata_schema(self, dummy_pdf_path: Path):
        """Every chunk must carry the metadata keys required by CONTEXT.md."""
        chunks = process_pdf(dummy_pdf_path)

        required_chunk_fields = {"source_file", "page_number"}
        required_metadata_keys = {"extracted_entities"}

        for chunk in chunks:
            # Fields on the Chunk dataclass
            assert chunk.source_file, "source_file must be non-empty"
            assert chunk.page_number >= 1, "page_number must be >= 1"
            # Keys in the metadata dict
            for key in required_metadata_keys:
                assert key in chunk.metadata, f"metadata missing key: {key}"

    def test_missing_pdf_raises(self, tmp_path: Path):
        """process_pdf should propagate FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            process_pdf(tmp_path / "does_not_exist.pdf")

    def test_to_dict_serialization(self, single_page_pdf: Path):
        """Chunk.to_dict() should return a JSON-serialisable dictionary."""
        chunks = process_pdf(single_page_pdf)
        assert len(chunks) >= 1

        d = chunks[0].to_dict()
        assert isinstance(d, dict)
        assert "text" in d
        assert "chunk_id" in d
        assert "source_file" in d
        assert "page_number" in d
        assert "entities" in d
        assert "metadata" in d


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
