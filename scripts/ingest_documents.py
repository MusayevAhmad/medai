#!/usr/bin/env python3
"""
Bulk PDF ingestion script.

Processes all PDFs in a directory, extracts text, chunks content,
runs NER on each chunk, and stores results in Qdrant.

Example:
    python scripts/ingest_documents.py --collection-name bio_guidelines
"""

import argparse
from pathlib import Path
from typing import List

from tqdm import tqdm

from src.ingest import process_pdf
from src.multimodal_ingest import process_pdf_multimodal
from src.inference import MedicalNER
from src.vector_store import QdrantStore


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Ingest PDFs into Qdrant.")
    parser.add_argument(
        "--collection-name",
        required=True,
        help="Qdrant collection name to store chunks.",
    )
    parser.add_argument(
        "--pdf-dir",
        default="data/raw_pdfs",
        help="Directory containing PDF files to ingest.",
    )
    parser.add_argument(
        "--model-path",
        default="outputs/models/final_model",
        help="Path to the trained NER model.",
    )
    parser.add_argument(
        "--qdrant-url",
        default=None,
        help="Qdrant URL (e.g. http://localhost:6333).",
    )
    parser.add_argument(
        "--qdrant-path",
        default=None,
        help="Local Qdrant storage path (mutually exclusive with --qdrant-url).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=500,
        help="Maximum approximate tokens per chunk.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=50,
        help="Token overlap between consecutive chunks.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="NER confidence threshold.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Number of chunks to embed per batch.",
    )
    parser.add_argument(
        "--multimodal",
        action="store_true",
        help="Enable multimodal ingestion (extract tables and figures).",
    )
    parser.add_argument(
        "--figures-dir",
        default="data/figures",
        help="Directory to save extracted figure images.",
    )
    return parser.parse_args()


def collect_pdfs(pdf_dir: Path) -> List[Path]:
    """Return a sorted list of PDF paths in *pdf_dir*."""
    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF directory does not exist: {pdf_dir}")
    if not pdf_dir.is_dir():
        raise NotADirectoryError(f"PDF path is not a directory: {pdf_dir}")
    return sorted([p for p in pdf_dir.glob("*.pdf") if p.is_file()])


def ingest_all(
    pdf_paths: List[Path],
    store: QdrantStore,
    ner: MedicalNER,
    max_tokens: int,
    overlap: int,
    threshold: float,
    batch_size: int,
    multimodal: bool = False,
    figures_dir: Path = Path("data/figures"),
) -> int:
    """Ingest all PDFs and return total chunks inserted."""
    total_inserted = 0
    for pdf_path in tqdm(pdf_paths, desc="Ingesting PDFs"):
        try:
            if multimodal:
                chunks = process_pdf_multimodal(
                    pdf_path,
                    ner=ner,
                    max_tokens=max_tokens,
                    overlap=overlap,
                    threshold=threshold,
                    figures_dir=figures_dir,
                )
            else:
                chunks = process_pdf(
                    pdf_path,
                    ner=ner,
                    max_tokens=max_tokens,
                    overlap=overlap,
                    threshold=threshold,
                )
            inserted = store.add_chunks(chunks, batch_size=batch_size)
            total_inserted += inserted
        except Exception as exc:
            tqdm.write(f"[WARN] Failed to ingest {pdf_path}: {exc}")
    return total_inserted


def main() -> None:
    """Entry point for bulk ingestion."""
    args = parse_args()

    if args.qdrant_url and args.qdrant_path:
        raise ValueError("Use only one of --qdrant-url or --qdrant-path.")

    pdf_dir = Path(args.pdf_dir)
    pdf_paths = collect_pdfs(pdf_dir)
    if not pdf_paths:
        print(f"No PDFs found in {pdf_dir}")
        return

    ner = MedicalNER(model_path=args.model_path)
    store = QdrantStore(
        collection_name=args.collection_name,
        qdrant_url=args.qdrant_url,
        qdrant_path=args.qdrant_path,
    )

    total_inserted = ingest_all(
        pdf_paths=pdf_paths,
        store=store,
        ner=ner,
        max_tokens=args.max_tokens,
        overlap=args.overlap,
        threshold=args.threshold,
        batch_size=args.batch_size,
        multimodal=args.multimodal,
        figures_dir=Path(args.figures_dir),
    )

    print(f"Done. Inserted {total_inserted} chunks into '{args.collection_name}'.")


if __name__ == "__main__":
    main()
