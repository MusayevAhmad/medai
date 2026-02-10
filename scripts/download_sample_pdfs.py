#!/usr/bin/env python3
"""
Download open-access medical PDFs for the BioScholar ingestion pipeline.

Sources:
    - PubMed Central (PMC) open-access articles
    - WHO technical guidelines (public domain)

Usage:
    python scripts/download_sample_pdfs.py
    python scripts/download_sample_pdfs.py --output-dir data/raw_pdfs --max-pdfs 10
"""

import argparse
import hashlib
import sys
import time
from pathlib import Path
from typing import List, Tuple

import httpx
from tqdm import tqdm

# -------------------------------------------------------------------
# Curated list of open-access medical PDFs
# These are all freely available under open-access licenses (CC-BY, public domain)
# -------------------------------------------------------------------

SAMPLE_PDFS: List[Tuple[str, str]] = [
    # ---- Batch 1: Original 10 PDFs ----
    # Europe PMC open-access articles (no auth required, CC-BY)
    (
        "EPMC_Diabetes_Type2_Review.pdf",
        "https://europepmc.org/backend/ptpmcrender.fcgi?accid=PMC6520897&blobtype=pdf",
    ),
    (
        "EPMC_Hypertension_Management.pdf",
        "https://europepmc.org/backend/ptpmcrender.fcgi?accid=PMC6477925&blobtype=pdf",
    ),
    (
        "EPMC_Antibiotic_Resistance.pdf",
        "https://europepmc.org/backend/ptpmcrender.fcgi?accid=PMC6929930&blobtype=pdf",
    ),
    (
        "EPMC_Asthma_Pathophysiology.pdf",
        "https://europepmc.org/backend/ptpmcrender.fcgi?accid=PMC7174566&blobtype=pdf",
    ),
    (
        "EPMC_Depression_Treatment.pdf",
        "https://europepmc.org/backend/ptpmcrender.fcgi?accid=PMC6584108&blobtype=pdf",
    ),
    (
        "EPMC_Heart_Failure_Review.pdf",
        "https://europepmc.org/backend/ptpmcrender.fcgi?accid=PMC6887664&blobtype=pdf",
    ),
    (
        "EPMC_Chronic_Pain_Review.pdf",
        "https://europepmc.org/backend/ptpmcrender.fcgi?accid=PMC6676152&blobtype=pdf",
    ),
    (
        "EPMC_Alzheimer_Disease.pdf",
        "https://europepmc.org/backend/ptpmcrender.fcgi?accid=PMC6514974&blobtype=pdf",
    ),
    (
        "EPMC_Cancer_Immunotherapy.pdf",
        "https://europepmc.org/backend/ptpmcrender.fcgi?accid=PMC6446023&blobtype=pdf",
    ),
    (
        "EPMC_COVID19_Clinical_Review.pdf",
        "https://europepmc.org/backend/ptpmcrender.fcgi?accid=PMC7169770&blobtype=pdf",
    ),
    # ---- Batch 2: 10 additional PDFs for broader coverage ----
    (
        "EPMC_Stroke_Management.pdf",
        "https://europepmc.org/backend/ptpmcrender.fcgi?accid=PMC6935818&blobtype=pdf",
    ),
    (
        "EPMC_COPD_Review.pdf",
        "https://europepmc.org/backend/ptpmcrender.fcgi?accid=PMC6400111&blobtype=pdf",
    ),
    (
        "EPMC_Rheumatoid_Arthritis.pdf",
        "https://europepmc.org/backend/ptpmcrender.fcgi?accid=PMC6422329&blobtype=pdf",
    ),
    (
        "EPMC_Epilepsy_Treatment.pdf",
        "https://europepmc.org/backend/ptpmcrender.fcgi?accid=PMC6723954&blobtype=pdf",
    ),
    (
        "EPMC_HIV_Treatment_Review.pdf",
        "https://europepmc.org/backend/ptpmcrender.fcgi?accid=PMC6541416&blobtype=pdf",
    ),
    (
        "EPMC_Liver_Disease_Review.pdf",
        "https://europepmc.org/backend/ptpmcrender.fcgi?accid=PMC6826798&blobtype=pdf",
    ),
    (
        "EPMC_Kidney_Disease_CKD.pdf",
        "https://europepmc.org/backend/ptpmcrender.fcgi?accid=PMC7004228&blobtype=pdf",
    ),
    (
        "EPMC_Tuberculosis_Review.pdf",
        "https://europepmc.org/backend/ptpmcrender.fcgi?accid=PMC6548090&blobtype=pdf",
    ),
    (
        "EPMC_Malaria_Treatment.pdf",
        "https://europepmc.org/backend/ptpmcrender.fcgi?accid=PMC6425757&blobtype=pdf",
    ),
    (
        "EPMC_Sepsis_Management.pdf",
        "https://europepmc.org/backend/ptpmcrender.fcgi?accid=PMC6429642&blobtype=pdf",
    ),
]


def download_pdf(url: str, dest: Path, timeout: float = 30.0) -> bool:
    """Download a single PDF from *url* to *dest*.

    Returns True on success, False on failure (non-fatal).
    """
    if dest.exists():
        print(f"  [skip] {dest.name} already exists")
        return True

    headers = {
        "User-Agent": "BioScholar/1.0 (academic research; mailto:user@example.com)",
        "Accept": "application/pdf",
    }
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            resp = client.get(url, headers=headers)
            resp.raise_for_status()

            # Basic sanity check: should be a PDF
            content_type = resp.headers.get("content-type", "")
            if "pdf" not in content_type and not resp.content[:5] == b"%PDF-":
                print(f"  [warn] {dest.name}: unexpected content-type {content_type}")

            dest.write_bytes(resp.content)
            size_kb = len(resp.content) / 1024
            print(f"  [ok]   {dest.name} ({size_kb:.0f} KB)")
            return True

    except httpx.HTTPStatusError as e:
        print(f"  [fail] {dest.name}: HTTP {e.response.status_code}")
        return False
    except httpx.ConnectError:
        print(f"  [fail] {dest.name}: connection error")
        return False
    except Exception as e:
        print(f"  [fail] {dest.name}: {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Download sample medical PDFs")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw_pdfs"),
        help="Directory to save PDFs (default: data/raw_pdfs)",
    )
    parser.add_argument(
        "--max-pdfs",
        type=int,
        default=len(SAMPLE_PDFS),
        help=f"Maximum PDFs to download (default: {len(SAMPLE_PDFS)})",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pdfs_to_download = SAMPLE_PDFS[: args.max_pdfs]

    print(f"Downloading {len(pdfs_to_download)} open-access medical PDFs")
    print(f"Output directory: {args.output_dir.resolve()}\n")

    success = 0
    for name, url in tqdm(pdfs_to_download, desc="Downloading"):
        dest = args.output_dir / name
        if download_pdf(url, dest):
            success += 1
        time.sleep(0.5)  # Be polite to servers

    print(f"\nDone: {success}/{len(pdfs_to_download)} PDFs downloaded.")

    if success == 0:
        print("No PDFs downloaded. Check your internet connection.")
        sys.exit(1)


if __name__ == "__main__":
    main()
