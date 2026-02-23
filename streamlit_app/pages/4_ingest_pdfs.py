"""
Ingest PDFs

Upload PDFs into a local staging directory and trigger ingestion via POST /ingest.
"""

from pathlib import Path
import sys

import streamlit as st

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

st.set_page_config(
    page_title="Ingest PDFs",
    page_icon="⬆️",
    layout="wide",
)


def main() -> None:
    st.title("Ingest PDFs")
    st.markdown("Upload guideline PDFs, then ingest them into the configured Qdrant collection.")
    st.markdown("---")

    from utils.api_client import BioScholarAPIError, get_client, ingest

    staging_dir = project_root / "data" / "raw_pdfs"
    staging_dir.mkdir(parents=True, exist_ok=True)

    with st.sidebar:
        st.markdown("### Ingestion settings")
        multimodal = st.toggle("Multimodal (tables + figures)", value=True)
        max_tokens = st.slider("max_tokens", min_value=100, max_value=1200, value=500, step=50)
        overlap = st.slider("overlap", min_value=0, max_value=200, value=50, step=10)
        threshold = st.slider("NER threshold", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
        batch_size = st.select_slider("batch_size", options=[16, 32, 64, 128, 256], value=64)
        collection_name = st.text_input("collection_name (optional check)", value="bio_guidelines")

    st.markdown("### Upload PDFs")
    uploaded = st.file_uploader(
        "Select one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded:
        saved = 0
        for f in uploaded:
            out_path = staging_dir / f.name
            out_path.write_bytes(f.getbuffer())
            saved += 1
        st.success(f"Saved {saved} PDF(s) to {staging_dir}")

    st.markdown("---")
    st.markdown("### Trigger ingestion")
    st.caption("This ingests all `*.pdf` currently in the staging directory.")
    st.code(str(staging_dir), language="text")

    if st.button("Run ingestion", type="primary", use_container_width=True):
        with st.spinner("Calling /ingest..."):
            try:
                with get_client(timeout_s=600.0) as client:
                    res = ingest(
                        client,
                        collection_name=collection_name.strip() or "bio_guidelines",
                        pdf_dir=str(staging_dir),
                        figures_dir=str(project_root / "data" / "figures"),
                        multimodal=multimodal,
                        max_tokens=max_tokens,
                        overlap=overlap,
                        threshold=threshold,
                        batch_size=batch_size,
                    )
            except BioScholarAPIError as e:
                st.error(str(e))
                return
            except Exception:
                st.error("Unexpected error while calling the API.")
                return

        st.success("Ingestion completed.")
        st.json(res)


if __name__ == "__main__":
    main()

