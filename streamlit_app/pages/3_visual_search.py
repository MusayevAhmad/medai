"""
Visual Search

Search for text, tables, and figures via POST /visual-search.
"""

from pathlib import Path
import sys

import streamlit as st

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

st.set_page_config(
    page_title="Visual Search",
    page_icon="🔎",
    layout="wide",
)


def main() -> None:
    st.title("Visual Search")
    st.markdown("Retrieve text, tables, and figures (with captions) from ingested PDFs.")
    st.markdown("---")

    from utils.api_client import BioScholarAPIError, get_client, visual_search

    with st.sidebar:
        st.markdown("### Controls")
        top_k = st.slider("top_k", min_value=1, max_value=20, value=5, step=1)
        chunk_types = st.multiselect(
            "chunk_types",
            options=["text", "table", "figure"],
            default=[],
            help="Empty = return all chunk types.",
        )

    query_text = st.text_input("Search query", value="Table 1")

    if st.button("Search", type="primary", use_container_width=True):
        if not query_text.strip():
            st.warning("Please enter a query.")
            return

        with st.spinner("Calling /visual-search..."):
            try:
                with get_client(timeout_s=120.0) as client:
                    res = visual_search(
                        client,
                        query_text=query_text,
                        top_k=top_k,
                        chunk_types=chunk_types or None,
                    )
            except BioScholarAPIError as e:
                st.error(str(e))
                return
            except Exception:
                st.error("Unexpected error while calling the API.")
                return

        stats_cols = st.columns(4)
        stats_cols[0].metric("Results", str(res.count))
        stats_cols[1].metric("Tables found", str(res.tables_found))
        stats_cols[2].metric("Figures found", str(res.figures_found))
        stats_cols[3].metric("Query entities", str(len(res.query_entities)))

        st.markdown("---")

        if not res.results:
            st.info("No results.")
            return

        for i, r in enumerate(res.results, start=1):
            title = (
                f"[{i}] {r.chunk_type} — {r.source_file} p{r.page_number} — score {r.score:.3f}"
            )
            with st.expander(title, expanded=(i == 1)):
                if r.caption:
                    st.markdown(f"**Caption:** {r.caption}")

                if r.chunk_type == "table":
                    st.markdown(r.text)
                elif r.chunk_type == "figure":
                    if r.image_url:
                        st.image(r.image_url, caption=r.caption or "Figure", use_container_width=True)
                        st.code(r.image_url, language="text")
                    elif r.image_path and Path(r.image_path).exists():
                        st.image(r.image_path, caption=r.caption or "Figure", use_container_width=True)
                        st.code(r.image_path, language="text")
                    st.markdown("**Figure text:**")
                    st.write(r.text)
                else:
                    st.write(r.text)

                if r.extracted_entities:
                    st.markdown("**Extracted entities:**")
                    st.code(", ".join(r.extracted_entities), language="text")


if __name__ == "__main__":
    main()

