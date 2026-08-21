"""
BioScholar - Medical RAG Demo

Demo-ready Streamlit UI for the BioScholar FastAPI backend.
"""

import streamlit as st
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Page configuration
st.set_page_config(
    page_title="BioScholar - Medical RAG Demo",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-box {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


def main():
    st.markdown('<p class="main-header">BioScholar</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Medical RAG demo (citations, tables, figures, agent)</p>',
        unsafe_allow_html=True,
    )

    from utils.api_client import BioScholarAPIError, get_client, health

    with st.sidebar:
        st.markdown("### Connection")
        st.caption("Set `BIOSCHOLAR_API_URL` if your API isn't on localhost.")
        st.code('export BIOSCHOLAR_API_URL="http://localhost:8000"', language="bash")

        try:
            with get_client(timeout_s=10.0) as client:
                h = health(client)
            st.success("API reachable")
            st.markdown(f"**NER loaded:** `{h.ner_loaded}`")
            st.markdown(f"**LLM available:** `{h.llm_available}`")
            st.markdown(f"**Collection count:** `{h.collection_count}`")
        except BioScholarAPIError as e:
            st.error(str(e))
        except Exception:
            st.error("API not reachable. Start it with `make serve` or `docker-compose up`.")

        st.markdown("---")
        st.markdown("### Pages")
        if st.button("Entities (NER)", use_container_width=True):
            st.switch_page("pages/1_text_analysis.py")
        if st.button("Ask BioScholar", use_container_width=True):
            st.switch_page("pages/2_ask_bioscholar.py")
        if st.button("Visual Search", use_container_width=True):
            st.switch_page("pages/3_visual_search.py")
        if st.button("Ingest PDFs", use_container_width=True):
            st.switch_page("pages/4_ingest_pdfs.py")
        if st.button("Case Analyzer", use_container_width=True):
            st.switch_page("pages/6_case_analyzer.py")
        if st.button("Scan Assistant", use_container_width=True):
            st.switch_page("pages/7_medical_scan_assistant.py")
        if st.button("About", use_container_width=True):
            st.switch_page("pages/5_about.py")

        st.markdown("---")
        st.markdown("### Disclaimer")
        st.caption("Educational demo only. Not medical advice.")

    st.markdown("### 🔬 System Capabilities & Interactive Modules")
    st.markdown(
        """
        Explore the end-to-end clinical AI workflow across specialized modules:
        """
    )

    # Row 1
    col1, col2, col3 = st.columns(3)
    with col1:
        with st.container():
            st.markdown("#### 🏷️ Clinical NER")
            st.caption("Fine-tuned BioBERT model extracting Diseases, Chemicals, and Symptoms with token offsets.")
            if st.button("Explore Entities (NER) →", use_container_width=True):
                st.switch_page("pages/1_text_analysis.py")
    with col2:
        with st.container():
            st.markdown("#### 💬 Grounded RAG Assistant")
            st.caption("Ask clinical questions with strict document grounding, confidence scoring, and verified citations.")
            if st.button("Ask BioScholar →", use_container_width=True):
                st.switch_page("pages/2_ask_bioscholar.py")
    with col3:
        with st.container():
            st.markdown("#### 📊 Visual Search")
            st.caption("Retrieve extracted tables in Markdown and figure diagrams embedded in clinical PDFs.")
            if st.button("Explore Visual Search →", use_container_width=True):
                st.switch_page("pages/3_visual_search.py")

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 2
    col4, col5, col6 = st.columns(3)
    with col4:
        with st.container():
            st.markdown("#### 📄 Document Ingestion")
            st.caption("Upload new guidelines, automatically chunk, annotate with NER, and index into Qdrant.")
            if st.button("Ingest Clinical PDFs →", use_container_width=True):
                st.switch_page("pages/4_ingest_pdfs.py")
    with col5:
        with st.container():
            st.markdown("#### 🩺 Patient Case Analyzer")
            st.caption("Complex patient vignette analysis combining ReAct agent reasoning and drug interaction checks.")
            if st.button("Analyze Patient Cases →", use_container_width=True):
                st.switch_page("pages/6_case_analyzer.py")
    with col6:
        with st.container():
            st.markdown("#### 👁️ Medical Scan Assistant")
            st.caption("Vision-Language Model (VLM) clinical interpretation of medical figures, scans, and reports.")
            if st.button("Scan Assistant →", use_container_width=True):
                st.switch_page("pages/7_medical_scan_assistant.py")


if __name__ == "__main__":
    main()
