"""
About Page

Information about the BioScholar application.
"""

from pathlib import Path
import sys

import streamlit as st

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

st.set_page_config(
    page_title="About - BioScholar",
    page_icon="ℹ️",
    layout="wide",
)


def main():
    st.title("ℹ️ About BioScholar")

    st.markdown("---")

    st.markdown("## 🎯 Overview")
    st.markdown(
        """
**BioScholar** is an AI-powered clinical evidence research assistant.\n
It extracts medical entities from text and uses Retrieval-Augmented Generation (RAG) to answer questions\n
grounded in ingested clinical guidelines, with citations back to source PDFs.\n
\n
This demo UI is a client of the FastAPI backend (`/query`, `/search`, `/entities`, `/visual-search`).\n
        """
    )

    st.markdown("---")

    st.markdown("## 🏗️ Architecture")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Entity extraction (NER)")
        st.markdown(
            """
**Model:** BioBERT (fine-tuned)\n
- Base: `dmis-lab/biobert-base-cased-v1.2`\n
- Task: Named Entity Recognition (NER)\n
- Used for query understanding and metadata-filtered retrieval\n
            """
        )

    with col2:
        st.markdown("### Retrieval + Generation (RAG)")
        st.markdown(
            """
**Vector DB:** Qdrant\n
**Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`\n
**Answering:** OpenAI-compatible LLM endpoint (e.g. Ollama)\n
\n
Features:\n
- Entity-filtered hybrid retrieval\n
- Citations with source file + page\n
- Visual search for tables and figures\n
- Optional agent routing for complex questions\n
            """
        )

    st.markdown("---")

    st.markdown("## 📖 Quick Start")
    st.code(
        """
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train NER model
python data/prepare_data.py --include-synthetic
python src/train.py --config config.yaml

# 3. Start the API
uvicorn app.main:app --reload --port 8000

# 4. Run the Streamlit demo UI
streamlit run streamlit_app/app.py
        """.strip(),
        language="bash",
    )

    st.markdown("---")
    st.markdown("## ⚠️ Disclaimer")
    st.error(
        """
**IMPORTANT: This application is for educational purposes only!**\n
- This is NOT a medical diagnostic tool\n
- Do not make healthcare decisions based on this system\n
        """
    )


if __name__ == "__main__":
    main()

