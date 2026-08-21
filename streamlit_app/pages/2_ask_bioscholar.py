"""
Ask BioScholar

Demo page for the full RAG pipeline via POST /query.
"""

from pathlib import Path
import sys

import streamlit as st

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

st.set_page_config(
    page_title="Ask BioScholar",
    page_icon="💬",
    layout="wide",
)


def _agent_mode_to_flag(mode: str):
    if mode == "Auto":
        return None
    if mode == "Always":
        return True
    if mode == "Never":
        return False
    return None


def main() -> None:
    st.title("Ask BioScholar")
    st.markdown("Ask questions and inspect citations returned by the RAG pipeline.")
    st.markdown("---")

    from utils.api_client import BioScholarAPIError, get_client, query

    with st.sidebar:
        st.markdown("### Controls")
        top_k = st.slider("top_k", min_value=1, max_value=20, value=5, step=1)
        score_threshold = st.slider(
            "score_threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05,
        )
        entity_filter = st.toggle("entity_filter", value=True)
        agent_mode = st.selectbox("Agent routing", ["Auto", "Always", "Never"], index=0)

    examples = [
        "What is recommended for fever treatment in children?",
        "What are contraindications for aspirin?",
        "Compare the side effects of ibuprofen vs acetaminophen.",
        "Summarize the guideline section on hypertension management.",
    ]
    example = st.selectbox("Example questions", [""] + examples)

    question = st.text_area(
        "Question",
        value=example or "What is recommended for fever treatment in children?",
        height=120,
    )

    if st.button("Ask", type="primary", use_container_width=True):
        if not question.strip():
            st.warning("Please enter a question.")
            return

        with st.spinner("Calling /query..."):
            try:
                with get_client(timeout_s=120.0) as client:
                    res = query(
                        client,
                        question=question,
                        top_k=top_k,
                        score_threshold=score_threshold,
                        entity_filter=entity_filter,
                        use_agent=_agent_mode_to_flag(agent_mode),
                    )
            except BioScholarAPIError as e:
                st.error(str(e))
                return
            except Exception:
                st.error("Unexpected error while calling the API.")
                return

        st.markdown("### Answer")
        st.write(res.answer)

        meta_cols = st.columns(4)
        meta_cols[0].metric("Model", res.model or "unknown")
        meta_cols[1].metric("Citations", str(len(res.citations)))
        meta_cols[2].metric("Agent used", str(res.agent_used))
        meta_cols[3].metric("Agent steps", str(res.agent_steps or "-"))

        if res.agent_trace:
            with st.expander("🕵️ Agent Trace (Reasoning Steps)", expanded=False):
                for step_idx, msg in enumerate(res.agent_trace):
                    role = msg.get("type", "unknown")
                    content = msg.get("content", "")
                    # Clean up content for display
                    if role == "human":
                        st.chat_message("user").write(content)
                    elif role == "ai":
                        # Check for tool calls
                        tool_calls = msg.get("tool_calls", [])
                        if tool_calls:
                            for tc in tool_calls:
                                st.chat_message("assistant").write(f"🛠️ **Tool Call:** `{tc.get('name')}`")
                                st.json(tc.get("args"))
                        else:
                            st.chat_message("assistant").write(content)
                    elif role == "tool":
                        st.chat_message("tool").caption(f"Tool Result ({msg.get('name', 'unknown')})")
                        # Try to format JSON if possible
                        try:
                            import json
                            json_content = json.loads(content)
                            st.json(json_content)
                        except:
                            st.text(content[:500] + "..." if len(content) > 500 else content)

        st.markdown("---")
        st.markdown("### Citations")

        if not res.citations:
            st.info("No citations returned (insufficient retrieval or below score threshold).")
        else:
            for i, c in enumerate(res.citations, start=1):
                title = f"[Source {i}] {c.source_file} — page {c.page_number} — score {c.score:.3f}"
                with st.expander(title, expanded=(i == 1)):
                    st.caption(c.chunk_id)
                    if c.extracted_entities:
                        st.markdown("**Extracted entities:**")
                        st.code(", ".join(c.extracted_entities), language="text")
                    st.markdown("**Preview:**")
                    st.write(c.text_preview)


if __name__ == "__main__":
    main()

