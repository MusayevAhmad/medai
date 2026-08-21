"""
Case Analyzer

Analyze patient cases against medical guidelines.
"""

from pathlib import Path
import sys
import streamlit as st

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

st.set_page_config(
    page_title="Case Analyzer",
    page_icon="🩺",
    layout="wide",
)

def main():
    st.title("🩺 Patient Case Analyzer")
    st.markdown("""
    Paste a patient case description below. BioScholar will:
    1. Extract clinical entities (Symptoms, Diseases, Medications).
    2. Search guidelines for relevant evidence.
    3. Propose a management plan based **only** on the retrieved guidelines.
    """)

    from utils.api_client import BioScholarAPIError, get_client, entities, query

    col1, col2 = st.columns([1, 1])

    with col1:
        default_case = """
Patient is a 45-year-old male presenting with persistent dry cough and shortness of breath for 2 weeks. 
History of Type 2 Diabetes (controlled with Metformin) and Hypertension. 
Reports fatigue and low-grade fever in the evenings. 
Oxygen saturation 94% on room air.
        """.strip()
        
        case_text = st.text_area("Patient Case Description", value=default_case, height=300)
        
        analyze_btn = st.button("Analyze Case", type="primary", use_container_width=True)

    if analyze_btn and case_text:
        client = get_client()
        
        # 1. Extract Entities
        with col2:
            st.subheader("1. Clinical Entities")
            with st.spinner("Extracting entities..."):
                try:
                    ents = entities(client, case_text)
                    
                    # Group by label
                    grouped = {}
                    for e in ents:
                        if e.label not in grouped:
                            grouped[e.label] = []
                        grouped[e.label].append(e.text)
                    
                    for label, texts in grouped.items():
                        st.markdown(f"**{label}**")
                        st.write(", ".join(set(texts)))
                        
                except Exception as e:
                    st.error(f"NER Error: {e}")

        # 2. RAG Analysis
        st.markdown("---")
        st.subheader("2. Guideline-Based Analysis")
        
        analysis_prompt = (
            "Analyze the following patient case based on clinical guidelines. "
            "Identify potential diagnoses and recommend next steps for management/treatment.\n\n"
            f"Case: {case_text}"
        )

        with st.spinner("Consulting guidelines (this may take a moment)..."):
            try:
                # Use the agent for deeper analysis
                res = query(
                    client, 
                    question=analysis_prompt, 
                    top_k=10, 
                    use_agent=True # Force agent for reasoning
                )
                
                st.markdown("### Recommendation")
                st.write(res.answer)
                
                if res.agent_trace:
                    with st.expander("🕵️ Analysis Logic (Agent Trace)"):
                        for msg in res.agent_trace:
                            role = msg.get("type", "unknown")
                            content = msg.get("content", "")
                            if role == "ai":
                                if not msg.get("tool_calls"):
                                    st.markdown(f"**Thought:** {content}")
                            elif role == "tool":
                                st.caption(f"Evidence found via {msg.get('name')}")

                st.markdown("### Supporting Evidence")
                for i, c in enumerate(res.citations, 1):
                    with st.expander(f"{i}. {c.source_file} (p.{c.page_number})"):
                        st.write(c.text_preview)
                        
            except Exception as e:
                st.error(f"Analysis Error: {e}")

if __name__ == "__main__":
    main()
