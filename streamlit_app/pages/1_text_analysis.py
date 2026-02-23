"""
Text Analysis Page

Extract entities from text using the BioScholar API (/entities).
"""

import streamlit as st
from pathlib import Path
import sys
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

st.set_page_config(
    page_title="Text Analysis - MedAI",
    page_icon="📝",
    layout="wide",
)

# Custom CSS
st.markdown("""
<style>
    .entity-symptom {
        background-color: #e3f2fd;
        padding: 2px 8px;
        border-radius: 4px;
        border: 1px solid #1976d2;
    }
    .entity-disease {
        background-color: #fce4ec;
        padding: 2px 8px;
        border-radius: 4px;
        border: 1px solid #c2185b;
    }
    .entity-chemical {
        background-color: #e8f5e9;
        padding: 2px 8px;
        border-radius: 4px;
        border: 1px solid #388e3c;
    }
</style>
""", unsafe_allow_html=True)


def highlight_entities(text, entities):
    """Create HTML with highlighted entities."""
    if not entities:
        return text
    
    # Sort entities by start position (reverse to replace from end)
    sorted_entities = sorted(entities, key=lambda x: x.span[0] if hasattr(x, 'span') else x.get('span', [0])[0], reverse=True)
    
    result = text
    for ent in sorted_entities:
        start = ent.span[0] if hasattr(ent, 'span') else ent.get('span', [0, 0])[0]
        end = ent.span[1] if hasattr(ent, 'span') else ent.get('span', [0, 0])[1]
        label = ent.label if hasattr(ent, 'label') else ent.get('label', '')
        ent_text = ent.text if hasattr(ent, 'text') else ent.get('text', '')
        
        css_class = f"entity-{label.lower()}"
        highlighted = f'<span class="{css_class}" title="{label}">{ent_text}</span>'
        result = result[:start] + highlighted + result[end:]
    
    return result


def main():
    st.title("Entities (NER)")
    st.markdown("Extract medical entities using the BioScholar API endpoint `POST /entities`.")
    
    st.markdown("---")
    
    from utils.api_client import BioScholarAPIError, entities, get_client
    
    # Input section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Enter Text to Analyze")
        
        # Example inputs
        examples = [
            "I have a persistent headache and fever for 3 days.",
            "Patient presents with chest pain and shortness of breath.",
            "I took ibuprofen for my migraine but it didn't help.",
            "My joints are swollen and painful, especially in the morning.",
            "I have diabetes and lately my vision has been blurry.",
        ]
        
        selected_example = st.selectbox(
            "Or select an example:",
            [""] + examples,
            key="example_select"
        )
        
        default_text = selected_example if selected_example else "I have a persistent headache and fever for 3 days."
        
        input_text = st.text_area(
            "Enter your symptoms:",
            value=default_text,
            height=150,
            key="input_text"
        )
    
    with col2:
        st.markdown("### Entity Types")
        st.markdown("""
        The model detects:
        
        - **Disease**\n
        - **Chemical**\n
        - (and other symptom-like mentions depending on training)\n
        """)
    
    # Analyze button
    threshold = st.slider(
        "Confidence threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.05,
    )

    if st.button("Extract entities", type="primary", use_container_width=True):
        if not input_text.strip():
            st.warning("Please enter some text to analyze.")
            return
        
        with st.spinner("Calling API..."):
            try:
                with get_client(timeout_s=60.0) as client:
                    ents = entities(client, input_text, threshold=threshold)
                
                st.markdown("---")
                st.markdown("### Results")
                
                if ents:
                    # Highlighted text
                    st.markdown("#### Highlighted Text")
                    highlighted = highlight_entities(input_text, ents)
                    st.markdown(highlighted, unsafe_allow_html=True)
                    
                    st.markdown("#### Extracted Entities")
                    
                    # Group by type
                    by_type = {}
                    for ent in ents:
                        label = ent.label
                        if label not in by_type:
                            by_type[label] = []
                        by_type[label].append(ent)
                    
                    # Display by type
                    cols = st.columns(len(by_type))
                    for i, (label, ents) in enumerate(by_type.items()):
                        with cols[i]:
                            st.markdown(f"**{label}**")
                            for ent in ents:
                                st.markdown(f"- {ent.text} ({ent.confidence:.1%})")
                    
                    # Entity table
                    st.markdown("#### Detailed Results")
                    table_data = []
                    for ent in ents:
                        table_data.append({
                            "Text": ent.text,
                            "Type": ent.label,
                            "Confidence": f"{ent.confidence:.1%}",
                            "Position": f"{ent.span}",
                        })
                    st.table(table_data)
                    
                    # JSON output
                    with st.expander("📋 JSON Output"):
                        json_output = {
                            "text": input_text,
                            "entities": [
                                {
                                    "text": ent.text,
                                    "label": ent.label,
                                    "confidence": ent.confidence,
                                    "span": list(ent.span),
                                }
                                for ent in ents
                            ]
                        }
                        st.json(json_output)
                else:
                    st.info("No medical entities detected in the text.")
                    
            except BioScholarAPIError as e:
                st.error(str(e))
            except Exception:
                st.error("Unexpected error while calling the API.")
    
    # Sidebar info
    with st.sidebar:
        st.markdown("### ℹ️ About")
        st.markdown("""
        This page calls the BioScholar backend `/entities` endpoint.
        """)
        
        st.markdown("### 💡 Tips")
        st.markdown("""
        - If the API isn't reachable, start it with `make serve`.\n
        - Adjust the confidence threshold to reduce noise.\n
        """)


if __name__ == "__main__":
    main()
