"""
Text Analysis Page

Analyze text descriptions of symptoms using the NER model.
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
    st.title("📝 Text Symptom Analysis")
    st.markdown("Analyze text descriptions of symptoms using our fine-tuned BioBERT NER model.")
    
    st.markdown("---")
    
    # Check model availability
    from utils.model_loader import get_ner_predictor
    
    with st.spinner("Loading model..."):
        predictor = get_ner_predictor()
    
    if not predictor:
        st.error("""
        **NER Model not available!**
        
        Please train the model first:
        ```bash
        python data/prepare_data.py --include-synthetic
        python src/train.py --config config.yaml
        ```
        """)
        return
    
    st.success("Model loaded successfully!")
    
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
        
        - 🔵 **Symptom** - Physical symptoms
        - 🔴 **Disease** - Medical conditions
        - 🟢 **Chemical** - Medications/drugs
        """)
    
    # Analyze button
    if st.button("🔍 Analyze Text", type="primary", use_container_width=True):
        if not input_text.strip():
            st.warning("Please enter some text to analyze.")
            return
        
        with st.spinner("Analyzing..."):
            try:
                result = predictor.predict(input_text)
                entities = result.entities if hasattr(result, 'entities') else result.get('entities', [])
                
                st.markdown("---")
                st.markdown("### Results")
                
                if entities:
                    # Highlighted text
                    st.markdown("#### Highlighted Text")
                    highlighted = highlight_entities(input_text, entities)
                    st.markdown(highlighted, unsafe_allow_html=True)
                    
                    st.markdown("#### Extracted Entities")
                    
                    # Group by type
                    by_type = {}
                    for ent in entities:
                        label = ent.label if hasattr(ent, 'label') else ent.get('label', 'Unknown')
                        if label not in by_type:
                            by_type[label] = []
                        by_type[label].append(ent)
                    
                    # Display by type
                    cols = st.columns(len(by_type))
                    for i, (label, ents) in enumerate(by_type.items()):
                        with cols[i]:
                            emoji = "🔵" if "symptom" in label.lower() else "🔴" if "disease" in label.lower() else "🟢"
                            st.markdown(f"**{emoji} {label}**")
                            for ent in ents:
                                ent_text = ent.text if hasattr(ent, 'text') else ent.get('text', '')
                                ent_conf = ent.confidence if hasattr(ent, 'confidence') else ent.get('confidence', 0)
                                st.markdown(f"- {ent_text} ({ent_conf:.1%})")
                    
                    # Entity table
                    st.markdown("#### Detailed Results")
                    table_data = []
                    for ent in entities:
                        table_data.append({
                            "Text": ent.text if hasattr(ent, 'text') else ent.get('text', ''),
                            "Type": ent.label if hasattr(ent, 'label') else ent.get('label', ''),
                            "Confidence": f"{(ent.confidence if hasattr(ent, 'confidence') else ent.get('confidence', 0)):.1%}",
                            "Position": f"{(ent.span if hasattr(ent, 'span') else ent.get('span', [0,0]))}"
                        })
                    st.table(table_data)
                    
                    # JSON output
                    with st.expander("📋 JSON Output"):
                        json_output = {
                            "text": input_text,
                            "entities": [
                                {
                                    "text": ent.text if hasattr(ent, 'text') else ent.get('text', ''),
                                    "label": ent.label if hasattr(ent, 'label') else ent.get('label', ''),
                                    "confidence": ent.confidence if hasattr(ent, 'confidence') else ent.get('confidence', 0),
                                    "span": list(ent.span if hasattr(ent, 'span') else ent.get('span', [0, 0]))
                                }
                                for ent in entities
                            ]
                        }
                        st.json(json_output)
                else:
                    st.info("No medical entities detected in the text.")
                    
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
    
    # Sidebar info
    with st.sidebar:
        st.markdown("### ℹ️ About")
        st.markdown("""
        This page uses a fine-tuned **BioBERT** model for Named Entity Recognition (NER).
        
        The model was trained to identify:
        - Symptoms
        - Diseases
        - Chemicals/Medications
        """)
        
        st.markdown("### 💡 Tips")
        st.markdown("""
        - Use complete sentences for better results
        - The model works with both casual and medical language
        - Longer text may contain more entities
        """)


if __name__ == "__main__":
    main()
