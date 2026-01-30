"""
MedAI - Medical Symptom Analyzer

Main Streamlit application for medical NER and X-ray analysis.
"""

import streamlit as st
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Page configuration
st.set_page_config(
    page_title="MedAI - Medical Symptom Analyzer",
    page_icon="🏥",
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
    # Header
    st.markdown('<p class="main-header">🏥 MedAI</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Medical Symptom Analyzer</p>', unsafe_allow_html=True)
    
    # Main content
    st.markdown("---")
    
    # Feature cards
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📝 Text Analysis")
        st.markdown("""
        Analyze text descriptions of symptoms using our fine-tuned BioBERT model.
        
        **Features:**
        - Extract medical entities (symptoms, diseases, medications)
        - High-confidence predictions
        - Support for casual and medical language
        """)
        if st.button("Go to Text Analysis →", key="text_btn"):
            st.switch_page("pages/1_text_analysis.py")
    
    with col2:
        st.markdown("### 🩻 X-Ray Analysis")
        st.markdown("""
        Analyze chest X-ray images for potential conditions.
        
        **Features:**
        - COVID-19 detection
        - Pneumonia classification
        - Normal vs abnormal classification
        """)
        if st.button("Go to X-Ray Analysis →", key="xray_btn"):
            st.switch_page("pages/2_xray_analysis.py")
    
    st.markdown("---")
    
    # Quick demo
    st.markdown("### 🚀 Quick Demo")
    
    demo_tab1, demo_tab2 = st.tabs(["Text Analysis Demo", "X-Ray Analysis Demo"])
    
    with demo_tab1:
        demo_text = st.text_area(
            "Enter symptoms to analyze:",
            value="I have a persistent headache and fever for 3 days.",
            height=100,
            key="demo_text"
        )
        
        if st.button("Analyze Text", key="demo_analyze"):
            with st.spinner("Analyzing..."):
                try:
                    from utils.model_loader import get_ner_predictor
                    predictor = get_ner_predictor()
                    
                    if predictor:
                        result = predictor.predict(demo_text)
                        entities = result.entities if hasattr(result, 'entities') else result.get('entities', [])
                        
                        if entities:
                            st.success(f"Found {len(entities)} medical entities:")
                            for ent in entities:
                                ent_text = ent.text if hasattr(ent, 'text') else ent.get('text', '')
                                ent_label = ent.label if hasattr(ent, 'label') else ent.get('label', '')
                                ent_conf = ent.confidence if hasattr(ent, 'confidence') else ent.get('confidence', 0)
                                st.markdown(f"- **{ent_text}** → `{ent_label}` ({ent_conf:.1%})")
                        else:
                            st.info("No medical entities detected.")
                    else:
                        st.warning("NER model not available. Please train the model first.")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    with demo_tab2:
        st.info("Upload a chest X-ray image to analyze.")
        demo_file = st.file_uploader(
            "Upload X-Ray Image",
            type=['png', 'jpg', 'jpeg'],
            key="demo_xray"
        )
        
        if demo_file:
            from PIL import Image
            image = Image.open(demo_file)
            
            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(image, caption="Uploaded X-Ray", use_container_width=True)
            
            with col2:
                if st.button("Analyze X-Ray", key="demo_xray_analyze"):
                    with st.spinner("Analyzing..."):
                        try:
                            from utils.model_loader import get_xray_predictor
                            predictor = get_xray_predictor()
                            
                            if predictor:
                                result = predictor.predict(image)
                                st.success(f"**Prediction:** {result.predicted_class}")
                                st.metric("Confidence", f"{result.confidence:.1%}")
                                
                                st.markdown("**All Probabilities:**")
                                for cls, prob in sorted(result.all_probabilities.items(), key=lambda x: -x[1]):
                                    st.progress(prob, text=f"{cls}: {prob:.1%}")
                            else:
                                st.warning("X-Ray model not available. Please train the model first.")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 Model Status")
        
        # Check NER model
        ner_model_path = project_root / "outputs" / "models"
        ner_available = any(ner_model_path.glob("run_*/final_model"))
        st.markdown(f"**NER Model:** {'✅ Available' if ner_available else '❌ Not trained'}")
        
        # Check X-Ray model
        xray_model_path = project_root / "outputs" / "xray_models"
        xray_available = any(xray_model_path.glob("*.pt")) or any(xray_model_path.glob("run_*/best_model.pt"))
        st.markdown(f"**X-Ray Model:** {'✅ Available' if xray_available else '❌ Not trained'}")
        
        st.markdown("---")
        st.markdown("### ⚠️ Disclaimer")
        st.markdown("""
        This tool is for **educational purposes only**.
        
        It should **NOT** be used for actual medical diagnosis.
        Always consult a healthcare professional.
        """)
        
        st.markdown("---")
        st.markdown("### 📚 About")
        st.markdown("""
        **MedAI** is an AI-powered medical symptom analyzer built with:
        - BioBERT for text analysis
        - DenseNet121 for X-ray classification
        - Streamlit for the web interface
        """)


if __name__ == "__main__":
    main()
