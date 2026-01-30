"""
About Page

Information about the MedAI application.
"""

import streamlit as st
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

st.set_page_config(
    page_title="About - MedAI",
    page_icon="ℹ️",
    layout="wide",
)


def main():
    st.title("ℹ️ About MedAI")
    
    st.markdown("---")
    
    # Overview
    st.markdown("## 🎯 Overview")
    st.markdown("""
    **MedAI** is an AI-powered medical symptom analyzer that combines Natural Language Processing (NLP) 
    and Computer Vision to help analyze medical symptoms from both text descriptions and chest X-ray images.
    
    This project was built as a learning exercise to understand:
    - Transfer learning in NLP and Computer Vision
    - Fine-tuning pre-trained models for domain-specific tasks
    - Building multi-modal AI applications
    - Creating interactive web interfaces for ML models
    """)
    
    st.markdown("---")
    
    # Architecture
    st.markdown("## 🏗️ Architecture")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Text Analysis (NER)")
        st.markdown("""
        **Model:** BioBERT (fine-tuned)
        
        - Base: `dmis-lab/biobert-base-cased-v1.2`
        - Task: Named Entity Recognition (NER)
        - Entities: Symptoms, Diseases, Chemicals
        - Training: BIO tagging scheme
        
        **Training Details:**
        - Dataset: BC5CDR + Synthetic symptoms
        - Fine-tuning: LoRA (optional) or full
        - Epochs: 3
        - Optimizer: AdamW
        """)
    
    with col2:
        st.markdown("### X-Ray Analysis (CNN)")
        st.markdown("""
        **Model:** DenseNet121 (transfer learning)
        
        - Base: ImageNet pre-trained
        - Task: Multi-class classification
        - Classes: COVID-19, Normal, Pneumonia
        - Input: 224x224 RGB images
        
        **Training Details:**
        - Dataset: COVID-19 Radiography Database
        - Transfer learning with fine-tuning
        - Class-weighted loss for imbalance
        - Optimizer: Adam with cosine scheduler
        """)
    
    st.markdown("---")
    
    # Tech Stack
    st.markdown("## 🛠️ Tech Stack")
    
    tech_col1, tech_col2, tech_col3 = st.columns(3)
    
    with tech_col1:
        st.markdown("### ML/AI")
        st.markdown("""
        - PyTorch
        - Transformers (HuggingFace)
        - TorchVision
        - scikit-learn
        """)
    
    with tech_col2:
        st.markdown("### Data")
        st.markdown("""
        - BC5CDR Dataset
        - COVID-19 Radiography DB
        - Custom synthetic data
        """)
    
    with tech_col3:
        st.markdown("### Web")
        st.markdown("""
        - Streamlit
        - Plotly
        - Pillow
        """)
    
    st.markdown("---")
    
    # Model Performance
    st.markdown("## 📊 Model Performance")
    
    perf_col1, perf_col2 = st.columns(2)
    
    with perf_col1:
        st.markdown("### NER Model")
        st.markdown("""
        | Metric | Score |
        |--------|-------|
        | Precision | >90% |
        | Recall | >90% |
        | F1 Score | >90% |
        
        *Note: Metrics depend on training data quality*
        """)
    
    with perf_col2:
        st.markdown("### X-Ray Model")
        st.markdown("""
        | Metric | Target |
        |--------|--------|
        | Accuracy | >85% |
        | Precision | >80% |
        | Recall | >80% |
        
        *Note: Requires dataset download and training*
        """)
    
    st.markdown("---")
    
    # Usage
    st.markdown("## 📖 Usage")
    
    st.markdown("### Quick Start")
    st.code("""
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train NER model
python data/prepare_data.py --include-synthetic
python src/train.py --config config.yaml

# 3. Train X-Ray model (requires dataset)
python src/image_train.py --config config_xray.yaml

# 4. Run the app
streamlit run app/app.py
    """, language="bash")
    
    st.markdown("### Command Line Usage")
    
    st.markdown("**Text Analysis:**")
    st.code("""
python src/predict.py --model-path outputs/models/run_*/final_model --text "I have a headache"
    """, language="bash")
    
    st.markdown("**X-Ray Analysis:**")
    st.code("""
python src/image_predict.py --model-path outputs/xray_models/best_model.pt --image path/to/xray.png
    """, language="bash")
    
    st.markdown("---")
    
    # Disclaimer
    st.markdown("## ⚠️ Disclaimer")
    st.error("""
    **IMPORTANT: This application is for educational purposes only!**
    
    - This is NOT a medical diagnostic tool
    - Results should NOT be used for self-diagnosis
    - Always consult qualified healthcare professionals for medical advice
    - The models may produce incorrect predictions
    - Do not make healthcare decisions based on these results
    """)
    
    st.markdown("---")
    
    # Project Structure
    st.markdown("## 📁 Project Structure")
    st.code("""
medai/
├── app/                    # Streamlit application
│   ├── app.py             # Main app entry point
│   ├── pages/             # Multi-page app pages
│   └── utils/             # Utility functions
├── src/                    # Core ML modules
│   ├── model.py           # NER model
│   ├── train.py           # NER training
│   ├── predict.py         # NER inference
│   ├── image_model.py     # CNN model
│   ├── image_train.py     # CNN training
│   └── image_predict.py   # CNN inference
├── data/                   # Datasets
├── notebooks/              # Jupyter notebooks
├── outputs/                # Trained models
├── config.yaml            # NER configuration
└── config_xray.yaml       # CNN configuration
    """, language="text")
    
    st.markdown("---")
    
    # Footer
    st.markdown("## 🙏 Acknowledgments")
    st.markdown("""
    - **BioBERT** - Pre-trained biomedical language model
    - **COVID-19 Radiography Database** - Chest X-ray dataset
    - **HuggingFace** - Transformers library
    - **Streamlit** - Web framework
    """)


if __name__ == "__main__":
    main()
