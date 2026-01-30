"""
X-Ray Analysis Page

Analyze chest X-ray images for potential conditions.
"""

import streamlit as st
from pathlib import Path
import sys
from PIL import Image
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

st.set_page_config(
    page_title="X-Ray Analysis - MedAI",
    page_icon="🩻",
    layout="wide",
)


def create_probability_chart(probabilities):
    """Create a horizontal bar chart for probabilities."""
    import plotly.graph_objects as go
    
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    labels = [p[0] for p in sorted_probs]
    values = [p[1] for p in sorted_probs]
    
    colors = ['#e74c3c' if v == max(values) else '#3498db' for v in values]
    
    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation='h',
        marker_color=colors,
        text=[f'{v:.1%}' for v in values],
        textposition='auto',
    ))
    
    fig.update_layout(
        title="Prediction Probabilities",
        xaxis_title="Probability",
        yaxis_title="Class",
        height=300,
        margin=dict(l=0, r=0, t=40, b=0),
        xaxis=dict(range=[0, 1]),
    )
    
    return fig


def main():
    st.title("🩻 X-Ray Image Analysis")
    st.markdown("Analyze chest X-ray images for potential conditions using our CNN classifier.")
    
    st.markdown("---")
    
    # Check model availability
    from app.utils.model_loader import get_xray_predictor
    
    with st.spinner("Loading model..."):
        predictor = get_xray_predictor()
    
    if not predictor:
        st.warning("""
        **X-Ray Model not available!**
        
        To use this feature, you need to:
        1. Download the COVID-19 Radiography Dataset from Kaggle
        2. Train the model:
        ```bash
        python src/image_train.py --config config_xray.yaml
        ```
        
        For now, you can still upload images and see the interface.
        """)
        demo_mode = True
    else:
        st.success("Model loaded successfully!")
        demo_mode = False
    
    # Main layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Upload X-Ray Image")
        
        uploaded_file = st.file_uploader(
            "Choose an X-ray image",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a chest X-ray image for analysis"
        )
        
        # Sample images option
        st.markdown("---")
        st.markdown("**Or use a sample image:**")
        
        sample_dir = project_root / "data" / "xray" / "COVID-19_Radiography_Dataset"
        sample_images = []
        
        if sample_dir.exists():
            for class_dir in sample_dir.iterdir():
                if class_dir.is_dir():
                    images_dir = class_dir / "images" if (class_dir / "images").exists() else class_dir
                    for img in list(images_dir.glob("*.png"))[:2]:
                        sample_images.append((class_dir.name, str(img)))
        
        if sample_images:
            sample_options = ["Select a sample..."] + [f"{cls}: {Path(p).name}" for cls, p in sample_images]
            selected_sample = st.selectbox("Sample images:", sample_options)
            
            if selected_sample != "Select a sample...":
                idx = sample_options.index(selected_sample) - 1
                uploaded_file = sample_images[idx][1]
        else:
            st.info("No sample images available. Please upload an image.")
    
    with col2:
        st.markdown("### Analysis Results")
        
        if uploaded_file:
            # Load and display image
            if isinstance(uploaded_file, str):
                image = Image.open(uploaded_file).convert('RGB')
                st.image(image, caption="X-Ray Image", use_container_width=True)
            else:
                image = Image.open(uploaded_file).convert('RGB')
                st.image(image, caption="Uploaded X-Ray", use_container_width=True)
            
            # Analyze button
            if st.button("🔍 Analyze X-Ray", type="primary", use_container_width=True):
                if demo_mode:
                    # Demo results
                    st.warning("Running in demo mode (model not trained)")
                    st.info("""
                    **Demo Results:**
                    
                    In production, the model would analyze this X-ray and provide:
                    - Predicted condition (COVID-19, Normal, Pneumonia)
                    - Confidence score
                    - Probability distribution across classes
                    """)
                else:
                    with st.spinner("Analyzing X-ray..."):
                        try:
                            result = predictor.predict(image)
                            
                            # Display results
                            st.markdown("---")
                            
                            # Main prediction
                            pred_class = result.predicted_class
                            confidence = result.confidence
                            
                            # Color based on prediction
                            if "covid" in pred_class.lower():
                                color = "red"
                                emoji = "⚠️"
                            elif "pneumonia" in pred_class.lower():
                                color = "orange"
                                emoji = "⚠️"
                            else:
                                color = "green"
                                emoji = "✅"
                            
                            st.markdown(f"### {emoji} Prediction: **:{color}[{pred_class}]**")
                            
                            # Confidence meter
                            st.metric("Confidence", f"{confidence:.1%}")
                            st.progress(confidence)
                            
                            # Probability chart
                            try:
                                fig = create_probability_chart(result.all_probabilities)
                                st.plotly_chart(fig, use_container_width=True)
                            except ImportError:
                                st.markdown("**All Probabilities:**")
                                for cls, prob in sorted(result.all_probabilities.items(), key=lambda x: -x[1]):
                                    st.progress(prob, text=f"{cls}: {prob:.1%}")
                            
                            # Detailed results
                            with st.expander("📋 Detailed Results"):
                                st.json({
                                    "predicted_class": result.predicted_class,
                                    "confidence": result.confidence,
                                    "all_probabilities": result.all_probabilities
                                })
                            
                        except Exception as e:
                            st.error(f"Error during analysis: {str(e)}")
        else:
            st.info("👆 Upload an X-ray image to get started")
            
            # Show expected classes
            st.markdown("#### What the model detects:")
            st.markdown("""
            - **COVID-19** - Signs of COVID-19 infection
            - **Normal** - Healthy lung appearance
            - **Viral Pneumonia** - Signs of viral pneumonia
            """)
    
    # Sidebar info
    with st.sidebar:
        st.markdown("### ℹ️ About")
        st.markdown("""
        This page uses a **DenseNet121** CNN trained on chest X-ray images.
        
        The model can identify:
        - COVID-19
        - Normal lungs
        - Viral Pneumonia
        """)
        
        st.markdown("### ⚠️ Important")
        st.markdown("""
        **This is for educational purposes only!**
        
        - Not a medical diagnostic tool
        - Results should not replace professional medical advice
        - Always consult a healthcare provider
        """)
        
        st.markdown("### 💡 Tips")
        st.markdown("""
        - Use clear, frontal chest X-ray images
        - Ensure good image quality
        - The model works best with standard PA/AP views
        """)


if __name__ == "__main__":
    main()
