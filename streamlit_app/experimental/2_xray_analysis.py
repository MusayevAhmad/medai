"""
X-Ray Analysis Page (Experimental)

Kept for reference, but excluded from the demo-ready BioScholar RAG app.
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
    page_title="X-Ray Analysis (Experimental)",
    page_icon="🩻",
    layout="wide",
)


def create_probability_chart(probabilities):
    """Create a horizontal bar chart for probabilities."""
    import plotly.graph_objects as go

    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    labels = [p[0] for p in sorted_probs]
    values = [p[1] for p in sorted_probs]

    colors = ["#e74c3c" if v == max(values) else "#3498db" for v in values]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=[f"{v:.1%}" for v in values],
            textposition="auto",
        )
    )

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
    st.title("🩻 X-Ray Image Analysis (Experimental)")
    st.markdown("This page is not part of the BioScholar demo.")

    st.markdown("---")

    from streamlit_app.utils.model_loader import get_xray_predictor

    with st.spinner("Loading model..."):
        predictor = get_xray_predictor()

    if not predictor:
        st.warning(
            """
        **X-Ray Model not available!**

        To use this feature, you need to:
        1. Download the COVID-19 Radiography Dataset from Kaggle
        2. Train the model:
        ```bash
        python src/image_train.py --config config_xray.yaml
        ```
        """
        )
        demo_mode = True
    else:
        st.success("Model loaded successfully!")
        demo_mode = False

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Upload X-Ray Image")
        uploaded_file = st.file_uploader(
            "Choose an X-ray image",
            type=["png", "jpg", "jpeg"],
            help="Upload a chest X-ray image for analysis",
        )

    with col2:
        st.markdown("### Analysis Results")

        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded X-Ray", use_container_width=True)

            if st.button("🔍 Analyze X-Ray", type="primary", use_container_width=True):
                if demo_mode:
                    st.info("Demo mode: no trained model found.")
                else:
                    with st.spinner("Analyzing X-ray..."):
                        result = predictor.predict(image)
                        st.success(f"Prediction: {result.predicted_class}")
                        st.metric("Confidence", f"{result.confidence:.1%}")
                        fig = create_probability_chart(result.all_probabilities)
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Upload an X-ray image to get started.")


if __name__ == "__main__":
    main()

