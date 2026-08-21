"""
Medical Scan Assistant

Analyze medical images (X-rays, MRIs, Lab Reports) using Vision-Language Models.
"""

import base64
from pathlib import Path
import sys
import streamlit as st

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

st.set_page_config(
    page_title="Medical Scan Assistant",
    page_icon="👁️",
    layout="wide",
)

def main():
    st.title("👁️ Medical Scan Assistant")
    st.markdown("""
    Upload a medical image (X-ray, MRI, CT, Ultrasound, or Lab Report) for analysis.
    
    **Powered by Vision-Language Models (VLM).**
    *Ensure you have a vision model (e.g., `llama3.2-vision`) pulled in Ollama.*
    """)

    from utils.api_client import BioScholarAPIError, get_client, analyze_image

    with st.sidebar:
        model = st.text_input("Vision Model", value="llama3.2-vision")
        st.caption("Make sure to run `ollama pull llama3.2-vision`")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1. Select or Upload Image")
        source_mode = st.radio(
            "Image Source",
            ["Upload Custom Image", "Choose Sample Guideline Figure"],
            horizontal=True,
        )

        base64_image = None
        display_caption = ""

        if source_mode == "Upload Custom Image":
            uploaded_file = st.file_uploader("Upload Medical Image", type=["png", "jpg", "jpeg"])
            if uploaded_file:
                st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
                bytes_data = uploaded_file.getvalue()
                base64_image = base64.b64encode(bytes_data).decode('utf-8')
                display_caption = uploaded_file.name
        else:
            figures_dir = project_root / "data" / "figures"
            sample_files = sorted(list(figures_dir.glob("*.png")) + list(figures_dir.glob("*.jpeg")) + list(figures_dir.glob("*.jpg")))
            if sample_files:
                sample_names = [f.name for f in sample_files[:15]]
                selected_name = st.selectbox("Select Sample Figure", sample_names)
                selected_path = figures_dir / selected_name
                st.image(str(selected_path), caption=f"Sample: {selected_name}", use_container_width=True)
                bytes_data = selected_path.read_bytes()
                base64_image = base64.b64encode(bytes_data).decode('utf-8')
                display_caption = selected_name
            else:
                st.info("No sample figures found in data/figures/.")

    with col2:
        st.subheader("2. VLM Clinical Analysis")
        if base64_image:
            default_prompt = "Analyze this medical image. Describe the key visual findings, tables/charts, and clinical implications."
            prompt = st.text_area("Question / Instruction", value=default_prompt, height=140)
            
            if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                with st.spinner(f"Analyzing {display_caption} with {model}..."):
                    try:
                        with get_client(timeout_s=120.0) as client:
                            analysis = analyze_image(client, base64_image, prompt, model)
                        
                        st.markdown("### 📋 Clinical Findings")
                        st.write(analysis)
                        
                    except BioScholarAPIError as e:
                        st.error(f"API Error: {e}")
                    except Exception as e:
                        st.error(f"Error: {e}")
        else:
            st.info("👈 Please select or upload a medical image to begin analysis.")

if __name__ == "__main__":
    main()
