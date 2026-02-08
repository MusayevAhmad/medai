"""
Model Loader Utilities

Handles loading and caching of ML models for the Streamlit app.
"""

import streamlit as st
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


@st.cache_resource
def get_ner_predictor():
    """
    Load and cache the NER predictor.
    
    Returns:
        NERPredictor instance or None if model not available
    """
    try:
        from src.predict import NERPredictor
        
        # Find the latest model
        model_dir = project_root / "outputs" / "models"
        model_runs = sorted(model_dir.glob("run_*"))
        
        if not model_runs:
            print("No NER models found")
            return None
        
        latest_model = model_runs[-1] / "final_model"
        
        if not latest_model.exists():
            print(f"Model not found at {latest_model}")
            return None
        
        print(f"Loading NER model from {latest_model}")
        predictor = NERPredictor(str(latest_model))
        
        return predictor
        
    except Exception as e:
        print(f"Error loading NER model: {e}")
        return None


@st.cache_resource
def get_xray_predictor():
    """
    Load and cache the X-Ray predictor.
    
    Returns:
        XRayPredictor instance or None if model not available
    """
    try:
        from src.image_predict import XRayPredictor
        
        # Find the model
        model_dir = project_root / "outputs" / "xray_models"
        
        # Check for best_model.pt in run directories
        model_path = None
        
        # First check for run_* directories
        run_dirs = sorted(model_dir.glob("run_*"))
        if run_dirs:
            for run_dir in reversed(run_dirs):
                best_model = run_dir / "best_model.pt"
                if best_model.exists():
                    model_path = best_model
                    break
        
        # Then check for direct .pt files
        if not model_path:
            pt_files = list(model_dir.glob("*.pt"))
            if pt_files:
                model_path = sorted(pt_files)[-1]
        
        if not model_path:
            print("No X-Ray models found")
            return None
        
        print(f"Loading X-Ray model from {model_path}")
        predictor = XRayPredictor(str(model_path))
        
        return predictor
        
    except Exception as e:
        print(f"Error loading X-Ray model: {e}")
        return None


def check_model_status():
    """
    Check the status of all models.
    
    Returns:
        Dict with model availability status
    """
    status = {
        "ner": {
            "available": False,
            "path": None,
        },
        "xray": {
            "available": False,
            "path": None,
        }
    }
    
    # Check NER model
    model_dir = project_root / "outputs" / "models"
    model_runs = sorted(model_dir.glob("run_*"))
    if model_runs:
        latest_model = model_runs[-1] / "final_model"
        if latest_model.exists():
            status["ner"]["available"] = True
            status["ner"]["path"] = str(latest_model)
    
    # Check X-Ray model
    xray_dir = project_root / "outputs" / "xray_models"
    run_dirs = sorted(xray_dir.glob("run_*"))
    if run_dirs:
        for run_dir in reversed(run_dirs):
            best_model = run_dir / "best_model.pt"
            if best_model.exists():
                status["xray"]["available"] = True
                status["xray"]["path"] = str(best_model)
                break
    
    if not status["xray"]["available"]:
        pt_files = list(xray_dir.glob("*.pt"))
        if pt_files:
            status["xray"]["available"] = True
            status["xray"]["path"] = str(sorted(pt_files)[-1])
    
    return status
