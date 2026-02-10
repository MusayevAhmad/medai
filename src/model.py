"""
Model Configuration for Medical NER

Creates BioBERT/PubMedBERT model with LoRA adapters for parameter-efficient
fine-tuning on token classification (NER) tasks.

Supports:
- BioBERT, PubMedBERT, and other BERT-based models
- LoRA (Low-Rank Adaptation) for efficient fine-tuning
- Automatic device detection (MPS/CUDA/CPU)
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)

# Handle different transformers versions for PreTrainedModel import
try:
    from transformers import PreTrainedModel
except ImportError:
    from transformers.modeling_utils import PreTrainedModel

# Alias for type hints
PreTrainedTokenizer = PreTrainedTokenizerBase


def _ensure_hf_cache() -> None:
    """Configure Hugging Face cache inside the repo to avoid permission issues."""
    project_root = Path(__file__).parent.parent
    hf_cache = project_root / "data" / "hf_cache"
    hf_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hf_cache))
    os.environ.setdefault("HF_DATASETS_CACHE", str(hf_cache / "datasets"))
    os.environ.setdefault("HF_HUB_CACHE", str(hf_cache / "hub"))
    os.environ.setdefault("HF_MODULES_CACHE", str(hf_cache / "modules"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_cache / "hub"))

# Optional PEFT import
try:
    from peft import (
        LoraConfig,
        TaskType,
        get_peft_model,
        PeftModel,
        prepare_model_for_kbit_training,
    )
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: PEFT not installed. LoRA fine-tuning will not be available.")
    print("Install with: pip install peft")


# Default label scheme
DEFAULT_LABEL_LIST = [
    "O",           # 0 - Outside any entity
    "B-Disease",   # 1 - Beginning of disease entity
    "I-Disease",   # 2 - Inside disease entity
    "B-Chemical",  # 3 - Beginning of chemical entity
    "I-Chemical",  # 4 - Inside chemical entity
    "B-Symptom",   # 5 - Beginning of symptom entity
    "I-Symptom",   # 6 - Inside symptom entity
]


@dataclass
class ModelConfig:
    """Configuration for the NER model."""
    
    # Model
    model_name: str = "dmis-lab/biobert-base-cased-v1.2"
    max_length: int = 128
    
    # Labels
    label_list: List[str] = None
    
    # LoRA
    lora_enabled: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = None
    lora_bias: str = "none"
    
    # Device
    device: str = None  # Auto-detect if None
    
    def __post_init__(self):
        if self.label_list is None:
            self.label_list = DEFAULT_LABEL_LIST
        
        if self.lora_target_modules is None:
            self.lora_target_modules = ["query", "value"]
        
        if self.device is None:
            self.device = get_device()


def get_device() -> str:
    """
    Automatically detect the best available device.
    
    Priority: MPS (Apple Silicon) > CUDA > CPU
    
    Returns:
        Device string for PyTorch
    """
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def get_label_mappings(label_list: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Create label to ID and ID to label mappings."""
    label_to_id = {label: idx for idx, label in enumerate(label_list)}
    id_to_label = {idx: label for idx, label in enumerate(label_list)}
    return label_to_id, id_to_label


def create_model(
    config: Union[ModelConfig, dict] = None,
    model_name: str = None,
    num_labels: int = None,
    label_list: List[str] = None,
    use_lora: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    lora_target_modules: List[str] = None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer, Dict[str, int], Dict[int, str]]:
    """
    Create a model for NER token classification with optional LoRA.
    
    Args:
        config: ModelConfig instance or dict with config values
        model_name: HuggingFace model name (overrides config)
        num_labels: Number of labels (overrides len(label_list))
        label_list: List of label strings
        use_lora: Whether to apply LoRA adapters
        lora_r: LoRA rank
        lora_alpha: LoRA alpha scaling factor
        lora_dropout: LoRA dropout rate
        lora_target_modules: List of module names to apply LoRA to
    
    Returns:
        Tuple of (model, tokenizer, label_to_id, id_to_label)
    """
    _ensure_hf_cache()
    # Handle config
    if config is None:
        config = ModelConfig()
    elif isinstance(config, dict):
        config = ModelConfig(**config)
    
    # Override with explicit arguments
    if model_name is not None:
        config.model_name = model_name
    if label_list is not None:
        config.label_list = label_list
    if lora_target_modules is not None:
        config.lora_target_modules = lora_target_modules
    
    config.lora_enabled = use_lora
    config.lora_r = lora_r
    config.lora_alpha = lora_alpha
    config.lora_dropout = lora_dropout
    
    # Get label mappings
    label_to_id, id_to_label = get_label_mappings(config.label_list)
    
    # Number of labels
    if num_labels is None:
        num_labels = len(config.label_list)
    
    print(f"Creating model: {config.model_name}")
    print(f"Number of labels: {num_labels}")
    print(f"Device: {config.device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # Load base model for token classification
    model_config = AutoConfig.from_pretrained(
        config.model_name,
        num_labels=num_labels,
        id2label=id_to_label,
        label2id=label_to_id,
    )
    
    model = AutoModelForTokenClassification.from_pretrained(
        config.model_name,
        config=model_config,
    )
    
    # Count parameters before LoRA
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Base model parameters: {total_params:,}")
    
    # Apply LoRA if enabled
    if config.lora_enabled and PEFT_AVAILABLE:
        model = apply_lora(
            model,
            r=config.lora_r,
            alpha=config.lora_alpha,
            dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
            bias=config.lora_bias,
        )
        
        # Count trainable parameters after LoRA
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters (LoRA): {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    elif config.lora_enabled and not PEFT_AVAILABLE:
        print("Warning: LoRA requested but PEFT not available. Using full fine-tuning.")
    
    # Move to device
    model = model.to(config.device)
    
    return model, tokenizer, label_to_id, id_to_label


def apply_lora(
    model: PreTrainedModel,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.1,
    target_modules: List[str] = None,
    bias: str = "none",
) -> PreTrainedModel:
    """
    Apply LoRA adapters to the model for parameter-efficient fine-tuning.
    
    Args:
        model: Base transformer model
        r: LoRA rank (low-rank dimension)
        alpha: LoRA alpha (scaling factor)
        dropout: Dropout probability for LoRA layers
        target_modules: List of module names to apply LoRA to
        bias: Bias type ("none", "all", "lora_only")
    
    Returns:
        Model with LoRA adapters
    """
    if not PEFT_AVAILABLE:
        raise ImportError("PEFT is required for LoRA. Install with: pip install peft")
    
    if target_modules is None:
        target_modules = ["query", "value"]
    
    print(f"Applying LoRA: r={r}, alpha={alpha}, dropout={dropout}")
    print(f"Target modules: {target_modules}")
    
    lora_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias=bias,
    )
    
    model = get_peft_model(model, lora_config)
    
    return model


def load_model(
    model_path: str,
    device: str = None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer, Dict[str, int], Dict[int, str]]:
    """
    Load a saved model (with or without LoRA adapters).
    
    Args:
        model_path: Path to saved model directory
        device: Device to load model to (auto-detect if None)
    
    Returns:
        Tuple of (model, tokenizer, label_to_id, id_to_label)
    """
    _ensure_hf_cache()
    if device is None:
        device = get_device()
    
    model_path = Path(model_path)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Check if this is a PEFT model
    adapter_config_path = model_path / "adapter_config.json"
    
    if adapter_config_path.exists() and PEFT_AVAILABLE:
        print(f"Loading PEFT/LoRA model from {model_path}")
        
        import json

        # Load the adapter config for base model name
        with open(adapter_config_path, "r") as f:
            adapter_config = json.load(f)
        
        base_model_name = adapter_config.get("base_model_name_or_path", "dmis-lab/biobert-base-cased-v1.2")
        
        # Get label mappings — prefer label_info.json (saved by save_model),
        # fall back to config.json (standard HF format)
        label_info_path = model_path / "label_info.json"
        if label_info_path.exists():
            with open(label_info_path, "r") as f:
                label_info = json.load(f)
            label_to_id = label_info["label_to_id"]
            id_to_label = {int(k): v for k, v in label_info["id_to_label"].items()}
        else:
            config = AutoConfig.from_pretrained(model_path)
            id_to_label = config.id2label
            label_to_id = config.label2id
        
        # Load base model with correct label count
        base_model = AutoModelForTokenClassification.from_pretrained(
            base_model_name,
            num_labels=len(id_to_label),
            id2label=id_to_label,
            label2id=label_to_id,
        )
        
        # Load LoRA adapters
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        print(f"Loading standard model from {model_path}")
        
        # Load standard model
        config = AutoConfig.from_pretrained(model_path)
        model = AutoModelForTokenClassification.from_pretrained(model_path)
        
        id_to_label = config.id2label
        label_to_id = config.label2id
    
    # Convert to proper types
    if isinstance(id_to_label, dict):
        id_to_label = {int(k): v for k, v in id_to_label.items()}
    if isinstance(label_to_id, dict):
        label_to_id = {k: int(v) for k, v in label_to_id.items()}
    
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded to {device}")
    
    return model, tokenizer, label_to_id, id_to_label


def save_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    save_path: str,
    label_to_id: Dict[str, int] = None,
    id_to_label: Dict[int, str] = None,
) -> None:
    """
    Save model and tokenizer to disk.
    
    Args:
        model: Model to save
        tokenizer: Tokenizer to save
        save_path: Directory to save to
        label_to_id: Label mapping to save in config
        id_to_label: ID mapping to save in config
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving model to {save_path}")
    
    # Save model
    model.save_pretrained(save_path)
    
    # Save tokenizer
    tokenizer.save_pretrained(save_path)
    
    # Save label mappings separately for easy access
    if label_to_id is not None:
        import json
        label_info = {
            "label_to_id": label_to_id,
            "id_to_label": {str(k): v for k, v in id_to_label.items()},
        }
        with open(save_path / "label_info.json", "w") as f:
            json.dump(label_info, f, indent=2)
    
    print(f"Model saved successfully")


def print_model_info(model: PreTrainedModel) -> None:
    """Print detailed information about the model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\n" + "=" * 50)
    print("Model Information")
    print("=" * 50)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable ratio: {100*trainable_params/total_params:.2f}%")
    print(f"Memory footprint (approx): {total_params * 4 / 1024**2:.1f} MB (FP32)")
    
    # Check if PEFT model
    if hasattr(model, 'peft_config'):
        print("\nPEFT/LoRA Configuration:")
        for name, config in model.peft_config.items():
            print(f"  Adapter: {name}")
            print(f"    r: {config.r}")
            print(f"    alpha: {config.lora_alpha}")
            print(f"    dropout: {config.lora_dropout}")
            print(f"    target_modules: {config.target_modules}")
    
    print("=" * 50 + "\n")


# Example usage and testing
if __name__ == "__main__":
    print("Testing model creation...")
    
    # Create model with LoRA
    model, tokenizer, label_to_id, id_to_label = create_model(
        model_name="dmis-lab/biobert-base-cased-v1.2",
        use_lora=True,
        lora_r=16,
        lora_alpha=32,
    )
    
    print_model_info(model)
    
    # Test forward pass
    test_text = "Patient has fever and headache."
    inputs = tokenizer(test_text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    print(f"Input text: {test_text}")
    print(f"Output logits shape: {outputs.logits.shape}")
    print(f"Predicted labels: {outputs.logits.argmax(dim=-1)[0].tolist()}")
    
    print("\nModel tests passed!")
