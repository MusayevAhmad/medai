# Symptom NER Fine-Tuning Source Package
"""
src/ - Core source code for the NER pipeline

Modules:
    - dataset: NER dataset class with tokenization and label alignment
    - model: BioBERT + LoRA model configuration
    - train: Training loop with HuggingFace Trainer
    - evaluate: Entity-level evaluation metrics
    - predict: CLI inference script
"""

__version__ = "0.1.0"
