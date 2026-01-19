#!/usr/bin/env python3
"""
Training Script for Medical NER

Fine-tunes BioBERT with LoRA on BC5CDR dataset using HuggingFace Trainer.
Supports automatic device detection (MPS/CUDA/CPU) and checkpoint saving.

Usage:
    python src/train.py
    python src/train.py --config config.yaml
    python src/train.py --epochs 5 --batch-size 16
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import create_model, save_model, print_model_info, get_device
from src.dataset import NERDataset, get_label_mappings, DEFAULT_LABEL_LIST
from src.evaluate import compute_ner_metrics, seqeval_compute_metrics

# Optional YAML config loading
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    if not YAML_AVAILABLE:
        raise ImportError("PyYAML required for config loading. Install with: pip install pyyaml")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    return config


def create_compute_metrics(label_list: List[str]):
    """
    Create a compute_metrics function for the Trainer.
    
    Args:
        label_list: List of label strings
    
    Returns:
        Function that computes metrics from predictions
    """
    label_to_id, id_to_label = get_label_mappings(label_list)
    
    def compute_metrics(eval_pred):
        """Compute NER metrics from Trainer predictions."""
        predictions, labels = eval_pred
        
        # Get predicted labels (argmax of logits)
        predictions = np.argmax(predictions, axis=-1)
        
        # Convert to lists and filter out -100 (ignored tokens)
        true_labels = []
        pred_labels = []
        
        for pred_seq, label_seq in zip(predictions, labels):
            true_seq = []
            pred_seq_filtered = []
            
            for pred, label in zip(pred_seq, label_seq):
                if label != -100:  # Ignore padding and special tokens
                    true_seq.append(id_to_label.get(label, "O"))
                    pred_seq_filtered.append(id_to_label.get(pred, "O"))
            
            true_labels.append(true_seq)
            pred_labels.append(pred_seq_filtered)
        
        # Compute metrics using seqeval
        return seqeval_compute_metrics(true_labels, pred_labels)
    
    return compute_metrics


def train(
    data_dir: str = "data/processed",
    output_dir: str = "outputs/models",
    model_name: str = "dmis-lab/biobert-base-cased-v1.2",
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-4,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    max_length: int = 128,
    use_lora: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    seed: int = 42,
    logging_steps: int = 50,
    eval_steps: int = 200,
    save_steps: int = 200,
    save_total_limit: int = 2,
    early_stopping_patience: int = 3,
    fp16: bool = False,
    gradient_accumulation_steps: int = 1,
) -> Dict:
    """
    Train the NER model.
    
    Args:
        data_dir: Directory containing processed dataset
        output_dir: Directory to save model checkpoints
        model_name: HuggingFace model name
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        weight_decay: Weight decay for AdamW
        warmup_ratio: Ratio of warmup steps
        max_length: Maximum sequence length
        use_lora: Whether to use LoRA
        lora_r: LoRA rank
        lora_alpha: LoRA alpha scaling
        lora_dropout: LoRA dropout
        seed: Random seed
        logging_steps: Steps between logging
        eval_steps: Steps between evaluation
        save_steps: Steps between checkpoints
        save_total_limit: Maximum checkpoints to keep
        early_stopping_patience: Patience for early stopping
        fp16: Use FP16 training (CUDA only)
        gradient_accumulation_steps: Gradient accumulation steps
    
    Returns:
        Dict with training results
    """
    # Set seed
    set_seed(seed)
    
    print("=" * 60)
    print("Medical NER Training Pipeline")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"LoRA: {use_lora} (r={lora_r}, alpha={lora_alpha})")
    print(f"Device: {get_device()}")
    print()
    
    # Load label info
    label_info_path = os.path.join(data_dir, "label_info.json")
    if os.path.exists(label_info_path):
        with open(label_info_path, "r") as f:
            label_info = json.load(f)
        label_list = label_info.get("label_list", DEFAULT_LABEL_LIST)
    else:
        label_list = DEFAULT_LABEL_LIST
    
    label_to_id, id_to_label = get_label_mappings(label_list)
    num_labels = len(label_list)
    
    print(f"Labels ({num_labels}): {label_list}")
    
    # Create model and tokenizer
    model, tokenizer, label_to_id, id_to_label = create_model(
        model_name=model_name,
        label_list=label_list,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
    
    print_model_info(model)
    
    # Load processed dataset
    print(f"Loading dataset from {data_dir}...")
    dataset_dict = load_from_disk(data_dir)
    
    # Create NERDataset instances
    train_data = [dict(ex) for ex in dataset_dict["train"]]
    val_data = [dict(ex) for ex in dataset_dict["validation"]]
    
    train_dataset = NERDataset(
        data=train_data,
        tokenizer=tokenizer,
        max_length=max_length,
        label_to_id=label_to_id,
    )
    
    val_dataset = NERDataset(
        data=val_data,
        tokenizer=tokenizer,
        max_length=max_length,
        label_to_id=label_to_id,
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Data collator for dynamic padding
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding=True,
        max_length=max_length,
    )
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(output_dir, f"run_{timestamp}")
    
    # Determine device-specific settings
    device = get_device()
    use_fp16 = fp16 and device == "cuda"  # FP16 only works reliably on CUDA
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=run_output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        gradient_accumulation_steps=gradient_accumulation_steps,
        
        # Evaluation
        eval_strategy="steps",
        eval_steps=eval_steps,
        
        # Saving
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        
        # Logging
        logging_dir=os.path.join(run_output_dir, "logs"),
        logging_steps=logging_steps,
        logging_first_step=True,
        report_to="none",  # Disable wandb by default
        
        # Performance
        fp16=use_fp16,
        dataloader_num_workers=0,  # Avoid multiprocessing issues on Mac
        
        # Misc
        seed=seed,
        remove_unused_columns=False,
        push_to_hub=False,
    )
    
    # Create compute_metrics function
    compute_metrics = create_compute_metrics(label_list)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
    )
    
    # Train
    print("\nStarting training...")
    print("-" * 60)
    
    train_result = trainer.train()
    
    print("-" * 60)
    print("Training completed!")
    
    # Evaluate on validation set
    print("\nFinal evaluation on validation set:")
    eval_results = trainer.evaluate()
    
    for key, value in eval_results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
    
    # Save final model
    final_model_dir = os.path.join(run_output_dir, "final_model")
    save_model(model, tokenizer, final_model_dir, label_to_id, id_to_label)
    
    # Save training info
    training_info = {
        "model_name": model_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "use_lora": use_lora,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "seed": seed,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "eval_results": {k: float(v) if isinstance(v, (int, float)) else v 
                        for k, v in eval_results.items()},
        "train_runtime": train_result.metrics.get("train_runtime", 0),
    }
    
    with open(os.path.join(run_output_dir, "training_info.json"), "w") as f:
        json.dump(training_info, f, indent=2)
    
    print(f"\nModel saved to: {final_model_dir}")
    print(f"Training info saved to: {run_output_dir}/training_info.json")
    
    return {
        "model_dir": final_model_dir,
        "eval_results": eval_results,
        "training_info": training_info,
    }


def main():
    parser = argparse.ArgumentParser(description="Train Medical NER model")
    
    # Data arguments
    parser.add_argument("--data-dir", type=str, default="data/processed",
                        help="Directory containing processed dataset")
    parser.add_argument("--output-dir", type=str, default="outputs/models",
                        help="Directory to save model checkpoints")
    
    # Model arguments
    parser.add_argument("--model-name", type=str, default="dmis-lab/biobert-base-cased-v1.2",
                        help="HuggingFace model name")
    parser.add_argument("--max-length", type=int, default=128,
                        help="Maximum sequence length")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--warmup-ratio", type=float, default=0.1,
                        help="Warmup ratio")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1,
                        help="Gradient accumulation steps")
    
    # LoRA arguments
    parser.add_argument("--no-lora", action="store_true",
                        help="Disable LoRA (full fine-tuning)")
    parser.add_argument("--lora-r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32,
                        help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.1,
                        help="LoRA dropout")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--fp16", action="store_true",
                        help="Use FP16 training (CUDA only)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file")
    
    args = parser.parse_args()
    
    # Load config file if provided
    config = {}
    if args.config and YAML_AVAILABLE:
        config = load_config(args.config)
    
    # Merge config with command line arguments (CLI takes precedence)
    train_config = {
        "data_dir": args.data_dir,
        "output_dir": args.output_dir,
        "model_name": config.get("model", {}).get("name", args.model_name),
        "max_length": config.get("model", {}).get("max_length", args.max_length),
        "epochs": config.get("training", {}).get("epochs", args.epochs),
        "batch_size": config.get("training", {}).get("batch_size", args.batch_size),
        "learning_rate": config.get("training", {}).get("learning_rate", args.learning_rate),
        "weight_decay": config.get("training", {}).get("weight_decay", args.weight_decay),
        "warmup_ratio": config.get("training", {}).get("warmup_ratio", args.warmup_ratio),
        "gradient_accumulation_steps": config.get("training", {}).get("gradient_accumulation_steps", args.gradient_accumulation_steps),
        "use_lora": not args.no_lora and config.get("lora", {}).get("enabled", True),
        "lora_r": config.get("lora", {}).get("r", args.lora_r),
        "lora_alpha": config.get("lora", {}).get("alpha", args.lora_alpha),
        "lora_dropout": config.get("lora", {}).get("dropout", args.lora_dropout),
        "seed": config.get("training", {}).get("seed", args.seed),
        "fp16": args.fp16 or config.get("training", {}).get("fp16", False),
    }
    
    # Run training
    result = train(**train_config)
    
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"Final F1 Score: {result['eval_results'].get('eval_f1', 'N/A'):.4f}")
    print(f"Model saved to: {result['model_dir']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
