#!/usr/bin/env python3
"""
Evaluation Metrics for NER

Computes entity-level metrics for Named Entity Recognition:
- Strict matching: entity text and type must match exactly
- Partial matching: overlapping spans count as partial match

Uses seqeval library for standard NER evaluation.

Usage:
    python src/evaluate.py --model-path outputs/models/final_model
    python src/evaluate.py --model-path outputs/models/final_model --data-dir data/processed
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_from_disk

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Optional seqeval import
try:
    from seqeval.metrics import (
        classification_report,
        f1_score,
        precision_score,
        recall_score,
        accuracy_score,
    )
    from seqeval.scheme import IOB2
    SEQEVAL_AVAILABLE = True
except ImportError:
    SEQEVAL_AVAILABLE = False
    print("Warning: seqeval not installed. Install with: pip install seqeval")


def seqeval_compute_metrics(
    true_labels: List[List[str]],
    pred_labels: List[List[str]],
) -> Dict[str, float]:
    """
    Compute NER metrics using seqeval library.
    
    Args:
        true_labels: List of sequences of true label strings
        pred_labels: List of sequences of predicted label strings
    
    Returns:
        Dict with precision, recall, f1, and accuracy
    """
    if not SEQEVAL_AVAILABLE:
        return compute_metrics_manual(true_labels, pred_labels)
    
    return {
        "precision": precision_score(true_labels, pred_labels),
        "recall": recall_score(true_labels, pred_labels),
        "f1": f1_score(true_labels, pred_labels),
        "accuracy": accuracy_score(true_labels, pred_labels),
    }


def compute_metrics_manual(
    true_labels: List[List[str]],
    pred_labels: List[List[str]],
) -> Dict[str, float]:
    """
    Manual implementation of NER metrics (fallback when seqeval not available).
    
    Uses strict entity matching: start, end, and type must all match.
    """
    def extract_entities(labels: List[str]) -> List[Tuple[str, int, int]]:
        """Extract entities as (type, start, end) tuples."""
        entities = []
        current_entity = None
        current_start = None
        
        for i, label in enumerate(labels):
            if label.startswith("B-"):
                if current_entity is not None:
                    entities.append((current_entity, current_start, i))
                current_entity = label[2:]
                current_start = i
            elif label.startswith("I-"):
                if current_entity is None:
                    # I- without B- (treat as B-)
                    current_entity = label[2:]
                    current_start = i
                elif label[2:] != current_entity:
                    # Type mismatch
                    entities.append((current_entity, current_start, i))
                    current_entity = label[2:]
                    current_start = i
            else:  # O or other
                if current_entity is not None:
                    entities.append((current_entity, current_start, i))
                    current_entity = None
                    current_start = None
        
        if current_entity is not None:
            entities.append((current_entity, current_start, len(labels)))
        
        return entities
    
    # Count true positives, false positives, false negatives
    tp = 0
    fp = 0
    fn = 0
    total_tokens = 0
    correct_tokens = 0
    
    for true_seq, pred_seq in zip(true_labels, pred_labels):
        true_entities = set(extract_entities(true_seq))
        pred_entities = set(extract_entities(pred_seq))
        
        tp += len(true_entities & pred_entities)
        fp += len(pred_entities - true_entities)
        fn += len(true_entities - pred_entities)
        
        # Token-level accuracy
        total_tokens += len(true_seq)
        correct_tokens += sum(t == p for t, p in zip(true_seq, pred_seq))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def compute_ner_metrics(
    true_labels: List[List[str]],
    pred_labels: List[List[str]],
    mode: str = "strict",
) -> Dict[str, float]:
    """
    Compute NER metrics with different matching modes.
    
    Args:
        true_labels: List of sequences of true label strings
        pred_labels: List of sequences of predicted label strings
        mode: "strict" (exact match) or "partial" (overlapping spans)
    
    Returns:
        Dict with metrics
    """
    if mode == "strict":
        return seqeval_compute_metrics(true_labels, pred_labels)
    elif mode == "partial":
        return compute_partial_metrics(true_labels, pred_labels)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'strict' or 'partial'.")


def compute_partial_metrics(
    true_labels: List[List[str]],
    pred_labels: List[List[str]],
) -> Dict[str, float]:
    """
    Compute NER metrics with partial matching.
    
    Partial match: predicted entity overlaps with true entity of same type.
    """
    def extract_entities_with_spans(labels: List[str]) -> List[Tuple[str, int, int]]:
        """Extract entities as (type, start, end) tuples."""
        entities = []
        current_entity = None
        current_start = None
        
        for i, label in enumerate(labels):
            if label.startswith("B-"):
                if current_entity is not None:
                    entities.append((current_entity, current_start, i))
                current_entity = label[2:]
                current_start = i
            elif label.startswith("I-"):
                if current_entity is None:
                    current_entity = label[2:]
                    current_start = i
                elif label[2:] != current_entity:
                    entities.append((current_entity, current_start, i))
                    current_entity = label[2:]
                    current_start = i
            else:
                if current_entity is not None:
                    entities.append((current_entity, current_start, i))
                    current_entity = None
        
        if current_entity is not None:
            entities.append((current_entity, current_start, len(labels)))
        
        return entities
    
    def spans_overlap(span1: Tuple[int, int], span2: Tuple[int, int]) -> bool:
        """Check if two spans overlap."""
        return span1[0] < span2[1] and span2[0] < span1[1]
    
    total_true = 0
    total_pred = 0
    correct = 0
    partial = 0
    
    for true_seq, pred_seq in zip(true_labels, pred_labels):
        true_entities = extract_entities_with_spans(true_seq)
        pred_entities = extract_entities_with_spans(pred_seq)
        
        total_true += len(true_entities)
        total_pred += len(pred_entities)
        
        # Match predictions to true entities
        matched_true = set()
        for pred_type, pred_start, pred_end in pred_entities:
            for i, (true_type, true_start, true_end) in enumerate(true_entities):
                if i in matched_true:
                    continue
                if pred_type == true_type and spans_overlap((pred_start, pred_end), (true_start, true_end)):
                    if pred_start == true_start and pred_end == true_end:
                        correct += 1
                    else:
                        partial += 1
                    matched_true.add(i)
                    break
    
    # Calculate metrics
    # For partial matching, count partial matches as 0.5
    effective_correct = correct + 0.5 * partial
    
    precision = effective_correct / total_pred if total_pred > 0 else 0.0
    recall = effective_correct / total_true if total_true > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_matches": correct,
        "partial_matches": partial,
        "total_true": total_true,
        "total_pred": total_pred,
    }


def get_classification_report(
    true_labels: List[List[str]],
    pred_labels: List[List[str]],
) -> str:
    """
    Get a detailed classification report per entity type.
    
    Returns:
        String with formatted classification report
    """
    if SEQEVAL_AVAILABLE:
        return classification_report(true_labels, pred_labels, digits=4)
    else:
        # Simple fallback report
        metrics = compute_metrics_manual(true_labels, pred_labels)
        return (
            f"Overall Metrics:\n"
            f"  Precision: {metrics['precision']:.4f}\n"
            f"  Recall: {metrics['recall']:.4f}\n"
            f"  F1: {metrics['f1']:.4f}\n"
            f"  Accuracy: {metrics['accuracy']:.4f}\n"
        )


def evaluate_model(
    model_path: str,
    data_dir: str = "data/processed",
    split: str = "test",
    batch_size: int = 16,
    mode: str = "strict",
) -> Dict[str, any]:
    """
    Evaluate a trained model on a dataset split.
    
    Args:
        model_path: Path to saved model
        data_dir: Directory containing processed dataset
        split: Dataset split to evaluate on
        batch_size: Batch size for inference
        mode: Evaluation mode ("strict" or "partial")
    
    Returns:
        Dict with evaluation results
    """
    from src.model import load_model
    from src.dataset import NERDataset, get_label_mappings
    from torch.utils.data import DataLoader
    
    print(f"Loading model from {model_path}...")
    model, tokenizer, label_to_id, id_to_label = load_model(model_path)
    
    print(f"Loading {split} data from {data_dir}...")
    dataset_dict = load_from_disk(data_dir)
    
    if split not in dataset_dict:
        raise ValueError(f"Split '{split}' not found. Available: {list(dataset_dict.keys())}")
    
    # Create dataset
    data = [dict(ex) for ex in dataset_dict[split]]
    dataset = NERDataset(
        data=data,
        tokenizer=tokenizer,
        max_length=128,
        label_to_id=label_to_id,
    )
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Run inference
    model.eval()
    all_true_labels = []
    all_pred_labels = []
    
    print(f"Running inference on {len(dataset)} examples...")
    
    with torch.no_grad():
        for batch in dataloader:
            # Move to device
            inputs = {
                "input_ids": batch["input_ids"].to(model.device),
                "attention_mask": batch["attention_mask"].to(model.device),
            }
            labels = batch["labels"]
            
            # Forward pass
            outputs = model(**inputs)
            predictions = outputs.logits.argmax(dim=-1).cpu()
            
            # Convert to label strings
            for pred_seq, label_seq in zip(predictions, labels):
                true_seq = []
                pred_seq_filtered = []
                
                for pred, label in zip(pred_seq, label_seq):
                    if label.item() != -100:
                        true_seq.append(id_to_label.get(label.item(), "O"))
                        pred_seq_filtered.append(id_to_label.get(pred.item(), "O"))
                
                all_true_labels.append(true_seq)
                all_pred_labels.append(pred_seq_filtered)
    
    # Compute metrics
    print(f"\nComputing {mode} matching metrics...")
    metrics = compute_ner_metrics(all_true_labels, all_pred_labels, mode=mode)
    
    # Get classification report
    report = get_classification_report(all_true_labels, all_pred_labels)
    
    print("\n" + "=" * 60)
    print(f"Evaluation Results ({split} split, {mode} matching)")
    print("=" * 60)
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    if 'accuracy' in metrics:
        print(f"Accuracy: {metrics['accuracy']:.4f}")
    print("\nDetailed Report:")
    print(report)
    
    return {
        "metrics": metrics,
        "report": report,
        "num_examples": len(dataset),
        "split": split,
        "mode": mode,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate NER model")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to saved model")
    parser.add_argument("--data-dir", type=str, default="data/processed",
                        help="Directory containing processed dataset")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "validation", "test"],
                        help="Dataset split to evaluate on")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for inference")
    parser.add_argument("--mode", type=str, default="strict",
                        choices=["strict", "partial"],
                        help="Matching mode for evaluation")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save evaluation results (JSON)")
    
    args = parser.parse_args()
    
    results = evaluate_model(
        model_path=args.model_path,
        data_dir=args.data_dir,
        split=args.split,
        batch_size=args.batch_size,
        mode=args.mode,
    )
    
    if args.output:
        # Save results to JSON
        output_data = {
            "model_path": args.model_path,
            "data_dir": args.data_dir,
            "split": args.split,
            "mode": args.mode,
            "metrics": results["metrics"],
            "num_examples": results["num_examples"],
        }
        
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
