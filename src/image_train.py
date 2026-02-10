"""
X-Ray Image Model Training Script

Trains a CNN classifier for chest X-ray classification.
Supports multiple backbones and training configurations.
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from tqdm import tqdm
import yaml

from src.image_dataset import create_data_loaders, compute_class_weights
from src.image_model import create_model, save_model, freeze_backbone


def train_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
    epoch: int,
) -> Tuple[float, float]:
    """
    Train for one epoch.
    
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{100.*correct/total:.2f}%"
        })
    
    avg_loss = running_loss / len(train_loader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: str,
) -> Tuple[float, float]:
    """
    Validate the model.
    
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    avg_loss = running_loss / len(val_loader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def evaluate(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: str,
    class_names: List[str],
) -> Dict:
    """
    Evaluate the model and compute detailed metrics.
    
    Returns:
        Dictionary with metrics
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Compute metrics
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        classification_report, confusion_matrix
    )
    import numpy as np
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    # Per-class metrics
    report = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        output_dict=True
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "predictions": all_preds,
        "labels": all_labels,
        "probabilities": [p.tolist() for p in all_probs],
    }


def train(
    data_dir: str,
    output_dir: str,
    config: Dict,
) -> Dict:
    """
    Main training function.
    
    Args:
        data_dir: Path to dataset directory
        output_dir: Path to save outputs
        config: Training configuration
    
    Returns:
        Training results dictionary
    """
    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_dir) / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("X-Ray Classification Training")
    print("=" * 60)
    
    # Set seed
    seed = config.get("training", {}).get("seed", 42)
    torch.manual_seed(seed)
    
    # Create data loaders
    print("\nLoading data...")
    class_names = config.get("dataset", {}).get("classes")
    train_loader, val_loader, test_loader, class_names = create_data_loaders(
        root_dir=data_dir,
        class_names=class_names,
        train_ratio=config.get("dataset", {}).get("train_ratio", 0.8),
        val_ratio=config.get("dataset", {}).get("val_ratio", 0.1),
        test_ratio=config.get("dataset", {}).get("test_ratio", 0.1),
        batch_size=config.get("training", {}).get("batch_size", 32),
        image_size=config.get("preprocessing", {}).get("image_size", 224),
        seed=seed,
    )
    
    # Create model
    print("\nCreating model...")
    model, device = create_model(
        num_classes=len(class_names),
        backbone=config.get("model", {}).get("backbone", "densenet121"),
        pretrained=config.get("model", {}).get("pretrained", True),
        dropout=config.get("model", {}).get("dropout", 0.3),
    )
    
    # Loss function with class weights
    train_labels = [label for _, label in train_loader.dataset]
    class_weights = compute_class_weights(train_labels, len(class_names))
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print(f"Class weights: {class_weights.tolist()}")
    
    # Optimizer
    lr = config.get("training", {}).get("learning_rate", 0.0001)
    weight_decay = config.get("training", {}).get("weight_decay", 0.0001)
    optimizer_name = config.get("training", {}).get("optimizer", "adam")
    
    if optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    
    # Learning rate scheduler
    epochs = config.get("training", {}).get("epochs", 20)
    scheduler_name = config.get("training", {}).get("scheduler", "cosine")
    
    if scheduler_name == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_name == "step":
        scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    else:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3)
    
    # Training loop
    print(f"\nStarting training for {epochs} epochs...")
    print("-" * 60)
    
    best_val_acc = 0.0
    patience = config.get("training", {}).get("early_stopping_patience", 5)
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        if scheduler_name == "plateau":
            scheduler.step(val_loss)
        else:
            scheduler.step()
        
        # Record history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        print(f"\nEpoch {epoch}/{epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            save_model(
                model,
                run_dir / "best_model.pt",
                class_names,
                config,
            )
            print(f"  New best model saved! (Val Acc: {val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.1f}s")
    
    # Final evaluation on test set
    print("\n" + "=" * 60)
    print("Final Evaluation on Test Set")
    print("=" * 60)
    
    # Load best model
    checkpoint = torch.load(run_dir / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    eval_results = evaluate(model, test_loader, device, class_names)
    
    print(f"\nTest Results:")
    print(f"  Accuracy: {eval_results['accuracy']:.4f}")
    print(f"  Precision: {eval_results['precision']:.4f}")
    print(f"  Recall: {eval_results['recall']:.4f}")
    print(f"  F1 Score: {eval_results['f1']:.4f}")
    
    print("\nPer-class Performance:")
    for cls in class_names:
        if cls in eval_results['classification_report']:
            metrics = eval_results['classification_report'][cls]
            print(f"  {cls}:")
            print(f"    Precision: {metrics['precision']:.4f}")
            print(f"    Recall: {metrics['recall']:.4f}")
            print(f"    F1: {metrics['f1-score']:.4f}")
    
    # Save training info
    training_info = {
        "timestamp": timestamp,
        "config": config,
        "class_names": class_names,
        "training_time": training_time,
        "epochs_trained": len(history["train_loss"]),
        "best_val_accuracy": best_val_acc,
        "test_results": {
            "accuracy": eval_results["accuracy"],
            "precision": eval_results["precision"],
            "recall": eval_results["recall"],
            "f1": eval_results["f1"],
        },
        "history": history,
    }
    
    with open(run_dir / "training_info.json", "w") as f:
        json.dump(training_info, f, indent=2)
    
    print(f"\nResults saved to: {run_dir}")
    
    return training_info


def main():
    parser = argparse.ArgumentParser(description="Train X-Ray classifier")
    parser.add_argument(
        "--config", type=str, default="config_xray.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Override data directory from config"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Override output directory from config"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override number of epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Override batch size"
    )
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Apply overrides
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    
    # Determine paths
    data_dir = args.data_dir or config.get("dataset", {}).get("root_dir", "data/xray")
    output_dir = args.output_dir or config.get("output", {}).get("model_dir", "outputs/xray_models")
    
    # Check if data exists
    data_path = Path(data_dir)
    expected_dataset = data_path / "COVID-19_Radiography_Dataset"
    
    if expected_dataset.exists():
        data_dir = str(expected_dataset)
    elif not data_path.exists():
        print(f"Error: Data directory not found: {data_dir}")
        print("\nPlease download the COVID-19 Radiography Dataset:")
        print("https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database")
        return
    
    # Train
    train(data_dir, output_dir, config)


if __name__ == "__main__":
    main()
