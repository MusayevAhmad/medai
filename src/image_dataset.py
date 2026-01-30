"""
X-Ray Image Dataset Module

Handles loading and preprocessing of chest X-ray images for classification.
Supports COVID-19 Radiography Dataset and similar structured datasets.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split


class XRayDataset(Dataset):
    """
    Dataset class for chest X-ray images.
    
    Expects directory structure:
    root_dir/
        Class1/
            images/
                img1.png
                img2.png
        Class2/
            images/
                img1.png
                ...
    
    Or:
    root_dir/
        Class1/
            img1.png
            img2.png
        Class2/
            img1.png
            ...
    """
    
    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        transform: Optional[Callable] = None,
        class_names: Optional[List[str]] = None,
    ):
        """
        Initialize the dataset.
        
        Args:
            image_paths: List of paths to images
            labels: List of integer labels
            transform: Optional transforms to apply
            class_names: Optional list of class names
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.class_names = class_names or []
        
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_name(self, label: int) -> str:
        """Get class name for a label."""
        if self.class_names and label < len(self.class_names):
            return self.class_names[label]
        return str(label)


def get_transforms(
    image_size: int = 224,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225],
    augment: bool = False,
    rotation: int = 10,
) -> transforms.Compose:
    """
    Get image transforms for training or inference.
    
    Args:
        image_size: Target image size
        mean: Normalization mean
        std: Normalization std
        augment: Whether to apply augmentation
        rotation: Max rotation degrees for augmentation
    
    Returns:
        Composed transforms
    """
    transform_list = []
    
    # Resize
    transform_list.append(transforms.Resize((image_size, image_size)))
    
    # Augmentation (training only)
    if augment:
        transform_list.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(rotation),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        ])
    
    # Convert to tensor and normalize
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    
    return transforms.Compose(transform_list)


def load_dataset_from_directory(
    root_dir: str,
    class_names: Optional[List[str]] = None,
) -> Tuple[List[str], List[int], List[str]]:
    """
    Load image paths and labels from directory structure.
    
    Args:
        root_dir: Root directory containing class folders
        class_names: Optional specific class names to load
    
    Returns:
        Tuple of (image_paths, labels, discovered_class_names)
    """
    root_path = Path(root_dir)
    
    # Discover classes if not specified
    if class_names is None:
        class_names = sorted([
            d.name for d in root_path.iterdir() 
            if d.is_dir() and not d.name.startswith('.')
        ])
    
    image_paths = []
    labels = []
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = root_path / class_name
        
        if not class_dir.exists():
            print(f"Warning: Class directory not found: {class_dir}")
            continue
        
        # Check for images subdirectory
        images_dir = class_dir / "images"
        if images_dir.exists():
            search_dir = images_dir
        else:
            search_dir = class_dir
        
        # Find all images
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
            for img_path in search_dir.glob(ext):
                image_paths.append(str(img_path))
                labels.append(class_idx)
    
    print(f"Loaded {len(image_paths)} images from {len(class_names)} classes")
    for i, name in enumerate(class_names):
        count = labels.count(i)
        print(f"  {name}: {count} images")
    
    return image_paths, labels, class_names


def create_data_loaders(
    root_dir: str,
    class_names: Optional[List[str]] = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        root_dir: Root directory containing class folders
        class_names: Optional specific class names
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        batch_size: Batch size
        image_size: Target image size
        num_workers: Number of data loading workers
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader, class_names)
    """
    # Load all images
    image_paths, labels, class_names = load_dataset_from_directory(
        root_dir, class_names
    )
    
    if len(image_paths) == 0:
        raise ValueError(f"No images found in {root_dir}")
    
    # Stratified split
    # First split: train+val vs test
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        image_paths, labels,
        test_size=test_ratio,
        stratify=labels,
        random_state=seed
    )
    
    # Second split: train vs val
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels,
        test_size=val_ratio_adjusted,
        stratify=train_val_labels,
        random_state=seed
    )
    
    print(f"\nData split:")
    print(f"  Train: {len(train_paths)} images")
    print(f"  Validation: {len(val_paths)} images")
    print(f"  Test: {len(test_paths)} images")
    
    # Create transforms
    train_transform = get_transforms(image_size=image_size, augment=True)
    eval_transform = get_transforms(image_size=image_size, augment=False)
    
    # Create datasets
    train_dataset = XRayDataset(
        train_paths, train_labels, train_transform, class_names
    )
    val_dataset = XRayDataset(
        val_paths, val_labels, eval_transform, class_names
    )
    test_dataset = XRayDataset(
        test_paths, test_labels, eval_transform, class_names
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader, class_names


def compute_class_weights(labels: List[int], num_classes: int) -> torch.Tensor:
    """
    Compute class weights for imbalanced datasets.
    
    Uses inverse frequency weighting: weight = total / (num_classes * count)
    
    Args:
        labels: List of integer labels
        num_classes: Total number of classes
    
    Returns:
        Tensor of class weights
    """
    from collections import Counter
    
    counts = Counter(labels)
    total = len(labels)
    
    weights = []
    for i in range(num_classes):
        count = counts.get(i, 1)  # Avoid division by zero
        weight = total / (num_classes * count)
        weights.append(weight)
    
    return torch.FloatTensor(weights)


if __name__ == "__main__":
    # Test the dataset loading
    import argparse
    
    parser = argparse.ArgumentParser(description="Test X-Ray dataset loading")
    parser.add_argument("--data-dir", type=str, required=True, help="Data directory")
    args = parser.parse_args()
    
    train_loader, val_loader, test_loader, class_names = create_data_loaders(
        args.data_dir,
        batch_size=4,
    )
    
    # Test loading a batch
    images, labels = next(iter(train_loader))
    print(f"\nBatch shape: {images.shape}")
    print(f"Labels: {labels.tolist()}")
    print(f"Class names: {[class_names[l] for l in labels.tolist()]}")
