#!/usr/bin/env python3
"""
Data Preparation Script for NER Fine-Tuning

Downloads BC5CDR dataset from HuggingFace, converts span annotations to BIO format,
and creates train/val/test splits with consistent label encoding.

Usage:
    python data/prepare_data.py
    python data/prepare_data.py --include-synthetic
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# Label scheme for BIO tagging
LABEL_LIST = [
    "O",           # 0 - Outside any entity
    "B-Disease",   # 1 - Beginning of disease entity
    "I-Disease",   # 2 - Inside disease entity
    "B-Chemical",  # 3 - Beginning of chemical entity
    "I-Chemical",  # 4 - Inside chemical entity
    "B-Symptom",   # 5 - Beginning of symptom entity
    "I-Symptom",   # 6 - Inside symptom entity
]

LABEL_TO_ID = {label: idx for idx, label in enumerate(LABEL_LIST)}
ID_TO_LABEL = {idx: label for idx, label in enumerate(LABEL_LIST)}


def tokenize_and_align_labels(
    text: str,
    entities: List[Dict[str, Any]],
    entity_type_map: Dict[str, str] = None
) -> Tuple[List[str], List[str]]:
    """
    Convert text with span annotations to token-level BIO labels.
    
    Args:
        text: Raw text string
        entities: List of entity dicts with 'offsets', 'text', and 'type' keys
        entity_type_map: Optional mapping from dataset entity types to our labels
    
    Returns:
        Tuple of (tokens, labels) where labels are BIO format strings
    """
    if entity_type_map is None:
        entity_type_map = {
            "Disease": "Disease",
            "Chemical": "Chemical",
            "Symptom": "Symptom",
        }
    
    # Simple whitespace tokenization (model tokenizer will handle subwords later)
    tokens = text.split()
    labels = ["O"] * len(tokens)
    
    # Build character offset to token index mapping
    char_to_token = {}
    current_char = 0
    for token_idx, token in enumerate(tokens):
        for char_offset in range(current_char, current_char + len(token)):
            char_to_token[char_offset] = token_idx
        current_char += len(token) + 1  # +1 for space
    
    # Process each entity
    for entity in entities:
        entity_type = entity.get("type", "Disease")
        mapped_type = entity_type_map.get(entity_type, entity_type)
        
        # Handle different offset formats
        offsets = entity.get("offsets", [])
        if not offsets:
            continue
            
        # Offsets can be list of [start, end] pairs or single pair
        if isinstance(offsets[0], list):
            start, end = offsets[0]
        else:
            start, end = offsets[0], offsets[1] if len(offsets) > 1 else offsets[0] + len(entity.get("text", [""])[0])
        
        # Find tokens that overlap with this entity span
        entity_token_indices = set()
        for char_offset in range(start, end):
            if char_offset in char_to_token:
                entity_token_indices.add(char_to_token[char_offset])
        
        # Assign BIO labels
        if entity_token_indices:
            sorted_indices = sorted(entity_token_indices)
            for i, token_idx in enumerate(sorted_indices):
                if i == 0:
                    labels[token_idx] = f"B-{mapped_type}"
                else:
                    labels[token_idx] = f"I-{mapped_type}"
    
    return tokens, labels


def process_bc5cdr_example(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a single BC5CDR example from BigBio format.
    
    BC5CDR BigBio format has:
    - passages: list of dicts with 'text' and 'offsets'
    - entities: list of dicts with 'offsets', 'text', 'type'
    """
    # Combine all passages into one text
    full_text = ""
    passage_offsets = []
    
    for passage in example.get("passages", []):
        if full_text:
            full_text += " "
        start_offset = len(full_text)
        passage_text = passage.get("text", [""])[0] if isinstance(passage.get("text"), list) else passage.get("text", "")
        full_text += passage_text
        passage_offsets.append((start_offset, len(full_text)))
    
    # Process entities
    entities = []
    for entity in example.get("entities", []):
        entity_type = entity.get("type", "Disease")
        # Map BC5CDR types
        if entity_type.lower() in ["disease", "disorder"]:
            mapped_type = "Disease"
        elif entity_type.lower() in ["chemical", "drug"]:
            mapped_type = "Chemical"
        else:
            mapped_type = entity_type
        
        entities.append({
            "type": mapped_type,
            "offsets": entity.get("offsets", []),
            "text": entity.get("text", []),
        })
    
    # Convert to BIO format
    tokens, labels = tokenize_and_align_labels(full_text, entities)
    
    # Convert string labels to IDs
    label_ids = [LABEL_TO_ID.get(label, 0) for label in labels]
    
    return {
        "id": example.get("id", ""),
        "tokens": tokens,
        "labels": label_ids,
        "ner_tags": labels,  # Keep string labels for debugging
        "text": full_text,
    }


def load_and_process_bc5cdr() -> DatasetDict:
    """
    Load BC5CDR dataset from HuggingFace and process to BIO format.
    
    Returns:
        DatasetDict with train, validation, and test splits
    """
    print("Loading BC5CDR dataset from HuggingFace...")
    
    try:
        # Try loading the BigBio version first
        dataset = load_dataset("bigbio/bc5cdr", "bc5cdr_bigbio_kb", trust_remote_code=True)
    except Exception as e:
        print(f"Could not load bigbio/bc5cdr: {e}")
        print("Trying alternative loading method...")
        try:
            # Try the standard version
            dataset = load_dataset("bc5cdr", trust_remote_code=True)
        except Exception as e2:
            print(f"Could not load bc5cdr: {e2}")
            print("Creating minimal example dataset for testing...")
            return create_example_dataset()
    
    print(f"Dataset loaded. Splits: {list(dataset.keys())}")
    
    # Process each split
    processed_splits = {}
    for split_name in ["train", "validation", "test"]:
        if split_name not in dataset:
            print(f"Warning: {split_name} split not found, skipping...")
            continue
            
        print(f"Processing {split_name} split ({len(dataset[split_name])} examples)...")
        
        processed_examples = []
        for example in dataset[split_name]:
            try:
                processed = process_bc5cdr_example(example)
                if processed["tokens"]:  # Only keep non-empty examples
                    processed_examples.append(processed)
            except Exception as e:
                print(f"Warning: Could not process example {example.get('id', 'unknown')}: {e}")
                continue
        
        processed_splits[split_name] = Dataset.from_list(processed_examples)
        print(f"  Processed {len(processed_examples)} examples")
    
    return DatasetDict(processed_splits)


def create_example_dataset() -> DatasetDict:
    """
    Create a minimal example dataset for testing when BC5CDR is unavailable.
    """
    examples = [
        {
            "id": "example_1",
            "tokens": ["Patient", "presents", "with", "fever", "and", "headache", "."],
            "labels": [0, 0, 0, 1, 0, 1, 0],  # B-Disease
            "ner_tags": ["O", "O", "O", "B-Disease", "O", "B-Disease", "O"],
            "text": "Patient presents with fever and headache.",
        },
        {
            "id": "example_2",
            "tokens": ["Aspirin", "is", "used", "to", "treat", "inflammation", "."],
            "labels": [3, 0, 0, 0, 0, 1, 0],  # B-Chemical, B-Disease
            "ner_tags": ["B-Chemical", "O", "O", "O", "O", "B-Disease", "O"],
            "text": "Aspirin is used to treat inflammation.",
        },
    ] * 50  # Repeat to have some training data
    
    # Create splits
    train_examples = examples[:80]
    val_examples = examples[80:90]
    test_examples = examples[90:]
    
    return DatasetDict({
        "train": Dataset.from_list(train_examples),
        "validation": Dataset.from_list(val_examples),
        "test": Dataset.from_list(test_examples),
    })


def load_synthetic_symptoms(csv_path: str) -> Dataset:
    """
    Load synthetic symptom examples from CSV file.
    
    CSV format:
        text,entities
        "persistent headache and nausea","headache:SYMPTOM:11:19;nausea:SYMPTOM:24:30"
    
    Args:
        csv_path: Path to the CSV file
    
    Returns:
        Dataset with processed examples
    """
    if not os.path.exists(csv_path):
        print(f"Synthetic symptoms file not found: {csv_path}")
        return None
    
    print(f"Loading synthetic symptoms from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    processed_examples = []
    for idx, row in df.iterrows():
        text = row["text"]
        entities_str = row.get("entities", "")
        
        # Parse entities string: "headache:SYMPTOM:11:19;nausea:SYMPTOM:24:30"
        entities = []
        if entities_str and pd.notna(entities_str):
            for entity_str in entities_str.split(";"):
                parts = entity_str.strip().split(":")
                if len(parts) >= 4:
                    entity_text, entity_type, start, end = parts[0], parts[1], int(parts[2]), int(parts[3])
                    entities.append({
                        "type": entity_type.capitalize(),
                        "offsets": [[start, end]],
                        "text": [entity_text],
                    })
        
        # Convert to BIO format
        tokens, labels = tokenize_and_align_labels(text, entities)
        label_ids = [LABEL_TO_ID.get(label, 0) for label in labels]
        
        processed_examples.append({
            "id": f"synthetic_{idx}",
            "tokens": tokens,
            "labels": label_ids,
            "ner_tags": labels,
            "text": text,
        })
    
    print(f"Loaded {len(processed_examples)} synthetic examples")
    return Dataset.from_list(processed_examples)


def save_processed_data(dataset: DatasetDict, output_dir: str) -> None:
    """
    Save processed dataset to disk.
    
    Args:
        dataset: DatasetDict to save
        output_dir: Directory to save to
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as HuggingFace dataset
    dataset.save_to_disk(output_dir)
    print(f"Saved processed dataset to {output_dir}")
    
    # Also save label mappings
    label_info = {
        "label_list": LABEL_LIST,
        "label_to_id": LABEL_TO_ID,
        "id_to_label": ID_TO_LABEL,
    }
    
    with open(os.path.join(output_dir, "label_info.json"), "w") as f:
        json.dump(label_info, f, indent=2)
    print(f"Saved label info to {output_dir}/label_info.json")


def main():
    parser = argparse.ArgumentParser(description="Prepare NER datasets for fine-tuning")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Directory to save processed data",
    )
    parser.add_argument(
        "--include-synthetic",
        action="store_true",
        help="Include synthetic symptom examples in training",
    )
    parser.add_argument(
        "--synthetic-path",
        type=str,
        default="data/synthetic_symptoms.csv",
        help="Path to synthetic symptoms CSV file",
    )
    args = parser.parse_args()
    
    # Change to project root directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    print("=" * 60)
    print("NER Data Preparation Pipeline")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print(f"Include synthetic: {args.include_synthetic}")
    print()
    
    # Load and process BC5CDR
    dataset = load_and_process_bc5cdr()
    
    # Optionally add synthetic symptoms
    if args.include_synthetic:
        synthetic_dataset = load_synthetic_symptoms(args.synthetic_path)
        if synthetic_dataset is not None:
            # Add synthetic examples to training set
            combined_train = Dataset.from_list(
                list(dataset["train"]) + list(synthetic_dataset)
            )
            dataset["train"] = combined_train
            print(f"Combined training set: {len(dataset['train'])} examples")
    
    # Print dataset statistics
    print("\nDataset Statistics:")
    print("-" * 40)
    for split_name, split_data in dataset.items():
        print(f"{split_name}: {len(split_data)} examples")
        
        # Count entity types
        entity_counts = {}
        for example in split_data:
            for tag in example["ner_tags"]:
                if tag != "O":
                    entity_type = tag.split("-")[1] if "-" in tag else tag
                    entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
        
        print(f"  Entity counts: {entity_counts}")
    
    # Save processed data
    save_processed_data(dataset, args.output_dir)
    
    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print("=" * 60)
    
    return dataset


if __name__ == "__main__":
    main()
