"""
NER Dataset Class for Medical Entity Recognition

Handles tokenization, subword alignment, and label encoding for NER fine-tuning.
Uses HuggingFace tokenizers with proper handling of BIO tags across subwords.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from datasets import load_from_disk, DatasetDict


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


def get_label_mappings(label_list: List[str] = None) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Create label to ID and ID to label mappings.
    
    Args:
        label_list: List of label strings. Uses DEFAULT_LABEL_LIST if None.
    
    Returns:
        Tuple of (label_to_id, id_to_label) dictionaries
    """
    if label_list is None:
        label_list = DEFAULT_LABEL_LIST
    
    label_to_id = {label: idx for idx, label in enumerate(label_list)}
    id_to_label = {idx: label for idx, label in enumerate(label_list)}
    
    return label_to_id, id_to_label


class NERDataset(Dataset):
    """
    PyTorch Dataset for NER with proper subword tokenization and label alignment.
    
    Handles:
    - Tokenization with subword splitting (WordPiece/BPE)
    - Label alignment: first subword gets label, rest get -100
    - Padding and truncation to max_length
    - Special token handling ([CLS], [SEP], [PAD])
    
    Args:
        data: List of dicts with 'tokens' and 'labels' (or 'ner_tags') keys
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length (default: 128)
        label_to_id: Label string to ID mapping (uses default if None)
        label_all_tokens: If True, label all subwords (not just first). Default: False
    """
    
    def __init__(
        self,
        data: List[Dict],
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        max_length: int = 128,
        label_to_id: Dict[str, int] = None,
        label_all_tokens: bool = False,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_all_tokens = label_all_tokens
        
        # Set up label mappings
        if label_to_id is None:
            self.label_to_id, self.id_to_label = get_label_mappings()
        else:
            self.label_to_id = label_to_id
            self.id_to_label = {v: k for k, v in label_to_id.items()}
        
        self.num_labels = len(self.label_to_id)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single tokenized and aligned example.
        
        Returns:
            Dict with input_ids, attention_mask, and labels tensors
        """
        example = self.data[idx]
        
        # Get tokens and labels
        tokens = example.get("tokens", [])
        
        # Labels can be either string tags or integer IDs
        labels = example.get("labels", example.get("ner_tags", []))
        
        # Convert string labels to IDs if needed
        if labels and isinstance(labels[0], str):
            labels = [self.label_to_id.get(label, 0) for label in labels]
        
        # Tokenize with subword handling
        tokenized = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        # Get word IDs for label alignment
        word_ids = tokenized.word_ids(0)
        
        # Align labels with subword tokens
        aligned_labels = self._align_labels(labels, word_ids)
        
        return {
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "labels": torch.tensor(aligned_labels, dtype=torch.long),
        }
    
    def _align_labels(
        self,
        labels: List[int],
        word_ids: List[Optional[int]],
    ) -> List[int]:
        """
        Align word-level labels with subword tokens.
        
        Strategy:
        - Special tokens (None word_id): -100 (ignored in loss)
        - First subword of a word: original label
        - Subsequent subwords: -100 (or I- label if label_all_tokens=True)
        
        Args:
            labels: Word-level label IDs
            word_ids: Word ID for each subword token (None for special tokens)
        
        Returns:
            List of aligned label IDs
        """
        aligned_labels = []
        previous_word_idx = None
        
        for word_idx in word_ids:
            if word_idx is None:
                # Special token ([CLS], [SEP], [PAD])
                aligned_labels.append(-100)
            elif word_idx != previous_word_idx:
                # First subword of a new word - use the original label
                if word_idx < len(labels):
                    aligned_labels.append(labels[word_idx])
                else:
                    aligned_labels.append(0)  # O label for out-of-bounds
            else:
                # Continuation subword
                if self.label_all_tokens:
                    # Option 1: Label all subwords with the word's label
                    # Convert B- to I- for continuation subwords
                    if word_idx < len(labels):
                        label = labels[word_idx]
                        label_str = self.id_to_label.get(label, "O")
                        if label_str.startswith("B-"):
                            # Convert B- to I- for continuation
                            i_label = "I-" + label_str[2:]
                            aligned_labels.append(self.label_to_id.get(i_label, label))
                        else:
                            aligned_labels.append(label)
                    else:
                        aligned_labels.append(0)
                else:
                    # Option 2: Ignore subwords in loss (standard approach)
                    aligned_labels.append(-100)
            
            previous_word_idx = word_idx
        
        return aligned_labels
    
    def get_labels(self) -> List[str]:
        """Get the list of label strings."""
        return list(self.label_to_id.keys())


class NERDataCollator:
    """
    Data collator for NER that handles dynamic padding.
    
    More memory-efficient than padding to max_length for every example.
    """
    
    def __init__(
        self,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        padding: str = "longest",
        max_length: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length
    
    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of examples with dynamic padding.
        
        Args:
            features: List of dicts with input_ids, attention_mask, labels
        
        Returns:
            Batched dict with padded tensors
        """
        # Separate labels from inputs
        labels = [feature["labels"] for feature in features]
        
        # Pad input tensors
        batch = {
            "input_ids": torch.stack([f["input_ids"] for f in features]),
            "attention_mask": torch.stack([f["attention_mask"] for f in features]),
        }
        
        # Pad labels to same length as inputs
        batch["labels"] = torch.stack(labels)
        
        return batch


def load_processed_dataset(
    data_dir: str,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    max_length: int = 128,
) -> Dict[str, NERDataset]:
    """
    Load processed dataset from disk and create NERDataset instances.
    
    Args:
        data_dir: Directory containing processed HuggingFace dataset
        tokenizer: HuggingFace tokenizer for the model
        max_length: Maximum sequence length
    
    Returns:
        Dict mapping split names to NERDataset instances
    """
    # Load the HuggingFace dataset from disk
    dataset_dict = load_from_disk(data_dir)
    
    # Load label mappings if available
    label_info_path = os.path.join(data_dir, "label_info.json")
    if os.path.exists(label_info_path):
        with open(label_info_path, "r") as f:
            label_info = json.load(f)
        label_to_id = label_info.get("label_to_id", None)
    else:
        label_to_id = None
    
    # Create NERDataset for each split
    datasets = {}
    for split_name in dataset_dict.keys():
        # Convert HuggingFace dataset to list of dicts
        data = [dict(example) for example in dataset_dict[split_name]]
        
        datasets[split_name] = NERDataset(
            data=data,
            tokenizer=tokenizer,
            max_length=max_length,
            label_to_id=label_to_id,
        )
    
    return datasets


def create_dataset_from_examples(
    examples: List[Dict],
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    max_length: int = 128,
    label_to_id: Dict[str, int] = None,
) -> NERDataset:
    """
    Create NERDataset from a list of example dictionaries.
    
    Each example should have:
    - 'tokens': List of word strings
    - 'labels' or 'ner_tags': List of label strings or IDs
    
    Args:
        examples: List of example dicts
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        label_to_id: Label mapping (uses default if None)
    
    Returns:
        NERDataset instance
    """
    return NERDataset(
        data=examples,
        tokenizer=tokenizer,
        max_length=max_length,
        label_to_id=label_to_id,
    )


# Example usage and testing
if __name__ == "__main__":
    from transformers import AutoTokenizer
    
    # Test with BioBERT tokenizer
    print("Testing NERDataset...")
    
    tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.2")
    
    # Sample data
    sample_data = [
        {
            "tokens": ["Patient", "has", "fever", "and", "headache", "."],
            "ner_tags": ["O", "O", "B-Disease", "O", "B-Disease", "O"],
        },
        {
            "tokens": ["Aspirin", "treats", "inflammation", "."],
            "ner_tags": ["B-Chemical", "O", "B-Disease", "O"],
        },
    ]
    
    # Create dataset
    dataset = NERDataset(
        data=sample_data,
        tokenizer=tokenizer,
        max_length=64,
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of labels: {dataset.num_labels}")
    print(f"Labels: {dataset.get_labels()}")
    
    # Get a sample
    sample = dataset[0]
    print(f"\nSample 0:")
    print(f"  input_ids shape: {sample['input_ids'].shape}")
    print(f"  attention_mask shape: {sample['attention_mask'].shape}")
    print(f"  labels shape: {sample['labels'].shape}")
    
    # Decode tokens
    tokens = tokenizer.convert_ids_to_tokens(sample['input_ids'])
    labels = sample['labels'].tolist()
    
    print(f"\nToken-label alignment:")
    for token, label in zip(tokens[:15], labels[:15]):
        label_str = dataset.id_to_label.get(label, "IGN") if label != -100 else "IGN"
        print(f"  {token:<15} -> {label} ({label_str})")
    
    print("\nNERDataset tests passed!")
