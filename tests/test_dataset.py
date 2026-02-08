"""
Tests for NER Dataset Class

Tests tokenization, label alignment, and dataset functionality.
"""

import os
import sys
from pathlib import Path

import pytest
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestLabelMappings:
    """Tests for label mappings and conversions."""
    
    def test_default_label_list(self):
        """Test that default label list has expected labels."""
        from src.dataset import DEFAULT_LABEL_LIST
        
        assert "O" in DEFAULT_LABEL_LIST
        assert "B-Disease" in DEFAULT_LABEL_LIST
        assert "I-Disease" in DEFAULT_LABEL_LIST
        assert "B-Chemical" in DEFAULT_LABEL_LIST
        assert "I-Chemical" in DEFAULT_LABEL_LIST
        assert "B-Symptom" in DEFAULT_LABEL_LIST
        assert "I-Symptom" in DEFAULT_LABEL_LIST
    
    def test_label_mappings(self):
        """Test label to ID and ID to label mappings."""
        from src.dataset import get_label_mappings, DEFAULT_LABEL_LIST
        
        label_to_id, id_to_label = get_label_mappings(DEFAULT_LABEL_LIST)
        
        # Check that O is label 0
        assert label_to_id["O"] == 0
        assert id_to_label[0] == "O"
        
        # Check bidirectional mapping
        for label in DEFAULT_LABEL_LIST:
            idx = label_to_id[label]
            assert id_to_label[idx] == label


class TestNERDataset:
    """Tests for NERDataset class."""
    
    @pytest.fixture
    def sample_data(self):
        """Sample NER data for testing."""
        return [
            {
                "tokens": ["Patient", "has", "fever", "and", "headache", "."],
                "ner_tags": ["O", "O", "B-Disease", "O", "B-Disease", "O"],
            },
            {
                "tokens": ["Aspirin", "treats", "inflammation", "."],
                "ner_tags": ["B-Chemical", "O", "B-Disease", "O"],
            },
            {
                "tokens": ["I", "have", "persistent", "chest", "pain", "."],
                "ner_tags": ["O", "O", "O", "B-Symptom", "I-Symptom", "O"],
            },
        ]
    
    @pytest.fixture
    def tokenizer(self):
        """Load tokenizer for testing."""
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("bert-base-cased")
    
    def test_dataset_creation(self, sample_data, tokenizer):
        """Test that dataset can be created from sample data."""
        from src.dataset import NERDataset
        
        dataset = NERDataset(
            data=sample_data,
            tokenizer=tokenizer,
            max_length=64,
        )
        
        assert len(dataset) == 3
        assert dataset.num_labels == 7  # Default labels
    
    def test_dataset_getitem(self, sample_data, tokenizer):
        """Test that __getitem__ returns correct format."""
        from src.dataset import NERDataset
        
        dataset = NERDataset(
            data=sample_data,
            tokenizer=tokenizer,
            max_length=64,
        )
        
        item = dataset[0]
        
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item
        
        assert isinstance(item["input_ids"], torch.Tensor)
        assert isinstance(item["attention_mask"], torch.Tensor)
        assert isinstance(item["labels"], torch.Tensor)
        
        assert item["input_ids"].shape == item["attention_mask"].shape == item["labels"].shape
    
    def test_label_alignment(self, sample_data, tokenizer):
        """Test that labels are properly aligned with subword tokens."""
        from src.dataset import NERDataset
        
        dataset = NERDataset(
            data=sample_data,
            tokenizer=tokenizer,
            max_length=64,
        )
        
        item = dataset[0]
        labels = item["labels"].tolist()
        
        # First token should be -100 (CLS)
        assert labels[0] == -100
        
        # Should have -100 for special tokens at the end
        # Find last non-padding position
        attention_mask = item["attention_mask"].tolist()
        seq_length = sum(attention_mask)
        
        # Last real token should be -100 (SEP)
        assert labels[seq_length - 1] == -100
    
    def test_multi_token_entity(self, tokenizer):
        """Test handling of multi-token entities."""
        from src.dataset import NERDataset
        
        data = [{
            "tokens": ["Patient", "has", "diabetes", "mellitus", "."],
            "ner_tags": ["O", "O", "B-Disease", "I-Disease", "O"],
        }]
        
        dataset = NERDataset(
            data=data,
            tokenizer=tokenizer,
            max_length=64,
        )
        
        item = dataset[0]
        labels = item["labels"].tolist()
        
        # Should have B-Disease followed by I-Disease (ignoring subwords)
        # Get the label_to_id mapping
        b_disease_id = dataset.label_to_id["B-Disease"]
        i_disease_id = dataset.label_to_id["I-Disease"]
        
        # Find positions with disease labels
        disease_positions = [i for i, l in enumerate(labels) if l in [b_disease_id, i_disease_id]]
        
        assert len(disease_positions) >= 2  # At least diabetes and mellitus
    
    def test_get_labels(self, sample_data, tokenizer):
        """Test get_labels method."""
        from src.dataset import NERDataset
        
        dataset = NERDataset(
            data=sample_data,
            tokenizer=tokenizer,
            max_length=64,
        )
        
        labels = dataset.get_labels()
        
        assert isinstance(labels, list)
        assert "O" in labels
        assert "B-Disease" in labels


class TestDataCollator:
    """Tests for NER data collator."""
    
    def test_collator(self):
        """Test that collator batches data correctly."""
        from src.dataset import NERDataset, NERDataCollator
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        
        data = [
            {"tokens": ["Hello", "world"], "ner_tags": ["O", "O"]},
            {"tokens": ["Test", "sentence", "here"], "ner_tags": ["O", "O", "O"]},
        ]
        
        dataset = NERDataset(data, tokenizer, max_length=32)
        collator = NERDataCollator(tokenizer)
        
        # Get individual items
        items = [dataset[i] for i in range(len(dataset))]
        
        # Collate
        batch = collator(items)
        
        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "labels" in batch
        
        assert batch["input_ids"].shape[0] == 2  # Batch size
        assert batch["labels"].shape[0] == 2


class TestDataPreparation:
    """Tests for data preparation functions."""
    
    def test_tokenize_and_align_labels(self):
        """Test BIO label alignment function."""
        from data.prepare_data import tokenize_and_align_labels
        
        text = "Patient has fever and headache."
        entities = [
            {"type": "Disease", "offsets": [[12, 17]], "text": ["fever"]},
            {"type": "Disease", "offsets": [[22, 30]], "text": ["headache"]},
        ]
        
        tokens, labels = tokenize_and_align_labels(text, entities)
        
        assert "fever" in tokens
        assert "headache" in tokens
        assert "B-Disease" in labels
    
    def test_label_to_id_mapping(self):
        """Test that label mappings are consistent."""
        from data.prepare_data import LABEL_TO_ID, ID_TO_LABEL, LABEL_LIST
        
        assert len(LABEL_TO_ID) == len(ID_TO_LABEL) == len(LABEL_LIST)
        
        for label in LABEL_LIST:
            assert label in LABEL_TO_ID
            idx = LABEL_TO_ID[label]
            assert ID_TO_LABEL[idx] == label


class TestSyntheticData:
    """Tests for synthetic symptoms data."""
    
    def test_synthetic_file_exists(self):
        """Test that synthetic symptoms file exists."""
        import pandas as pd
        
        csv_path = Path(__file__).parent.parent / "data" / "synthetic_symptoms.csv"
        
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            assert len(df) >= 50  # At least 50 examples
            assert "text" in df.columns
            assert "entities" in df.columns
    
    def test_synthetic_data_format(self):
        """Test that synthetic data has correct format."""
        import pandas as pd
        
        csv_path = Path(__file__).parent.parent / "data" / "synthetic_symptoms.csv"
        
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            
            # Check first row format
            first_row = df.iloc[0]
            assert isinstance(first_row["text"], str)
            
            # Check entities format (should be semicolon-separated)
            entities_str = first_row["entities"]
            if pd.notna(entities_str):
                parts = entities_str.split(";")
                for part in parts:
                    # Format: text:type:start:end
                    entity_parts = part.split(":")
                    assert len(entity_parts) >= 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
