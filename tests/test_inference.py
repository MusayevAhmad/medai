#!/usr/bin/env python3
"""
Unit tests for the inference module.

Tests the MedicalNER class with mocked models to ensure correct behavior
without requiring a trained model.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import torch

from src.inference import MedicalNER, Entity, predict_text


@pytest.fixture
def mock_model_components():
    """
    Create mock model components for testing.
    
    Returns a tuple of (model, tokenizer, label_to_id, id_to_label)
    """
    # Mock model
    model = Mock()
    model.eval = Mock()
    model.to = Mock(return_value=model)
    
    # Mock tokenizer
    tokenizer = Mock()
    
    # Label mappings
    label_to_id = {
        "O": 0,
        "B-Disease": 1,
        "I-Disease": 2,
        "B-Chemical": 3,
        "I-Chemical": 4,
        "B-Symptom": 5,
        "I-Symptom": 6,
    }
    id_to_label = {v: k for k, v in label_to_id.items()}
    
    return model, tokenizer, label_to_id, id_to_label


@pytest.fixture
def mock_ner(mock_model_components):
    """Create a MedicalNER instance with mocked components."""
    model, tokenizer, label_to_id, id_to_label = mock_model_components
    
    # Patch the imports within MedicalNER.__init__
    with patch('src.model.load_model') as mock_load:
        with patch('src.model.get_device', return_value='cpu'):
            mock_load.return_value = (model, tokenizer, label_to_id, id_to_label)
            
            # Mock Path.exists to return True
            with patch.object(Path, 'exists', return_value=True):
                ner = MedicalNER(model_path="fake/path", verbose=False)
    
    return ner


class TestEntity:
    """Test the Entity dataclass."""
    
    def test_entity_creation(self):
        """Test creating an Entity object."""
        entity = Entity(
            text="fever",
            label="Symptom",
            confidence=0.95,
            span=(12, 17)
        )
        
        assert entity.text == "fever"
        assert entity.label == "Symptom"
        assert entity.confidence == 0.95
        assert entity.span == (12, 17)
    
    def test_entity_to_dict(self):
        """Test Entity serialization to dictionary."""
        entity = Entity(
            text="diabetes",
            label="Disease",
            confidence=0.8567,
            span=(0, 8)
        )
        
        result = entity.to_dict()
        
        assert result["text"] == "diabetes"
        assert result["label"] == "Disease"
        assert result["confidence"] == 0.8567  # Rounded to 4 decimals
        assert result["span"] == [0, 8]  # Converted to list


class TestMedicalNER:
    """Test the MedicalNER class."""
    
    def test_init_with_valid_path(self, mock_model_components):
        """Test initialization with valid model path."""
        model, tokenizer, label_to_id, id_to_label = mock_model_components
        
        with patch('src.model.load_model') as mock_load:
            with patch('src.model.get_device', return_value='cpu'):
                mock_load.return_value = (model, tokenizer, label_to_id, id_to_label)
                
                with patch.object(Path, 'exists', return_value=True):
                    ner = MedicalNER(model_path="fake/path", verbose=False)
        
        assert ner.device == "cpu"
        assert ner.model_path == "fake/path"
        assert ner.label_to_id == label_to_id
        assert ner.id_to_label == id_to_label
        model.eval.assert_called_once()
    
    def test_init_with_invalid_path(self):
        """Test initialization with non-existent path."""
        with patch.object(Path, 'exists', return_value=False):
            with pytest.raises(FileNotFoundError):
                MedicalNER(model_path="nonexistent/path", verbose=False)
    
    def test_init_auto_device_detection(self, mock_model_components):
        """Test automatic device detection."""
        model, tokenizer, label_to_id, id_to_label = mock_model_components
        
        with patch('src.model.load_model') as mock_load:
            with patch('src.model.get_device', return_value='cuda'):
                mock_load.return_value = (model, tokenizer, label_to_id, id_to_label)
                
                with patch.object(Path, 'exists', return_value=True):
                    ner = MedicalNER(model_path="fake/path", device=None, verbose=False)
        
        assert ner.device == "cuda"
    
    def test_predict_entities_simple(self, mock_ner):
        """Test entity prediction on simple text."""
        # Setup mock tokenizer output
        mock_ner.tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
            "offset_mapping": torch.tensor([[[0, 0], [0, 5], [5, 6], [6, 13], [0, 0]]]),
        }
        
        # Setup mock model output
        # Predictions: O, B-Symptom, I-Symptom, O, O
        logits = torch.tensor([
            [
                [10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # O (special token)
                [0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0],  # B-Symptom (fever)
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0],  # I-Symptom (space)
                [10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # O (patient)
                [10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # O (special token)
            ]
        ])
        
        mock_output = Mock()
        mock_output.logits = logits
        mock_ner.model.return_value = mock_output
        
        # Test prediction
        entities = mock_ner.predict_entities("fever patient", threshold=0.0)
        
        # Should extract one entity "fever " (characters 0-6)
        assert len(entities) == 1
        assert entities[0].label == "Symptom"
        assert entities[0].text == "fever "
        assert entities[0].span == (0, 6)
    
    def test_predict_entities_with_threshold(self, mock_ner):
        """Test that threshold filters low-confidence entities."""
        # Setup mock with low confidence
        mock_ner.tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "offset_mapping": torch.tensor([[[0, 0], [0, 5], [0, 0]]]),
        }
        
        # Low confidence logits
        logits = torch.tensor([
            [
                [10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0],  # Very low confidence B-Symptom
                [10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ])
        
        mock_output = Mock()
        mock_output.logits = logits
        mock_ner.model.return_value = mock_output
        
        # With threshold=0.9, should filter out the entity
        entities = mock_ner.predict_entities("fever", threshold=0.9)
        assert len(entities) == 0
    
    def test_predict_batch(self, mock_ner):
        """Test batch prediction."""
        texts = ["fever", "cough", "pain"]
        
        # Mock tokenizer for batch
        mock_ner.tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3], [1, 4, 3], [1, 5, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
            "offset_mapping": torch.tensor([
                [[0, 0], [0, 5], [0, 0]],
                [[0, 0], [0, 5], [0, 0]],
                [[0, 0], [0, 4], [0, 0]],
            ]),
        }
        
        # Mock model output for batch
        logits = torch.tensor([
            [
                [10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0],  # B-Symptom
                [10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            [
                [10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0],  # B-Symptom
                [10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            [
                [10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0],  # B-Symptom
                [10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
        ])
        
        mock_output = Mock()
        mock_output.logits = logits
        mock_ner.model.return_value = mock_output
        
        # Test batch prediction
        results = mock_ner.predict_batch(texts)
        
        assert len(results) == 3
        for entities in results:
            assert len(entities) == 1  # Each text has one Symptom entity
            assert entities[0].label == "Symptom"
    
    def test_get_label_info(self, mock_ner):
        """Test getting label information."""
        info = mock_ner.get_label_info()
        
        assert "labels" in info
        assert "num_labels" in info
        assert "label_to_id" in info
        assert "id_to_label" in info
        
        assert info["num_labels"] == 7
        assert "O" in info["labels"]
        assert "B-Disease" in info["labels"]


class TestExtractEntities:
    """Test the _extract_entities private method."""
    
    def test_extract_single_token_entity(self, mock_ner):
        """Test extraction of single-token entity."""
        text = "fever"
        predictions = [0, 5, 0]  # O, B-Symptom, O
        confidences = [0.9, 0.95, 0.9]
        offset_mapping = [(0, 0), (0, 5), (0, 0)]
        
        entities = mock_ner._extract_entities(
            text, predictions, confidences, offset_mapping, threshold=0.0
        )
        
        assert len(entities) == 1
        assert entities[0].text == "fever"
        assert entities[0].label == "Symptom"
        assert entities[0].confidence == 0.95
    
    def test_extract_multi_token_entity(self, mock_ner):
        """Test extraction of multi-token entity."""
        text = "diabetes mellitus"
        predictions = [0, 1, 2, 0]  # O, B-Disease, I-Disease, O
        confidences = [0.9, 0.95, 0.93, 0.9]
        offset_mapping = [(0, 0), (0, 8), (9, 17), (0, 0)]
        
        entities = mock_ner._extract_entities(
            text, predictions, confidences, offset_mapping, threshold=0.0
        )
        
        assert len(entities) == 1
        assert entities[0].text == "diabetes mellitus"
        assert entities[0].label == "Disease"
        # Confidence should be average of 0.95 and 0.93
        assert abs(entities[0].confidence - 0.94) < 0.01
    
    def test_extract_multiple_entities(self, mock_ner):
        """Test extraction of multiple entities."""
        text = "fever and cough"
        predictions = [0, 5, 0, 5, 0]  # O, B-Symptom, O, B-Symptom, O
        confidences = [0.9, 0.95, 0.9, 0.92, 0.9]
        offset_mapping = [(0, 0), (0, 5), (6, 9), (10, 15), (0, 0)]
        
        entities = mock_ner._extract_entities(
            text, predictions, confidences, offset_mapping, threshold=0.0
        )
        
        assert len(entities) == 2
        assert entities[0].text == "fever"
        assert entities[0].label == "Symptom"
        assert entities[1].text == "cough"
        assert entities[1].label == "Symptom"
    
    def test_extract_with_threshold(self, mock_ner):
        """Test threshold filtering."""
        text = "fever"
        predictions = [0, 5, 0]  # O, B-Symptom, O
        confidences = [0.9, 0.4, 0.9]  # Low confidence for entity
        offset_mapping = [(0, 0), (0, 5), (0, 0)]
        
        # With threshold=0.5, entity should be filtered out
        entities = mock_ner._extract_entities(
            text, predictions, confidences, offset_mapping, threshold=0.5
        )
        
        assert len(entities) == 0
    
    def test_extract_handles_malformed_sequence(self, mock_ner):
        """Test handling of I- tag without preceding B- tag."""
        text = "fever patient"
        predictions = [0, 6, 0]  # O, I-Symptom (without B-), O
        confidences = [0.9, 0.95, 0.9]
        offset_mapping = [(0, 0), (0, 5), (0, 0)]
        
        # Should still extract the entity
        entities = mock_ner._extract_entities(
            text, predictions, confidences, offset_mapping, threshold=0.0
        )
        
        assert len(entities) == 1
        assert entities[0].text == "fever"
        assert entities[0].label == "Symptom"


class TestConvenienceFunction:
    """Test the convenience predict_text function."""
    
    def test_predict_text(self, mock_model_components):
        """Test the predict_text convenience function."""
        model, tokenizer, label_to_id, id_to_label = mock_model_components
        
        with patch('src.inference.MedicalNER') as MockNER:
            # Create mock instance
            mock_instance = Mock()
            mock_instance.predict_entities.return_value = [
                Entity("fever", "Symptom", 0.95, (0, 5))
            ]
            MockNER.return_value = mock_instance
            
            # Call function
            entities = predict_text("fever", model_path="fake/path")
            
            # Verify
            assert len(entities) == 1
            assert entities[0].text == "fever"
            mock_instance.predict_entities.assert_called_once()


# Integration test markers
@pytest.mark.integration
class TestMedicalNERIntegration:
    """
    Integration tests that require a trained model.
    
    Run with: pytest tests/test_inference.py -m integration
    Skip with: pytest tests/test_inference.py -m "not integration"
    """
    
    @pytest.mark.skipif(
        not Path("outputs/models/final_model").exists(),
        reason="Trained model not available"
    )
    def test_with_real_model(self):
        """Test with actual trained model if available."""
        ner = MedicalNER("outputs/models/final_model")
        
        entities = ner.predict_entities(
            "Patient diagnosed with diabetes and prescribed metformin"
        )
        
        # Should extract at least disease and chemical entities
        assert len(entities) > 0
        
        labels = {e.label for e in entities}
        assert any(label in ["Disease", "Chemical"] for label in labels)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
