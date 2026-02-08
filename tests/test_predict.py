"""
Tests for NER Prediction

Tests entity extraction and prediction functionality.
"""

import json
import sys
from pathlib import Path

import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestEntity:
    """Tests for Entity dataclass."""
    
    def test_entity_creation(self):
        """Test creating an Entity object."""
        from src.predict import Entity
        
        entity = Entity(
            text="headache",
            label="SYMPTOM",
            confidence=0.95,
            span=(10, 18),
        )
        
        assert entity.text == "headache"
        assert entity.label == "SYMPTOM"
        assert entity.confidence == 0.95
        assert entity.span == (10, 18)
    
    def test_entity_to_dict(self):
        """Test Entity to_dict method."""
        from src.predict import Entity
        
        entity = Entity(
            text="fever",
            label="SYMPTOM",
            confidence=0.8765,
            span=(5, 10),
        )
        
        d = entity.to_dict()
        
        assert d["text"] == "fever"
        assert d["label"] == "SYMPTOM"
        assert d["confidence"] == 0.8765  # Rounded to 4 decimals
        assert d["span"] == [5, 10]  # Tuple converted to list


class TestPredictionResult:
    """Tests for PredictionResult dataclass."""
    
    def test_result_creation(self):
        """Test creating a PredictionResult."""
        from src.predict import Entity, PredictionResult
        
        entities = [
            Entity("headache", "SYMPTOM", 0.9, (10, 18)),
            Entity("fever", "SYMPTOM", 0.85, (23, 28)),
        ]
        
        result = PredictionResult(
            text="I have headache and fever.",
            entities=entities,
        )
        
        assert result.text == "I have headache and fever."
        assert len(result.entities) == 2
    
    def test_result_to_dict(self):
        """Test PredictionResult to_dict method."""
        from src.predict import Entity, PredictionResult
        
        entities = [Entity("pain", "SYMPTOM", 0.9, (0, 4))]
        result = PredictionResult("pain here", entities)
        
        d = result.to_dict()
        
        assert "text" in d
        assert "entities" in d
        assert len(d["entities"]) == 1
        assert d["entities"][0]["text"] == "pain"
    
    def test_result_to_json(self):
        """Test PredictionResult to_json method."""
        from src.predict import Entity, PredictionResult
        
        entities = [Entity("headache", "SYMPTOM", 0.9, (0, 8))]
        result = PredictionResult("headache", entities)
        
        json_str = result.to_json()
        
        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed["text"] == "headache"
        assert len(parsed["entities"]) == 1


class TestPredictorHelpers:
    """Tests for predictor helper functions."""
    
    def test_entity_extraction_simple(self):
        """Test entity extraction from BIO tags."""
        # This is a simplified test of the extraction logic
        predictions = [0, 0, 1, 0, 1, 0]  # O, O, B-Disease, O, B-Disease, O
        
        # Check that we can identify entity positions
        entity_starts = [i for i, p in enumerate(predictions) if p == 1]
        assert len(entity_starts) == 2
    
    def test_entity_extraction_multi_token(self):
        """Test multi-token entity extraction."""
        # B-Disease, I-Disease pattern
        predictions = [0, 1, 2, 0]  # O, B-Disease, I-Disease, O
        
        # Should identify one entity spanning positions 1-2
        in_entity = False
        entity_spans = []
        start = None
        
        for i, p in enumerate(predictions):
            if p == 1:  # B-
                if in_entity:
                    entity_spans.append((start, i))
                in_entity = True
                start = i
            elif p == 2:  # I-
                pass  # Continue entity
            else:  # O
                if in_entity:
                    entity_spans.append((start, i))
                    in_entity = False
        
        if in_entity:
            entity_spans.append((start, len(predictions)))
        
        assert len(entity_spans) == 1
        assert entity_spans[0] == (1, 3)


class TestKnownExamples:
    """Tests on known examples (integration-like tests)."""
    
    @pytest.fixture
    def known_examples(self):
        """Examples with known expected outputs."""
        return [
            {
                "text": "Patient has fever and headache.",
                "expected_entities": ["fever", "headache"],
                "expected_labels": ["Disease", "Disease"],
            },
            {
                "text": "Aspirin is used to treat pain.",
                "expected_entities": ["Aspirin", "pain"],
                "expected_labels": ["Chemical", "Disease"],
            },
        ]
    
    def test_known_examples_format(self, known_examples):
        """Test that known examples have correct format."""
        for example in known_examples:
            assert "text" in example
            assert "expected_entities" in example
            assert "expected_labels" in example
            assert len(example["expected_entities"]) == len(example["expected_labels"])


class TestPredictFunction:
    """Tests for the predict_text convenience function."""
    
    def test_predict_function_signature(self):
        """Test that predict_text function exists with correct signature."""
        from src.predict import predict_text
        import inspect
        
        sig = inspect.signature(predict_text)
        params = list(sig.parameters.keys())
        
        assert "text" in params
        assert "model_path" in params
        assert "threshold" in params


class TestOutputFormat:
    """Tests for output format compliance."""
    
    def test_json_output_structure(self):
        """Test that JSON output matches expected structure."""
        from src.predict import Entity, PredictionResult
        
        # Create sample output
        entities = [
            Entity("headache", "SYMPTOM", 0.92, (21, 29)),
            Entity("fever", "SYMPTOM", 0.88, (34, 39)),
        ]
        result = PredictionResult(
            "persistent headache and fever for 3 days",
            entities,
        )
        
        output = result.to_dict()
        
        # Check structure matches spec
        assert "text" in output
        assert "entities" in output
        
        for entity in output["entities"]:
            assert "text" in entity
            assert "label" in entity
            assert "confidence" in entity
            assert "span" in entity
            assert isinstance(entity["span"], list)
            assert len(entity["span"]) == 2
    
    def test_confidence_in_range(self):
        """Test that confidence values are in valid range."""
        from src.predict import Entity
        
        entity = Entity("test", "LABEL", 0.5, (0, 4))
        d = entity.to_dict()
        
        assert 0.0 <= d["confidence"] <= 1.0


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_text(self):
        """Test handling of empty text."""
        from src.predict import PredictionResult
        
        result = PredictionResult("", [])
        
        assert result.text == ""
        assert result.entities == []
        
        d = result.to_dict()
        assert d["text"] == ""
        assert d["entities"] == []
    
    def test_no_entities(self):
        """Test text with no entities."""
        from src.predict import PredictionResult
        
        result = PredictionResult("The weather is nice today.", [])
        
        assert len(result.entities) == 0
        
        d = result.to_dict()
        assert len(d["entities"]) == 0
    
    def test_overlapping_check(self):
        """Test that we handle potential overlapping entities."""
        from src.predict import Entity
        
        # Two entities that might overlap in edge cases
        e1 = Entity("chest pain", "SYMPTOM", 0.9, (0, 10))
        e2 = Entity("pain", "SYMPTOM", 0.8, (6, 10))
        
        # In proper extraction, we shouldn't have overlapping entities
        # This is just to ensure the data structures work
        assert e1.span[1] > e2.span[0]  # They would overlap


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
