#!/usr/bin/env python3
"""
Medical NER Inference Engine

Production-ready NER inference with model caching and batch processing.
This module provides a reusable MedicalNER class for extracting medical entities
from text using a fine-tuned BioBERT model.

Usage:
    from src.inference import MedicalNER, Entity
    
    # Initialize once (model is cached)
    ner = MedicalNER(model_path="outputs/models/final_model")
    
    # Extract entities
    entities = ner.predict_entities("Patient has fever and headache")
    
    # Batch processing
    results = ner.predict_batch(["text 1", "text 2", "text 3"])
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer


@dataclass
class Entity:
    """
    Represents an extracted medical entity.
    
    Attributes:
        text: The entity text as it appears in the original input
        label: Entity type (e.g., "Disease", "Symptom", "Chemical")
        confidence: Average confidence score (0.0-1.0)
        span: Character offsets as (start, end) tuple
    """
    text: str
    label: str
    confidence: float
    span: Tuple[int, int]
    
    def to_dict(self) -> dict:
        """Convert to dictionary format for JSON serialization."""
        return {
            "text": self.text,
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "span": list(self.span),
        }


class MedicalNER:
    """
    Medical Named Entity Recognition inference engine.
    
    This class handles loading a trained BioBERT+LoRA model and provides
    methods for single and batch entity extraction. The model is loaded
    once during initialization and cached for subsequent predictions.
    
    Example:
        >>> ner = MedicalNER("outputs/models/final_model")
        >>> entities = ner.predict_entities("Patient diagnosed with diabetes")
        >>> for entity in entities:
        ...     print(f"{entity.label}: {entity.text} ({entity.confidence:.2f})")
        Disease: diabetes (0.95)
    """
    
    def __init__(
        self,
        model_path: str = "outputs/models/final_model",
        device: Optional[str] = None,
        verbose: bool = True,
    ):
        """
        Initialize the NER inference engine.
        
        Args:
            model_path: Path to saved model directory containing model weights,
                       tokenizer, and config files
            device: Device to run inference on ("cpu", "cuda", "mps").
                   Auto-detects best available device if None
            verbose: Whether to print loading information
        
        Raises:
            FileNotFoundError: If model_path does not exist
            RuntimeError: If model loading fails
        """
        from src.model import load_model, get_device
        
        # Validate path
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
        
        # Auto-detect device if not specified
        if device is None:
            device = get_device()
        
        self.device = device
        self.model_path = str(model_path)
        self.verbose = verbose
        
        # Load model components (cached for reuse)
        if self.verbose:
            print(f"Loading model from {self.model_path}...")
        
        self.model, self.tokenizer, self.label_to_id, self.id_to_label = load_model(
            self.model_path, device=device
        )
        self.model.eval()  # Set to evaluation mode
        
        if self.verbose:
            print(f"Model loaded successfully")
            print(f"Device: {self.device}")
            print(f"Available labels: {list(self.label_to_id.keys())}")
    
    def predict_entities(
        self,
        text: str,
        threshold: float = 0.0,
        max_length: int = 128,
    ) -> List[Entity]:
        """
        Extract medical entities from a single text.
        
        Args:
            text: Input text string to process
            threshold: Minimum confidence threshold (0.0-1.0). Entities with
                      lower confidence are filtered out
            max_length: Maximum sequence length for tokenization (will truncate
                       longer inputs)
        
        Returns:
            List of Entity objects sorted by appearance in text
        
        Example:
            >>> ner = MedicalNER()
            >>> entities = ner.predict_entities(
            ...     "Patient has fever and was prescribed aspirin",
            ...     threshold=0.5
            ... )
            >>> len(entities)
            2
        """
        # Tokenize with offset mapping for character-level spans
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        
        # Extract offset mapping before moving to device
        offset_mapping = encoded.pop("offset_mapping")[0].tolist()
        
        # Move input tensors to device
        inputs = {k: v.to(self.device) for k, v in encoded.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0]  # Shape: (seq_len, num_labels)
            
            # Compute probabilities and predictions
            probs = F.softmax(logits, dim=-1)
            predictions = logits.argmax(dim=-1).cpu().tolist()
            confidences = probs.max(dim=-1).values.cpu().tolist()
        
        # Extract entities from BIO tags
        entities = self._extract_entities(
            text=text,
            predictions=predictions,
            confidences=confidences,
            offset_mapping=offset_mapping,
            threshold=threshold,
        )
        
        return entities
    
    def predict_batch(
        self,
        texts: List[str],
        threshold: float = 0.0,
        batch_size: int = 16,
        max_length: int = 128,
    ) -> List[List[Entity]]:
        """
        Extract entities from multiple texts efficiently.
        
        Processes texts in batches for improved throughput.
        
        Args:
            texts: List of input text strings
            threshold: Minimum confidence threshold for entities
            batch_size: Number of texts to process simultaneously
            max_length: Maximum sequence length for tokenization
        
        Returns:
            List of entity lists, one per input text
        
        Example:
            >>> ner = MedicalNER()
            >>> texts = ["fever and cough", "diabetes treatment"]
            >>> results = ner.predict_batch(texts)
            >>> for text, entities in zip(texts, results):
            ...     print(f"{text}: {len(entities)} entities")
            fever and cough: 2 entities
            diabetes treatment: 1 entities
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize batch
            encoded = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                return_offsets_mapping=True,
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            
            # Extract offset mappings
            offset_mappings = encoded.pop("offset_mapping").tolist()
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in encoded.items()}
            
            # Inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = F.softmax(outputs.logits, dim=-1)
                predictions = outputs.logits.argmax(dim=-1).cpu().tolist()
                confidences = probs.max(dim=-1).values.cpu().tolist()
            
            # Extract entities for each text in batch
            for text, preds, confs, offsets in zip(
                batch_texts, predictions, confidences, offset_mappings
            ):
                entities = self._extract_entities(
                    text=text,
                    predictions=preds,
                    confidences=confs,
                    offset_mapping=offsets,
                    threshold=threshold,
                )
                results.append(entities)
        
        return results
    
    def _extract_entities(
        self,
        text: str,
        predictions: List[int],
        confidences: List[float],
        offset_mapping: List[Tuple[int, int]],
        threshold: float = 0.0,
    ) -> List[Entity]:
        """
        Extract entities from BIO-tagged predictions.
        
        Handles:
        - Multi-token entities (combines consecutive B-/I- tags)
        - Confidence averaging across entity tokens
        - Malformed sequences (I- tag without B- tag)
        
        Args:
            text: Original input text
            predictions: Predicted label IDs for each token
            confidences: Confidence scores for each prediction
            offset_mapping: Token-to-character offset mapping
            threshold: Minimum average confidence to include entity
        
        Returns:
            List of extracted Entity objects
        """
        entities = []
        current_entity = None
        current_start = None
        current_end = None
        current_confidences = []
        
        for pred_id, conf, (start, end) in zip(predictions, confidences, offset_mapping):
            # Skip special tokens (have offset 0,0)
            if start == end == 0:
                continue
            
            label = self.id_to_label.get(pred_id, "O")
            
            if label.startswith("B-"):
                # Save previous entity if exists
                if current_entity is not None:
                    avg_conf = sum(current_confidences) / len(current_confidences)
                    if avg_conf >= threshold:
                        entities.append(Entity(
                            text=text[current_start:current_end],
                            label=current_entity,
                            confidence=avg_conf,
                            span=(current_start, current_end),
                        ))
                
                # Start new entity
                current_entity = label[2:]  # Remove "B-" prefix
                current_start = start
                current_end = end
                current_confidences = [conf]
                
            elif label.startswith("I-"):
                entity_type = label[2:]  # Remove "I-" prefix
                
                if current_entity == entity_type:
                    # Continue current entity
                    current_end = end
                    current_confidences.append(conf)
                else:
                    # Type mismatch - save previous and start new
                    if current_entity is not None:
                        avg_conf = sum(current_confidences) / len(current_confidences)
                        if avg_conf >= threshold:
                            entities.append(Entity(
                                text=text[current_start:current_end],
                                label=current_entity,
                                confidence=avg_conf,
                                span=(current_start, current_end),
                            ))
                    
                    # Start new entity (handle malformed I- tag)
                    current_entity = entity_type
                    current_start = start
                    current_end = end
                    current_confidences = [conf]
            else:
                # "O" tag - save previous entity if exists
                if current_entity is not None:
                    avg_conf = sum(current_confidences) / len(current_confidences)
                    if avg_conf >= threshold:
                        entities.append(Entity(
                            text=text[current_start:current_end],
                            label=current_entity,
                            confidence=avg_conf,
                            span=(current_start, current_end),
                        ))
                    current_entity = None
                    current_start = None
                    current_end = None
                    current_confidences = []
        
        # Don't forget the last entity
        if current_entity is not None:
            avg_conf = sum(current_confidences) / len(current_confidences)
            if avg_conf >= threshold:
                entities.append(Entity(
                    text=text[current_start:current_end],
                    label=current_entity,
                    confidence=avg_conf,
                    span=(current_start, current_end),
                ))
        
        return entities
    
    def get_label_info(self) -> Dict:
        """
        Get information about available labels.
        
        Returns:
            Dictionary with label mappings and statistics
        """
        return {
            "labels": list(self.label_to_id.keys()),
            "num_labels": len(self.label_to_id),
            "label_to_id": self.label_to_id,
            "id_to_label": self.id_to_label,
        }


# Convenience function for backward compatibility
def predict_text(
    text: str,
    model_path: str = "outputs/models/final_model",
    threshold: float = 0.0,
    device: Optional[str] = None,
) -> List[Entity]:
    """
    Convenience function to predict entities from a single text.
    
    Creates a new MedicalNER instance for each call. For repeated
    predictions, create a MedicalNER instance once and reuse it.
    
    Args:
        text: Input text
        model_path: Path to saved model
        threshold: Minimum confidence threshold
        device: Device to use (auto-detect if None)
    
    Returns:
        List of Entity objects
    """
    ner = MedicalNER(model_path, device=device, verbose=False)
    return ner.predict_entities(text, threshold=threshold)


# Example usage
if __name__ == "__main__":
    import sys
    
    # Test with example text
    test_text = "Patient diagnosed with diabetes mellitus and prescribed metformin. Reports fever and headache."
    
    print("=" * 70)
    print("Medical NER Inference Test")
    print("=" * 70)
    print(f"\nInput: {test_text}\n")
    
    try:
        # Initialize NER
        ner = MedicalNER()
        
        # Extract entities
        entities = ner.predict_entities(test_text, threshold=0.3)
        
        # Display results
        if entities:
            print(f"Found {len(entities)} entities:")
            print("-" * 70)
            for entity in entities:
                print(f"  [{entity.label:10s}] \"{entity.text}\"")
                print(f"                Confidence: {entity.confidence:.4f}, Span: {entity.span}")
        else:
            print("No entities found.")
        
        # Test batch processing
        print("\n" + "=" * 70)
        print("Batch Processing Test")
        print("=" * 70)
        
        batch_texts = [
            "fever and cough",
            "aspirin for pain relief",
            "diagnosed with hypertension"
        ]
        
        results = ner.predict_batch(batch_texts)
        
        for text, entities in zip(batch_texts, results):
            print(f"\n'{text}' → {len(entities)} entities")
            for entity in entities:
                print(f"  - {entity.label}: {entity.text} ({entity.confidence:.2f})")
        
        print("\n" + "=" * 70)
        print("All tests passed!")
        print("=" * 70)
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please train a model first or specify a valid model path.")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
