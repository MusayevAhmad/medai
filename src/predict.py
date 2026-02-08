#!/usr/bin/env python3
"""
NER Prediction Script

Load a trained model and extract medical entities from text.
Outputs results in JSON format with entity text, label, confidence, and spans.

Usage:
    python src/predict.py --text "I have persistent headache and fever"
    python src/predict.py --text "Patient diagnosed with diabetes mellitus" --model-path outputs/models/final_model
    python src/predict.py --file input.txt --output results.json
"""

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class Entity:
    """Represents an extracted entity."""
    text: str
    label: str
    confidence: float
    span: Tuple[int, int]  # Character offsets (start, end)
    
    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "span": list(self.span),
        }


@dataclass
class PredictionResult:
    """Represents the result of NER prediction on a text."""
    text: str
    entities: List[Entity]
    
    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "entities": [e.to_dict() for e in self.entities],
        }
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class NERPredictor:
    """
    NER Predictor for extracting medical entities from text.
    
    Loads a trained model and provides methods for inference.
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = None,
    ):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to saved model directory
            device: Device to run inference on (auto-detect if None)
        """
        from src.model import load_model, get_device
        
        if device is None:
            device = get_device()
        
        self.device = device
        self.model_path = model_path
        
        # Load model
        print(f"Loading model from {model_path}...")
        self.model, self.tokenizer, self.label_to_id, self.id_to_label = load_model(
            model_path, device=device
        )
        self.model.eval()
        
        print(f"Model loaded. Device: {device}")
        print(f"Labels: {list(self.label_to_id.keys())}")
    
    def predict(
        self,
        text: str,
        threshold: float = 0.0,
    ) -> PredictionResult:
        """
        Extract entities from a single text.
        
        Args:
            text: Input text string
            threshold: Minimum confidence threshold for entities (0.0-1.0)
        
        Returns:
            PredictionResult with extracted entities
        """
        # Tokenize
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            padding=True,
            truncation=True,
            max_length=128,
        )
        
        # Get offset mapping before moving to device
        offset_mapping = encoded.pop("offset_mapping")[0].tolist()
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in encoded.items()}
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0]
            
            # Get probabilities
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
        
        return PredictionResult(text=text, entities=entities)
    
    def predict_batch(
        self,
        texts: List[str],
        threshold: float = 0.0,
        batch_size: int = 16,
    ) -> List[PredictionResult]:
        """
        Extract entities from multiple texts.
        
        Args:
            texts: List of input text strings
            threshold: Minimum confidence threshold
            batch_size: Batch size for inference
        
        Returns:
            List of PredictionResult objects
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
                max_length=128,
            )
            
            offset_mappings = encoded.pop("offset_mapping").tolist()
            inputs = {k: v.to(self.device) for k, v in encoded.items()}
            
            # Forward pass
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
                results.append(PredictionResult(text=text, entities=entities))
        
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
        Extract entities from predicted BIO tags.
        
        Handles multi-token entities by combining consecutive B-/I- tags.
        
        Args:
            text: Original input text
            predictions: List of predicted label IDs
            confidences: List of confidence scores
            offset_mapping: Token to character offset mapping
            threshold: Minimum confidence threshold
        
        Returns:
            List of Entity objects
        """
        entities = []
        current_entity = None
        current_start = None
        current_end = None
        current_confidences = []
        
        for i, (pred_id, conf, (start, end)) in enumerate(
            zip(predictions, confidences, offset_mapping)
        ):
            # Skip special tokens (offset 0,0)
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
                    
                    # Start new entity with I- tag (handle malformed sequences)
                    current_entity = entity_type
                    current_start = start
                    current_end = end
                    current_confidences = [conf]
            else:
                # O tag - save previous entity if exists
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


def predict_text(
    text: str,
    model_path: str = "outputs/models/final_model",
    threshold: float = 0.0,
    device: str = None,
) -> PredictionResult:
    """
    Convenience function to predict entities from a single text.
    
    Args:
        text: Input text
        model_path: Path to saved model
        threshold: Minimum confidence threshold
        device: Device to use (auto-detect if None)
    
    Returns:
        PredictionResult with extracted entities
    """
    predictor = NERPredictor(model_path, device=device)
    return predictor.predict(text, threshold=threshold)


def pretty_print_result(result: PredictionResult) -> None:
    """Print prediction result in a human-readable format."""
    print("\n" + "=" * 60)
    print(f"Text: {result.text}")
    print("=" * 60)
    
    if result.entities:
        print(f"\nFound {len(result.entities)} entities:")
        print("-" * 40)
        for entity in result.entities:
            print(f"  [{entity.label}] \"{entity.text}\"")
            print(f"      Confidence: {entity.confidence:.4f}")
            print(f"      Span: {entity.span}")
        print("-" * 40)
    else:
        print("\nNo entities found.")
    
    print("\nJSON output:")
    print(result.to_json())


def main():
    parser = argparse.ArgumentParser(description="Extract medical entities from text")
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--text", type=str,
                             help="Text to extract entities from")
    input_group.add_argument("--file", type=str,
                             help="File containing texts (one per line)")
    
    # Model options
    parser.add_argument("--model-path", type=str, default="outputs/models/final_model",
                        help="Path to saved model directory")
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="Minimum confidence threshold (0.0-1.0)")
    parser.add_argument("--device", type=str, default=None,
                        choices=["cpu", "cuda", "mps"],
                        help="Device to use for inference")
    
    # Output options
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path for JSON results")
    parser.add_argument("--quiet", action="store_true",
                        help="Only output JSON, no pretty printing")
    
    args = parser.parse_args()
    
    # Load predictor
    predictor = NERPredictor(args.model_path, device=args.device)
    
    # Process input
    if args.text:
        # Single text
        result = predictor.predict(args.text, threshold=args.threshold)
        results = [result]
        
        if not args.quiet:
            pretty_print_result(result)
    else:
        # File with multiple texts
        with open(args.file, "r") as f:
            texts = [line.strip() for line in f if line.strip()]
        
        print(f"Processing {len(texts)} texts...")
        results = predictor.predict_batch(texts, threshold=args.threshold)
        
        if not args.quiet:
            for result in results:
                pretty_print_result(result)
    
    # Save output if requested
    if args.output:
        output_data = [r.to_dict() for r in results]
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")
    elif args.quiet:
        # Print JSON to stdout
        output_data = [r.to_dict() for r in results]
        print(json.dumps(output_data, indent=2))


if __name__ == "__main__":
    main()
