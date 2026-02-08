#!/usr/bin/env python3
"""
NER Prediction Script (CLI Wrapper)

Command-line interface for the Medical NER system.
Load a trained model and extract medical entities from text.
Outputs results in JSON format with entity text, label, confidence, and spans.

This is a CLI wrapper around src/inference.py. For programmatic usage,
import MedicalNER from src.inference instead.

Usage:
    python src/predict.py --text "I have persistent headache and fever"
    python src/predict.py --text "Patient diagnosed with diabetes mellitus" --model-path outputs/models/final_model
    python src/predict.py --file input.txt --output results.json
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from inference module
from src.inference import MedicalNER, Entity


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


# Backward compatibility: alias for old class name
NERPredictor = MedicalNER


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
    ner = MedicalNER(model_path, device=device, verbose=True)
    entities = ner.predict_entities(text, threshold=threshold)
    return PredictionResult(text=text, entities=entities)


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
    
    # Load predictor (using new MedicalNER class from inference.py)
    ner = MedicalNER(args.model_path, device=args.device, verbose=not args.quiet)
    
    # Process input
    if args.text:
        # Single text
        entities = ner.predict_entities(args.text, threshold=args.threshold)
        result = PredictionResult(text=args.text, entities=entities)
        results = [result]
        
        if not args.quiet:
            pretty_print_result(result)
    else:
        # File with multiple texts
        with open(args.file, "r") as f:
            texts = [line.strip() for line in f if line.strip()]
        
        print(f"Processing {len(texts)} texts...")
        batch_entities = ner.predict_batch(texts, threshold=args.threshold)
        
        # Convert to PredictionResult objects
        results = [PredictionResult(text=t, entities=e) for t, e in zip(texts, batch_entities)]
        
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
