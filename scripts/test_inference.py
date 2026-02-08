#!/usr/bin/env python3
"""
Test script for the new inference module.

Demonstrates usage of the MedicalNER class with examples.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference import MedicalNER, Entity


def test_basic_usage():
    """Test basic usage of MedicalNER."""
    print("=" * 70)
    print("TEST 1: Basic Entity Extraction")
    print("=" * 70)
    
    # Example texts
    texts = [
        "Patient diagnosed with diabetes mellitus and prescribed metformin.",
        "Symptoms include fever, headache, and nausea.",
        "Treatment: Aspirin 100mg daily for hypertension.",
    ]
    
    try:
        # Initialize NER (model is loaded once and cached)
        print("\nInitializing MedicalNER...")
        ner = MedicalNER(model_path="outputs/models/final_model")
        
        # Process each text
        for text in texts:
            print(f"\nText: {text}")
            entities = ner.predict_entities(text, threshold=0.3)
            
            if entities:
                print(f"Found {len(entities)} entities:")
                for entity in entities:
                    print(f"  [{entity.label:10s}] \"{entity.text}\" (confidence: {entity.confidence:.2f})")
            else:
                print("  No entities found")
        
        print("\n✓ Test passed!")
        return True
        
    except FileNotFoundError:
        print("\n⚠ Model not found. Please train a model first.")
        print("Run: python src/train.py")
        return False
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_processing():
    """Test batch processing for efficiency."""
    print("\n" + "=" * 70)
    print("TEST 2: Batch Processing")
    print("=" * 70)
    
    texts = [
        "fever and cough",
        "aspirin for pain",
        "diabetes treatment",
        "headache with nausea",
        "chest pain",
    ]
    
    try:
        ner = MedicalNER(model_path="outputs/models/final_model", verbose=False)
        
        print(f"\nProcessing {len(texts)} texts in batch...")
        results = ner.predict_batch(texts, threshold=0.3, batch_size=3)
        
        for text, entities in zip(texts, results):
            print(f"\n'{text}' → {len(entities)} entities")
            for entity in entities:
                print(f"  - {entity.label}: {entity.text}")
        
        print("\n✓ Batch processing successful!")
        return True
        
    except FileNotFoundError:
        print("\n⚠ Model not found. Skipping batch test.")
        return False


def test_backward_compatibility():
    """Test that old NERPredictor alias still works."""
    print("\n" + "=" * 70)
    print("TEST 3: Backward Compatibility")
    print("=" * 70)
    
    try:
        # Import using old name
        from src.predict import NERPredictor, Entity
        
        print("\n✓ NERPredictor alias imported successfully")
        print(f"  NERPredictor is MedicalNER: {NERPredictor.__name__}")
        
        return True
        
    except ImportError as e:
        print(f"\n✗ Import failed: {e}")
        return False


def test_label_info():
    """Test label information retrieval."""
    print("\n" + "=" * 70)
    print("TEST 4: Label Information")
    print("=" * 70)
    
    try:
        ner = MedicalNER(model_path="outputs/models/final_model", verbose=False)
        
        info = ner.get_label_info()
        
        print(f"\nTotal labels: {info['num_labels']}")
        print(f"Available labels: {', '.join(info['labels'])}")
        
        print("\n✓ Label info retrieved successfully!")
        return True
        
    except FileNotFoundError:
        print("\n⚠ Model not found. Skipping label info test.")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("MEDICAL NER INFERENCE MODULE - TEST SUITE")
    print("=" * 70)
    
    results = []
    
    # Run tests
    results.append(("Basic Usage", test_basic_usage()))
    results.append(("Batch Processing", test_batch_processing()))
    results.append(("Backward Compatibility", test_backward_compatibility()))
    results.append(("Label Info", test_label_info()))
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8s} {test_name}")
    
    total_passed = sum(passed for _, passed in results)
    print(f"\nTotal: {total_passed}/{len(results)} tests passed")
    
    if total_passed == len(results):
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n⚠ Some tests failed or were skipped")
        return 1


if __name__ == "__main__":
    sys.exit(main())
