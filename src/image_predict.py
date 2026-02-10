"""
X-Ray Image Prediction Module

Provides inference functionality for chest X-ray classification.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.image_model import load_model


@dataclass
class XRayPrediction:
    """Container for X-ray prediction results."""
    predicted_class: str
    confidence: float
    all_probabilities: Dict[str, float]
    

class XRayPredictor:
    """
    X-Ray image classifier for inference.
    
    Example:
        predictor = XRayPredictor("outputs/xray_models/run_xxx/best_model.pt")
        result = predictor.predict("path/to/xray.png")
        print(f"Predicted: {result.predicted_class} ({result.confidence:.1%})")
    """
    
    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        image_size: int = 224,
    ):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to saved model checkpoint
            device: Device to run inference on (auto-detect if None)
            image_size: Input image size
        """
        self.model, self.class_names, self.device = load_model(model_path, device)
        self.image_size = image_size
        
        # Preprocessing transform (ImageNet normalization)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        
        print(f"XRay Predictor initialized")
        print(f"  Model loaded from: {model_path}")
        print(f"  Device: {self.device}")
        print(f"  Classes: {self.class_names}")
    
    def preprocess(self, image: Union[str, Path, Image.Image]) -> torch.Tensor:
        """
        Preprocess an image for inference.
        
        Args:
            image: Image path or PIL Image
        
        Returns:
            Preprocessed tensor
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            raise ValueError("Input must be a path or PIL Image")
        
        # Apply transforms
        tensor = self.transform(image)
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor
    
    def predict(
        self,
        image: Union[str, Path, Image.Image],
    ) -> XRayPrediction:
        """
        Predict the class of an X-ray image.
        
        Args:
            image: Image path or PIL Image
        
        Returns:
            XRayPrediction with class, confidence, and all probabilities
        """
        # Preprocess
        tensor = self.preprocess(image)
        tensor = tensor.to(self.device)
        
        # Inference
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = F.softmax(outputs, dim=1)
        
        # Get prediction
        probs = probabilities[0].cpu().numpy()
        predicted_idx = probs.argmax()
        confidence = probs[predicted_idx]
        predicted_class = self.class_names[predicted_idx]
        
        # All probabilities
        all_probs = {
            self.class_names[i]: float(probs[i])
            for i in range(len(self.class_names))
        }
        
        return XRayPrediction(
            predicted_class=predicted_class,
            confidence=float(confidence),
            all_probabilities=all_probs,
        )
    
    def predict_batch(
        self,
        images: List[Union[str, Path, Image.Image]],
        batch_size: int = 16,
    ) -> List[XRayPrediction]:
        """
        Predict classes for multiple images.
        
        Args:
            images: List of image paths or PIL Images
            batch_size: Batch size for inference
        
        Returns:
            List of XRayPrediction objects
        """
        results = []
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            
            # Preprocess batch
            tensors = [self.preprocess(img) for img in batch_images]
            batch_tensor = torch.cat(tensors, dim=0).to(self.device)
            
            # Inference
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(batch_tensor)
                probabilities = F.softmax(outputs, dim=1)
            
            # Process results
            for j, probs in enumerate(probabilities):
                probs = probs.cpu().numpy()
                predicted_idx = probs.argmax()
                
                results.append(XRayPrediction(
                    predicted_class=self.class_names[predicted_idx],
                    confidence=float(probs[predicted_idx]),
                    all_probabilities={
                        self.class_names[k]: float(probs[k])
                        for k in range(len(self.class_names))
                    },
                ))
        
        return results
    
    def predict_with_visualization(
        self,
        image: Union[str, Path, Image.Image],
    ) -> Tuple[XRayPrediction, Image.Image]:
        """
        Predict and return the original image for visualization.
        
        Args:
            image: Image path or PIL Image
        
        Returns:
            Tuple of (prediction, original_image)
        """
        # Load image if path
        if isinstance(image, (str, Path)):
            original = Image.open(image).convert('RGB')
        else:
            original = image.copy()
        
        prediction = self.predict(image)
        
        return prediction, original


def main():
    parser = argparse.ArgumentParser(description="X-Ray image prediction")
    parser.add_argument(
        "--model-path", type=str, required=True,
        help="Path to saved model"
    )
    parser.add_argument(
        "--image", type=str, required=True,
        help="Path to X-ray image"
    )
    parser.add_argument(
        "--top-k", type=int, default=3,
        help="Show top-k predictions"
    )
    args = parser.parse_args()
    
    # Check if image exists
    if not Path(args.image).exists():
        print(f"Error: Image not found: {args.image}")
        return
    
    # Initialize predictor
    predictor = XRayPredictor(args.model_path)
    
    # Predict
    print(f"\n{'='*60}")
    print(f"Image: {args.image}")
    print(f"{'='*60}")
    
    result = predictor.predict(args.image)
    
    print(f"\nPrediction: {result.predicted_class}")
    print(f"Confidence: {result.confidence:.1%}")
    
    print(f"\nAll probabilities:")
    sorted_probs = sorted(
        result.all_probabilities.items(),
        key=lambda x: x[1],
        reverse=True
    )
    for cls, prob in sorted_probs[:args.top_k]:
        bar = "█" * int(prob * 20)
        print(f"  {cls:20s} {prob:6.1%} {bar}")
    
    # JSON output
    print(f"\nJSON output:")
    import json
    output = {
        "image": args.image,
        "predicted_class": result.predicted_class,
        "confidence": result.confidence,
        "probabilities": result.all_probabilities,
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
