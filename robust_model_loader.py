"""
🚀 Robust PyTorch Model Loading for Cattle Breed Recognition
Production-ready error handling and fallback mechanisms
"""

import torch
import torch.nn as nn
import os
import sys
import warnings
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

class ModelLoadingError(Exception):
    """Custom exception for model loading issues"""
    pass

class RobustModelLoader:
    """
    Robust model loader with comprehensive error handling
    Handles various PyTorch versions, missing dependencies, and model loading issues
    """
    
    def __init__(self, checkpoint_path: str = "best_breed_classifier.pth"):
        self.checkpoint_path = checkpoint_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.breed_classes = None
        self.transform = None
        self.is_loaded = False
        
    def check_dependencies(self) -> Dict[str, bool]:
        """Check if all required dependencies are available"""
        dependencies = {
            "torch": False,
            "torchvision": False,
            "efficientnet_pytorch": False,
            "PIL": False,
            "numpy": False
        }
        
        try:
            import torch
            dependencies["torch"] = True
        except ImportError:
            pass
            
        try:
            import torchvision
            dependencies["torchvision"] = True
        except ImportError:
            pass
            
        try:
            from efficientnet_pytorch import EfficientNet
            dependencies["efficientnet_pytorch"] = True
        except ImportError:
            pass
            
        try:
            from PIL import Image
            dependencies["PIL"] = True
        except ImportError:
            pass
            
        try:
            import numpy
            dependencies["numpy"] = True
        except ImportError:
            pass
            
        return dependencies
    
    def get_image_transform(self):
        """Get image transformation with error handling"""
        try:
            from torchvision import transforms
            
            return transforms.Compose([
                transforms.Resize((300, 300)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        except ImportError as e:
            raise ModelLoadingError(f"TorchVision not available for transforms: {e}")
    
    def load_efficientnet(self, num_classes: int) -> nn.Module:
        """Load EfficientNet with multiple fallback methods"""
        
        # Method 1: efficientnet-pytorch library (preferred)
        try:
            from efficientnet_pytorch import EfficientNet
            
            model = EfficientNet.from_pretrained("efficientnet-b3")
            model._fc = nn.Linear(model._fc.in_features, num_classes)
            
            print("✅ EfficientNet loaded from efficientnet-pytorch library")
            return model
            
        except ImportError:
            print("⚠️ efficientnet-pytorch not available, trying torchvision...")
            
        # Method 2: torchvision EfficientNet (PyTorch 1.12+)
        try:
            import torchvision.models as models
            
            # Try new API first
            try:
                from torchvision.models import EfficientNet_B3_Weights
                model = models.efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
            except (ImportError, AttributeError):
                # Fall back to legacy API
                model = models.efficientnet_b3(pretrained=True)
            
            # Modify final layer for our classes
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
            
            print("✅ EfficientNet loaded from torchvision")
            return model
            
        except Exception as e:
            print(f"❌ TorchVision EfficientNet failed: {e}")
            
        # Method 3: Create empty EfficientNet without pretrained weights
        try:
            from efficientnet_pytorch import EfficientNet
            
            model = EfficientNet.from_name("efficientnet-b3", num_classes=num_classes)
            print("⚠️ EfficientNet created without pretrained weights")
            return model
            
        except Exception as e:
            raise ModelLoadingError(f"All EfficientNet loading methods failed: {e}")
    
    def load_checkpoint(self) -> Tuple[Optional[Dict], Optional[list]]:
        """Load model checkpoint with error handling"""
        
        if not os.path.exists(self.checkpoint_path):
            raise ModelLoadingError(f"Checkpoint file not found: {self.checkpoint_path}")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            
            # Validate checkpoint structure
            required_keys = ["model_state_dict", "breed_classes"]
            missing_keys = [key for key in required_keys if key not in checkpoint]
            
            if missing_keys:
                raise ModelLoadingError(f"Checkpoint missing required keys: {missing_keys}")
            
            breed_classes = checkpoint["breed_classes"]
            
            if not isinstance(breed_classes, list) or len(breed_classes) == 0:
                raise ModelLoadingError("Invalid breed_classes in checkpoint")
            
            print(f"✅ Checkpoint loaded: {len(breed_classes)} breed classes found")
            return checkpoint, breed_classes
            
        except Exception as e:
            raise ModelLoadingError(f"Failed to load checkpoint: {e}")
    
    def load_model(self) -> bool:
        """Load the complete model with comprehensive error handling"""
        
        try:
            # Check dependencies
            deps = self.check_dependencies()
            missing_deps = [name for name, available in deps.items() if not available]
            
            if missing_deps:
                print(f"⚠️ Missing dependencies: {missing_deps}")
                print("Install with: pip install torch torchvision efficientnet-pytorch pillow numpy")
                return False
            
            # Load checkpoint
            checkpoint, breed_classes = self.load_checkpoint()
            
            # Load model architecture
            model = self.load_efficientnet(len(breed_classes))
            
            # Load trained weights
            try:
                model.load_state_dict(checkpoint["model_state_dict"])
                print("✅ Model weights loaded successfully")
            except Exception as e:
                print(f"⚠️ Failed to load model weights: {e}")
                print("Model architecture loaded without trained weights")
            
            # Move to device and set to eval mode
            model = model.to(self.device)
            model.eval()
            
            # Get image transform
            transform = self.get_image_transform()
            
            # Store loaded components
            self.model = model
            self.breed_classes = breed_classes
            self.transform = transform
            self.is_loaded = True
            
            print(f"🎉 Model loaded successfully on {self.device}")
            return True
            
        except ModelLoadingError as e:
            print(f"❌ Model loading failed: {e}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return False
    
    def predict(self, image) -> Tuple[Optional[str], Optional[float], Optional[list]]:
        """Make prediction with loaded model"""
        
        if not self.is_loaded:
            print("❌ Model not loaded. Call load_model() first.")
            return None, None, None
        
        try:
            from PIL import Image
            import numpy as np
            
            # Ensure image is PIL Image
            if not isinstance(image, Image.Image):
                raise ValueError("Input must be a PIL Image")
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Transform image
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
            
            # Get prediction results
            pred_idx = int(np.argmax(probs))
            breed = self.breed_classes[pred_idx]
            confidence = float(probs[pred_idx])
            
            return breed, confidence, probs.tolist()
            
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            return None, None, None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        
        return {
            "is_loaded": self.is_loaded,
            "device": str(self.device),
            "num_classes": len(self.breed_classes) if self.breed_classes else 0,
            "breed_classes": self.breed_classes,
            "checkpoint_path": self.checkpoint_path,
            "model_type": "EfficientNet-B3" if self.is_loaded else None
        }

def demo_usage():
    """Demonstrate robust model loading"""
    print("🧪 Testing Robust Model Loader")
    print("=" * 50)
    
    # Initialize loader
    loader = RobustModelLoader()
    
    # Load model
    success = loader.load_model()
    
    if success:
        # Test prediction with dummy image
        try:
            from PIL import Image
            import numpy as np
            
            # Create dummy image
            dummy_image = Image.fromarray(
                np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
            )
            
            # Make prediction
            breed, confidence, probs = loader.predict(dummy_image)
            
            if breed:
                print(f"🎯 Test prediction: {breed} ({confidence:.3f} confidence)")
            
            # Show model info
            info = loader.get_model_info()
            print(f"📊 Model info: {info['num_classes']} classes on {info['device']}")
            
        except Exception as e:
            print(f"⚠️ Test prediction failed: {e}")
    else:
        print("❌ Model loading failed")

if __name__ == "__main__":
    demo_usage()