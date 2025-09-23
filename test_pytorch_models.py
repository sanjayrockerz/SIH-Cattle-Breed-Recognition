#!/usr/bin/env python3
"""
Comprehensive PyTorch Model Loading with Error Handling
Tests various pretrained models and provides robust error handling
"""

import torch
import torch.nn as nn
import sys
import os
from pathlib import Path

def test_pytorch_installation():
    """Test basic PyTorch installation and capabilities"""
    print("🔧 Testing PyTorch Installation")
    print("=" * 50)
    
    try:
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✅ CUDA version: {torch.version.cuda}")
            print(f"✅ GPU devices: {torch.cuda.device_count()}")
        print(f"✅ Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
        return True
    except Exception as e:
        print(f"❌ PyTorch installation issue: {e}")
        return False

def load_resnet18_robust():
    """Load ResNet18 with robust error handling for different PyTorch versions"""
    print("\n🏗️ Loading ResNet18 Model")
    print("=" * 50)
    
    try:
        # Method 1: New API (PyTorch 1.13+)
        import torchvision.models as models
        from torchvision.models import ResNet18_Weights
        
        resnet18 = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        resnet18.eval()
        print("✅ ResNet18 loaded with new API (weights=ResNet18_Weights.IMAGENET1K_V1)")
        return resnet18, "new_api"
        
    except (ImportError, AttributeError) as e:
        print(f"⚠️ New API failed: {e}")
        
        try:
            # Method 2: Legacy API (older PyTorch versions)
            import torchvision.models as models
            resnet18 = models.resnet18(pretrained=True)
            resnet18.eval()
            print("✅ ResNet18 loaded with legacy API (pretrained=True)")
            return resnet18, "legacy_api"
            
        except Exception as e2:
            print(f"❌ Legacy API also failed: {e2}")
            
            try:
                # Method 3: Manual loading without pretrained weights
                resnet18 = models.resnet18(pretrained=False)
                resnet18.eval()
                print("⚠️ ResNet18 loaded without pretrained weights (pretrained=False)")
                return resnet18, "no_weights"
                
            except Exception as e3:
                print(f"❌ All ResNet18 loading methods failed: {e3}")
                return None, "failed"

def load_efficientnet_robust():
    """Load EfficientNet with robust error handling"""
    print("\n🔬 Loading EfficientNet Model")
    print("=" * 50)
    
    try:
        # Method 1: efficientnet-pytorch library
        from efficientnet_pytorch import EfficientNet
        
        model = EfficientNet.from_pretrained('efficientnet-b3')
        model.eval()
        print("✅ EfficientNet-B3 loaded from efficientnet-pytorch library")
        return model, "efficientnet_pytorch"
        
    except ImportError as e:
        print(f"⚠️ efficientnet-pytorch not available: {e}")
        
        try:
            # Method 2: torchvision EfficientNet (PyTorch 1.12+)
            import torchvision.models as models
            try:
                from torchvision.models import EfficientNet_B3_Weights
                model = models.efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
            except ImportError:
                # Legacy API
                model = models.efficientnet_b3(pretrained=True)
            
            model.eval()
            print("✅ EfficientNet-B3 loaded from torchvision")
            return model, "torchvision"
            
        except Exception as e2:
            print(f"❌ EfficientNet loading failed: {e2}")
            return None, "failed"

def test_model_checkpoint_loading():
    """Test loading your specific cattle breed classifier checkpoint"""
    print("\n🐄 Testing Cattle Breed Classifier Checkpoint")
    print("=" * 50)
    
    checkpoint_path = "best_breed_classifier.pth"
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint file not found: {checkpoint_path}")
        return None, None
    
    try:
        # Load checkpoint
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        print(f"✅ Checkpoint loaded successfully")
        print(f"📋 Keys in checkpoint: {list(checkpoint.keys())}")
        
        # Get breed classes
        breed_classes = checkpoint.get("breed_classes")
        if breed_classes:
            print(f"🎯 Number of breed classes: {len(breed_classes)}")
            print(f"📝 Sample breeds: {breed_classes[:5]}...")
        else:
            print("⚠️ No breed_classes found in checkpoint")
        
        # Load EfficientNet model
        try:
            from efficientnet_pytorch import EfficientNet
            
            model = EfficientNet.from_pretrained("efficientnet-b3")
            if breed_classes:
                model._fc = nn.Linear(model._fc.in_features, len(breed_classes))
            
            # Load state dict
            model.load_state_dict(checkpoint["model_state_dict"])
            model = model.to(device)
            model.eval()
            
            print("✅ Cattle breed classifier model loaded successfully!")
            return model, breed_classes
            
        except Exception as e:
            print(f"❌ Failed to load cattle breed model: {e}")
            return None, breed_classes
            
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        return None, None

def test_model_prediction():
    """Test model prediction with a dummy image"""
    print("\n🧪 Testing Model Prediction")
    print("=" * 50)
    
    try:
        from torchvision import transforms
        from PIL import Image
        import numpy as np
        
        # Create a dummy image
        dummy_image = Image.fromarray(np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8))
        
        # Define transform
        transform = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Test with your cattle breed model
        model, breed_classes = test_model_checkpoint_loading()
        
        if model is not None and breed_classes is not None:
            # Transform image
            input_tensor = transform(dummy_image).unsqueeze(0)
            
            # Predict
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
            
            pred_idx = int(np.argmax(probs))
            breed = breed_classes[pred_idx]
            confidence = float(probs[pred_idx])
            
            print(f"✅ Prediction successful!")
            print(f"🎯 Predicted breed: {breed}")
            print(f"📊 Confidence: {confidence:.3f}")
            return True
        else:
            print("⚠️ Could not test prediction - model not loaded")
            return False
            
    except Exception as e:
        print(f"❌ Prediction test failed: {e}")
        return False

def install_missing_dependencies():
    """Suggest installation commands for missing dependencies"""
    print("\n📦 Dependency Installation Guide")
    print("=" * 50)
    
    print("If you encounter import errors, try installing these packages:")
    print()
    print("# Core PyTorch (CPU version)")
    print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
    print()
    print("# Core PyTorch (CUDA 11.8 version)")
    print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print()
    print("# EfficientNet")
    print("pip install efficientnet-pytorch")
    print()
    print("# Image processing")
    print("pip install pillow opencv-python")
    print()
    print("# All dependencies for your project")
    print("pip install streamlit pillow numpy pandas plotly matplotlib opencv-python torch torchvision efficientnet-pytorch")

def main():
    """Run comprehensive PyTorch model testing"""
    print("🚀 PyTorch Model Loading Test Suite")
    print("=" * 60)
    
    # Test 1: Basic PyTorch installation
    if not test_pytorch_installation():
        install_missing_dependencies()
        return
    
    # Test 2: ResNet18 loading
    resnet_model, resnet_method = load_resnet18_robust()
    
    # Test 3: EfficientNet loading
    efficientnet_model, efficientnet_method = load_efficientnet_robust()
    
    # Test 4: Your specific model checkpoint
    cattle_model, breed_classes = test_model_checkpoint_loading()
    
    # Test 5: Model prediction
    prediction_success = test_model_prediction()
    
    # Summary
    print("\n📋 Test Summary")
    print("=" * 50)
    print(f"✅ PyTorch Installation: Working")
    print(f"{'✅' if resnet_model else '❌'} ResNet18 Loading: {resnet_method}")
    print(f"{'✅' if efficientnet_model else '❌'} EfficientNet Loading: {efficientnet_method}")
    print(f"{'✅' if cattle_model else '❌'} Cattle Model Loading: {'Success' if cattle_model else 'Failed'}")
    print(f"{'✅' if prediction_success else '❌'} Model Prediction: {'Working' if prediction_success else 'Failed'}")
    
    if not efficientnet_model:
        print("\n🔧 Troubleshooting Suggestions:")
        print("1. Install efficientnet-pytorch: pip install efficientnet-pytorch")
        print("2. Update PyTorch: pip install --upgrade torch torchvision")
        print("3. Check internet connection for model downloads")
        
    install_missing_dependencies()

if __name__ == "__main__":
    main()