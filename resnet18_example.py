#!/usr/bin/env python3
"""
Robust PyTorch Model Loading Example - ResNet18
Demonstrates proper error handling for pretrained models
"""

import torch
import torch.nn as nn

def load_resnet18_robust():
    """
    Load ResNet18 with comprehensive error handling
    Handles PyTorch version differences and network issues
    """
    
    print("🔄 Loading ResNet18 with robust error handling...")
    
    try:
        # Method 1: New API (PyTorch 1.13+) - Recommended
        import torchvision.models as models
        from torchvision.models import ResNet18_Weights
        
        try:
            # Use DEFAULT weights (recommended)
            resnet18 = models.resnet18(weights=ResNet18_Weights.DEFAULT)
            resnet18.eval()
            print("✅ ResNet18 loaded with DEFAULT weights (new API)")
            return resnet18, "new_api_default"
            
        except AttributeError:
            # Use IMAGENET1K_V1 weights (specific)
            resnet18 = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            resnet18.eval()
            print("✅ ResNet18 loaded with IMAGENET1K_V1 weights (new API)")
            return resnet18, "new_api_imagenet"
            
    except (ImportError, AttributeError) as e:
        print(f"⚠️ New API failed: {e}")
        
        try:
            # Method 2: Legacy API (PyTorch < 1.13)
            import torchvision.models as models
            resnet18 = models.resnet18(pretrained=True)
            resnet18.eval()
            print("✅ ResNet18 loaded with legacy API (pretrained=True)")
            return resnet18, "legacy_api"
            
        except Exception as e2:
            print(f"⚠️ Legacy API failed: {e2}")
            
            try:
                # Method 3: No pretrained weights (fallback)
                resnet18 = models.resnet18(pretrained=False)
                resnet18.eval()
                print("⚠️ ResNet18 loaded without pretrained weights")
                return resnet18, "no_weights"
                
            except Exception as e3:
                print(f"❌ All ResNet18 loading methods failed: {e3}")
                return None, "failed"

def modify_resnet_for_custom_classes(model, num_classes):
    """Modify ResNet18 for custom number of classes"""
    if model is None:
        return None
        
    try:
        # Replace the final fully connected layer
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        print(f"✅ Modified ResNet18 for {num_classes} classes")
        return model
    except Exception as e:
        print(f"❌ Failed to modify ResNet18: {e}")
        return None

def test_resnet_prediction(model):
    """Test ResNet18 with dummy data"""
    if model is None:
        return False
        
    try:
        import torch
        
        # Create dummy input (batch_size=1, channels=3, height=224, width=224)
        dummy_input = torch.randn(1, 3, 224, 224)
        
        # Make prediction
        with torch.no_grad():
            output = model(dummy_input)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)
        
        print(f"✅ ResNet18 prediction successful!")
        print(f"🎯 Output shape: {output.shape}")
        print(f"📊 Predicted class: {predicted_class.item()}")
        print(f"🔢 Max probability: {probabilities.max().item():.3f}")
        return True
        
    except Exception as e:
        print(f"❌ ResNet18 prediction failed: {e}")
        return False

def handle_common_errors():
    """Handle common PyTorch model loading errors"""
    
    print("\n🔧 Common Error Solutions:")
    print("=" * 50)
    
    print("1. ModuleNotFoundError: No module named 'torchvision'")
    print("   Solution: pip install torchvision")
    print()
    
    print("2. AttributeError: module has no attribute 'ResNet18_Weights'")
    print("   Solution: Update PyTorch - pip install --upgrade torch torchvision")
    print()
    
    print("3. URLError or ConnectionError during model download")
    print("   Solution: Check internet connection or use offline weights")
    print()
    
    print("4. RuntimeError: CUDA out of memory")
    print("   Solution: Use CPU device - torch.device('cpu')")
    print()
    
    print("5. UserWarning about deprecated 'pretrained' parameter")
    print("   Solution: Use new weights API - weights=ResNet18_Weights.DEFAULT")

def main():
    """Main demonstration function"""
    
    print("🚀 Robust ResNet18 Loading Example")
    print("=" * 60)
    
    # Test PyTorch installation
    try:
        print(f"📦 PyTorch version: {torch.__version__}")
        print(f"🔧 CUDA available: {torch.cuda.is_available()}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🎯 Using device: {device}")
    except Exception as e:
        print(f"❌ PyTorch issue: {e}")
        return
    
    print("\n" + "="*60)
    
    # Load ResNet18
    model, method = load_resnet18_robust()
    
    if model is not None:
        # Move to appropriate device
        model = model.to(device)
        
        # Test original model (1000 classes)
        print(f"\n🧪 Testing original ResNet18 (1000 classes)...")
        test_resnet_prediction(model)
        
        # Modify for custom classes (e.g., for cattle breeds)
        print(f"\n🔧 Modifying for custom classes...")
        num_custom_classes = 41  # For your cattle breed project
        custom_model = modify_resnet_for_custom_classes(model, num_custom_classes)
        
        if custom_model is not None:
            custom_model = custom_model.to(device)
            print(f"\n🧪 Testing modified ResNet18 ({num_custom_classes} classes)...")
            test_resnet_prediction(custom_model)
    
    # Show common error solutions
    handle_common_errors()
    
    print(f"\n✅ ResNet18 loading demonstration completed!")
    print(f"📋 Method used: {method}")

if __name__ == "__main__":
    main()