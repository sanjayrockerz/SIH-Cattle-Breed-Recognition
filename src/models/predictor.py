"""
ML Model prediction functionality
Handles both real ML model and demo mode predictions
"""

import numpy as np
import streamlit as st
from src.validation.cattle_validator import validate_cattle_image


def predict_breed_ml(image, model, breed_classes, device, transform_func):
    """ML-based breed prediction with enhanced cattle validation"""
    try:
        # First validate if image contains cattle
        is_cattle, confidence, reason = validate_cattle_image(image)
        
        if not is_cattle:
            error_msg = _get_rejection_message(reason)
            return None, None, None, error_msg
        
        image_rgb = image.convert("RGB")
        input_tensor = transform_func(image_rgb).unsqueeze(0).to(device)
        
        import torch
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
        
        pred_idx = int(np.argmax(probs))
        breed = breed_classes[pred_idx]
        conf = float(probs[pred_idx])
        
        validation_msg = f"✅ **Cattle Detected** ({confidence:.1%} confidence): {reason}"
        return breed, conf, probs, validation_msg
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None, None, f"Error: {str(e)}"


def predict_breed_demo(image, breed_classes):
    """Demo prediction function with enhanced cattle validation"""
    # First validate if image contains cattle
    is_cattle, confidence, reason = validate_cattle_image(image)
    
    if not is_cattle:
        error_msg = _get_rejection_message(reason)
        return None, None, None, error_msg
    
    np.random.seed(hash(str(image.size)) % 2**32)  # Consistent results per image
    probs = np.random.random(len(breed_classes))
    probs = probs / probs.sum()
    pred_idx = int(np.argmax(probs))
    breed = breed_classes[pred_idx]
    conf = float(probs[pred_idx])
    
    validation_msg = f"✅ **Cattle Detected** ({confidence:.1%} confidence): {reason}"
    return breed, conf, probs, validation_msg


def load_ml_model():
    """Load the ML model if available"""
    try:
        import torch
        import torch.nn as nn
        from efficientnet_pytorch import EfficientNet
        
        checkpoint_path = "best_breed_classifier.pth"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            breed_classes = checkpoint.get("breed_classes")
            
            if breed_classes is None:
                return None, None, device
            
            model = EfficientNet.from_pretrained("efficientnet-b3")
            model._fc = nn.Linear(model._fc.in_features, len(breed_classes))
            model.load_state_dict(checkpoint["model_state_dict"])
            model = model.to(device)
            model.eval()
            
            # Success! Return model components
            return model, breed_classes, device
            
        except FileNotFoundError:
            return None, None, device
            
    except ImportError:
        return None, None, "cpu"


def get_image_transform():
    """Get image transformation for ML model"""
    try:
        from torchvision import transforms
        return transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    except ImportError:
        return None


def _get_rejection_message(reason):
    """Get formatted rejection message for invalid images"""
    error_msg = f"❌ **Image Rejected**: {reason}\n\n"
    error_msg += "**Please upload an image containing:**\n"
    error_msg += "• 🐄 Cattle (cows, bulls, oxen)\n"
    error_msg += "• 🐃 Buffalo (water buffalo)\n\n"
    error_msg += "**Avoid images with:**\n"
    error_msg += "• 🚫 Humans or people\n"
    error_msg += "• 🚫 Dogs, cats, or other pets\n"
    error_msg += "• 🚫 Other animals (goats, sheep, horses, etc.)\n"
    error_msg += "• 🚫 Objects, landscapes, or buildings"
    return error_msg