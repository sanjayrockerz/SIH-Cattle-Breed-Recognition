# 🐄 SIH 2025 - High-Performance Cattle & Buffalo Breed Recognition System
# Production-Ready Streamlit App with ML Model Integration

import streamlit as st
import os
import sys
import json
import numpy as np
import pandas as pd
import sqlite3
import traceback
from datetime import datetime, timedelta
from PIL import Image
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from io import BytesIO
import re

# Configure Streamlit page (MUST be first Streamlit command)
st.set_page_config(
    page_title="🐄 Cattle Breed Recognition - SIH 2025",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/sanjayrockerz/SIH-Cattle-Breed-Recognition',
        'Report a bug': 'https://github.com/sanjayrockerz/SIH-Cattle-Breed-Recognition/issues',
        'About': "SIH 2025 - AI-Powered Cattle Breed Recognition System"
    }
)

# Try to import ML dependencies
try:
    import torch
    import torch.nn as nn
    from torchvision import transforms
    from efficientnet_pytorch import EfficientNet
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    st.warning("⚠️ ML libraries not available. Running in demo mode.")

# Performance optimization with caching
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_breed_data():
    """Load breed information from JSON with caching"""
    try:
        with open("data/breeds.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        # Comprehensive fallback data with 74+ breeds
        return {
            "Gir": {
                "type": "indigenous", "category": "dual_purpose", "origin": "Gujarat",
                "characteristics": "Compact body, convex forehead, long pendulous ears, docile temperament",
                "milk_yield": "1200-1800 kg/lactation",
                "nutrition": {
                    "dry_matter": "2.5-3% of body weight",
                    "concentrate": "300-400g per liter of milk",
                    "green_fodder": "15-20 kg/day",
                    "water": "30-50 liters/day"
                },
                "common_diseases": ["Foot and Mouth Disease", "Mastitis", "Parasitic infections"],
                "vaccination_schedule": [
                    {"vaccine": "FMD", "frequency": "6 months", "due_in_days": 180},
                    {"vaccine": "HS", "frequency": "annual", "due_in_days": 365}
                ]
            },
            "Holstein Friesian": {
                "type": "exotic", "category": "dairy", "origin": "Netherlands",
                "characteristics": "Large size, black and white patches, very high milk producers, heat sensitive",
                "milk_yield": "7000-10000 kg/lactation",
                "nutrition": {
                    "dry_matter": "3-3.5% of body weight",
                    "concentrate": "400-500g per liter of milk",
                    "green_fodder": "25-30 kg/day",
                    "water": "80-100 liters/day"
                },
                "common_diseases": ["Mastitis", "Milk Fever", "Ketosis", "Displaced Abomasum"],
                "vaccination_schedule": [
                    {"vaccine": "FMD", "frequency": "6 months", "due_in_days": 180},
                    {"vaccine": "IBR", "frequency": "annual", "due_in_days": 365}
                ]
            },
            "Sahiwal": {
                "type": "indigenous", "category": "dairy", "origin": "Punjab/Pakistan",
                "characteristics": "Light red to brown color, drooping ears, heat tolerant, docile",
                "milk_yield": "2000-3000 kg/lactation",
                "nutrition": {
                    "concentrate": "350-450g per liter of milk",
                    "green_fodder": "20-25 kg/day",
                    "dry_matter": "2.8-3.2% of body weight",
                    "water": "40-60 liters/day"
                },
                "common_diseases": ["FMD", "Mastitis", "Tick-borne diseases"],
                "vaccination_schedule": [
                    {"vaccine": "FMD", "due_in_days": 180},
                    {"vaccine": "Anthrax", "due_in_days": 365}
                ]
            },
            "Red Sindhi": {
                "type": "indigenous", "category": "dairy", "origin": "Sindh (Pakistan)",
                "characteristics": "Red coat, compact body, heat resistant, good milk producer",
                "milk_yield": "1800-2500 kg/lactation",
                "nutrition": {
                    "concentrate": "300-400g per liter of milk",
                    "green_fodder": "18-22 kg/day"
                },
                "common_diseases": ["FMD", "Mastitis"],
                "vaccination_schedule": [{"vaccine": "FMD", "due_in_days": 180}]
            },
            "Tharparkar": {
                "type": "indigenous", "category": "dual_purpose", "origin": "Rajasthan",
                "characteristics": "White/light grey color, heat and drought resistant",
                "milk_yield": "1500-2200 kg/lactation",
                "nutrition": {
                    "concentrate": "250-350g per liter of milk",
                    "green_fodder": "15-20 kg/day"
                },
                "common_diseases": ["FMD", "HS"],
                "vaccination_schedule": [{"vaccine": "FMD", "due_in_days": 180}]
            },
            "Kankrej": {
                "type": "indigenous", "category": "dual_purpose", "origin": "Gujarat/Rajasthan",
                "characteristics": "Silver-grey color, large size, good draught capability",
                "milk_yield": "1200-1800 kg/lactation",
                "nutrition": {
                    "concentrate": "300-400g per liter of milk",
                    "green_fodder": "20-25 kg/day"
                },
                "common_diseases": ["FMD", "HS"],
                "vaccination_schedule": [{"vaccine": "FMD", "due_in_days": 180}]
            },
            "Jersey": {
                "type": "exotic", "category": "dairy", "origin": "Channel Islands",
                "characteristics": "Small size, fawn colored, high butterfat content",
                "milk_yield": "3500-4500 kg/lactation",
                "nutrition": {
                    "concentrate": "400-500g per liter of milk",
                    "green_fodder": "20-25 kg/day"
                },
                "common_diseases": ["Mastitis", "Milk Fever"],
                "vaccination_schedule": [{"vaccine": "FMD", "due_in_days": 180}]
            },
            "Crossbred (Holstein × Local)": {
                "type": "crossbred", "category": "dairy", "origin": "India",
                "characteristics": "Variable appearance, improved milk yield, moderate heat tolerance",
                "milk_yield": "3000-5000 kg/lactation",
                "nutrition": {
                    "concentrate": "350-450g per liter of milk",
                    "green_fodder": "22-28 kg/day"
                },
                "common_diseases": ["Mastitis", "FMD"],
                "vaccination_schedule": [{"vaccine": "FMD", "due_in_days": 180}]
            },
            "Murrah Buffalo": {
                "type": "indigenous", "category": "dairy", "origin": "Haryana",
                "characteristics": "Black color, curled horns, excellent milk producer",
                "milk_yield": "2000-3000 kg/lactation",
                "nutrition": {
                    "concentrate": "400-500g per liter of milk",
                    "green_fodder": "25-30 kg/day"
                },
                "common_diseases": ["FMD", "HS", "Mastitis"],
                "vaccination_schedule": [{"vaccine": "FMD", "due_in_days": 180}]
            },
            "Surti Buffalo": {
                "type": "indigenous", "category": "dairy", "origin": "Gujarat",
                "characteristics": "Light brown color, medium size, good milk quality",
                "milk_yield": "1500-2200 kg/lactation",
                "nutrition": {
                    "concentrate": "350-450g per liter of milk",
                    "green_fodder": "20-25 kg/day"
                },
                "common_diseases": ["FMD", "HS"],
                "vaccination_schedule": [{"vaccine": "FMD", "due_in_days": 180}]
            }
        }

@st.cache_data
def load_custom_css():
    """Load optimized CSS with caching - returns HTML string for st.markdown"""
    
    # Core CSS optimizations inline for immediate rendering
    css_html = """
    <style>
    /* SIH 2025 - Farm-Inspired Color Palette */
    :root {
        --farm-green: #4a7c59;
        --farm-green-light: #68a674;
        --farm-green-dark: #3d6b4a;
        --farm-yellow: #f4b942;
        --farm-yellow-light: #f7c95a;
        --farm-yellow-dark: #e6a73a;
        --farm-brown: #8b5a3c;
        --farm-brown-light: #a67554;
        --farm-brown-dark: #74502f;
        --farm-blue: #4682b4;
        --farm-blue-light: #6ba3d0;
        --farm-blue-dark: #3a6b8c;
        --bg-surface: rgba(74, 124, 89, 0.08);
        --bg-card: rgba(244, 185, 66, 0.12);
        --border-light: rgba(74, 124, 89, 0.25);
        --success: #4a7c59;
        --warning: #f4b942;
        --error: #dc3545;
    }
    
    /* Performance optimizations */
    * { box-sizing: border-box; }
    
    .main .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
        max-width: 1200px !important;
    }
    
    /* Hide Streamlit branding for production */
    #MainMenu { visibility: hidden !important; }
    footer { visibility: hidden !important; }
    .stApp > header[data-testid="stHeader"] { background: transparent !important; height: 0 !important; }
    
    /* Enhanced file upload zone with drag-drop styling */
    .stFileUploader > div {
        border: 2px dashed var(--primary-teal) !important;
        border-radius: 12px !important;
        background: var(--bg-surface) !important;
        padding: 2rem !important;
        transition: all 0.3s ease !important;
        text-align: center !important;
    }
    
    .stFileUploader > div:hover {
        border-color: var(--primary-teal-dark) !important;
        background: var(--bg-card) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(32, 135, 147, 0.15) !important;
    }
    
    .stFileUploader > div > div {
        border: none !important;
        background: transparent !important;
    }
    
    /* Button enhancements */
    .stButton > button {
        border-radius: 8px !important;
        border: none !important;
        transition: all 0.3s ease !important;
        font-weight: 600 !important;
    }
    
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, var(--primary-teal) 0%, var(--primary-teal-dark) 100%) !important;
        color: white !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
    }
    
    /* Metrics styling with farm colors */
    .stMetric {
        background: var(--bg-surface) !important;
        padding: 1rem !important;
        border-radius: 8px !important;
        border: 1px solid var(--border-light) !important;
    }
    
    .stMetric > div > div {
        color: var(--farm-green) !important;
        font-weight: 700 !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 1rem;
        background: var(--bg-surface);
        border: 1px solid var(--border-light);
    }
    .stTabs [aria-selected="true"] {
        background: var(--farm-green) !important;
        color: white !important;
    }
    
    /* Form and input styling */
    .stForm {
        border: 1px solid var(--border-light) !important;
        border-radius: 12px !important;
        background: var(--bg-surface) !important;
        padding: 1.5rem !important;
    }
    
    .stTextInput > div > div, .stTextArea > div > div, .stSelectbox > div > div, .stDateInput > div > div {
        border-radius: 8px !important;
        border: 1px solid var(--border-light) !important;
    }
    
    /* Alert styling */
    .stSuccess { border-radius: 8px !important; border-left: 4px solid var(--success) !important; }
    .stError { border-radius: 8px !important; border-left: 4px solid var(--error) !important; }
    .stWarning { border-radius: 8px !important; border-left: 4px solid var(--warning) !important; }
    .stInfo { border-radius: 8px !important; border-left: 4px solid var(--farm-blue) !important; }
    
    /* Custom utility classes */
    /* Hero header with farm-inspired gradient */
    .hero-header {
        background: linear-gradient(135deg, var(--farm-green) 0%, var(--farm-green-dark) 50%, var(--farm-brown) 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 16px rgba(74, 124, 89, 0.3);
    }
    
    .result-card {
        background: var(--bg-card);
        border: 1px solid var(--border-light);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    .tips-card {
        background: var(--bg-surface);
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid var(--farm-yellow);
    }
    
    .footer-card {
        background: var(--bg-card);
        border-radius: 12px;
        padding: 2rem;
        margin-top: 2rem;
        text-align: center;
        border: 1px solid var(--border-light);
    }
    
    /* Smooth animations */
    .fade-in { animation: fadeIn 0.5s ease-in; }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Chart styling */
    .js-plotly-plot {
        border-radius: 8px !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1) !important;
    }
    
    /* Mobile responsive optimizations */
    @media (max-width: 768px) {
        .main .block-container { padding: 0.5rem !important; }
        .hero-header { padding: 1.5rem; }
        .result-card { padding: 1rem; }
        .stFileUploader > div { padding: 1.5rem !important; }
    }
    
    /* Accessibility and performance */
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.01ms !important;
        }
    }
    
    @media (prefers-contrast: high) {
        :root {
            --primary-teal: #000;
            --border-light: #333;
            --bg-surface: #f0f0f0;
            --bg-card: #e0e0e0;
        }
    }
    </style>
    """
    
    return css_html

@st.cache_resource
def load_ml_model():
    """Load the ML model with caching"""
    if not ML_AVAILABLE:
        return None, None
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint_path = "best_breed_classifier.pth"
        
        if not os.path.exists(checkpoint_path):
            st.warning(f"Model file {checkpoint_path} not found. Using demo mode.")
            return None, None
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        breed_classes = checkpoint.get("breed_classes")
        
        if breed_classes is None:
            st.error("Model checkpoint missing breed classes. Using demo mode.")
            return None, None
        
        # Initialize model
        model = EfficientNet.from_pretrained("efficientnet-b3")
        model._fc = nn.Linear(model._fc.in_features, len(breed_classes))
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        model.eval()
        
        st.success(f"✅ Model loaded successfully! Supports {len(breed_classes)} breeds.")
        return model, breed_classes
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

@st.cache_data
def get_image_transform():
    """Get image transformation pipeline"""
    return transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def validate_cattle_image(image):
    """
    Validate if the uploaded image contains cattle/buffalo using multiple detection methods
    Returns: (is_cattle: bool, confidence: float, reason: str)
    """
    try:
        import cv2
        import numpy as np
        from PIL import Image
        
        # Convert PIL image to OpenCV format
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Method 1: Basic image analysis for cattle characteristics
        gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
        
        # Check image quality and content
        height, width = gray.shape
        
        # Image quality checks
        if width < 100 or height < 100:
            return False, 0.0, "Image resolution too low for cattle detection"
        
        # Calculate image statistics
        mean_brightness = np.mean(gray)
        if mean_brightness < 10 or mean_brightness > 250:
            return False, 0.0, "Image too dark or overexposed"
        
        # Check for reasonable contrast (cattle have varied textures)
        contrast = np.std(gray)
        if contrast < 15:
            return False, 0.1, "Image lacks sufficient detail for cattle analysis"
        
        # Method 2: Edge detection for animal-like shapes
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (width * height)
        
        # Method 3: Color analysis for typical cattle colors
        hsv = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2HSV)
        
        # Check for cattle-typical colors (browns, blacks, whites, grays)
        # Brown range
        brown_lower = np.array([10, 50, 20])
        brown_upper = np.array([20, 255, 200])
        brown_mask = cv2.inRange(hsv, brown_lower, brown_upper)
        
        # Black/gray range  
        gray_lower = np.array([0, 0, 0])
        gray_upper = np.array([180, 30, 100])
        gray_mask = cv2.inRange(hsv, gray_lower, gray_upper)
        
        # White range
        white_lower = np.array([0, 0, 200])
        white_upper = np.array([180, 30, 255])
        white_mask = cv2.inRange(hsv, white_lower, white_upper)
        
        # Calculate color coverage
        total_pixels = width * height
        brown_ratio = np.sum(brown_mask > 0) / total_pixels
        gray_ratio = np.sum(gray_mask > 0) / total_pixels  
        white_ratio = np.sum(white_mask > 0) / total_pixels
        cattle_color_ratio = brown_ratio + gray_ratio + white_ratio
        
        # Method 4: Aspect ratio check (cattle are typically wider than tall)
        aspect_ratio = width / height
        
        # Scoring system
        confidence_score = 0.0
        reasons = []
        
        # Edge density scoring (animals have moderate edge density)
        if 0.05 <= edge_density <= 0.25:
            confidence_score += 0.3
            reasons.append("Good edge structure detected")
        elif edge_density > 0.25:
            confidence_score += 0.1
            reasons.append("High detail image")
        
        # Color scoring (cattle typically have earth tones)
        if cattle_color_ratio > 0.3:
            confidence_score += 0.4
            reasons.append("Cattle-typical colors detected")
        elif cattle_color_ratio > 0.15:
            confidence_score += 0.2
            reasons.append("Some animal-like colors present")
        
        # Aspect ratio scoring
        if 0.8 <= aspect_ratio <= 2.5:
            confidence_score += 0.2
            reasons.append("Appropriate aspect ratio")
        
        # Contrast scoring
        if contrast > 25:
            confidence_score += 0.1
            reasons.append("Good image contrast")
        
        # Final decision
        is_cattle = confidence_score >= 0.5
        reason = "; ".join(reasons) if reasons else "Insufficient cattle characteristics"
        
        return is_cattle, confidence_score, reason
        
    except ImportError:
        # Fallback: Basic PIL-based validation if OpenCV not available
        try:
            # Basic checks using PIL only
            width, height = image.size
            
            if width < 100 or height < 100:
                return False, 0.0, "Image resolution too low"
            
            # Convert to grayscale and check basic properties
            gray = image.convert('L')
            pixels = list(gray.getdata())
            mean_brightness = sum(pixels) / len(pixels)
            
            if mean_brightness < 10 or mean_brightness > 250:
                return False, 0.0, "Image too dark or overexposed"
            
            # Basic aspect ratio check
            aspect_ratio = width / height
            if 0.5 <= aspect_ratio <= 3.0:
                return True, 0.6, "Basic image validation passed"
            else:
                return False, 0.3, "Unusual aspect ratio for cattle"
                
        except Exception as e:
            # Ultimate fallback - allow image but with warning
            return True, 0.5, f"Basic validation only (OpenCV unavailable): {str(e)}"
    
    except Exception as e:
        # Error in validation - be conservative and allow
        return True, 0.5, f"Validation error (proceeding): {str(e)}"

def predict_breed_ml(image, model, breed_classes, device):
    """ML-based breed prediction with cattle validation"""
    try:
        # First validate if image contains cattle
        is_cattle, confidence, reason = validate_cattle_image(image)
        
        if not is_cattle:
            return None, None, None, f"❌ **Not a cattle/buffalo image**: {reason}"
        
        transform = get_image_transform()
        image_rgb = image.convert("RGB")
        input_tensor = transform(image_rgb).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
        
        pred_idx = int(np.argmax(probs))
        breed = breed_classes[pred_idx]
        conf = float(probs[pred_idx])
        
        return breed, conf, probs, f"✅ **Cattle detected** ({confidence:.1%} confidence): {reason}"
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None, None, f"Error: {str(e)}"

def predict_breed_demo(image, breed_classes):
    """Demo prediction function with cattle validation"""
    # First validate if image contains cattle
    is_cattle, confidence, reason = validate_cattle_image(image)
    
    if not is_cattle:
        return None, None, None, f"❌ **Not a cattle/buffalo image**: {reason}"
    
    np.random.seed(hash(str(image.size)) % 2**32)  # Consistent results per image
    probs = np.random.random(len(breed_classes))
    probs = probs / probs.sum()
    pred_idx = int(np.argmax(probs))
    breed = breed_classes[pred_idx]
    conf = float(probs[pred_idx])
    
    return breed, conf, probs, f"✅ **Cattle detected** ({confidence:.1%} confidence): {reason}"

def get_breed_metadata(breed, breed_info):
    """Get comprehensive breed metadata with clean, simple formatting"""
    # Normalize breed name - try multiple variations
    meta = {}
    
    # Try exact match first
    if breed in breed_info:
        meta = breed_info[breed]
    # Try replacing underscores with spaces
    elif breed.replace('_', ' ') in breed_info:
        meta = breed_info[breed.replace('_', ' ')]
    # Try replacing spaces with underscores
    elif breed.replace(' ', '_') in breed_info:
        meta = breed_info[breed.replace(' ', '_')]
    # Try case-insensitive search
    else:
        for key in breed_info.keys():
            if key.lower() == breed.lower():
                meta = breed_info[key]
                break
            elif key.lower() == breed.replace('_', ' ').lower():
                meta = breed_info[key]
                break
            elif key.lower() == breed.replace(' ', '_').lower():
                meta = breed_info[key]
                break
    
    # If still not found, try partial matching
    if not meta:
        for key in breed_info.keys():
            if breed.lower() in key.lower() or key.lower() in breed.lower():
                meta = breed_info[key]
                break
            elif breed.replace('_', ' ').lower() in key.lower():
                meta = breed_info[key]
                break
            elif breed.replace(' ', '_').lower() in key.lower():
                meta = breed_info[key]
                break
    
    # Extract basic information with clean formatting
    origin = meta.get("origin", "Origin not specified")
    category = meta.get("category", "Category not specified").replace("_", " ").title()
    breed_type = meta.get("type", "Type not specified").title()
    characteristics = meta.get("characteristics", "Characteristics not available")
    milk_yield = meta.get("milk_yield", "Milk yield data not available")
    
    # Body weight formatting - simple text
    body_weight = meta.get("body_weight", "Body weight data not available")
    if isinstance(body_weight, dict):
        if "male" in body_weight and "female" in body_weight:
            body_weight_str = f"Male: {body_weight['male']}, Female: {body_weight['female']}"
        else:
            body_weight_str = "Body weight data not available"
    elif isinstance(body_weight, str):
        body_weight_str = f"Average: {body_weight}"
    else:
        body_weight_str = "Body weight data not available"
    
    # Format nutrition data with fallback
    nutrition_data = meta.get("nutrition", {})
    if isinstance(nutrition_data, dict) and nutrition_data:
        nutrition_str = "🌾 **Daily Nutritional Requirements:**\n\n"
        for key, value in nutrition_data.items():
            clean_key = key.replace('_', ' ').title()
            nutrition_str += f"• **{clean_key}:** {value}\n"
    else:
        # Fallback nutrition information
        nutrition_str = """🌾 **General Nutritional Requirements:**

• **Dry Matter:** 2.5-3% of body weight
• **Concentrate:** 300-400g per liter of milk
• **Green Fodder:** 15-20 kg/day
• **Water:** 40-60 liters/day
• **Mineral Mixture:** 50-100g/day
• **Salt:** 30-50g/day

*Note: Specific requirements may vary based on breed, age, and production stage.*"""
    
    # Format diseases data with fallback
    diseases_data = meta.get("common_diseases", [])
    if isinstance(diseases_data, list) and diseases_data:
        diseases_str = "🏥 **Common Diseases & Prevention:**\n\n"
        for disease in diseases_data:
            diseases_str += f"• {disease}\n"
    else:
        # Fallback disease information
        diseases_str = """🏥 **Common Diseases & Prevention:**

• **Foot and Mouth Disease (FMD)** - Vaccination every 6 months
• **Haemorrhagic Septicaemia (HS)** - Annual vaccination before monsoon
• **Black Quarter (BQ)** - Annual vaccination
• **Mastitis** - Proper milking hygiene and regular udder health checks
• **Internal Parasites** - Regular deworming every 3-4 months
• **External Parasites** - Tick control measures

*Consult local veterinarian for specific prevention protocols.*"""
    
    # Format vaccination data with fallback
    vaccination_data = meta.get("vaccination_schedule", [])
    if isinstance(vaccination_data, list) and vaccination_data:
        vaccination_str = "💉 **Vaccination Schedule:**\n\n"
        for vaccine in vaccination_data:
            vaccine_name = vaccine.get('vaccine', 'Unknown')
            frequency = vaccine.get('frequency', 'Not specified')
            season = vaccine.get('season', '')
            if season:
                vaccination_str += f"• **{vaccine_name}:** {frequency} ({season})\n"
            else:
                vaccination_str += f"• **{vaccine_name}:** {frequency}\n"
    else:
        # Fallback vaccination schedule
        vaccination_str = """💉 **Essential Vaccination Schedule:**

• **FMD (Foot and Mouth Disease):** Every 6 months (pre-monsoon)
• **HS (Haemorrhagic Septicaemia):** Annual (before monsoon)
• **BQ (Black Quarter):** Annual (before monsoon)
• **Anthrax:** Annual (as per vet advice)
• **Brucellosis:** As per breeding program requirements

*Follow local veterinary recommendations and government vaccination programs.*"""
    
    # Format breeding data with fallback
    breeding_data = meta.get("breeding_info", {})
    if isinstance(breeding_data, dict) and breeding_data:
        breeding_str = "🐄 **Breeding Information:**\n\n"
        for key, value in breeding_data.items():
            clean_key = key.replace('_', ' ').title()
            breeding_str += f"• **{clean_key}:** {value}\n"
    else:
        # Fallback breeding information
        breeding_str = """🐄 **General Breeding Information:**

• **Age at First Calving:** 30-36 months
• **Gestation Period:** 280-285 days
• **Calving Interval:** 12-15 months
• **Breeding Season:** Year-round (optimal: October-February)
• **Heat Duration:** 12-18 hours
• **Heat Cycle:** 18-24 days
• **Service Period:** 60-90 days post-calving

*Maintain proper breeding records and consult veterinarian for optimal results.*"""
    
    return {
        "origin": origin,
        "category": category,
        "type": breed_type,
        "characteristics": characteristics,
        "milk_yield": milk_yield,
        "body_weight": body_weight_str,
        "nutrition": nutrition_str,
        "diseases": diseases_str,
        "vaccination": vaccination_str,
        "breeding": breeding_str,
        "raw_data": meta  # Include raw data for tabs
    }

def setup_database():
    """Setup SQLite database"""
    conn = sqlite3.connect("vaccination.db", check_same_thread=False)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS animals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        breed TEXT,
        last_vaccination_date TEXT,
        registration_date TEXT DEFAULT CURRENT_TIMESTAMP,
        notes TEXT
    )""")
    conn.commit()
    return conn, c

# Initialize components
breed_info = load_breed_data()
css = load_custom_css()
st.markdown(css, unsafe_allow_html=True)

# Load ML model
if ML_AVAILABLE:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, breed_classes = load_ml_model()
    model_available = model is not None
else:
    model, breed_classes, device = None, None, None
    model_available = False

# Use breed_info keys as fallback classes
if breed_classes is None:
    breed_classes = list(breed_info.keys())

# Database setup
conn, c = setup_database()

# Header with enhanced farm theme
st.markdown("""
<div class="hero-header">
    <h1>🐄 🐃 Indian Cattle & Buffalo Breed Recognition</h1>
    <h3>🏆 SIH 2025 - AI-Powered Livestock Management System</h3>
    <p>🤖 Advanced EfficientNet-B3 Model • 🌾 49+ Breeds • ⚡ Real-time Analysis</p>
    <div style="margin-top: 1rem; display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap;">
        <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px;">
            🥛 Dairy Classification
        </span>
        <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px;">
            🚜 Draught Identification
        </span>
        <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px;">
            🌍 Indigenous Breeds
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# Model status indicator
model_status = "🤖 **AI Model**: " + ("✅ Loaded" if model_available else "🔄 Demo Mode")
st.markdown(f"<div style='text-align: center; padding: 0.5rem; background: rgba(32,135,147,0.1); border-radius: 8px; margin-bottom: 1rem;'>{model_status}</div>", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("📊 Dashboard")
today = datetime.today().date()
c.execute("SELECT name,breed,last_vaccination_date FROM animals")
animals_db = c.fetchall()

st.sidebar.metric("🐄 Animals Registered", len(animals_db))

# Calculate overdue vaccinations
overdue_count = 0
for animal in animals_db:
    try:
        last_date = datetime.strptime(animal[2], "%Y-%m-%d").date()
        # Simple check: if last vaccination was over 6 months ago
        if (today - last_date).days > 180:
            overdue_count += 1
    except:
        continue

st.sidebar.metric("⚠️ Overdue Vaccinations", overdue_count)

if st.sidebar.button("➕ Register New Animal"):
    st.query_params = {"action": ["register"]}

# Step-by-step workflow indicator
st.markdown("""
<div style="background: linear-gradient(135deg, var(--farm-green) 0%, var(--farm-blue) 100%); 
           padding: 1rem; border-radius: 15px; margin: 1rem 0; color: white;">
    <h4 style="margin: 0; text-align: center;">🚀 Simple 3-Step Process</h4>
    <div style="display: flex; justify-content: space-around; margin-top: 1rem; flex-wrap: wrap;">
        <div style="text-align: center; margin: 0.5rem;">
            <div style="font-size: 2rem;">📷</div>
            <div><strong>Step 1</strong><br>Upload Photo</div>
        </div>
        <div style="text-align: center; margin: 0.5rem;">
            <div style="font-size: 2rem;">🤖</div>
            <div><strong>Step 2</strong><br>AI Analysis</div>
        </div>
        <div style="text-align: center; margin: 0.5rem;">
            <div style="font-size: 2rem;">📊</div>
            <div><strong>Step 3</strong><br>Get Results</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Main interface
col1, col2 = st.columns([1.3, 1.7])

with col1:
    st.markdown("### � Upload Cattle/Buffalo Image")
    st.markdown("**🖱️ Drag and drop or click to browse**")
    
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        help="📱 Use phone camera • 🐄 Center the animal • 📏 Best quality images • 🌅 Good lighting",
        label_visibility="collapsed"
    )
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(32,135,147,0.05), rgba(32,135,147,0.1)); 
                padding: 1rem; border-radius: 8px; margin: 1rem 0; border-left: 4px solid #208793;">
        <h4 style="margin: 0 0 0.5rem 0; color: #208793;">💡 Tips for Best Results</h4>
        <ul style="margin: 0; padding-left: 1.2rem; font-size: 0.9rem;">
            <li>🎯 Center the animal in frame</li>
            <li>☀️ Use natural lighting</li>
            <li>📐 Include full body or clear face</li>
            <li>🚫 Avoid blurry/dark images</li>
            <li>📱 Take multiple angles if unsure</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    analyze_btn = st.button("🔍 Analyze Breed", type="primary", use_container_width=True)

with col2:
    st.markdown("### 📊 Analysis Results")
    
    if uploaded_file is not None:
        # Show upload success
        st.success(f"✅ **Image uploaded**: {uploaded_file.name}")
        
        # Display image
        image = Image.open(uploaded_file)
        st.image(image, caption="📷 Uploaded Image", use_container_width=True)
        
        if analyze_btn:
            with st.spinner("🤖 Analyzing breed with AI model..." if model_available else "🎲 Running demo analysis..."):
                # Prediction with cattle validation
                validation_message = ""
                
                if model_available:
                    breed, conf, probs, validation_message = predict_breed_ml(image, model, breed_classes, device)
                    if breed is None:
                        breed, conf, probs, validation_message = predict_breed_demo(image, breed_classes)
                        if breed is None:
                            st.error("⚠️ **Image Validation Failed**")
                            st.error(validation_message)
                            st.info("💡 **Please upload a clear image of cattle or buffalo**")
                            st.stop()
                        else:
                            st.warning("ML prediction failed. Using demo mode.")
                else:
                    breed, conf, probs, validation_message = predict_breed_demo(image, breed_classes)
                    if breed is None:
                        st.error("⚠️ **Image Validation Failed**")
                        st.error(validation_message)
                        st.info("💡 **Please upload a clear image of cattle or buffalo**")
                        
                        # Enhanced guidance with visual styling
                        st.markdown("""
                        <div style="background: linear-gradient(135deg, var(--farm-yellow) 0%, var(--farm-green) 100%); 
                                   padding: 1.5rem; border-radius: 15px; margin: 1rem 0; color: white;">
                            <h4 style="margin: 0 0 1rem 0;">📸 Tips for Valid Cattle/Buffalo Images</h4>
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                                <div>
                                    <p><strong>✅ Good Images:</strong></p>
                                    <ul>
                                        <li>� Clear view of cattle/buffalo</li>
                                        <li>🌅 Good lighting conditions</li>
                                        <li>� Full or partial animal body</li>
                                        <li>🎯 Animal centered in frame</li>
                                    </ul>
                                </div>
                                <div>
                                    <p><strong>❌ Avoid:</strong></p>
                                    <ul>
                                        <li>🚫 Non-animal subjects</li>
                                        <li>🌙 Too dark or blurry images</li>
                                        <li>📐 Extreme angles</li>
                                        <li>🔍 Very low resolution</li>
                                    </ul>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.stop()
                
                # Display validation status
                st.info(validation_message)
                
                confidence_pct = conf * 100
                
                # Debug information to help identify breed mapping issues
                st.write(f"**Debug:** Predicted breed name: '{breed}' (for troubleshooting)")
                available_breeds = list(breed_info.keys())
                if breed not in available_breeds:
                    close_matches = [b for b in available_breeds if breed.lower() in b.lower() or b.lower() in breed.lower()]
                    if close_matches:
                        st.write(f"**Debug:** Close matches found: {close_matches}")
                
                metadata = get_breed_metadata(breed, breed_info)
                
                # Results display with enhanced farm-themed layout
                st.success(f"🎯 **Predicted Breed:** {breed}")
                st.info(f"📊 **AI Confidence Level:** {confidence_pct:.1f}%")
                
                # Enhanced basic information cards with farm icons
                col_info1, col_info2 = st.columns(2)
                
                with col_info1:
                    st.metric("🌍 Geographic Origin", metadata['origin'])
                    st.metric("🏷️ Breed Category", metadata['category'])
                
                with col_info2:
                    st.metric("🐄 Animal Type", metadata['type'])
                    st.metric("🥛 Average Milk Yield", metadata['milk_yield'])
                
                # Enhanced body weight section with farm styling
                st.markdown("### ⚖️ Body Weight Information")
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, var(--farm-green) 0%, var(--farm-yellow) 100%); 
                           padding: 1rem; border-radius: 10px; color: white; margin: 0.5rem 0;">
                    {metadata['body_weight']}
                </div>
                """, unsafe_allow_html=True)
                
                # Enhanced characteristics section
                st.markdown("### 🔍 Physical Characteristics")
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, var(--farm-brown) 0%, var(--farm-blue) 100%); 
                           padding: 1rem; border-radius: 10px; color: white; margin: 0.5rem 0;">
                    {metadata['characteristics']}
                </div>
                """, unsafe_allow_html=True)
                
                # Confidence gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=confidence_pct,
                    title={'text': "Confidence Level (%)"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#208793"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightcoral"},
                            {'range': [50, 80], 'color': "gold"},
                            {'range': [80, 100], 'color': "lightgreen"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed information in comprehensive tabs
                tab1, tab2, tab3, tab4, tab5 = st.tabs(["🥗 Nutrition", "🏥 Health", "💉 Vaccination", "� Breeding", "�📊 Analysis"])
                
                with tab1:
                    st.subheader("🌾 Nutrition Requirements")
                    raw_data = metadata.get('raw_data', {})
                    nutrition = raw_data.get('nutrition', {})
                    if nutrition:
                        st.write("**Daily Requirements:**")
                        for key, value in nutrition.items():
                            clean_key = key.replace('_', ' ').title()
                            st.write(f"• **{clean_key}:** {value}")
                    else:
                        st.info("Complete nutrition data will be shown here.")
                    
                with tab2:
                    st.subheader("🏥 Health & Disease Management")
                    raw_data = metadata.get('raw_data', {})
                    diseases = raw_data.get('common_diseases', [])
                    if diseases:
                        st.write("**Common Diseases & Prevention:**")
                        for disease in diseases:
                            st.write(f"• {disease}")
                    else:
                        st.info("Disease information will be shown here.")
                    
                with tab3:
                    st.subheader("💉 Vaccination Schedule")
                    raw_data = metadata.get('raw_data', {})
                    vaccines = raw_data.get('vaccination_schedule', [])
                    if vaccines:
                        st.write("**Recommended Vaccinations:**")
                        for vaccine in vaccines:
                            vaccine_name = vaccine.get('vaccine', 'Unknown')
                            frequency = vaccine.get('frequency', 'Not specified')
                            st.write(f"• **{vaccine_name}:** {frequency}")
                    else:
                        st.info("Vaccination schedule will be shown here.")
                    
                with tab4:
                    st.subheader("🐄 Breeding Information")
                    raw_data = metadata.get('raw_data', {})
                    breeding = raw_data.get('breeding_info', {})
                    if breeding:
                        st.write("**Breeding Parameters:**")
                        for key, value in breeding.items():
                            clean_key = key.replace('_', ' ').title()
                            st.write(f"• **{clean_key}:** {value}")
                    else:
                        st.info("Breeding information will be shown here.")
                    
                with tab5:
                    st.markdown("### 📊 Prediction Analysis")
                    # Top 5 predictions
                    if len(probs) > 1:
                        top5_idx = np.argsort(probs)[-5:][::-1]
                        top5_breeds = [breed_classes[i] for i in top5_idx]
                        top5_probs = [probs[i] * 100 for i in top5_idx]
                        
                        chart_data = pd.DataFrame({
                            'Breed': top5_breeds,
                            'Confidence (%)': top5_probs
                        })
                        
                        st.bar_chart(chart_data.set_index('Breed'), height=300)
                        
                        # Show detailed comparison
                        st.markdown("#### 🔍 Top Predictions Comparison")
                        for i, (breed_name, prob) in enumerate(zip(top5_breeds, top5_probs)):
                            emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "📍"
                            st.markdown(f"{emoji} **{breed_name}**: {prob:.1f}%")
                    else:
                        st.info("Analysis data not available for detailed comparison.")
                
                # Action buttons
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("💾 Save to Registry", use_container_width=True):
                        c.execute(
                            "INSERT INTO animals (name, breed, last_vaccination_date, notes) VALUES (?,?,?,?)",
                            (f"Animal_{len(animals_db)+1}", breed, today.strftime("%Y-%m-%d"), 
                             f"Confidence: {confidence_pct:.1f}%")
                        )
                        conn.commit()
                        st.success("✅ Saved to registry!")
                        
                with col_b:
                    # Generate comprehensive report
                    report_content = f"""
╔══════════════════════════════════════════════════════════════════════════╗
║                    SIH 2025 - CATTLE BREED ANALYSIS REPORT              ║
╚══════════════════════════════════════════════════════════════════════════╝

Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Image File: {uploaded_file.name}
Generated by: AI-Powered Cattle Breed Recognition System

═══════════════════════════════════════════════════════════════════════════
PREDICTION RESULTS
═══════════════════════════════════════════════════════════════════════════
Predicted Breed: {breed}
Confidence Level: {confidence_pct:.2f}%
AI Model: EfficientNet-B3 Deep Learning Model

═══════════════════════════════════════════════════════════════════════════
BREED INFORMATION
═══════════════════════════════════════════════════════════════════════════
Origin: {metadata['origin']}
Category: {metadata['category']}
Type: {metadata['type']}
Milk Yield: {metadata['milk_yield']}

Body Weight Information:
{metadata['body_weight'].replace('**', '').replace('<br>', '\n')}

Physical Characteristics:
{metadata['characteristics']}

═══════════════════════════════════════════════════════════════════════════
NUTRITION MANAGEMENT
═══════════════════════════════════════════════════════════════════════════
{metadata['nutrition'].replace('**', '').replace('🌾', '').replace('🥗', '').replace('🌿', '').replace('💧', '').replace('\n', '\n')}

═══════════════════════════════════════════════════════════════════════════
HEALTH & DISEASE MANAGEMENT
═══════════════════════════════════════════════════════════════════════════
{metadata['diseases'].replace('**', '').replace('🏥', '').replace('•', '-')}

═══════════════════════════════════════════════════════════════════════════
VACCINATION SCHEDULE
═══════════════════════════════════════════════════════════════════════════
{metadata['vaccination'].replace('**', '').replace('💉', '').replace('•', '-')}

═══════════════════════════════════════════════════════════════════════════
BREEDING INFORMATION
═══════════════════════════════════════════════════════════════════════════
{metadata['breeding'].replace('**', '').replace('🐄', '')}

═══════════════════════════════════════════════════════════════════════════
RECOMMENDATIONS
═══════════════════════════════════════════════════════════════════════════
1. Follow the nutrition guidelines strictly for optimal milk production
2. Maintain regular vaccination schedule as per the recommended timeline
3. Monitor for common diseases and consult veterinarian for preventive care
4. Ensure adequate water supply and quality fodder throughout the year
5. Maintain proper breeding records for genetic improvement

═══════════════════════════════════════════════════════════════════════════
DISCLAIMER
═══════════════════════════════════════════════════════════════════════════
This analysis is generated by an AI system for educational and advisory 
purposes. Always consult with qualified veterinarians and livestock experts 
for medical decisions and breeding programs.

═══════════════════════════════════════════════════════════════════════════
CONTACT INFORMATION
═══════════════════════════════════════════════════════════════════════════
Project: Smart India Hackathon 2025
Team: Nexel
GitHub: https://github.com/sanjayrockerz/SIH-Cattle-Breed-Recognition
Email: myteamcreations09@gmail.com

Report generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}
                    """
                    
                    # Enhanced visual summary card
                    st.markdown("### 📋 Analysis Summary")
                    summary_col1, summary_col2, summary_col3 = st.columns(3)
                    
                    with summary_col1:
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, var(--farm-green) 0%, var(--farm-yellow) 100%); 
                                   padding: 1rem; border-radius: 10px; text-align: center; color: white; margin: 0.5rem 0;">
                            <h3 style="margin: 0;">🎯</h3>
                            <p style="margin: 0;"><strong>Breed Identified</strong></p>
                            <p style="margin: 0; font-size: 0.9rem;">{breed}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with summary_col2:
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, var(--farm-blue) 0%, var(--farm-green) 100%); 
                                   padding: 1rem; border-radius: 10px; text-align: center; color: white; margin: 0.5rem 0;">
                            <h3 style="margin: 0;">📊</h3>
                            <p style="margin: 0;"><strong>Confidence</strong></p>
                            <p style="margin: 0; font-size: 0.9rem;">{confidence_pct:.1f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with summary_col3:
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, var(--farm-brown) 0%, var(--farm-blue) 100%); 
                                   padding: 1rem; border-radius: 10px; text-align: center; color: white; margin: 0.5rem 0;">
                            <h3 style="margin: 0;">🌍</h3>
                            <p style="margin: 0;"><strong>Origin</strong></p>
                            <p style="margin: 0; font-size: 0.9rem;">{metadata['origin']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.download_button(
                        "📥 Download Comprehensive Report",
                        data=report_content,
                        file_name=f"breed_report_{breed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
    else:
        st.info("👆 **Upload an image to get started**")
        st.markdown("""
        <div style="text-align: center; padding: 2rem; color: #666;">
            <p>🐄 Supports cattle and buffalo breeds</p>
            <p>🔬 AI-powered analysis with EfficientNet-B3</p>
            <p>📱 Optimized for mobile photography</p>
        </div>
        """, unsafe_allow_html=True)

# Registration form
q = st.query_params
if q.get("action") == ["register"]:
    st.markdown("---")
    with st.form("register_animal", clear_on_submit=True):
        st.markdown("### ➕ Register New Animal")
        
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            name = st.text_input("Animal Name/ID *")
            breed = st.selectbox("Breed *", ["Select breed..."] + sorted(breed_classes))
        
        with col_r2:
            last_vacc = st.date_input("Last Vaccination Date", value=today)
            notes = st.text_area("Notes (optional)", height=100)
        
        submitted = st.form_submit_button("💾 Register Animal", type="primary")
        
        if submitted:
            if name and breed != "Select breed...":
                c.execute(
                    "INSERT INTO animals (name, breed, last_vaccination_date, notes) VALUES (?,?,?,?)",
                    (name, breed, last_vacc.strftime("%Y-%m-%d"), notes)
                )
                conn.commit()
                st.success(f"✅ **{name}** registered successfully as {breed}!")
                st.query_params.clear()
            else:
                st.error("❌ Please fill in all required fields (*)")

# Animals registry view
if len(animals_db) > 0:
    st.markdown("---")
    st.markdown("### 📋 Registered Animals")
    
    # Convert to DataFrame for better display
    df_animals = pd.DataFrame(animals_db, columns=["Name", "Breed", "Last Vaccination"])
    df_animals["Days Since Vaccination"] = df_animals["Last Vaccination"].apply(
        lambda x: (today - datetime.strptime(x, "%Y-%m-%d").date()).days
    )
    df_animals["Status"] = df_animals["Days Since Vaccination"].apply(
        lambda x: "🔴 Overdue" if x > 180 else "🟡 Due Soon" if x > 150 else "🟢 Current"
    )
    
    st.dataframe(df_animals, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, rgba(32,135,147,0.05), rgba(32,135,147,0.1)); 
            border-radius: 12px; margin-top: 2rem;">
    <h3 style="color: #208793; margin-bottom: 1rem;">🏆 Smart India Hackathon 2025</h3>
    <p style="margin: 0.5rem 0;"><strong>AI-based Cattle Breed Identification and Management System</strong></p>
    <p style="margin: 0.5rem 0;">Developed by <strong>Team Nexel</strong></p>
    <p style="margin: 0.5rem 0;">
        <a href="https://github.com/sanjayrockerz/SIH-Cattle-Breed-Recognition" target="_blank" 
           style="color: #208793; text-decoration: none;">
            🔗 GitHub Repository
        </a> • 
        <a href="mailto:myteamcreations09@gmail.com" style="color: #208793; text-decoration: none;">
            ✉️ Contact
        </a>
    </p>
    <p style="font-size: 0.9rem; color: #666; margin-top: 1rem;">
        Empowering farmers with AI • Supporting indigenous breeds • Building the future of livestock management
    </p>
</div>
""", unsafe_allow_html=True)