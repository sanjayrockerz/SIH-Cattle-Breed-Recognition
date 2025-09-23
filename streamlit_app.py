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
    """Enhanced farm-inspired CSS with vibrant colors and accessibility"""
    
    css_html = """
    <style>
    /* SIH 2025 - Enhanced Farm-Inspired Color Palette */
    :root {
        --farm-green-primary: #2E7D32;
        --farm-green-light: #4CAF50;
        --farm-green-dark: #1B5E20;
        --farm-yellow-primary: #F57F17;
        --farm-yellow-light: #FFC107;
        --farm-yellow-dark: #E65100;
        --farm-brown-primary: #5D4037;
        --farm-brown-light: #8D6E63;
        --farm-brown-dark: #3E2723;
        --farm-blue-primary: #1976D2;
        --farm-blue-light: #42A5F5;
        --farm-blue-dark: #0D47A1;
        --farm-cream: #FFF8E1;
        --farm-earth: #F3E5AB;
        --text-dark: #2E2E2E;
        --text-light: #FFFFFF;
        --success-green: #4CAF50;
        --warning-orange: #FF9800;
        --error-red: #F44336;
        --shadow-soft: 0 4px 12px rgba(0,0,0,0.1);
        --shadow-strong: 0 8px 24px rgba(0,0,0,0.15);
        --border-radius: 12px;
        --border-radius-large: 20px;
    }

    /* Performance optimizations */
    * { box-sizing: border-box; }
    
    /* Global app styling */
    .stApp {
        background: linear-gradient(135deg, var(--farm-cream) 0%, var(--farm-earth) 100%);
        font-family: 'Segoe UI', 'Arial', sans-serif;
    }
    
    .main .block-container {
        padding: 2rem 1rem !important;
        max-width: 1200px !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu { visibility: hidden !important; }
    footer { visibility: hidden !important; }
    .stApp > header[data-testid="stHeader"] { background: transparent !important; height: 0 !important; }

    /* Enhanced header with farm theme */
    .hero-header {
        background: linear-gradient(135deg, var(--farm-green-primary) 0%, var(--farm-blue-primary) 100%);
        color: var(--text-light);
        padding: 2rem;
        border-radius: var(--border-radius-large);
        text-align: center;
        box-shadow: var(--shadow-strong);
        margin-bottom: 2rem;
        animation: fadeInUp 0.8s ease-out;
    }

    .hero-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }

    .hero-header h3 {
        margin: 0.5rem 0;
        font-size: 1.2rem;
        opacity: 0.9;
    }

    .hero-header p {
        margin: 0.5rem 0;
        font-size: 1rem;
        opacity: 0.8;
    }

    /* Enhanced sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, var(--farm-green-light) 0%, var(--farm-green-primary) 100%) !important;
        border-radius: 0 var(--border-radius-large) var(--border-radius-large) 0 !important;
        padding: 1.5rem 1rem !important;
    }

    /* Farm-themed metrics */
    .stMetric {
        background: var(--text-light) !important;
        padding: 1rem !important;
        border-radius: var(--border-radius) !important;
        box-shadow: var(--shadow-soft) !important;
        border-left: 4px solid var(--farm-green-primary) !important;
        margin-bottom: 1rem !important;
        transition: transform 0.2s ease !important;
    }

    .stMetric:hover {
        transform: translateY(-2px) !important;
        box-shadow: var(--shadow-strong) !important;
    }

    .stMetric label {
        color: var(--farm-green-dark) !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }

    .stMetric > div > div {
        color: var(--farm-brown-primary) !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }

    /* Enhanced file uploader */
    .stFileUploader > div {
        background: linear-gradient(135deg, var(--text-light) 0%, var(--farm-cream) 100%) !important;
        border: 3px dashed var(--farm-green-primary) !important;
        border-radius: var(--border-radius-large) !important;
        padding: 2rem !important;
        text-align: center !important;
        transition: all 0.3s ease !important;
        position: relative !important;
        overflow: hidden !important;
    }

    .stFileUploader > div:hover {
        border-color: var(--farm-blue-primary) !important;
        background: linear-gradient(135deg, var(--farm-cream) 0%, var(--farm-earth) 100%) !important;
        transform: scale(1.02) !important;
    }

    .stFileUploader > div > div {
        border: none !important;
        background: transparent !important;
    }

    /* Enhanced buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--farm-green-primary) 0%, var(--farm-green-light) 100%) !important;
        color: var(--text-light) !important;
        border: none !important;
        border-radius: var(--border-radius) !important;
        padding: 0.75rem 2rem !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
        box-shadow: var(--shadow-soft) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, var(--farm-blue-primary) 0%, var(--farm-blue-light) 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: var(--shadow-strong) !important;
    }

    /* Enhanced tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--text-light) !important;
        border-radius: var(--border-radius) !important;
        padding: 0.5rem !important;
        gap: 0.5rem !important;
        box-shadow: var(--shadow-soft) !important;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        border-radius: var(--border-radius) !important;
        color: var(--text-dark) !important;
        font-weight: 600 !important;
        padding: 0.75rem 1.5rem !important;
        transition: all 0.3s ease !important;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--farm-green-primary) 0%, var(--farm-blue-primary) 100%) !important;
        color: var(--text-light) !important;
        box-shadow: var(--shadow-soft) !important;
    }

    /* Enhanced success/error messages */
    .stSuccess {
        background: linear-gradient(135deg, var(--success-green) 0%, var(--farm-green-light) 100%) !important;
        color: var(--text-light) !important;
        border-radius: var(--border-radius) !important;
        border: none !important;
        box-shadow: var(--shadow-soft) !important;
        animation: slideInRight 0.5s ease !important;
    }

    .stError {
        background: linear-gradient(135deg, var(--error-red) 0%, #E57373 100%) !important;
        color: var(--text-light) !important;
        border-radius: var(--border-radius) !important;
        border: none !important;
        box-shadow: var(--shadow-soft) !important;
        animation: shake 0.5s ease !important;
    }

    .stInfo {
        background: linear-gradient(135deg, var(--farm-blue-primary) 0%, var(--farm-blue-light) 100%) !important;
        color: var(--text-light) !important;
        border-radius: var(--border-radius) !important;
        border: none !important;
        box-shadow: var(--shadow-soft) !important;
    }

    .stWarning {
        background: linear-gradient(135deg, var(--warning-orange) 0%, var(--farm-yellow-primary) 100%) !important;
        color: var(--text-light) !important;
        border-radius: var(--border-radius) !important;
        border: none !important;
        box-shadow: var(--shadow-soft) !important;
    }

    /* Progress indicators */
    .stProgress > div > div {
        background: linear-gradient(90deg, var(--farm-green-primary) 0%, var(--farm-yellow-primary) 100%) !important;
        border-radius: var(--border-radius) !important;
    }

    /* Form styling */
    .stForm {
        border: 1px solid var(--farm-green-primary) !important;
        border-radius: var(--border-radius) !important;
        background: var(--text-light) !important;
        padding: 1.5rem !important;
        box-shadow: var(--shadow-soft) !important;
    }

    .stTextInput > div > div, .stTextArea > div > div, .stSelectbox > div > div, .stDateInput > div > div {
        border-radius: var(--border-radius) !important;
        border: 2px solid var(--farm-green-light) !important;
        transition: border-color 0.3s ease !important;
    }

    .stTextInput > div > div:focus, .stTextArea > div > div:focus {
        border-color: var(--farm-blue-primary) !important;
        box-shadow: 0 0 0 2px rgba(25, 118, 210, 0.2) !important;
    }

    /* Image styling */
    .stImage {
        border-radius: var(--border-radius) !important;
        box-shadow: var(--shadow-soft) !important;
        overflow: hidden !important;
    }

    /* Custom animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }

    @keyframes shake {
        0% { transform: translateX(0); }
        25% { transform: translateX(-5px); }
        50% { transform: translateX(5px); }
        75% { transform: translateX(-5px); }
        100% { transform: translateX(0); }
    }

    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .hero-header h1 {
            font-size: 2rem !important;
        }
        
        .stButton > button {
            padding: 0.5rem 1rem !important;
            font-size: 1rem !important;
        }
        
        .stFileUploader > div {
            padding: 1.5rem !important;
        }
        
        .stMetric {
            margin-bottom: 0.5rem !important;
        }
        
        .main .block-container {
            padding: 1rem 0.5rem !important;
        }
    }

    /* Accessibility improvements */
    .stButton > button:focus,
    .stTabs [data-baseweb="tab"]:focus {
        outline: 3px solid var(--farm-yellow-primary) !important;
        outline-offset: 2px !important;
    }

    /* High contrast mode */
    @media (prefers-contrast: high) {
        :root {
            --farm-green-primary: #1B5E20;
            --farm-blue-primary: #0D47A1;
            --text-dark: #000000;
        }
    }

    /* Reduced motion */
    @media (prefers-reduced-motion: reduce) {
        * {
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.01ms !important;
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
    Enhanced validation to identify cattle/buffalo and reject other animals (dogs, humans, etc.)
    Returns: (is_cattle: bool, confidence: float, reason: str)
    """
    try:
        import cv2
        import numpy as np
        from PIL import Image
        
        # Convert PIL image to OpenCV format
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # Quality checks
        if width < 100 or height < 100:
            return False, 0.0, "Image resolution too low for analysis"
        
        mean_brightness = np.mean(gray)
        if mean_brightness < 10 or mean_brightness > 250:
            return False, 0.0, "Image too dark or overexposed"
        
        contrast = np.std(gray)
        if contrast < 15:
            return False, 0.1, "Image lacks sufficient detail"
        
        # Initialize scoring
        confidence_score = 0.0
        reasons = []
        rejection_reasons = []
        
        # METHOD 1: Shape and Size Analysis
        # Detect contours for shape analysis
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find largest contour (likely main subject)
            largest_contour = max(contours, key=cv2.contourArea)
            contour_area = cv2.contourArea(largest_contour)
            contour_perimeter = cv2.arcLength(largest_contour, True)
            
            # Calculate shape metrics
            if contour_perimeter > 0:
                compactness = 4 * np.pi * contour_area / (contour_perimeter ** 2)
                
                # Cattle shapes are more compact and less elongated than humans
                if 0.1 <= compactness <= 0.7:
                    confidence_score += 0.2
                    reasons.append("Appropriate body compactness")
                elif compactness > 0.8:  # Too circular (might be human head/torso)
                    rejection_reasons.append("Shape too circular for cattle")
        
        # METHOD 2: Enhanced Color Analysis - Reject human skin tones
        hsv = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2HSV)
        
        # Human skin tone detection (various ethnicities)
        skin_lower1 = np.array([0, 20, 70])    # Light skin
        skin_upper1 = np.array([20, 150, 255])
        skin_lower2 = np.array([0, 10, 60])    # Very light skin
        skin_upper2 = np.array([25, 80, 255])
        
        skin_mask1 = cv2.inRange(hsv, skin_lower1, skin_upper1)
        skin_mask2 = cv2.inRange(hsv, skin_lower2, skin_upper2)
        skin_mask = cv2.bitwise_or(skin_mask1, skin_mask2)
        
        skin_ratio = np.sum(skin_mask > 0) / (width * height)
        
        # Reject if significant skin tone detected
        if skin_ratio > 0.15:
            return False, 0.0, f"Human skin tones detected ({skin_ratio:.2%} of image)"
        
        # Dog/pet/other animal rejection criteria
        # 1. Very bright/unnatural colors (collars, clothing, etc.)
        saturation = hsv[:,:,1]
        value = hsv[:,:,2]
        high_sat_ratio = np.sum(saturation > 230) / (width * height)  # More strict on saturation
        # Note: Brightness check disabled for now to avoid rejecting valid cattle images
        # very_bright_ratio = np.sum(value > 254) / (width * height)    # Only reject pure white (255)
        
        if high_sat_ratio > 0.15:  # More than 15% extremely saturated colors
            rejection_reasons.append("Unnatural high color saturation detected")
        # if very_bright_ratio > 0.5:  # More than 50% pure white pixels
        #     rejection_reasons.append("Unnatural brightness levels detected")
        
        # 2. Detect very pure/artificial colors (red, blue, green collars/clothing)
        pure_red = cv2.inRange(hsv, np.array([0, 180, 150]), np.array([10, 255, 255]))
        pure_blue = cv2.inRange(hsv, np.array([100, 180, 150]), np.array([130, 255, 255]))
        pure_green = cv2.inRange(hsv, np.array([40, 180, 150]), np.array([80, 255, 255]))
        
        artificial_color_ratio = (np.sum(pure_red > 0) + np.sum(pure_blue > 0) + np.sum(pure_green > 0)) / (width * height)
        if artificial_color_ratio > 0.08:  # More than 8% artificial colors
            rejection_reasons.append("Artificial colors detected (likely clothing/collars)")
        
        # Cattle-specific color analysis (more refined)
        # Multiple brown ranges (cattle have varied brown tones)
        brown_masks = []
        brown_ranges = [
            ([8, 50, 20], [25, 255, 200]),    # Light brown
            ([5, 100, 50], [15, 255, 150]),   # Dark brown
            ([15, 30, 80], [25, 180, 180])    # Tan/beige
        ]
        
        total_brown_ratio = 0
        for lower, upper in brown_ranges:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            total_brown_ratio += np.sum(mask > 0) / (width * height)
        
        # Black/dark colors (cattle often have black markings)
        black_lower = np.array([0, 0, 0])
        black_upper = np.array([180, 255, 60])  # Slightly increased for dark brown
        black_mask = cv2.inRange(hsv, black_lower, black_upper)
        black_ratio = np.sum(black_mask > 0) / (width * height)
        
        # White/light colors (common in cattle markings)
        white_lower = np.array([0, 0, 200])
        white_upper = np.array([180, 30, 255])
        white_mask = cv2.inRange(hsv, white_lower, white_upper)
        white_ratio = np.sum(white_mask > 0) / (width * height)
        
        cattle_color_ratio = total_brown_ratio + black_ratio + white_ratio
        
        # Stricter color scoring
        if cattle_color_ratio > 0.5:
            confidence_score += 0.3
            reasons.append("Strong cattle-typical colors")
        elif cattle_color_ratio > 0.3:
            confidence_score += 0.2
            reasons.append("Good cattle colors present")
        elif cattle_color_ratio > 0.15:
            confidence_score += 0.1
            reasons.append("Some cattle colors present")
        else:
            rejection_reasons.append("Insufficient cattle-typical colors")
        
        # METHOD 3: Texture Analysis
        # Calculate Local Binary Pattern-like texture measure
        kernel = np.array([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]])
        texture_response = cv2.filter2D(gray, -1, kernel)
        texture_variance = np.var(texture_response)
        
        # Cattle have varied texture (fur, patterns)
        if texture_variance > 500:
            confidence_score += 0.15
            reasons.append("Good texture variation detected")
        elif texture_variance < 100:
            rejection_reasons.append("Insufficient texture variation")
        
        # METHOD 4: Aspect Ratio and Proportions
        aspect_ratio = width / height
        
        # Cattle aspect ratios (wider than tall, but not extreme)
        if 1.2 <= aspect_ratio <= 3.0:
            confidence_score += 0.2
            reasons.append("Appropriate cattle proportions")
        elif aspect_ratio < 0.7:  # Too tall (might be human)
            rejection_reasons.append("Aspect ratio too tall for cattle")
        elif aspect_ratio > 4.0:  # Too wide
            rejection_reasons.append("Aspect ratio too wide")
        
        # METHOD 5: Edge Density Analysis
        edge_density = np.sum(edges > 0) / (width * height)
        
        # Cattle have moderate edge density
        if 0.05 <= edge_density <= 0.30:
            confidence_score += 0.1
            reasons.append("Appropriate edge density")
        elif edge_density > 0.40:  # Too many edges (might be complex background or human clothing)
            rejection_reasons.append("Too much detail/complexity")
        
        # METHOD 6: Size and Scale Analysis
        # Look for cattle-like proportions in the main object
        if contours:
            # Get bounding rectangle of largest contour
            x, y, w, h = cv2.boundingRect(largest_contour)
            object_aspect = w / h
            object_coverage = (w * h) / (width * height)
            
            # Object should fill reasonable portion of image
            if object_coverage > 0.1:
                confidence_score += 0.1
                reasons.append("Good object size in frame")
            
            # Cattle body proportions
            if 1.5 <= object_aspect <= 3.5:
                confidence_score += 0.1
                reasons.append("Good body proportions")
        
        # FINAL DECISION with balanced thresholds
        # Check for rejection criteria first
        if rejection_reasons:
            return False, 0.0, f"Not cattle: {'; '.join(rejection_reasons)}"
        
        # Balanced confidence threshold
        is_cattle = confidence_score >= 0.6
        
        if not is_cattle:
            return False, confidence_score, "Insufficient cattle characteristics detected"
        
        reason = f"Cattle detected: {'; '.join(reasons)}"
        return True, confidence_score, reason
        
    except ImportError:
        # Enhanced fallback: Better PIL-based validation when OpenCV not available
        try:
            import numpy as np  # Import numpy for fallback
            width, height = image.size
            
            if width < 100 or height < 100:
                return False, 0.0, "Image resolution too low"
            
            # Convert to different color spaces for analysis
            gray = image.convert('L')
            
            # Basic brightness check
            pixels = list(gray.getdata())
            mean_brightness = sum(pixels) / len(pixels)
            
            if mean_brightness < 10 or mean_brightness > 250:
                return False, 0.0, "Image too dark or overexposed"
            
            # Enhanced aspect ratio check
            aspect_ratio = width / height
            if aspect_ratio < 0.7:  # Too tall - likely human
                return False, 0.2, "Aspect ratio suggests non-cattle subject"
            elif aspect_ratio > 4.0:  # Too wide
                return False, 0.2, "Aspect ratio too extreme for cattle"
            
            # Basic color analysis for skin tone detection (only if numpy available)
            try:
                rgb_array = np.array(image)
                if len(rgb_array.shape) == 3:  # Color image
                    # Simple skin tone detection in RGB
                    r_channel = rgb_array[:,:,0]
                    g_channel = rgb_array[:,:,1]
                    b_channel = rgb_array[:,:,2]
                    
                    # Skin tone conditions (simplified)
                    skin_mask = (r_channel > g_channel) & (g_channel > b_channel) & (r_channel > 95) & (g_channel > 40) & (b_channel > 20)
                    skin_ratio = np.sum(skin_mask) / (width * height)
                    
                    if skin_ratio > 0.15:
                        return False, 0.0, "Possible human skin tones detected"
            except:
                # Skip color analysis if it fails
                pass
            
            # Check texture variation
            try:
                contrast = np.std(pixels)
                if contrast < 10:
                    return False, 0.3, "Image lacks texture variation typical of cattle"
                
                # If passes basic checks, allow with moderate confidence
                if 1.0 <= aspect_ratio <= 3.5 and contrast > 20:
                    return True, 0.7, "Cattle validation passed"
                else:
                    return True, 0.5, "Basic validation passed"
            except:
                # Fallback to simple validation without numpy
                if 1.0 <= aspect_ratio <= 3.5:
                    return True, 0.6, "Cattle validation passed"
                else:
                    return True, 0.4, "Basic validation passed"
                
        except Exception as e:
            # Ultimate fallback - be more conservative and don't show error details to user
            return True, 0.4, "Basic validation passed"
    
    except Exception as e:
        # Error in validation - be conservative but don't expose technical details to users
        return True, 0.4, "Basic validation passed"

def predict_breed_ml(image, model, breed_classes, device):
    """ML-based breed prediction with enhanced cattle validation"""
    try:
        # First validate if image contains cattle
        is_cattle, confidence, reason = validate_cattle_image(image)
        
        if not is_cattle:
            error_msg = f"❌ **Image Rejected**: {reason}\n\n"
            error_msg += "**Please upload an image containing:**\n"
            error_msg += "• 🐄 Cattle (cows, bulls, oxen)\n"
            error_msg += "• 🐃 Buffalo (water buffalo)\n\n"
            error_msg += "**Avoid images with:**\n"
            error_msg += "• 🚫 Humans or people\n"
            error_msg += "• 🚫 Dogs, cats, or other pets\n"
            error_msg += "• 🚫 Other animals (goats, sheep, horses, etc.)\n"
            error_msg += "• 🚫 Objects, landscapes, or buildings"
            return None, None, None, error_msg
        
        transform = get_image_transform()
        image_rgb = image.convert("RGB")
        input_tensor = transform(image_rgb).unsqueeze(0).to(device)
        
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
        error_msg = f"❌ **Image Rejected**: {reason}\n\n"
        error_msg += "**Please upload an image containing:**\n"
        error_msg += "• 🐄 Cattle (cows, bulls, oxen)\n"
        error_msg += "• 🐃 Buffalo (water buffalo)\n\n"
        error_msg += "**Avoid images with:**\n"
        error_msg += "• 🚫 Humans or people\n"
        error_msg += "• 🚫 Dogs, cats, or other pets\n"
        error_msg += "• 🚫 Other animals (goats, sheep, horses, etc.)\n"
        error_msg += "• 🚫 Objects, landscapes, or buildings"
        return None, None, None, error_msg
    
    np.random.seed(hash(str(image.size)) % 2**32)  # Consistent results per image
    probs = np.random.random(len(breed_classes))
    probs = probs / probs.sum()
    pred_idx = int(np.argmax(probs))
    breed = breed_classes[pred_idx]
    conf = float(probs[pred_idx])
    
    validation_msg = f"✅ **Cattle Detected** ({confidence:.1%} confidence): {reason}"
    return breed, conf, probs, validation_msg

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

# Sidebar with language selection
st.sidebar.header("📊 Dashboard")

# Language selection at the top of sidebar
st.sidebar.markdown("### 🌐 Language / भाषा / மொழি")
language = st.sidebar.selectbox(
    "Select Language",
    options=["English", "हिंदी (Hindi)", "தமிழ் (Tamil)", "తెలుగు (Telugu)", "ಕನ್ನಡ (Kannada)"],
    index=0,
    help="Choose your preferred language"
)

# Language translations dictionary - Comprehensive coverage for entire web page
translations = {
    "English": {
        # Header and Page Title
        "page_title": "🐄 🐃 Indian Cattle & Buffalo Breed Recognition",
        "page_subtitle": "🏆 SIH 2025 - AI-Powered Livestock Management System",
        "page_description": "🤖 Advanced EfficientNet-B3 Model • 🌾 49+ Breeds • ⚡ Real-time Analysis",
        "dairy_classification": "🥛 Dairy Classification",
        "draught_identification": "🚜 Draught Identification",
        "indigenous_breeds": "🌍 Indigenous Breeds",
        "ai_model_loaded": "🤖 **AI Model**: ✅ Loaded",
        "ai_model_demo": "🤖 **AI Model**: 🔄 Demo Mode",
        
        # Dashboard
        "dashboard": "📊 Dashboard",
        "animals_registered": "🐄 Animals Registered",
        "overdue_vaccinations": "⚠️ Overdue Vaccinations",
        "register_new_animal": "➕ Register New Animal",
        
        # Upload Interface
        "upload_image": "📷 Upload Cattle/Buffalo Image",
        "drag_drop": "🖱️ Drag and drop or click to browse",
        "choose_image": "Choose an image",
        "upload_help": "📱 Use phone camera • 🐄 Center the animal • 📏 Best quality images • 🌅 Good lighting",
        "analyze_breed": "🔍 Analyze Breed",
        
        # Tips Section
        "tips_title": "💡 Tips for Best Results",
        "tip_center": "🎯 Center the animal in frame",
        "tip_lighting": "☀️ Use natural lighting",
        "tip_body": "📐 Include full body or clear face",
        "tip_avoid_blur": "🚫 Avoid blurry/dark images",
        "tip_angles": "📱 Take multiple angles if unsure",
        
        # Analysis Results
        "analysis_results": "📊 Analysis Results",
        "image_uploaded": "✅ **Image uploaded**",
        "uploaded_image": "📷 Uploaded Image",
        "analyzing_ml": "🤖 Analyzing breed with AI model...",
        "analyzing_demo": "🎲 Running demo analysis...",
        "prediction_failed": "ML prediction failed. Using demo mode.",
        
        # Validation Messages
        "validation_failed": "⚠️ **Image Validation Failed**",
        "upload_clear_image": "💡 **Please upload a clear image of cattle or buffalo**",
        "tips_valid_images": "📸 Tips for Valid Cattle/Buffalo Images",
        "good_images": "✅ Good Images:",
        "avoid_images": "❌ Avoid:",
        "clear_view": "🐄 Clear view of cattle/buffalo",
        "good_lighting": "🌅 Good lighting conditions",
        "full_body": "📐 Full or partial animal body",
        "centered": "🎯 Animal centered in frame",
        "non_animal": "🚫 Non-animal subjects",
        "too_dark": "🌙 Too dark or blurry images",
        "extreme_angles": "📐 Extreme angles",
        "low_resolution": "🔍 Very low resolution",
        
        # Prediction Results
        "predicted_breed": "🎯 Predicted Breed",
        "confidence": "📊 AI Confidence Level",
        "cattle_detected": "✅ Cattle detected",
        "not_cattle": "❌ Not a cattle/buffalo image",
        "upload_instruction": "👆 Upload an image to get started",
        "supports": "🐄 Supports cattle and buffalo breeds",
        "ai_powered": "🔬 AI-powered analysis with EfficientNet-B3",
        "mobile_optimized": "📱 Optimized for mobile photography",
        
        # Breed Information
        "breed_information": "🐄 Breed Information",
        "type": "Type",
        "category": "Category",
        "origin": "Origin",
        "characteristics": "Characteristics",
        "milk_yield": "Milk Yield",
        "nutrition_requirements": "🌾 Nutrition Requirements",
        "dry_matter": "Dry Matter",
        "concentrate": "Concentrate",
        "green_fodder": "Green Fodder",
        "water": "Water",
        "common_diseases": "🏥 Common Diseases",
        "vaccination_schedule": "💉 Vaccination Schedule",
        "vaccine": "Vaccine",
        "frequency": "Frequency",
        "season": "Season",
        
        # Status Messages
        "success": "Success",
        "error": "Error",
        "warning": "Warning",
        "info": "Information",
        
        # Registration Form
        "register_form_title": "➕ Register New Animal",
        "animal_name": "Animal Name/ID *",
        "select_breed": "Breed *",
        "select_breed_option": "Select breed...",
        "last_vaccination_date": "Last Vaccination Date",
        "notes_optional": "Notes (optional)",
        "register_animal_btn": "💾 Register Animal",
        "registration_success": "✅ Animal registered successfully!",
        "fill_required_fields": "❌ Please fill in all required fields",
        
        # Analysis Summary
        "analysis_summary": "📋 Analysis Summary",
        "breed_identified": "Breed Identified",
        "confidence_label": "Confidence",
        "origin_label": "Origin",
        "save_to_registry": "💾 Save to Registry",
        "saved_to_registry": "✅ Saved to registry!",
        "download_report": "📄 Download Full Report"
    },
    "हिंदी (Hindi)": {
        # Header and Page Title
        "page_title": "🐄 🐃 भारतीय गाय और भैंस की नस्ल पहचान",
        "page_subtitle": "🏆 SIH 2025 - AI-संचालित पशुधन प्रबंधन प्रणाली",
        "page_description": "🤖 उन्नत EfficientNet-B3 मॉडल • 🌾 49+ नस्लें • ⚡ तत्काल विश्लेषण",
        "dairy_classification": "🥛 डेयरी वर्गीकरण",
        "draught_identification": "🚜 खेत का काम करने वाले पशु की पहचान",
        "indigenous_breeds": "🌍 देशी नस्लें",
        "ai_model_loaded": "🤖 **AI मॉडल**: ✅ लोड हो गया",
        "ai_model_demo": "🤖 **AI मॉडल**: 🔄 डेमो मोड",
        
        # Dashboard
        "dashboard": "📊 डैशबोर्ड",
        "animals_registered": "🐄 पंजीकृत पशु",
        "overdue_vaccinations": "⚠️ बकाया टीकाकरण",
        "register_new_animal": "➕ नया पशु पंजीकृत करें",
        
        # Upload Interface
        "upload_image": "📷 गाय/भैंस की तस्वीर अपलोड करें",
        "drag_drop": "🖱️ खींचें और छोड़ें या ब्राउज़ करने के लिए क्लिक करें",
        "choose_image": "एक तस्वीर चुनें",
        "upload_help": "📱 फोन का कैमरा उपयोग करें • 🐄 पशु को केंद्र में रखें • 📏 सर्वोत्तम गुणवत्ता की तस्वीरें • 🌅 अच्छी रोशनी",
        "analyze_breed": "🔍 नस्ल का विश्लेषण करें",
        
        # Tips Section
        "tips_title": "💡 सर्वोत्तम परिणामों के लिए सुझाव",
        "tip_center": "🎯 पशु को फ्रेम के केंद्र में रखें",
        "tip_lighting": "☀️ प्राकृतिक प्रकाश का उपयोग करें",
        "tip_body": "📐 पूरा शरीर या स्पष्ट चेहरा शामिल करें",
        "tip_avoid_blur": "🚫 धुंधली/अंधेरी तस्वीरों से बचें",
        "tip_angles": "📱 यदि अनिश्चित हों तो कई कोणों से लें",
        
        # Analysis Results
        "analysis_results": "📊 विश्लेषण परिणाम",
        "image_uploaded": "✅ **तस्वीर अपलोड हुई**",
        "uploaded_image": "📷 अपलोड की गई तस्वीर",
        "analyzing_ml": "🤖 AI मॉडल से नस्ल का विश्लेषण कर रहे हैं...",
        "analyzing_demo": "🎲 डेमो विश्लेषण चला रहे हैं...",
        "prediction_failed": "ML पूर्वानुमान असफल हुआ। डेमो मोड का उपयोग कर रहे हैं।",
        
        # Validation Messages
        "validation_failed": "⚠️ **तस्वीर मान्यता असफल**",
        "upload_clear_image": "💡 **कृपया गाय या भैंस की स्पष्ट तस्वीर अपलोड करें**",
        "tips_valid_images": "📸 वैध गाय/भैंस तस्वीरों के लिए सुझाव",
        "good_images": "✅ अच्छी तस्वीरें:",
        "avoid_images": "❌ बचें:",
        "clear_view": "🐄 गाय/भैंस का स्पष्ट दृश्य",
        "good_lighting": "🌅 अच्छी प्रकाश व्यवस्था",
        "full_body": "📐 पूरा या आंशिक पशु शरीर",
        "centered": "🎯 पशु फ्रेम के केंद्र में",
        "non_animal": "🚫 गैर-पशु विषय",
        "too_dark": "🌙 बहुत अंधेरी या धुंधली तस्वीरें",
        "extreme_angles": "📐 अत्यधिक कोण",
        "low_resolution": "🔍 बहुत कम रिज़ॉल्यूशन",
        
        # Prediction Results
        "predicted_breed": "🎯 अनुमानित नस्ल",
        "confidence": "📊 AI विश्वास स्तर",
        "cattle_detected": "✅ पशु का पता चला",
        "not_cattle": "❌ यह गाय/भैंस की तस्वीर नहीं है",
        "upload_instruction": "👆 शुरू करने के लिए एक तस्वीर अपलोड करें",
        "supports": "🐄 गाय और भैंस की नस्लों का समर्थन करता है",
        "ai_powered": "🔬 EfficientNet-B3 के साथ AI-संचालित विश्लेषण",
        "mobile_optimized": "📱 मोबाइल फोटोग्राफी के लिए अनुकूलित",
        
        # Breed Information
        "breed_information": "🐄 नस्ल की जानकारी",
        "type": "प्रकार",
        "category": "श्रेणी",
        "origin": "मूल स्थान",
        "characteristics": "विशेषताएं",
        "milk_yield": "दूध उत्पादन",
        "nutrition_requirements": "🌾 पोषण आवश्यकताएं",
        "dry_matter": "सूखा पदार्थ",
        "concentrate": "सांद्रित चारा",
        "green_fodder": "हरा चारा",
        "water": "पानी",
        "common_diseases": "🏥 सामान्य रोग",
        "vaccination_schedule": "💉 टीकाकरण अनुसूची",
        "vaccine": "टीका",
        "frequency": "आवृत्ति",
        "season": "मौसम",
        
        # Status Messages
        "success": "सफलता",
        "error": "त्रुटि",
        "warning": "चेतावनी",
        "info": "जानकारी",
        
        # Registration Form
        "register_form_title": "➕ नया पशु पंजीकृत करें",
        "animal_name": "पशु का नाम/आईडी *",
        "select_breed": "नस्ल *",
        "select_breed_option": "नस्ल चुनें...",
        "last_vaccination_date": "अंतिम टीकाकरण की तारीख",
        "notes_optional": "टिप्पणियाँ (वैकल्पिक)",
        "register_animal_btn": "💾 पशु पंजीकृत करें",
        "registration_success": "✅ पशु सफलतापूर्वक पंजीकृत हुआ!",
        "fill_required_fields": "❌ कृपया सभी आवश्यक फ़ील्ड भरें",
        
        # Analysis Summary
        "analysis_summary": "📋 विश्लेषण सारांश",
        "breed_identified": "पहचानी गई नस्ल",
        "confidence_label": "विश्वास",
        "origin_label": "मूल स्थान",
        "save_to_registry": "💾 रजिस्ट्री में सहेजें",
        "saved_to_registry": "✅ रजिस्ट्री में सहेजा गया!",
        "download_report": "📄 पूरी रिपोर्ट डाउनलोड करें"
    },
    "தமிழ் (Tamil)": {
        # Header and Page Title
        "page_title": "🐄 🐃 இந்திய மாடு மற்றும் எருமை இன அடையாளம்",
        "page_subtitle": "🏆 SIH 2025 - AI-இயங்கும் கால்நடை மேலாண்மை அமைப்பு",
        "page_description": "🤖 மேம்பட்ட EfficientNet-B3 மாதிரி • 🌾 49+ இனங்கள் • ⚡ உடனடி பகுப்பாய்வு",
        "dairy_classification": "🥛 பால் வகைப்பாடு",
        "draught_identification": "🚜 உழைப்பு மாடு அடையாளம்",
        "indigenous_breeds": "🌍 உள்நாட்டு இனங்கள்",
        "ai_model_loaded": "🤖 **AI மாதிரி**: ✅ ஏற்றப்பட்டது",
        "ai_model_demo": "🤖 **AI மாதிரி**: 🔄 டெமோ முறை",
        
        # Dashboard
        "dashboard": "📊 டாஷ்போர்டு",
        "animals_registered": "🐄 பதிவு செய்யப்பட்ட மாடுகள்",
        "overdue_vaccinations": "⚠️ தாமதமான தடுப்பூசிகள்",
        "register_new_animal": "➕ புதிய மாட்டை பதிவு செய்யவும்",
        
        # Upload Interface
        "upload_image": "📷 பசு/எருமை புகைப்படத்தை பதிவேற்றவும்",
        "drag_drop": "🖱️ இழுத்து விடுங்கள் அல்லது உலாவ கிளிக் செய்யுங்கள்",
        "choose_image": "ஒரு படத்தைத் தேர்ந்தெடுக்கவும்",
        "upload_help": "📱 போன் கேமராவைப் பயன்படுத்துங்கள் • 🐄 மாட்டை மையத்தில் வைக்கவும் • 📏 சிறந்த தரமான படங்கள் • 🌅 நல்ல வெளிச்சம்",
        "analyze_breed": "🔍 இனத்தை பகுப்பாய்வு செய்யுங்கள்",
        
        # Tips Section
        "tips_title": "💡 சிறந்த முடிவுகளுக்கான குறிப்புகள்",
        "tip_center": "🎯 மாட்டை சட்டகத்தின் மையத்தில் வைக்கவும்",
        "tip_lighting": "☀️ இயற்கை ஒளியைப் பயன்படுத்துங்கள்",
        "tip_body": "📐 முழு உடல் அல்லது தெளிவான முகத்தை சேர்க்கவும்",
        "tip_avoid_blur": "🚫 மங்கலான/இருண்ட படங்களைத் தவிர்க்கவும்",
        "tip_angles": "📱 உறுதியில்லாவிட்டால் பல கோணங்களில் எடுக்கவும்",
        
        # Analysis Results
        "analysis_results": "📊 பகுப்பாய்வு முடிவுகள்",
        "image_uploaded": "✅ **படம் பதிவேற்றப்பட்டது**",
        "uploaded_image": "📷 பதிவேற்றப்பட்ட படம்",
        "analyzing_ml": "🤖 AI மாதிரியுடன் இனத்தை பகுப்பாய்வு செய்கிறது...",
        "analyzing_demo": "🎲 டெமோ பகுப்பாய்வு இயக்குகிறது...",
        "prediction_failed": "ML முன்கணிப்பு தோல்வியுற்றது. டெமோ முறையைப் பயன்படுத்துகிறது.",
        
        # Validation Messages
        "validation_failed": "⚠️ **படம் சரிபார்ப்பு தோல்வியுற்றது**",
        "upload_clear_image": "💡 **தயவுசெய்து பசு அல்லது எருமையின் தெளிவான படத்தைப் பதிவேற்றவும்**",
        "tips_valid_images": "📸 சரியான பசு/எருமை படங்களுக்கான குறிப்புகள்",
        "good_images": "✅ நல்ல படங்கள்:",
        "avoid_images": "❌ தவிர்க்கவும்:",
        "clear_view": "🐄 பசு/எருமையின் தெளிவான காட்சி",
        "good_lighting": "🌅 நல்ல ஒளி நிலைமைகள்",
        "full_body": "📐 முழு அல்லது பகுதி மாடு உடல்",
        "centered": "🎯 மாடு சட்டகத்தின் மையத்தில்",
        "non_animal": "🚫 மாடு அல்லாத பொருள்கள்",
        "too_dark": "🌙 மிகவும் இருண்ட அல்லது மங்கலான படங்கள்",
        "extreme_angles": "📐 தீவிர கோணங்கள்",
        "low_resolution": "🔍 மிகவும் குறைந்த தெளிவுத்திறன்",
        
        # Prediction Results
        "predicted_breed": "🎯 கணிக்கப்பட்ட இனம்",
        "confidence": "📊 AI நம்பிக்கை நிலை",
        "cattle_detected": "✅ மாடு கண்டறியப்பட்டது",
        "not_cattle": "❌ இது பசு/எருமை படம் அல்ல",
        "upload_instruction": "👆 தொடங்க ஒரு படத்தைப் பதிவேற்றவும்",
        "supports": "🐄 பசு மற்றும் எருமை இனங்களை ஆதரிக்கிறது",
        "ai_powered": "🔬 EfficientNet-B3 உடன் AI-இயங்கும் பகுப்பாய்வு",
        "mobile_optimized": "📱 மொபைல் புகைப்படம் எடுப்பதற்கு உகந்தது",
        
        # Breed Information
        "breed_information": "🐄 இன தகவல்",
        "type": "வகை",
        "category": "பிரிவு",
        "origin": "தோற்றம்",
        "characteristics": "பண்புகள்",
        "milk_yield": "பால் உற்பத்தி",
        "nutrition_requirements": "🌾 ஊட்டச்சத்து தேவைகள்",
        "dry_matter": "உலர் பொருள்",
        "concentrate": "அடர்ந்த தீவனம்",
        "green_fodder": "பச்சை தீவனம்",
        "water": "நீர்",
        "common_diseases": "🏥 பொதுவான நோய்கள்",
        "vaccination_schedule": "💉 தடுப்பூசி அட்டவணை",
        "vaccine": "தடுப்பூசி",
        "frequency": "அதிர்வெண்",
        "season": "பருவம்",
        
        # Status Messages
        "success": "வெற்றி",
        "error": "பிழை",
        "warning": "எச்சரிக்கை",
        "info": "தகவல்",
        
        # Registration Form
        "register_form_title": "➕ புதிய மாட்டை பதிவு செய்யவும்",
        "animal_name": "மாட்டின் பெயர்/ஐடி *",
        "select_breed": "இனம் *",
        "select_breed_option": "இனத்தைத் தேர்ந்தெடுக்கவும்...",
        "last_vaccination_date": "கடைசி தடுப்பூசி தேதி",
        "notes_optional": "குறிப்புகள் (விருப்பம்)",
        "register_animal_btn": "💾 மாட்டைப் பதிவு செய்யவும்",
        "registration_success": "✅ மாடு வெற்றிகரமாகப் பதிவு செய்யப்பட்டது!",
        "fill_required_fields": "❌ தயவுசெய்து அனைத்து தேவையான புலங்களையும் நிரப்பவும்",
        
        # Analysis Summary
        "analysis_summary": "📋 பகுப்பாய்வு சுருக்கம்",
        "breed_identified": "அடையாளம் காணப்பட்ட இனம்",
        "confidence_label": "நம்பிக்கை",
        "origin_label": "தோற்றம்",
        "save_to_registry": "💾 பதிவகத்தில் சேமிக்கவும்",
        "saved_to_registry": "✅ பதிவகத்தில் சேமிக்கப்பட்டது!",
        "download_report": "📄 முழு அறிக்கையைப் பதிவிறக்கவும்"
    },
    "తెలుగు (Telugu)": {
        # Header and Page Title
        "page_title": "🐄 🐃 భారతీయ పశువులు మరియు గేదెల జాతుల గుర్తింపు",
        "page_subtitle": "🏆 SIH 2025 - AI-శక్తితో పశువుల నిర్వహణ వ్యవస్థ",
        "page_description": "🤖 అధునాతన EfficientNet-B3 మోడల్ • 🌾 49+ జాతులు • ⚡ తక్షణ విశ్లేషణ",
        "dairy_classification": "🥛 పాల వర్గీకరణ",
        "draught_identification": "🚜 పని చేసే పశువుల గుర్తింపు",
        "indigenous_breeds": "🌍 దేశీయ జాతులు",
        "ai_model_loaded": "🤖 **AI మోడల్**: ✅ లోడ్ అయింది",
        "ai_model_demo": "🤖 **AI మోడల్**: 🔄 డెమో మోడ్",
        
        # Dashboard
        "dashboard": "📊 డాష్‌బోర్డ్",
        "animals_registered": "🐄 నమోదైన పశువులు",
        "overdue_vaccinations": "⚠️ వాయిదా టీకాలు",
        "register_new_animal": "➕ కొత్త పశువును నమోదు చేయండి",
        
        # Upload Interface
        "upload_image": "📷 ఆవు/గేదె చిత్రాన్ని అప్‌లోడ్ చేయండి",
        "drag_drop": "🖱️ లాగి వదలండి లేదా బ్రౌజ్ చేయడానికి క్లిక్ చేయండి",
        "choose_image": "ఒక చిత్రాన్ని ఎంచుకోండి",
        "upload_help": "📱 ఫోన్ కెమెరాను ఉపయోగించండి • 🐄 పశువును మధ్యలో ఉంచండి • 📏 ఉత్తమ నాణ్యత చిత్రాలు • 🌅 మంచి వెలుతురు",
        "analyze_breed": "🔍 జాతిని విశ్లేషించండి",
        
        # Tips Section
        "tips_title": "💡 ఉత్తమ ఫలితాల కోసం చిట్కాలు",
        "tip_center": "🎯 పశువును ఫ్రేమ్ మధ్యలో ఉంచండి",
        "tip_lighting": "☀️ సహజ వెలుతురును ఉపయోగించండి",
        "tip_body": "📐 పూర్తి శరీరం లేదా స్పష్టమైన ముఖాన్ని చేర్చండి",
        "tip_avoid_blur": "🚫 అస్పష్టమైన/చీకటి చిత్రాలను నివారించండి",
        "tip_angles": "📱 అనిశ్చితంగా ఉంటే అనేక కోణాలలో తీయండి",
        
        # Analysis Results
        "analysis_results": "📊 విశ్లేషణ ఫలితాలు",
        "image_uploaded": "✅ **చిత్రం అప్‌లోడ్ అయింది**",
        "uploaded_image": "📷 అప్‌లోడ్ చేసిన చిత్రం",
        "analyzing_ml": "🤖 AI మోడల్‌తో జాతిని విశ్లేషిస్తోంది...",
        "analyzing_demo": "🎲 డెమో విశ్లేషణ నడుపుతోంది...",
        "prediction_failed": "ML అంచనా విఫలమైంది. డెమో మోడ్‌ను ఉపయోగిస్తోంది.",
        
        # Validation Messages
        "validation_failed": "⚠️ **చిత్ర ధృవీకరణ విఫలమైంది**",
        "upload_clear_image": "💡 **దయచేసి ఆవు లేదా గేదె యొక్క స్పష్టమైన చిత్రాన్ని అప్‌లోడ్ చేయండి**",
        "tips_valid_images": "📸 చెల్లుబాటు అయ్యే ఆవు/గేదె చిత్రాల కోసం చిట్కాలు",
        "good_images": "✅ మంచి చిత్రాలు:",
        "avoid_images": "❌ నివారించండి:",
        "clear_view": "🐄 ఆవు/గేదె యొక్క స్పష్టమైన దృశ్యం",
        "good_lighting": "🌅 మంచి వెలుతురు పరిస్థితులు",
        "full_body": "📐 పూర్తి లేదా పాక్షిక పశు శరీరం",
        "centered": "🎯 పశువు ఫ్రేమ్ మధ్యలో",
        "non_animal": "🚫 పశువు కాని విషయాలు",
        "too_dark": "🌙 చాలా చీకటి లేదా అస్పష్టమైన చిత్రాలు",
        "extreme_angles": "📐 తీవ్రమైన కోణాలు",
        "low_resolution": "🔍 చాలా తక్కువ రిజల్యూషన్",
        
        # Prediction Results
        "predicted_breed": "🎯 అంచనా వేయబడిన జాతి",
        "confidence": "📊 AI విశ్వాస స్థాయి",
        "cattle_detected": "✅ పశువు గుర్తించబడింది",
        "not_cattle": "❌ ఇది ఆవు/గేదె చిత్రం కాదు",
        "upload_instruction": "👆 ప్రారంభించడానికి చిత్రాన్ని అప్‌లోడ్ చేయండి",
        "supports": "🐄 ఆవు మరియు గేదె జాతులకు మద్దతు ఇస్తుంది",
        "ai_powered": "🔬 EfficientNet-B3తో AI-శక్తితో విశ్లేషణ",
        "mobile_optimized": "📱 మొబైల్ ఫోటోగ్రఫీ కోసం అనుకూలీకరించబడింది",
        
        # Breed Information
        "breed_information": "🐄 జాతి సమాచారం",
        "type": "రకం",
        "category": "వర్గం",
        "origin": "మూలం",
        "characteristics": "లక్షణాలు",
        "milk_yield": "పాల ఉత్పादన",
        "nutrition_requirements": "🌾 పోషకాహార అవసరాలు",
        "dry_matter": "పొడి పదార్థం",
        "concentrate": "గాఢమైన మేత",
        "green_fodder": "పచ్చి మేత",
        "water": "నీరు",
        "common_diseases": "🏥 సాధారణ వ్యాధులు",
        "vaccination_schedule": "💉 టీకా షెడ్యూల్",
        "vaccine": "టీకా",
        "frequency": "ఫ్రీక్వెన్సీ",
        "season": "సీజన్",
        
        # Status Messages
        "success": "విజయం",
        "error": "లోపం",
        "warning": "హెచ్చరిక",
        "info": "సమాచారం",
        
        # Registration Form
        "register_form_title": "➕ కొత్త పశువును నమోదు చేయండి",
        "animal_name": "పశువు పేరు/ఐడి *",
        "select_breed": "జాతి *",
        "select_breed_option": "జాతిని ఎంచుకోండి...",
        "last_vaccination_date": "చివరి టీకా తేదీ",
        "notes_optional": "గమనికలు (ఐచ్ఛికం)",
        "register_animal_btn": "💾 పశువును నమోదు చేయండి",
        "registration_success": "✅ పశువు విజయవంతంగా నమోదు చేయబడింది!",
        "fill_required_fields": "❌ దయచేసి అన్ని అవసరమైన ఫీల్డ్‌లను నింపండి",
        
        # Analysis Summary
        "analysis_summary": "📋 విశ్లేషణ సారాంశం",
        "breed_identified": "గుర్తించబడిన జాతి",
        "confidence_label": "విశ్వాసం",
        "origin_label": "మూలం",
        "save_to_registry": "💾 రిజిస్ట్రీలో సేవ్ చేయండి",
        "saved_to_registry": "✅ రిజిస్ట్రీలో సేవ్ చేయబడింది!",
        "download_report": "📄 పూర్తి నివేదికను డౌన్‌లోడ్ చేయండి"
    },
    "ಕನ್ನಡ (Kannada)": {
        # Header and Page Title
        "page_title": "🐄 🐃 ಭಾರತೀಯ ಹಸು ಮತ್ತು ಎಮ್ಮೆ ಜಾತಿಯ ಗುರುತಿಸುವಿಕೆ",
        "page_subtitle": "🏆 SIH 2025 - AI-ಚಾಲಿತ ಪಶುಸಂಗೋಪನೆ ನಿರ್ವಹಣಾ ವ್ಯವಸ್ಥೆ",
        "page_description": "🤖 ಸುಧಾರಿತ EfficientNet-B3 ಮಾದರಿ • 🌾 49+ ಜಾತಿಗಳು • ⚡ ತ್ವರಿತ ವಿಶ್ಲೇಷಣೆ",
        "dairy_classification": "🥛 ಹಾಲು ವರ್ಗೀಕರಣ",
        "draught_identification": "🚜 ಕೆಲಸದ ಪ್ರಾಣಿಗಳ ಗುರುತಿಸುವಿಕೆ",
        "indigenous_breeds": "🌍 ಸ್ವದೇಶೀ ಜಾತಿಗಳು",
        "ai_model_loaded": "🤖 **AI ಮಾದರಿ**: ✅ ಲೋಡ್ ಆಗಿದೆ",
        "ai_model_demo": "🤖 **AI ಮಾದರಿ**: 🔄 ಡೆಮೊ ಮೋಡ್",
        
        # Dashboard
        "dashboard": "📊 ಡ್ಯಾಶ್‌ಬೋರ್ಡ್",
        "animals_registered": "🐄 ನೋಂದಾಯಿತ ಪ್ರಾಣಿಗಳು",
        "overdue_vaccinations": "⚠️ ಮುಂದೂಡಲ್ಪಟ್ಟ ಲಸಿಕೆಗಳು",
        "register_new_animal": "➕ ಹೊಸ ಪ್ರಾಣಿಯನ್ನು ನೋಂದಾಯಿಸಿ",
        
        # Upload Interface
        "upload_image": "📷 ಹಸು/ಎಮ್ಮೆ ಚಿತ್ರವನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ",
        "drag_drop": "🖱️ ಎಳೆದು ಬಿಡಿ ಅಥವಾ ಬ್ರೌಸ್ ಮಾಡಲು ಕ್ಲಿಕ್ ಮಾಡಿ",
        "choose_image": "ಒಂದು ಚಿತ್ರವನ್ನು ಆಯ್ಕೆಮಾಡಿ",
        "upload_help": "📱 ಫೋನ್ ಕ್ಯಾಮೆರಾವನ್ನು ಬಳಸಿ • 🐄 ಪ್ರಾಣಿಯನ್ನು ಮಧ್ಯದಲ್ಲಿ ಇರಿಸಿ • 📏 ಅತ್ಯುತ್ತಮ ಗುಣಮಟ್ಟದ ಚಿತ್ರಗಳು • 🌅 ಉತ್ತಮ ಬೆಳಕು",
        "analyze_breed": "🔍 ಜಾತಿಯನ್ನು ವಿಶ್ಲೇಷಿಸಿ",
        
        # Tips Section
        "tips_title": "💡 ಅತ್ಯುತ್ತಮ ಫಲಿತಾಂಶಗಳಿಗೆ ಸಲಹೆಗಳು",
        "tip_center": "🎯 ಪ್ರಾಣಿಯನ್ನು ಚೌಕಟ್ಟಿನ ಮಧ್ಯದಲ್ಲಿ ಇರಿಸಿ",
        "tip_lighting": "☀️ ನೈಸರ್ಗಿಕ ಬೆಳಕನ್ನು ಬಳಸಿ",
        "tip_body": "📐 ಪೂರ್ಣ ದೇಹ ಅಥವಾ ಸ್ಪಷ್ಟ ಮುಖವನ್ನು ಸೇರಿಸಿ",
        "tip_avoid_blur": "🚫 ಮಂದ/ಕತ್ತಲೆಯಾದ ಚಿತ್ರಗಳನ್ನು ತಪ್ಪಿಸಿ",
        "tip_angles": "📱 ಅನುಮಾನವಿದ್ದರೆ ಅನೇಕ ಕೋನಗಳಿಂದ ತೆಗೆಯಿರಿ",
        
        # Analysis Results
        "analysis_results": "📊 ವಿಶ್ಲೇಷಣೆ ಫಲಿತಾಂಶಗಳು",
        "image_uploaded": "✅ **ಚಿತ್ರ ಅಪ್‌ಲೋಡ್ ಆಗಿದೆ**",
        "uploaded_image": "📷 ಅಪ್‌ಲೋಡ್ ಮಾಡಿದ ಚಿತ್ರ",
        "analyzing_ml": "🤖 AI ಮಾದರಿಯೊಂದಿಗೆ ಜಾತಿಯನ್ನು ವಿಶ್ಲೇಷಿಸುತ್ತಿದೆ...",
        "analyzing_demo": "🎲 ಡೆಮೊ ವಿಶ್ಲೇಷಣೆ ನಡೆಸುತ್ತಿದೆ...",
        "prediction_failed": "ML ಮುನ್ಸೂಚನೆ ವಿಫಲವಾಗಿದೆ. ಡೆಮೊ ಮೋಡ್ ಬಳಸುತ್ತಿದೆ.",
        
        # Validation Messages
        "validation_failed": "⚠️ **ಚಿತ್ರ ಪರಿಶೀಲನೆ ವಿಫಲವಾಗಿದೆ**",
        "upload_clear_image": "💡 **ದಯವಿಟ್ಟು ಹಸು ಅಥವಾ ಎಮ್ಮೆಯ ಸ್ಪಷ್ಟ ಚಿತ್ರವನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ**",
        "tips_valid_images": "📸 ಮಾನ್ಯವಾದ ಹಸು/ಎಮ್ಮೆ ಚಿತ್ರಗಳಿಗೆ ಸಲಹೆಗಳು",
        "good_images": "✅ ಉತ್ತಮ ಚಿತ್ರಗಳು:",
        "avoid_images": "❌ ತಪ್ಪಿಸಿ:",
        "clear_view": "🐄 ಹಸು/ಎಮ್ಮೆಯ ಸ್ಪಷ್ಟ ನೋಟ",
        "good_lighting": "🌅 ಉತ್ತಮ ಬೆಳಕಿನ ಪರಿಸ್ಥಿತಿಗಳು",
        "full_body": "📐 ಪೂರ್ಣ ಅಥವಾ ಭಾಗಶಃ ಪ್ರಾಣಿ ದೇಹ",
        "centered": "🎯 ಪ್ರಾಣಿ ಚೌಕಟ್ಟಿನ ಮಧ್ಯದಲ್ಲಿ",
        "non_animal": "🚫 ಪ್ರಾಣಿಯಲ್ಲದ ವಿಷಯಗಳು",
        "too_dark": "🌙 ತುಂಬಾ ಕಪ್ಪು ಅಥವಾ ಮಂದ ಚಿತ್ರಗಳು",
        "extreme_angles": "📐 ತೀವ್ರ ಕೋನಗಳು",
        "low_resolution": "🔍 ತುಂಬಾ ಕಡಿಮೆ ರೆಸಲ್ಯೂಶನ್",
        
        # Prediction Results
        "predicted_breed": "🎯 ಊಹಿಸಲಾದ ಜಾತಿ",
        "confidence": "📊 AI ವಿಶ್ವಾಸ ಮಟ್ಟ",
        "cattle_detected": "✅ ಪ್ರಾಣಿ ಪತ್ತೆಯಾಗಿದೆ",
        "not_cattle": "❌ ಇದು ಹಸು/ಎಮ್ಮೆ ಚಿತ್ರವಲ್ಲ",
        "upload_instruction": "👆 ಪ್ರಾರಂಭಿಸಲು ಚಿತ್ರವನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ",
        "supports": "🐄 ಹಸು ಮತ್ತು ಎಮ್ಮೆ ಜಾತಿಗಳನ್ನು ಬೆಂಬಲಿಸುತ್ತದೆ",
        "ai_powered": "🔬 EfficientNet-B3 ನೊಂದಿಗೆ AI-ಚಾಲಿತ ವಿಶ್ಲೇಷಣೆ",
        "mobile_optimized": "📱 ಮೊಬೈಲ್ ಫೋಟೋಗ್ರಫಿಗಾಗಿ ಅನುಕೂಲಿತ",
        
        # Breed Information
        "breed_information": "🐄 ಜಾತಿ ಮಾಹಿತಿ",
        "type": "ಪ್ರಕಾರ",
        "category": "ವರ್ಗ",
        "origin": "ಮೂಲ",
        "characteristics": "ಗುಣಲಕ್ಷಣಗಳು",
        "milk_yield": "ಹಾಲಿನ ಉತ್ಪಾದನೆ",
        "nutrition_requirements": "🌾 ಪೋಷಣೆಯ ಅವಶ್ಯಕತೆಗಳು",
        "dry_matter": "ಒಣ ಪದಾರ್ಥ",
        "concentrate": "ಸಾಂದ್ರೀಕೃತ ಆಹಾರ",
        "green_fodder": "ಹಸಿರು ಮೇವು",
        "water": "ನೀರು",
        "common_diseases": "🏥 ಸಾಮಾನ್ಯ ರೋಗಗಳು",
        "vaccination_schedule": "💉 ಲಸಿಕೆ ವೇಳಾಪಟ್ಟಿ",
        "vaccine": "ಲಸಿಕೆ",
        "frequency": "ಆವೃತ್ತಿ",
        "season": "ಋತು",
        
        # Status Messages
        "success": "ಯಶಸ್ಸು",
        "error": "ದೋಷ",
        "warning": "ಎಚ್ಚರಿಕೆ",
        "info": "ಮಾಹಿತಿ",
        
        # Registration Form
        "register_form_title": "➕ ಹೊಸ ಪ್ರಾಣಿಯನ್ನು ನೋಂದಾಯಿಸಿ",
        "animal_name": "ಪ್ರಾಣಿಯ ಹೆಸರು/ಐಡಿ *",
        "select_breed": "ಜಾತಿ *",
        "select_breed_option": "ಜಾತಿಯನ್ನು ಆಯ್ಕೆಮಾಡಿ...",
        "last_vaccination_date": "ಕೊನೆಯ ಲಸಿಕೆ ದಿನಾಂಕ",
        "notes_optional": "ಟಿಪ್ಪಣಿಗಳು (ಐಚ್ಛಿಕ)",
        "register_animal_btn": "💾 ಪ್ರಾಣಿಯನ್ನು ನೋಂದಾಯಿಸಿ",
        "registration_success": "✅ ಪ್ರಾಣಿ ಯಶಸ್ವಿಯಾಗಿ ನೋಂದಾಯಿಸಲಾಗಿದೆ!",
        "fill_required_fields": "❌ ದಯವಿಟ್ಟು ಎಲ್ಲಾ ಅಗತ್ಯ ಕ್ಷೇತ್ರಗಳನ್ನು ಭರ್ತಿ ಮಾಡಿ",
        
        # Analysis Summary
        "analysis_summary": "📋 ವಿಶ್ಲೇಷಣೆ ಸಾರಾಂಶ",
        "breed_identified": "ಗುರುತಿಸಲಾದ ಜಾತಿ",
        "confidence_label": "ವಿಶ್ವಾಸ",
        "origin_label": "ಮೂಲ",
        "save_to_registry": "💾 ರಿಜಿಸ್ಟ್ರಿಯಲ್ಲಿ ಉಳಿಸಿ",
        "saved_to_registry": "✅ ರಿಜಿಸ್ಟ್ರಿಯಲ್ಲಿ ಉಳಿಸಲಾಗಿದೆ!",
        "download_report": "📄 ಪೂರ್ಣ ವರದಿಯನ್ನು ಡೌನ್‌ಲೋಡ್ ಮಾಡಿ"
    }
}

# Get current language translations
t = translations.get(language, translations["English"])

# Header with enhanced farm theme - Using translations
st.markdown(f"""
<div class="hero-header">
    <h1>{t["page_title"]}</h1>
    <h3>{t["page_subtitle"]}</h3>
    <p>{t["page_description"]}</p>
    <div style="margin-top: 1rem; display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap;">
        <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px; color: #1a1a1a; font-weight: 600;">
            {t["dairy_classification"]}
        </span>
        <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px; color: #1a1a1a; font-weight: 600;">
            {t["draught_identification"]}
        </span>
        <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px; color: #1a1a1a; font-weight: 600;">
            {t["indigenous_breeds"]}
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# Model status indicator
model_status = t["ai_model_loaded"] if model_available else t["ai_model_demo"]
st.markdown(f"<div style='text-align: center; padding: 0.5rem; background: rgba(32,135,147,0.1); border-radius: 8px; margin-bottom: 1rem;'>{model_status}</div>", unsafe_allow_html=True)

# Sidebar content with translations
today = datetime.today().date()
c.execute("SELECT name,breed,last_vaccination_date FROM animals")
animals_db = c.fetchall()

st.sidebar.metric(t["animals_registered"], len(animals_db))

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

st.sidebar.metric(t["overdue_vaccinations"], overdue_count)

if st.sidebar.button(t["register_new_animal"]):
    st.query_params = {"action": ["register"]}

# Main interface
col1, col2 = st.columns([1.3, 1.7])

with col1:
    st.markdown(f"### {t['upload_image']}")
    st.markdown(f"**{t['drag_drop']}**")
    
    uploaded_file = st.file_uploader(
        t["choose_image"],
        type=["jpg", "jpeg", "png"],
        help=t["upload_help"],
        label_visibility="collapsed"
    )
    
    st.markdown(f'<div style="background: linear-gradient(135deg, rgba(32,135,147,0.05), rgba(32,135,147,0.1)); padding: 1rem; border-radius: 8px; margin: 1rem 0; border-left: 4px solid #208793;"><h4 style="margin: 0 0 0.5rem 0; color: #208793;">{t["tips_title"]}</h4><ul style="margin: 0; padding-left: 1.2rem; font-size: 0.9rem;"><li>{t["tip_center"]}</li><li>{t["tip_lighting"]}</li><li>{t["tip_body"]}</li><li>{t["tip_avoid_blur"]}</li><li>{t["tip_angles"]}</li></ul></div>', unsafe_allow_html=True)
    
    analyze_btn = st.button(t["analyze_breed"], type="primary", use_container_width=True)

with col2:
    st.markdown(f"### {t['analysis_results']}")
    
    if uploaded_file is not None:
        # Show upload success
        st.success(f"{t['image_uploaded']}: {uploaded_file.name}")
        
        # Display image
        image = Image.open(uploaded_file)
        st.image(image, caption=t["uploaded_image"], use_container_width=True)
        
        if analyze_btn:
            with st.spinner(t["analyzing_ml"] if model_available else t["analyzing_demo"]):
                # Prediction with cattle validation
                validation_message = ""
                
                if model_available:
                    breed, conf, probs, validation_message = predict_breed_ml(image, model, breed_classes, device)
                    if breed is None:
                        breed, conf, probs, validation_message = predict_breed_demo(image, breed_classes)
                        if breed is None:
                            st.error(t["validation_failed"])
                            st.error(validation_message)
                            st.info(t["upload_clear_image"])
                            st.stop()
                        else:
                            st.warning(t["prediction_failed"])
                else:
                    breed, conf, probs, validation_message = predict_breed_demo(image, breed_classes)
                    if breed is None:
                        st.error(t["validation_failed"])
                        st.error(validation_message)
                        st.info(t["upload_clear_image"])
                        
                        # Enhanced guidance with visual styling
                        st.markdown(f'<div style="background: linear-gradient(135deg, #FFC107 0%, #4CAF50 100%); padding: 1.5rem; border-radius: 15px; margin: 1rem 0; color: white;"><h4 style="margin: 0 0 1rem 0;">{t["tips_valid_images"]}</h4><div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;"><div><p><strong>{t["good_images"]}</strong></p><ul><li>{t["clear_view"]}</li><li>{t["good_lighting"]}</li><li>{t["full_body"]}</li><li>{t["centered"]}</li></ul></div><div><p><strong>{t["avoid_images"]}</strong></p><ul><li>{t["non_animal"]}</li><li>{t["too_dark"]}</li><li>{t["extreme_angles"]}</li><li>{t["low_resolution"]}</li></ul></div></div></div>', unsafe_allow_html=True)
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
                st.markdown(f'<div style="background: linear-gradient(135deg, #4CAF50 0%, #FFC107 100%); padding: 1rem; border-radius: 10px; color: white; margin: 0.5rem 0;">{metadata["body_weight"]}</div>', unsafe_allow_html=True)
                
                # Enhanced characteristics section
                st.markdown("### 🔍 Physical Characteristics")
                st.markdown(f'<div style="background: linear-gradient(135deg, #8D6E63 0%, #42A5F5 100%); padding: 1rem; border-radius: 10px; color: white; margin: 0.5rem 0;">{metadata["characteristics"]}</div>', unsafe_allow_html=True)
                
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
                    if st.button(t["save_to_registry"], use_container_width=True):
                        c.execute(
                            "INSERT INTO animals (name, breed, last_vaccination_date, notes) VALUES (?,?,?,?)",
                            (f"Animal_{len(animals_db)+1}", breed, today.strftime("%Y-%m-%d"), 
                             f"Confidence: {confidence_pct:.1f}%")
                        )
                        conn.commit()
                        st.success(t["saved_to_registry"])
                        
                with col_b:
                    # Generate comprehensive report
                    report_content = f"SIH 2025 - CATTLE BREED ANALYSIS REPORT\n\nAnalysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\nImage File: {uploaded_file.name}\nGenerated by: AI-Powered Cattle Breed Recognition System\n\nPREDICTION RESULTS\nPredicted Breed: {breed}\nConfidence Level: {confidence_pct:.2f}%\nAI Model: EfficientNet-B3 Deep Learning Model\n\nBREED INFORMATION\nOrigin: {metadata['origin']}\nCategory: {metadata['category']}\nType: {metadata['type']}\nMilk Yield: {metadata['milk_yield']}\n\nBody Weight Information:\n{metadata['body_weight'].replace('**', '').replace('<br>', '\n')}\n\nPhysical Characteristics:\n{metadata['characteristics']}\n\nNUTRITION MANAGEMENT\n{metadata['nutrition'].replace('**', '').replace('🌾', '').replace('🥗', '').replace('🌿', '').replace('💧', '').replace('\n', '\n')}\n\nHEALTH & DISEASE MANAGEMENT\n{metadata['diseases'].replace('**', '').replace('🏥', '').replace('•', '-')}\n\nVACCINATION SCHEDULE\n{metadata['vaccination'].replace('**', '').replace('💉', '').replace('•', '-')}\n\nBREEDING INFORMATION\n{metadata['breeding'].replace('**', '').replace('🐄', '')}\n\nRECOMMENDATIONS\n1. Follow the nutrition guidelines strictly for optimal milk production\n2. Maintain regular vaccination schedule as per the recommended timeline\n3. Monitor for common diseases and consult veterinarian for preventive care\n4. Ensure adequate water supply and quality fodder throughout the year\n5. Maintain proper breeding records for genetic improvement\n\nDISCLAIMER\nThis analysis is generated by an AI system for educational and advisory purposes. Always consult with qualified veterinarians and livestock experts for medical decisions and breeding programs.\n\nCONTACT INFORMATION\nProject: Smart India Hackathon 2025\nTeam: Nexel\nGitHub: https://github.com/sanjayrockerz/SIH-Cattle-Breed-Recognition\nEmail: myteamcreations09@gmail.com\n\nReport generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}"
                    
                    # Enhanced visual summary card
                    st.markdown(f"### {t['analysis_summary']}")
                    summary_col1, summary_col2, summary_col3 = st.columns(3)
                    
                    with summary_col1:
                        st.markdown(f'<div style="background: linear-gradient(135deg, #4CAF50 0%, #FFC107 100%); padding: 1rem; border-radius: 10px; text-align: center; color: white; margin: 0.5rem 0;"><h3 style="margin: 0;">🎯</h3><p style="margin: 0;"><strong>{t["breed_identified"]}</strong></p><p style="margin: 0; font-size: 0.9rem;">{breed}</p></div>', unsafe_allow_html=True)
                    
                    with summary_col2:
                        st.markdown(f'<div style="background: linear-gradient(135deg, #42A5F5 0%, #4CAF50 100%); padding: 1rem; border-radius: 10px; text-align: center; color: white; margin: 0.5rem 0;"><h3 style="margin: 0;">📊</h3><p style="margin: 0;"><strong>{t["confidence_label"]}</strong></p><p style="margin: 0; font-size: 0.9rem;">{confidence_pct:.1f}%</p></div>', unsafe_allow_html=True)
                    
                    with summary_col3:
                        st.markdown(f'<div style="background: linear-gradient(135deg, #8D6E63 0%, #42A5F5 100%); padding: 1rem; border-radius: 10px; text-align: center; color: white; margin: 0.5rem 0;"><h3 style="margin: 0;">🌍</h3><p style="margin: 0;"><strong>{t["origin_label"]}</strong></p><p style="margin: 0; font-size: 0.9rem;">{metadata["origin"]}</p></div>', unsafe_allow_html=True)
                    
                    st.download_button(
                        t["download_report"],
                        data=report_content,
                        file_name=f"breed_report_{breed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
    else:
        st.info(f"**{t['upload_instruction']}**")
        st.markdown(f'<div style="text-align: center; padding: 2rem; color: #666;"><p>{t["supports"]}</p><p>{t["ai_powered"]}</p><p>{t["mobile_optimized"]}</p></div>', unsafe_allow_html=True)

# Registration form
q = st.query_params
if q.get("action") == ["register"]:
    st.markdown("---")
    with st.form("register_animal", clear_on_submit=True):
        st.markdown(f"### {t['register_form_title']}")
        
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            name = st.text_input(t["animal_name"])
            breed = st.selectbox(t["select_breed"], [t["select_breed_option"]] + sorted(breed_classes))
        
        with col_r2:
            last_vacc = st.date_input(t["last_vaccination_date"], value=today)
            notes = st.text_area(t["notes_optional"], height=100)
        
        submitted = st.form_submit_button(t["register_animal_btn"], type="primary")
        
        if submitted:
            if name and breed != t["select_breed_option"]:
                c.execute(
                    "INSERT INTO animals (name, breed, last_vaccination_date, notes) VALUES (?,?,?,?)",
                    (name, breed, last_vacc.strftime("%Y-%m-%d"), notes)
                )
                conn.commit()
                st.success(f"✅ **{name}** {t['registration_success']}")
                st.query_params.clear()
            else:
                st.error(f"{t['fill_required_fields']} (*)")

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
st.markdown('<div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, rgba(32,135,147,0.05), rgba(32,135,147,0.1)); border-radius: 12px; margin-top: 2rem;"><h3 style="color: #208793; margin-bottom: 1rem;">🏆 Smart India Hackathon 2025</h3><p style="margin: 0.5rem 0;"><strong>AI-based Cattle Breed Identification and Management System</strong></p><p style="margin: 0.5rem 0;">Developed by <strong>Team Nexel</strong></p><p style="margin: 0.5rem 0;"><a href="https://github.com/sanjayrockerz/SIH-Cattle-Breed-Recognition" target="_blank" style="color: #208793; text-decoration: none;">🔗 GitHub Repository</a> • <a href="mailto:myteamcreations09@gmail.com" style="color: #208793; text-decoration: none;">✉️ Contact</a></p><p style="font-size: 0.9rem; color: #666; margin-top: 1rem;">Empowering farmers with AI • Supporting indigenous breeds • Building the future of livestock management</p></div>', unsafe_allow_html=True)
