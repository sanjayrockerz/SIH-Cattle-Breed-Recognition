#!/usr/bin/env python3
"""
Test script for enhanced cattle image validation
Tests the improved validation function with different image scenarios
"""

import sys
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Add the current directory to path to import from streamlit_app
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_test_image(image_type, size=(400, 300)):
    """Create synthetic test images for different scenarios"""
    img = Image.new('RGB', size, color='white')
    draw = ImageDraw.Draw(img)
    
    if image_type == "cattle_brown":
        # Brown cattle-like colors and shapes
        draw.rectangle([50, 100, 350, 250], fill=(101, 67, 33))  # Brown body
        draw.ellipse([60, 80, 120, 120], fill=(139, 69, 19))     # Head
        draw.rectangle([80, 250, 100, 290], fill=(101, 67, 33))  # Legs
        draw.rectangle([120, 250, 140, 290], fill=(101, 67, 33))
        draw.rectangle([260, 250, 280, 290], fill=(101, 67, 33))
        draw.rectangle([300, 250, 320, 290], fill=(101, 67, 33))
        
    elif image_type == "cattle_black_white":
        # Black and white cattle pattern
        draw.rectangle([50, 100, 350, 250], fill=(255, 255, 255))  # White body
        draw.ellipse([60, 80, 120, 120], fill=(0, 0, 0))          # Black head
        draw.rectangle([150, 120, 250, 180], fill=(0, 0, 0))      # Black spots
        draw.rectangle([80, 250, 100, 290], fill=(0, 0, 0))       # Black legs
        draw.rectangle([120, 250, 140, 290], fill=(0, 0, 0))
        draw.rectangle([260, 250, 280, 290], fill=(0, 0, 0))
        draw.rectangle([300, 250, 320, 290], fill=(0, 0, 0))
        
    elif image_type == "human_skin":
        # Human skin tones
        draw.ellipse([150, 50, 250, 150], fill=(222, 184, 135))   # Head
        draw.rectangle([175, 150, 225, 280], fill=(222, 184, 135)) # Body
        draw.rectangle([160, 200, 190, 270], fill=(222, 184, 135)) # Arms
        draw.rectangle([210, 200, 240, 270], fill=(222, 184, 135))
        
    elif image_type == "dog_colors":
        # Dog-like colors and proportions (more compact)
        draw.ellipse([150, 80, 250, 150], fill=(160, 82, 45))     # Head
        draw.rectangle([120, 140, 280, 200], fill=(160, 82, 45))  # Body
        draw.rectangle([100, 190, 120, 220], fill=(160, 82, 45))  # Legs
        draw.rectangle([130, 190, 150, 220], fill=(160, 82, 45))
        draw.rectangle([250, 190, 270, 220], fill=(160, 82, 45))
        draw.rectangle([280, 190, 300, 220], fill=(160, 82, 45))
        # Add some bright collar color (unnatural for cattle)
        draw.rectangle([170, 140, 230, 155], fill=(255, 0, 0))    # Red collar
        
    elif image_type == "landscape":
        # Landscape scene
        draw.rectangle([0, 0, 400, 150], fill=(135, 206, 235))    # Sky
        draw.rectangle([0, 150, 400, 200], fill=(34, 139, 34))    # Hills
        draw.rectangle([0, 200, 400, 300], fill=(154, 205, 50))   # Grass
        
    return img

def test_validation_function():
    """Test the enhanced validation function"""
    try:
        # Import the validation function
        from streamlit_app import validate_cattle_image
        
        test_cases = [
            ("cattle_brown", "Brown cattle", True),
            ("cattle_black_white", "Black and white cattle", True),
            ("human_skin", "Human with skin tones", False),
            ("dog_colors", "Dog with bright collar", False),
            ("landscape", "Landscape scene", False),
        ]
        
        print("🧪 Testing Enhanced Cattle Image Validation")
        print("=" * 50)
        
        for image_type, description, expected_result in test_cases:
            print(f"\n📸 Testing: {description}")
            
            # Create test image
            test_img = create_test_image(image_type)
            
            # Test validation
            is_cattle, confidence, reason = validate_cattle_image(test_img)
            
            # Check result
            status = "✅ PASS" if (is_cattle == expected_result) else "❌ FAIL"
            print(f"{status} - Expected: {expected_result}, Got: {is_cattle}")
            print(f"   Confidence: {confidence:.2f}")
            print(f"   Reason: {reason}")
            
            # Save test image for manual inspection
            test_img.save(f"test_{image_type}.png")
            print(f"   Saved: test_{image_type}.png")
        
        print("\n" + "=" * 50)
        print("✨ Validation testing complete!")
        print("📁 Test images saved for manual inspection")
        
    except ImportError as e:
        print(f"❌ Error importing validation function: {e}")
        print("Make sure streamlit_app.py is in the same directory")
    except Exception as e:
        print(f"❌ Error during testing: {e}")

def test_edge_cases():
    """Test edge cases and boundary conditions"""
    try:
        from streamlit_app import validate_cattle_image
        
        print("\n🔍 Testing Edge Cases")
        print("=" * 30)
        
        # Very small image
        small_img = Image.new('RGB', (50, 50), color=(101, 67, 33))
        is_cattle, conf, reason = validate_cattle_image(small_img)
        print(f"Small image (50x50): {is_cattle}, {reason}")
        
        # Very dark image
        dark_img = Image.new('RGB', (300, 200), color=(5, 5, 5))
        is_cattle, conf, reason = validate_cattle_image(dark_img)
        print(f"Dark image: {is_cattle}, {reason}")
        
        # Very bright image
        bright_img = Image.new('RGB', (300, 200), color=(250, 250, 250))
        is_cattle, conf, reason = validate_cattle_image(bright_img)
        print(f"Bright image: {is_cattle}, {reason}")
        
        # Extreme aspect ratio (very wide)
        wide_img = Image.new('RGB', (800, 100), color=(101, 67, 33))
        is_cattle, conf, reason = validate_cattle_image(wide_img)
        print(f"Wide image (8:1): {is_cattle}, {reason}")
        
        # Extreme aspect ratio (very tall)
        tall_img = Image.new('RGB', (100, 800), color=(101, 67, 33))
        is_cattle, conf, reason = validate_cattle_image(tall_img)
        print(f"Tall image (1:8): {is_cattle}, {reason}")
        
    except Exception as e:
        print(f"❌ Error testing edge cases: {e}")

if __name__ == "__main__":
    print("🐄 Enhanced Cattle Image Validation Test Suite")
    print("=" * 55)
    
    test_validation_function()
    test_edge_cases()
    
    print("\n🎯 Test Summary:")
    print("• Enhanced validation should reject human skin tones")
    print("• Should reject unnatural colors (bright collars, etc.)")
    print("• Should accept typical cattle colors (brown, black, white)")
    print("• Should reject extreme aspect ratios")
    print("• Should handle edge cases gracefully")
    
    print("\n💡 To manually verify:")
    print("• Check saved test images: test_*.png")
    print("• Run the main app and test with real images")
    print("• Test with photos of dogs, humans, and actual cattle")