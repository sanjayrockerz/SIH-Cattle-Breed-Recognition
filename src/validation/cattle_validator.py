"""
Enhanced cattle image validation module
Provides robust validation to distinguish cattle/buffalo from other animals
"""

import numpy as np
from PIL import Image


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
        
        # METHOD 2: Enhanced Color Analysis - Prioritize cattle colors, then check for obvious non-cattle
        hsv = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2HSV)
        
        # First, check for typical cattle colors (more permissive approach)
        cattle_color_masks = []
        
        # Brown cattle colors (various shades) - EXPANDED RANGES
        brown_ranges = [
            ([8, 30, 40], [25, 255, 220]),    # Light brown to dark brown (broad range)
            ([0, 30, 50], [30, 255, 200]),    # Reddish brown
            ([25, 20, 80], [35, 180, 200])    # Tan/beige
        ]
        
        # Black/dark colors (cattle often have black markings)
        black_lower = np.array([0, 0, 0])
        black_upper = np.array([180, 255, 80])  # Expanded dark range
        
        # White/light colors (common in cattle markings)
        white_lower = np.array([0, 0, 180])     # Lighter threshold
        white_upper = np.array([180, 40, 255])
        
        # Calculate cattle color coverage
        total_cattle_ratio = 0
        for lower, upper in brown_ranges:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            total_cattle_ratio += np.sum(mask > 0) / (width * height)
        
        black_mask = cv2.inRange(hsv, black_lower, black_upper)
        white_mask = cv2.inRange(hsv, white_lower, white_upper)
        
        black_ratio = np.sum(black_mask > 0) / (width * height)
        white_ratio = np.sum(white_mask > 0) / (width * height)
        
        total_cattle_color_ratio = total_cattle_ratio + black_ratio + white_ratio
        
        # If we have good cattle colors, be very conservative about rejecting
        has_strong_cattle_colors = total_cattle_color_ratio > 0.4
        
        # Only check for human skin if we don't have strong cattle colors
        if not has_strong_cattle_colors:
            # Much more restrictive human skin detection (only obvious human skin)
            # Focus on pinkish skin tones, not brown tones that could be cattle
            skin_lower1 = np.array([0, 40, 120])   # Pinkish skin (higher saturation, brightness)
            skin_upper1 = np.array([15, 180, 255])
            skin_lower2 = np.array([340, 30, 150]) # Reddish skin (converted to 0-179 range)
            skin_upper2 = np.array([20, 120, 255])
            
            skin_mask1 = cv2.inRange(hsv, skin_lower1, skin_upper1)
            skin_mask2 = cv2.inRange(hsv, skin_lower2, skin_upper2)
            skin_mask = cv2.bitwise_or(skin_mask1, skin_mask2)
            
            skin_ratio = np.sum(skin_mask > 0) / (width * height)
            
            # Much higher threshold - only reject if very obvious human skin
            if skin_ratio > 0.25:  # Increased from 0.15 to 0.25
                return False, 0.0, f"Clear human skin tones detected ({skin_ratio:.2%} of image)"
        
        # Dog/pet/other animal rejection criteria
        # 1. Very bright/unnatural colors (collars, clothing, etc.)
        saturation = hsv[:,:,1]
        value = hsv[:,:,2]
        high_sat_ratio = np.sum(saturation > 230) / (width * height)
        
        if high_sat_ratio > 0.15:  # More than 15% extremely saturated colors
            rejection_reasons.append("Unnatural high color saturation detected")
        
        # 2. Detect very pure/artificial colors (red, blue, green collars/clothing)
        pure_red = cv2.inRange(hsv, np.array([0, 180, 150]), np.array([10, 255, 255]))
        pure_blue = cv2.inRange(hsv, np.array([100, 180, 150]), np.array([130, 255, 255]))
        pure_green = cv2.inRange(hsv, np.array([40, 180, 150]), np.array([80, 255, 255]))
        
        artificial_color_ratio = (np.sum(pure_red > 0) + np.sum(pure_blue > 0) + np.sum(pure_green > 0)) / (width * height)
        if artificial_color_ratio > 0.08:  # More than 8% artificial colors
            rejection_reasons.append("Artificial colors detected (likely clothing/collars)")
        
        # Cattle-specific color analysis
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
        black_upper = np.array([180, 255, 60])
        black_mask = cv2.inRange(hsv, black_lower, black_upper)
        black_ratio = np.sum(black_mask > 0) / (width * height)
        
        # White/light colors (common in cattle markings)
        white_lower = np.array([0, 0, 200])
        white_upper = np.array([180, 30, 255])
        white_mask = cv2.inRange(hsv, white_lower, white_upper)
        white_ratio = np.sum(white_mask > 0) / (width * height)
        
        cattle_color_ratio = total_brown_ratio + black_ratio + white_ratio
        
        # Color scoring - enhanced with cattle color priority
        if has_strong_cattle_colors:
            confidence_score += 0.4  # Strong boost for cattle colors
            reasons.append("Strong cattle-typical colors detected")
        elif cattle_color_ratio > 0.5:
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
        kernel = np.array([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]])
        texture_response = cv2.filter2D(gray, -1, kernel)
        texture_variance = np.var(texture_response)
        
        if texture_variance > 500:
            confidence_score += 0.15
            reasons.append("Good texture variation detected")
        elif texture_variance < 100:
            rejection_reasons.append("Insufficient texture variation")
        
        # METHOD 4: Aspect Ratio and Proportions
        aspect_ratio = width / height
        
        if 1.2 <= aspect_ratio <= 3.0:
            confidence_score += 0.2
            reasons.append("Appropriate cattle proportions")
        elif aspect_ratio < 0.7:
            rejection_reasons.append("Aspect ratio too tall for cattle")
        elif aspect_ratio > 4.0:
            rejection_reasons.append("Aspect ratio too wide")
        
        # METHOD 5: Edge Density Analysis
        edge_density = np.sum(edges > 0) / (width * height)
        
        if 0.05 <= edge_density <= 0.30:
            confidence_score += 0.1
            reasons.append("Appropriate edge density")
        elif edge_density > 0.40:
            rejection_reasons.append("Too much detail/complexity")
        
        # METHOD 6: Size and Scale Analysis
        if contours:
            x, y, w, h = cv2.boundingRect(largest_contour)
            object_aspect = w / h
            object_coverage = (w * h) / (width * height)
            
            if object_coverage > 0.1:
                confidence_score += 0.1
                reasons.append("Good object size in frame")
            
            if 1.5 <= object_aspect <= 3.5:
                confidence_score += 0.1
                reasons.append("Good body proportions")
        
        # FINAL DECISION
        if rejection_reasons:
            # If we have strong cattle colors, be more forgiving of other issues
            if has_strong_cattle_colors and len(rejection_reasons) <= 2:
                reasons.append(f"Override minor issues due to strong cattle colors: {'; '.join(rejection_reasons)}")
            else:
                return False, 0.0, f"Not cattle: {'; '.join(rejection_reasons)}"
        
        # Adaptive threshold based on cattle color presence
        threshold = 0.5 if has_strong_cattle_colors else 0.6
        is_cattle = confidence_score >= threshold
        
        if not is_cattle:
            return False, confidence_score, "Insufficient cattle characteristics detected"
        
        reason = f"Cattle detected: {'; '.join(reasons)}"
        return True, confidence_score, reason
        
    except ImportError:
        # Fallback validation without OpenCV
        return _fallback_validation(image)
    
    except Exception as e:
        return True, 0.4, "Basic validation passed"


def _fallback_validation(image):
    """Fallback validation when OpenCV is not available"""
    try:
        width, height = image.size
        
        if width < 100 or height < 100:
            return False, 0.0, "Image resolution too low"
        
        gray = image.convert('L')
        pixels = list(gray.getdata())
        mean_brightness = sum(pixels) / len(pixels)
        
        if mean_brightness < 10 or mean_brightness > 250:
            return False, 0.0, "Image too dark or overexposed"
        
        aspect_ratio = width / height
        if aspect_ratio < 0.7:
            return False, 0.2, "Aspect ratio suggests non-cattle subject"
        elif aspect_ratio > 4.0:
            return False, 0.2, "Aspect ratio too extreme for cattle"
        
        # Basic color analysis for skin tone detection
        try:
            rgb_array = np.array(image)
            if len(rgb_array.shape) == 3:
                r_channel = rgb_array[:,:,0]
                g_channel = rgb_array[:,:,1]
                b_channel = rgb_array[:,:,2]
                
                skin_mask = (r_channel > g_channel) & (g_channel > b_channel) & (r_channel > 95) & (g_channel > 40) & (b_channel > 20)
                skin_ratio = np.sum(skin_mask) / (width * height)
                
                if skin_ratio > 0.15:
                    return False, 0.0, "Possible human skin tones detected"
        except:
            pass
        
        # Check texture variation
        try:
            contrast = np.std(pixels)
            if contrast < 10:
                return False, 0.3, "Image lacks texture variation typical of cattle"
            
            if 1.0 <= aspect_ratio <= 3.5 and contrast > 20:
                return True, 0.7, "Cattle validation passed"
            else:
                return True, 0.5, "Basic validation passed"
        except:
            if 1.0 <= aspect_ratio <= 3.5:
                return True, 0.6, "Cattle validation passed"
            else:
                return True, 0.4, "Basic validation passed"
                
    except Exception as e:
        return True, 0.4, "Basic validation passed"