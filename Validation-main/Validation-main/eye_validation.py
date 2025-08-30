import os
import cv2
import numpy as np
import argparse
import math

# --- Pre-processing and Utility Functions (Largely Unchanged) ---
def is_blurry(image, threshold=60.0):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold

def enhance_image(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced_bgr = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    gray_enhanced = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY)
    filtered_image = cv2.bilateralFilter(gray_enhanced, 9, 75, 75)
    return filtered_image, enhanced_bgr

def remove_reflections(image):
    _, thresh = cv2.threshold(image, 225, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(thresh, kernel, iterations=2)
    inpainted_image = cv2.inpaint(image, mask, 5, cv2.INPAINT_TELEA)
    return inpainted_image

# --- Detection and Analysis Pipeline ---

def detect_eyes(image):
    """
    Detects the eye and returns BOTH the original and padded coordinates.
    This allows us to use the original size for calculations and the padded
    size for creating the visual ROI.
    """
    eye_cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_eye.xml')
    if not os.path.exists(eye_cascade_path):
        return None, None
    eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(50, 50))
    if len(eyes) == 0:
        return None, None
        
    x, y, w, h = max(eyes, key=lambda rect: rect[2] * rect[3])
    original_coords = (x, y, w, h)
    
    # Add padding for a better visual ROI
    padding_w = int(w * 0.15)
    padding_h = int(h * 0.15)
    img_h, img_w = image.shape[:2]
    x1 = max(0, x - padding_w)
    y1 = max(0, y - padding_h)
    x2 = min(img_w, x + w + padding_w)
    y2 = min(img_h, y + h + padding_h)
    padded_coords = (x1, y1, x2 - x1, y2 - y1)
    
    return original_coords, padded_coords

def find_pupil(eye_image_gray, original_eye_width):
    """
    ## REVISED ##: A much more robust pupil detector using a scoring system.
    It now evaluates candidate circles based on a combination of their
    centrality, and most importantly, the darkness of their interior.
    This prevents it from locking onto bright reflections or the iris.
    """
    # 1. Pre-processing to reduce noise and reflections
    # Morphological opening removes small bright spots (like reflections)
    kernel = np.ones((5,5),np.uint8)
    processed_eye = cv2.morphologyEx(eye_image_gray, cv2.MORPH_OPEN, kernel)
    processed_eye = cv2.bilateralFilter(processed_eye, 10, 50, 50)

    # 2. Hough Circle Transform (parameters are stable)
    min_rad = int(original_eye_width * 0.08)
    max_rad = int(original_eye_width * 0.30)
    
    circles = cv2.HoughCircles(
        processed_eye, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
        param1=50, param2=30, minRadius=min_rad, maxRadius=max_rad
    )
    
    # 3. ## NEW ## Scoring and Selection Logic
    if circles is not None:
        best_circle = None
        min_score = float('inf')
        eye_center = (processed_eye.shape[1] // 2, processed_eye.shape[0] // 2)

        for c in circles[0, :]:
            x, y, r = int(c[0]), int(c[1]), int(c[2])
            
            # Create a mask for the current circle to analyze its interior
            mask = np.zeros(processed_eye.shape, dtype=np.uint8)
            cv2.circle(mask, (x, y), r, 255, -1)
            
            # --- Calculate Score ---
            # a) Darkness Score: Lower average intensity is better.
            if np.sum(mask) == 0: continue
            avg_intensity = np.mean(processed_eye[mask == 255])
            
            # b) Centrality Score: Lower distance from the center is better.
            dist_from_center = np.sqrt((x - eye_center[0])**2 + (y - eye_center[1])**2)
            
            # Combine scores: We want to minimize both.
            # We penalize distance from center and high brightness.
            score = dist_from_center + (avg_intensity * 1.5)

            if score < min_score:
                min_score = score
                best_circle = (x, y, r)

        if best_circle is not None:
            x, y, r = best_circle
            final_mask = np.zeros(processed_eye.shape, dtype=np.uint8)
            cv2.circle(final_mask, (x, y), r, 255, -1)
            return final_mask, (x, y, r)

    # Fallback if no circles are found
    return None, None

def analyze_cataract_features(pupil_pixels_gray, pupil_pixels_color, pupil_area, edges):
    """ Feature extraction remains the same. """
    if pupil_pixels_gray.size < 100: return None
    avg_intensity = np.mean(pupil_pixels_gray)
    texture_std_dev = np.std(pupil_pixels_gray)
    edge_pixels = np.count_nonzero(edges)
    edge_density = edge_pixels / pupil_area if pupil_area > 0 else 0
    opacity_pixels = np.sum(pupil_pixels_gray > 190)
    opacity_ratio = opacity_pixels / pupil_pixels_gray.size
    lab_pixels = cv2.cvtColor(pupil_pixels_color.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB)
    avg_b_channel = np.mean(lab_pixels[:, 0, 2])
    return {'intensity': avg_intensity, 'texture': texture_std_dev, 'edge_density': edge_density, 'opacity_ratio': opacity_ratio, 'yellowness': avg_b_channel}

def classify_cataract(features):
    """
    ## REVISED ##: A more robust and sensitive scoring system.
    This new logic is better at detecting clear cataracts and less prone
    to false negatives. It uses a weighted score instead of rigid rules.
    """
    if features is None:
        return False, "Analysis failed: Could not extract features.", []

    # --- Weighted Scoring System ---
    # A healthy pupil is dark (intensity < 60) and has low texture (< 15).
    # A cataract makes the pupil brighter and increases texture.
    score = 0
    reasons = []

    # 1. Intensity Score (heavily weighted)
    # This is the most reliable indicator.
    intensity = features['intensity']
    if intensity > 100:
        score += (intensity - 100) / 20.0  # Add 1 point for every 20 units over 100
        reasons.append(f"High Intensity ({intensity:.1f})")

    # 2. Texture Score
    texture = features['texture']
    if texture > 20:
        score += (texture - 20) / 10.0 # Add 1 point for every 10 units over 20
        reasons.append(f"High Texture ({texture:.1f})")

    # 3. Opacity Score (for very dense cataracts)
    opacity = features['opacity_ratio']
    if opacity > 0.1:
        score += opacity * 15 # Add up to 1.5 points for opacity
        reasons.append(f"High Opacity ({opacity:.2f})")

    # --- Final Decision ---
    # A score of 1.5 or higher is a strong indication of a cataract.
    is_cataract = score >= 1.5
    
    full_reason = "Analysis Results:\n"
    for key, val in features.items():
        full_reason += f" - {key.replace('_', ' ').title()}: {val:.2f}\n"
    full_reason += f"\nCalculated Score: {score:.2f} (Threshold: >= 1.5)"
    full_reason += f"\nDetected Indicators: {', '.join(reasons) if reasons else 'None'}"
    
    return is_cataract, full_reason, reasons

def visualize_analysis(image, eye_coords, pupil_circle, is_cataract, reasons):
    """ Visualization function remains the same. """
    output_image = image.copy()
    if eye_coords is not None:
        x, y, w, h = eye_coords
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if pupil_circle is not None:
            px, py, pr = pupil_circle
            full_px, full_py = px + x, py + y
            cv2.circle(output_image, (full_px, full_py), pr, (255, 0, 0), 2)
    status_text = "Cataract Detected" if is_cataract else "No Cataract Detected"
    color = (0, 0, 255) if is_cataract else (0, 255, 0)
    cv2.putText(output_image, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    if reasons:
       cv2.putText(output_image, reasons[0], (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return output_image

# --- Main Execution Logic ---
def analyze_image_cli(image_path, visualize=False):
    original_image = cv2.imread(image_path)
    if original_image is None: return
    
    # Resize
    max_width = 800
    h, w = original_image.shape[:2]
    if w > max_width:
        r = max_width / float(w)
        original_image = cv2.resize(original_image, (max_width, int(h * r)), interpolation=cv2.INTER_AREA)

    if is_blurry(original_image):
        print("Result: Invalid (Blurry Image)")
        return

    # ## REVISED LOGIC ##
    # Get both original and padded eye coordinates
    original_eye_coords, padded_eye_coords = detect_eyes(original_image)
    if original_eye_coords is None:
        print("Result: Invalid (No Eye Found)")
        return
        
    # Use the padded coordinates to define the ROI for processing
    px, py, pw, ph = padded_eye_coords
    eye_roi_color_full = original_image[py:py+ph, px:px+pw]
    
    enhanced_eye_gray, enhanced_eye_color = enhance_image(eye_roi_color_full)
    clean_eye = remove_reflections(enhanced_eye_gray)

    # ## REVISED LOGIC ##
    # Pass the ORIGINAL eye width to the pupil finder
    _, _, original_w, _ = original_eye_coords
    pupil_mask, pupil_circle = find_pupil(clean_eye, original_w)
    
    if pupil_mask is None:
        print("Result: Invalid (Pupil Not Found)")
        return

    pupil_pixels_gray = clean_eye[pupil_mask == 255]
    pupil_pixels_color = enhanced_eye_color[pupil_mask == 255]
    _, _, r = pupil_circle
    pupil_area = np.pi * (r ** 2)
    edges_in_pupil = cv2.bitwise_and(cv2.Canny(clean_eye, 50, 150), pupil_mask)
    
    features = analyze_cataract_features(pupil_pixels_gray, pupil_pixels_color, pupil_area, edges_in_pupil)
    is_cataract, analysis_reason, reasons_list = classify_cataract(features)

    print("\n--- Final Diagnosis ---")
    print(f"Status: {'Cataract Detected' if is_cataract else 'No Cataract Detected'}")
    print(f"Reasoning:\n{analysis_reason}")

    if visualize:
        # Define the name for the output directory 📁
        output_dir = "analysis_outputs"
        
        # Create the directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create the visualization image
        vis_image = visualize_analysis(original_image, padded_eye_coords, pupil_circle, is_cataract, reasons_list)
        
        # --- Create the new output path ---
        # 1. Get the base filename from the input path (e.g., "image_d28700.jpg")
        base_filename = os.path.basename(image_path)
        
        # 2. Get the name without the extension (e.g., "image_d28700")
        name_only = os.path.splitext(base_filename)[0]
        
        # 3. Create the full path for the new file inside the output directory
        output_path = os.path.join(output_dir, f"{name_only}_analysis.jpg")
        
        # Save the image to the new path
        cv2.imwrite(output_path, vis_image)
        print(f"\nSaved visualization to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="An improved tool to analyze eye images for cataracts.")
    parser.add_argument("image_path", type=str, help="The path to the eye image file.")
    parser.add_argument("--visualize", action="store_true", help="Generate and save an image with analysis overlays.")
    args = parser.parse_args()
    analyze_image_cli(args.image_path, args.visualize)