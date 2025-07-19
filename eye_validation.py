# main_cli.py
import os
import cv2
import numpy as np
import argparse

# --- Utility Functions ---

def is_blurry(image, threshold=100.0):
    """
    Detects if an image is blurry by calculating the variance of the Laplacian.
    A low variance suggests the image is blurry.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold

def analyze_cataract(eye_image):
    """
    Analyzes a cropped eye image for cataracts.
    This version includes logic to remove specular reflections (light glare)
    before analysis to prevent false positives.
    """
    gray_eye = cv2.cvtColor(eye_image, cv2.COLOR_BGR2GRAY)
    
    # Use a bilateral filter to reduce noise while keeping edges sharp.
    # This helps in more accurate circle detection.
    blurred_eye = cv2.bilateralFilter(gray_eye, 9, 75, 75)

    circles = cv2.HoughCircles(
        blurred_eye,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=100,
        param1=50,
        param2=30,
        minRadius=10,
        maxRadius=int(gray_eye.shape[0] * 0.4)
    )

    if circles is None:
        return False, "Pupil not clearly visible."

    circles = np.uint16(np.around(circles))
    i = circles[0, 0]
    x, y, r = i[0], i[1], i[2]
    
    # --- SPECULAR REFLECTION REMOVAL ---
    # Isolate the pupil region from the grayscale eye image
    pupil_roi = gray_eye[y-r:y+r, x-r:x+r]
    
    # Create a mask to identify extremely bright spots (likely reflections)
    # We use a high threshold, as reflections are much brighter than cataracts.
    _, reflection_mask = cv2.threshold(pupil_roi, 230, 255, cv2.THRESH_BINARY)
    
    # Dilate the reflection mask slightly to ensure the entire reflection is covered
    kernel = np.ones((3,3), np.uint8)
    reflection_mask = cv2.dilate(reflection_mask, kernel, iterations=1)

    # Inpaint the original pupil region using the reflection mask.
    # This "heals" the area of the reflection, filling it with information
    # from the surrounding pixels.
    healed_pupil = cv2.inpaint(pupil_roi, reflection_mask, 3, cv2.INPAINT_TELEA)
    # --- END OF REFLECTION REMOVAL ---

    # Create a circular mask for the pupil to ensure we only analyze pixels within it.
    circular_mask = np.zeros(healed_pupil.shape, dtype=np.uint8)
    cv2.circle(circular_mask, (r, r), r, 255, -1)

    # Calculate statistics on the "healed" pupil image
    mean, std_dev = cv2.meanStdDev(healed_pupil, mask=circular_mask)
    mean_intensity = mean[0][0]
    std_dev_intensity = std_dev[0][0]

    # --- Cataract Heuristics (applied on the cleaned image) ---
    mature_cataract_threshold = 95.0
    immature_cataract_std_dev_threshold = 15.0

    if mean_intensity > mature_cataract_threshold:
        return True, f"Potential mature cataract detected. Pupil is unusually bright (Intensity: {mean_intensity:.2f})."
    
    if std_dev_intensity > immature_cataract_std_dev_threshold:
        return True, f"Potential immature cataract detected. Pupil shows non-uniform texture (Std Dev: {std_dev_intensity:.2f})."

    return False, f"No significant cataract detected. Pupil appears clear and uniform (Intensity: {mean_intensity:.2f}, Std Dev: {std_dev_intensity:.2f})."

def analyze_image_cli(image_path):
    """
    Handles the image analysis logic for the command-line interface.
    """
    if not os.path.exists(image_path):
        print(f"Error: File not found at '{image_path}'")
        return

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image file at '{image_path}'.")
        return

    print("--- Starting Analysis ---")

    max_width = 1024
    height, width, _ = image.shape
    if width > max_width:
        print(f"Image is large ({width}x{height}). Resizing to {max_width}px width...")
        ratio = max_width / float(width)
        new_height = int(height * ratio)
        image = cv2.resize(image, (max_width, new_height), interpolation=cv2.INTER_AREA)
        print("Resizing complete.")

    print("Step 1: Checking for blurriness...")
    if is_blurry(image):
        print("\n--- Result ---")
        print("Status: Invalid")
        print("Reason: Image is too blurry or out of focus.")
        return

    print("Step 2: Detecting eyes...")
    try:
        eye_cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_eye.xml')
        if not os.path.exists(eye_cascade_path):
             print(f"\nError: Could not find 'haarcascade_eye.xml'. Please ensure OpenCV is installed correctly. Expected at: {eye_cascade_path}")
             return
        eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
    except Exception as e:
         print(f"\nError: Could not load eye detector model. {e}")
         return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # The Haar cascade is generally robust to moderate rotation.
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    print(f"Step 3: Validating number of eyes... Found {len(eyes)} eye(s).")
    if len(eyes) == 0:
        print("\n--- Result ---")
        print("Status: Invalid")
        print("Reason: No human eye was detected in the image.")
        return
    if len(eyes) > 1:
        print("\n--- Result ---")
        print("Status: Invalid")
        print("Reason: More than one eye was detected. Please upload a photo of a single eye.")
        return

    print("Step 4: Analyzing detected eye for cataracts (with reflection removal)...")
    x, y, w, h = eyes[0]
    eye_roi = image[y:y+h, x:x+w]

    is_cataract, analysis_reason = analyze_cataract(eye_roi)

    print("\n--- Result ---")
    if is_cataract:
        print("Status: Valid")
        print(f"Reason: {analysis_reason}")
    else:
        print("Status: Invalid")
        print(f"Reason: {analysis_reason}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Analyze an eye image for cataracts from the command line."
    )
    parser.add_argument(
        "image_path",
        type=str,
        help="The full path to the eye image file (e.g., C:/Users/YourUser/Pictures/eye.jpg)"
    )
    args = parser.parse_args()
    analyze_image_cli(args.image_path)
