import os
import cv2
import numpy as np
import argparse


def is_blurry(image, threshold=100.0):
    """
    Detects if an image is blurry by calculating the variance of the Laplacian.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold

def analyze_cataract_edge_detection(eye_image):
    """
    Analyzes a cropped eye image for cataracts using Canny edge detection
    to find structural formations within the lens.
    """
    # --- 1. Preprocessing and Pupil Isolation ---
    gray_eye = cv2.cvtColor(eye_image, cv2.COLOR_BGR2GRAY)
    # Use a median blur which is effective at removing noise while preserving edges
    blurred_eye = cv2.medianBlur(gray_eye, 5)

    circles = cv2.HoughCircles(
        blurred_eye,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=100,
        param1=50,
        param2=30,
        minRadius=10,
        maxRadius=int(gray_eye.shape[0] * 0.45) # Increased radius slightly
    )

    if circles is None:
        return False, "Pupil not clearly visible for analysis."

    circles = np.uint16(np.around(circles))
    i = circles[0, 0]
    x, y, r = i[0], i[1], i[2]

    # Create a circular mask to isolate the pupil (lens)
    mask = np.zeros(gray_eye.shape, dtype=np.uint8)
    cv2.circle(mask, (x, y), r, 255, -1)
    
    # Isolate the pupil using the mask
    pupil_only = cv2.bitwise_and(gray_eye, gray_eye, mask=mask)
    
    # Crop the image to the bounding box of the pupil
    pupil_cropped = pupil_only[y-r:y+r, x-r:x+r]

    if pupil_cropped.size == 0:
        return False, "Failed to isolate pupil region for analysis."

    # --- 2. Canny Edge Detection within the Pupil ---
    # We use a relatively low threshold to be sensitive to faint edges from immature cataracts
    edges = cv2.Canny(pupil_cropped, 50, 150)

    # --- 3. Quantify the Edges ---
    # Calculate the total number of pixels in the pupil (area of the circle)
    pupil_area = np.pi * (r ** 2)
    # Count the number of white pixels (the edges)
    edge_pixels = np.count_nonzero(edges)
    
    # Calculate the density of edges
    edge_density = 0
    if pupil_area > 0:
        edge_density = edge_pixels / pupil_area

    # --- 4. Classification based on Edge Density ---
    # A healthy lens is uniform and has very low edge density.
    # A cataractous lens has structures that create a higher edge density.
    # We set a threshold that is sensitive enough for immature cases.
    edge_density_threshold = 0.08 # Calibrated threshold

    reason = f"Analysis results:\n - Pupil Area: {pupil_area:.2f} pixels\n - Edge Pixels: {edge_pixels}\n - Edge Density: {edge_density:.4f}"

    if edge_density > edge_density_threshold:
        return True, "Cataract Detected. " + reason
    else:
        return False, "No Cataract Detected. " + reason


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
        print("Status: Invalid (Blurry Image)")
        print("Reason: Image is too blurry or out of focus.")
        return

    print("Step 2: Detecting eyes...")
    try:
        eye_cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_eye.xml')
        if not os.path.exists(eye_cascade_path):
             print(f"\nError: Could not find 'haarcascade_eye.xml'. Please ensure OpenCV is installed correctly.")
             return
        eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
    except Exception as e:
         print(f"\nError: Could not load eye detector model. {e}")
         return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # --- MODIFICATION FOR MULTIPLE EYE DETECTION ---
    print(f"Step 3: Validating number of eyes... Found {len(eyes)} potential eye region(s).")
    
    if len(eyes) == 0:
        print("\n--- Result ---")
        print("Status: Invalid (No Eye Found)")
        print("Reason: No human eye was detected in the image.")
        return
    
    # If multiple "eyes" are found, choose the largest one.
    if len(eyes) > 1:
        print("Multiple regions found. Selecting the largest one as the primary eye.")
        largest_area = 0
        main_eye = None
        for (ex, ey, ew, eh) in eyes:
            area = ew * eh
            if area > largest_area:
                largest_area = area
                main_eye = (ex, ey, ew, eh)
        # We now have our single, most likely eye.
        eyes = np.array([main_eye])


    print("Step 4: Analyzing eye structure using Edge Detection...")
    x, y, w, h = eyes[0]
    eye_roi = image[y:y+h, x:x+w]

    is_cataract, analysis_reason = analyze_cataract_edge_detection(eye_roi)

    print("\n--- Result ---")
    if is_cataract:
        print("Status: Valid (Cataract Detected)")
    else:
        print("Status: Invalid (No Cataract Detected)")
    
    print(f"Reason: {analysis_reason}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Analyze an eye image for cataracts using edge detection."
    )
    parser.add_argument(
        "image_path",
        type=str,
        help="The full path to the eye image file."
    )
    args = parser.parse_args()
    analyze_image_cli(args.image_path)
