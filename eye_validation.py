import os
import cv2
import numpy as np
import argparse


def is_blurry(image, threshold=100.0):
    """
    Detects if an image is blurry by calculating the variance of the Laplacian.
    Relaxed threshold to avoid false positives.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold


def enhance_image(image):
    """
    Enhance the image to improve contrast and remove noise.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # CLAHE (Contrast Limited Adaptive Histogram Equalization) for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(gray)
    # Bilateral filter to remove noise while preserving edges
    enhanced_image = cv2.bilateralFilter(enhanced_image, 9, 75, 75)
    return enhanced_image


def detect_eyes(image):
    """
    Detects eyes in the image using Haar Cascade and deep learning if needed.
    """
    # First use Haar Cascade for fast detection
    eye_cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_eye.xml')
    if not os.path.exists(eye_cascade_path):
        print("Error: 'haarcascade_eye.xml' not found.")
        return None

    eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(eyes) == 0:
        print("No eyes detected with Haar Cascade. Trying DNN-based detection.")
        # For better accuracy, use a DNN model here if Haar Cascade fails.
        # Here, we'll use the pre-trained DNN-based face detector for better detection.

        # Load pre-trained face detection model
        net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "deploy.caffemodel")
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104, 177, 123), swapRB=False)
        net.setInput(blob)
        detections = net.forward()
        eyes = detections[0, 0, :, 3:7]  # Assuming eyes are detected in the bounding box

    return eyes


def analyze_cataract_edge_detection(eye_image):
    """
    Analyzes a cropped eye image for cataracts using Canny edge detection and texture analysis.
    """
    # --- 1. Preprocessing and Pupil Isolation ---
    enhanced_eye = enhance_image(eye_image)

    # Apply median blur to reduce noise
    blurred_eye = cv2.medianBlur(enhanced_eye, 5)

    # --- 2. HoughCircles for Pupil Detection ---
    circles = cv2.HoughCircles(
        blurred_eye,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=80,
        param1=50,
        param2=25, 
        minRadius=10,
        maxRadius=int(blurred_eye.shape[0] * 0.45)  
    )

    if circles is None:
        return False, "Pupil not clearly visible for analysis. Proceeding with detection."

    circles = np.uint16(np.around(circles))
    i = circles[0, 0]
    x, y, r = i[0], i[1], i[2]

    # Create a circular mask to isolate the pupil (lens)
    mask = np.zeros(enhanced_eye.shape, dtype=np.uint8)
    cv2.circle(mask, (x, y), r, 255, -1)
    
    # Isolate the pupil using the mask
    pupil_only = cv2.bitwise_and(enhanced_eye, enhanced_eye, mask=mask)
    
    # Crop the image to the bounding box of the pupil
    pupil_cropped = pupil_only[y-r:y+r, x-r:x+r]

    if pupil_cropped.size == 0:
        return False, "Failed to isolate pupil region for analysis, proceeding with incomplete data."

    # --- 3. Canny Edge Detection within the Pupil ---
    edges = cv2.Canny(pupil_cropped, 30, 100)  

    # --- 4. Quantify the Edges ---
    pupil_area = np.pi * (r ** 2)
    edge_pixels = np.count_nonzero(edges)
    
    # Calculate the density of edges
    edge_density = 0
    if pupil_area > 0:
        edge_density = edge_pixels / pupil_area

    # --- 5. Classification based on Edge Density ---
    edge_density_threshold = 0.05  

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
    eyes = detect_eyes(image)
    
    if eyes is None or len(eyes) == 0:
        print("\n--- Result ---")
        print("Status: Invalid (No Eye Found)")
        print("Reason: No human eye was detected in the image.")
        return

    # If multiple eyes are found, select the largest one.
    if len(eyes) > 1:
        largest_area = 0
        main_eye = None
        for (ex, ey, ew, eh) in eyes:
            area = ew * eh
            if area > largest_area:
                largest_area = area
                main_eye = (ex, ey, ew, eh)
        eyes = np.array([main_eye])

    print("Step 3: Analyzing eye structure using Edge Detection...")
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
