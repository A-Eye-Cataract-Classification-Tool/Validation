# Cataract Detection using Computer Vision 👁️ (A-Eye model Validation)

This code uses OpenCV to analyze eye images and automatically detect the presence of cataracts.

The script processes an image by locating the eye, isolating the pupil, and then analyzing features like intensity, texture, and opacity to make a diagnosis.

---
## Features
* **Eye Detection**: Automatically locates the eye region in a photo.
* **Pupil Isolation**: Precisely detects the pupil for a focused analysis.
* **Feature Analysis**: Measures key indicators within the pupil.
* **Classification**: Uses a weighted scoring system to determine if a cataract is present.
* **Visualization**: Can generate an output image with the analysis results overlaid for easy review.

---
## Setup ⚙️

### Prerequisites
* Python 3.6 or newer

### Installation
1.  Clone or download the project files.
2.  Open your terminal or command prompt and navigate to the project directory.
3.  Install the required Python libraries using pip:
    ```bash
    pip install opencv-python numpy
    ```

---
## How to Run the Script 🚀

1.  Place all the eye images you want to analyze inside the **`test_inputs`** folder.
2.  From your terminal, run the script using one of the commands below.

### Example 1: Basic Analysis
This command will run the analysis on a single image and print the diagnosis directly to your terminal.

**Replace `your_image_name.jpg` with the actual filename.**

```bash
python eye_validation.py test_inputs/your_image_name.jpg
```

### Example 2: Analysis with Visualization
Use the --visualize flag to generate an annotated output image. The resulting image will be saved automatically inside the analysis_outputs folder.

```bash
python eye_validation.py test_inputs/your_image_name.jpg --visualize
```