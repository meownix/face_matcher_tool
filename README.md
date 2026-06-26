# Face Matcher Calibration Tool

This tool is a standalone utility designed to calibrate the face matching
thresholds for the `MobileFaceNet` TFLite model. It simulates the face
comparison process used in the Flutter KYC system, extracts facial embeddings,
calculates cosine similarity, and outputs an interactive HTML report where you
can tune similarity mappings and matching thresholds in real-time.

---

## Prerequisites

- **Python 3.11** (Highly recommended, as newer versions like 3.14 do not yet
  have pre-compiled binaries for packages like `tflite-runtime` or
  `tensorflow`).

---

## Installation & Setup (Virtual Environment)

1. **Navigate to the Project Directory**:
   ```bash
   cd $HOME/Dev/python/face_matcher_tool
   ```

2. **Create the Virtual Environment**:
   Initialize a Python 3.11 virtual environment named `venv`:
   ```bash
   python3.11 -m venv venv
   ```

3. **Activate the Virtual Environment**:
   To install packages and run the tool, activate the environment first:
   ```bash
   # On Linux/macOS:
   source venv/bin/activate

   # On Windows (PowerShell):
   # .\venv\Scripts\Activate.ps1

   # On Windows (CMD):
   # .\venv\Scripts\activate.bat
   ```
   *(Once activated, your terminal prompt will be prefixed with `(venv)`)*.

4. **Install Required Dependencies**:
   Install all dependencies listed in the `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```

5. **Deactivate when finished**:
   After you are done working on the project, you can exit the virtual environment by running:
   ```bash
   deactivate
   ```

---

## How to Use

### 1. Add Image Pairs to Compare
The tool scans the `samples/` directory for subdirectories containing image
pairs. Create subdirectories inside `samples/` and put **exactly two images** in
each subdirectory:

```text
samples/
├── example_pair/
│   ├── image1.jpg
│   └── image2.jpg
├── test_case_same_person/
│   ├── photo_ktp.png
│   └── photo_selfie.jpg
└── test_case_different_people/
    ├── female_ktp.jpg
    └── male_db.jpg
```

### 2. Run the Comparison Script
With the virtual environment activated, run the script:
```bash
python compare_faces.py
```
*(Or, run it directly without activating using: `venv/bin/python compare_faces.py`)*

The script will:
- Automatically detect and crop the face region using OpenCV's Haar Cascade.
- Pass the cropped faces to the `mobilefacenet.tflite` model to generate 192-dimensional embeddings.
- Compute the raw cosine similarity between the embeddings.
- Generate an interactive report under `report/report.html`.

### 3. Open the Interactive Calibration Report
Open the generated report in any web browser:
```bash
# On Linux
xdg-open report/report.html
```

Use the sliders at the top of the webpage to calibrate parameters dynamically:
* **Min Cosine Similarity (0% Match Score)**: The similarity threshold below
  which faces are considered `0%` match (default: `0.40`).
* **Perfect Cosine Similarity (100% Match Score)**: The similarity threshold
  above which faces are considered a `100%` match (default: `0.85`).
* **Pass Verification Threshold %**: The minimum match score percentage required
  to pass verification (default: `55%`).

Adjusting these sliders will instantly update the calculated match percentages
and PASS/FAIL status for all test cases.

---

## Project Structure

* `compare_faces.py` - The core face verification execution script.
* `mobilefacenet.tflite` - The TensorFlow Lite model used for extracting face embeddings.
* `samples/` - The directory containing subfolders of image pairs to compare.
* `report/` - Contains the generated `report.html` and the cropped face assets.
* `venv/` - The Python virtual environment holding installed packages.
