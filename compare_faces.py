import os
import sys
import shutil
import json
import math

# Check for dependencies
try:
    import cv2
    import numpy as np
    try:
        import tflite_runtime.interpreter as tflite
    except ImportError:
        import tensorflow.lite as tflite
except ImportError as e:
    print(f"Error: Missing required library: {e.name}")
    print("\nPlease install the dependencies using the following command:")
    print("pip install opencv-python numpy tflite-runtime")
    sys.exit(1)

# Paths relative to the script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "mobilefacenet.tflite")
SAMPLES_DIR = os.path.join(SCRIPT_DIR, "samples")
REPORT_DIR = os.path.join(SCRIPT_DIR, "report")
REPORT_ASSETS_DIR = os.path.join(REPORT_DIR, "assets")

# Ensure directories exist
os.makedirs(SAMPLES_DIR, exist_ok=True)
os.makedirs(REPORT_ASSETS_DIR, exist_ok=True)

class FaceInference:
    def __init__(self, model_path):
        print(f"Loading TFLite model from {model_path}...")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Load Haar Cascade for face detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
    def detect_and_crop_face(self, img_path, output_crop_path):
        """Detects a face in the image, crops it, and saves the cropped image."""
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not read image: {img_path}")
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) > 0:
            # Crop the first detected face with a small padding (15%)
            x, y, w, h = faces[0]
            height, width, _ = img.shape
            
            pad_w = int(w * 0.15)
            pad_h = int(h * 0.15)
            
            x1 = max(0, x - pad_w)
            y1 = max(0, y - pad_h)
            x2 = min(width, x + w + pad_w)
            y2 = min(height, y + h + pad_h)
            
            cropped = img[y1:y2, x1:x2]
            cv2.imwrite(output_crop_path, cropped)
            return cropped
        else:
            # If no face is detected, copy original as fallback crop
            shutil.copy2(img_path, output_crop_path)
            return img

    def get_embedding(self, face_img):
        """Resizes, normalizes, and extracts face embedding vector."""
        # Convert BGR to RGB
        rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        # Resize to 112x112 (MobileFaceNet input dimensions)
        resized = cv2.resize(rgb, (112, 112))
        
        # Normalize to [-1.0, 1.0] matching Dart's (pixel - 127.5) / 127.5
        input_data = (resized.astype(np.float32) - 127.5) / 127.5
        input_data = np.expand_dims(input_data, axis=0) # [1, 112, 112, 3]
        
        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        
        embedding = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        return embedding

def calculate_cosine_similarity(e1, e2):
    dot_product = np.dot(e1, e2)
    norm_a = np.linalg.norm(e1)
    norm_b = np.linalg.norm(e2)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot_product / (norm_a * norm_b)

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        print("Please ensure mobilefacenet.tflite is in the same directory as this script.")
        sys.exit(1)
        
    # Check if there are sample directories
    subdirs = [d for d in os.listdir(SAMPLES_DIR) if os.path.isdir(os.path.join(SAMPLES_DIR, d))]
    
    if not subdirs:
        print("="*60)
        print("No sample image pairs found!")
        print(f"Please create subdirectories inside:\n  {SAMPLES_DIR}/")
        print("\nFor example:")
        print(f"  mkdir -p {SAMPLES_DIR}/case_1_different")
        print(f"  mkdir -p {SAMPLES_DIR}/case_2_same")
        print("\nThen place 2 images to compare in each directory.")
        print("="*60)
        
        # Create a mock structure so the user can easily start
        mock_dir = os.path.join(SAMPLES_DIR, "example_pair")
        os.makedirs(mock_dir, exist_ok=True)
        print(f"\nCreated an empty example directory at: {mock_dir}")
        print("Place your images there and run this script again.")
        sys.exit(0)

    try:
        infer = FaceInference(MODEL_PATH)
    except Exception as e:
        print(f"Failed to initialize model interpreter: {e}")
        sys.exit(1)

    results = []
    
    for case_id, case_name in enumerate(sorted(subdirs)):
        case_path = os.path.join(SAMPLES_DIR, case_name)
        # Find images in the directory
        valid_exts = ('.jpg', '.jpeg', '.png', '.webp')
        images = sorted([f for f in os.listdir(case_path) if f.lower().endswith(valid_exts)])
        
        if len(images) < 2:
            print(f"Skipping directory '{case_name}': needs at least 2 images (found {len(images)}).")
            continue
            
        img1_name, img2_name = images[0], images[1]
        img1_path = os.path.join(case_path, img1_name)
        img2_path = os.path.join(case_path, img2_name)
        
        # Define destination paths in report_assets
        dest_img1 = f"case_{case_id}_1_{img1_name}"
        dest_img2 = f"case_{case_id}_2_{img2_name}"
        dest_crop1 = f"case_{case_id}_1_crop.jpg"
        dest_crop2 = f"case_{case_id}_2_crop.jpg"
        
        shutil.copy2(img1_path, os.path.join(REPORT_ASSETS_DIR, dest_img1))
        shutil.copy2(img2_path, os.path.join(REPORT_ASSETS_DIR, dest_img2))
        
        print(f"\nProcessing Case: {case_name}")
        try:
            # Crop faces
            crop1 = infer.detect_and_crop_face(img1_path, os.path.join(REPORT_ASSETS_DIR, dest_crop1))
            crop2 = infer.detect_and_crop_face(img2_path, os.path.join(REPORT_ASSETS_DIR, dest_crop2))
            
            # Extract embeddings
            e1 = infer.get_embedding(crop1)
            e2 = infer.get_embedding(crop2)
            
            # Compute cosine similarity
            similarity = float(calculate_cosine_similarity(e1, e2))
            print(f"  -> Cosine Similarity: {similarity:.4f}")
            
            results.append({
                "id": f"case_{case_id}",
                "name": case_name,
                "img1": f"assets/{dest_img1}",
                "img2": f"assets/{dest_img2}",
                "crop1": f"assets/{dest_crop1}",
                "crop2": f"assets/{dest_crop2}",
                "similarity": similarity
            })
        except Exception as e:
            print(f"  -> Error processing: {e}")
            
    # Generate HTML file
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MobileFaceNet Similarity Calibration Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: #121214;
            color: #e4e4e7;
            margin: 0;
            padding: 24px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1 {{
            font-size: 28px;
            margin-bottom: 8px;
            color: #ffffff;
            text-align: center;
        }}
        .subtitle {{
            text-align: center;
            color: #a1a1aa;
            margin-bottom: 32px;
        }}
        .control-panel {{
            background: #1e1e24;
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 32px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
            border: 1px solid #2d2d34;
        }}
        .control-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 24px;
        }}
        .slider-group {{
            display: flex;
            flex-direction: column;
        }}
        .slider-group label {{
            font-size: 14px;
            font-weight: 600;
            margin-bottom: 8px;
            color: #a1a1aa;
            display: flex;
            justify-content: space-between;
        }}
        .slider-group label span {{
            color: #60a5fa;
            font-family: monospace;
        }}
        input[type="range"] {{
            -webkit-appearance: none;
            width: 100%;
            height: 6px;
            background: #2d2d34;
            border-radius: 3px;
            outline: none;
        }}
        input[type="range"]::-webkit-slider-thumb {{
            -webkit-appearance: none;
            width: 18px;
            height: 18px;
            border-radius: 50%;
            background: #3b82f6;
            cursor: pointer;
            transition: transform 0.1s;
        }}
        input[type="range"]::-webkit-slider-thumb:hover {{
            transform: scale(1.2);
        }}
        .case-card {{
            background: #1e1e24;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 24px;
            border: 1px solid #2d2d34;
            transition: border-color 0.2s, box-shadow 0.2s;
        }}
        .case-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #2d2d34;
            padding-bottom: 12px;
            margin-bottom: 16px;
        }}
        .case-title {{
            font-size: 18px;
            font-weight: bold;
            color: #ffffff;
        }}
        .status-badge {{
            padding: 6px 12px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 12px;
            text-transform: uppercase;
        }}
        .status-badge.pass {{
            background-color: rgba(16, 185, 129, 0.2);
            color: #10b981;
            border: 1px solid rgba(16, 185, 129, 0.4);
        }}
        .status-badge.fail {{
            background-color: rgba(239, 68, 68, 0.2);
            color: #ef4444;
            border: 1px solid rgba(239, 68, 68, 0.4);
        }}
        .case-card.pass-card {{
            border-color: rgba(16, 185, 129, 0.3);
        }}
        .case-card.fail-card {{
            border-color: rgba(239, 68, 68, 0.3);
        }}
        .comparison-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            margin-bottom: 16px;
        }}
        .img-container {{
            background: #121214;
            border-radius: 8px;
            padding: 12px;
            text-align: center;
            border: 1px solid #2d2d34;
        }}
        .img-container img {{
            max-width: 100%;
            max-height: 180px;
            border-radius: 6px;
            object-fit: contain;
        }}
        .img-label {{
            font-size: 12px;
            color: #a1a1aa;
            margin-top: 8px;
            display: block;
        }}
        .results-row {{
            display: flex;
            justify-content: space-around;
            background: #121214;
            border-radius: 8px;
            padding: 16px;
            border: 1px solid #2d2d34;
        }}
        .metric-box {{
            text-align: center;
        }}
        .metric-label {{
            font-size: 12px;
            color: #a1a1aa;
            margin-bottom: 4px;
        }}
        .metric-value {{
            font-size: 20px;
            font-weight: bold;
            color: #ffffff;
            font-family: monospace;
        }}
        .metric-value.highlight {{
            color: #60a5fa;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Face Verification Threshold Calibration</h1>
        <div class="subtitle">Interactively tune the matching parameters using samples from <code>{SAMPLES_DIR}</code></div>
        
        <div class="control-panel">
            <div class="control-grid">
                <div class="slider-group">
                    <label>Min Cosine Similarity (0% Match Score): <span id="minSimVal">0.40</span></label>
                    <input type="range" id="minSimSlider" min="0.0" max="0.9" step="0.05" value="0.40" oninput="updateScores()">
                </div>
                <div class="slider-group">
                    <label>Perfect Cosine Similarity (100% Match Score): <span id="maxSimVal">0.85</span></label>
                    <input type="range" id="maxSimSlider" min="0.5" max="1.0" step="0.05" value="0.85" oninput="updateScores()">
                </div>
                <div class="slider-group">
                    <label>Pass Verification Threshold %: <span id="thresholdVal">55%</span></label>
                    <input type="range" id="thresholdSlider" min="10" max="95" step="5" value="55" oninput="updateScores()">
                </div>
            </div>
        </div>

        <div id="cases-container">
            <!-- Dynamic cases rendered here -->
        </div>
    </div>

    <script>
        const cases = {json.dumps(results)};
        
        function updateScores() {{
            const minSim = parseFloat(document.getElementById('minSimSlider').value);
            const maxSim = parseFloat(document.getElementById('maxSimSlider').value);
            const threshold = parseFloat(document.getElementById('thresholdSlider').value);
            
            document.getElementById('minSimVal').innerText = minSim.toFixed(2);
            document.getElementById('maxSimVal').innerText = maxSim.toFixed(2);
            document.getElementById('thresholdVal').innerText = threshold.toFixed(0) + '%';
            
            const container = document.getElementById('cases-container');
            container.innerHTML = '';
            
            cases.forEach(c => {{
                // Calculate new match percentage based on clamped formula
                let pct = 0.0;
                if (c.similarity > minSim) {{
                    if (c.similarity >= maxSim) {{
                        pct = 100.0;
                    }} else {{
                        pct = ((c.similarity - minSim) / (maxSim - minSim)) * 100.0;
                    }}
                }}
                
                // Format decimal
                pct = Math.round(pct * 10) / 10;
                const passed = pct >= threshold;
                
                // Render HTML for card
                const card = document.createElement('div');
                card.className = `case-card ${{passed ? 'pass-card' : 'fail-card'}}`;
                
                card.innerHTML = `
                    <div class="case-header">
                        <div class="case-title">${{c.name}}</div>
                        <div class="status-badge ${{passed ? 'pass' : 'fail'}}">${{passed ? 'PASSED' : 'FAILED'}}</div>
                    </div>
                    <div class="comparison-grid">
                        <div class="img-container">
                            <img src="${{c.img1}}" alt="Photo 1">
                            <span class="img-label">Photo 1 (Original)</span>
                        </div>
                        <div class="img-container">
                            <img src="${{c.crop1}}" alt="Face 1">
                            <span class="img-label">Face 1 (Cropped)</span>
                        </div>
                        <div class="img-container">
                            <img src="${{c.img2}}" alt="Photo 2">
                            <span class="img-label">Photo 2 (Original)</span>
                        </div>
                        <div class="img-container">
                            <img src="${{c.crop2}}" alt="Face 2">
                            <span class="img-label">Face 2 (Cropped)</span>
                        </div>
                    </div>
                    <div class="results-row">
                        <div class="metric-box">
                            <div class="metric-label">Cosine Similarity</div>
                            <div class="metric-value highlight">${{c.similarity.toFixed(4)}}</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-label">Calculated Match Score</div>
                            <div class="metric-value" style="color: ${{passed ? '#10b981' : '#ef4444'}}">${{pct.toFixed(1)}}%</div>
                        </div>
                    </div>
                `;
                
                container.appendChild(card);
            }});
        }}
        
        // Initial render
        updateScores();
    </script>
</body>
</html>
"""

    with open(os.path.join(REPORT_DIR, "report.html"), "w") as f:
        f.write(html_content)
        
    print("\n" + "="*60)
    print("HTML Report Generated Successfully!")
    print(f"Report location:\n  {os.path.join(REPORT_DIR, 'report.html')}")
    print("\nOpen this file in a web browser to interactively adjust and tune the threshold parameters.")
    print("="*60)

if __name__ == "__main__":
    main()
