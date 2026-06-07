# PCB Defect Analyzer (Anomaly Detection System)

This project is a high-performance visual inspection system designed to detect manufacturing defects in printed circuit boards (PCBs). By comparing a test target board against a defect-free reference master board, it automatically identifies, maps, and segments defect areas in real-time.

---

## 1. How It Works (Methodology)

### A. Deep Feature Extraction
The system utilizes a pre-trained **Wide ResNet-50-2** convolutional neural network as a feature extractor.
- Traditional pixel-level subtraction is highly sensitive to minor lighting, alignment, and noise variations.
- Instead, this system extracts high-dimensional, mid-level representative feature maps from different layers of the network:
  - **Layer 1**: Captures fine-grained local textures and edge details.
  - **Layer 2**: Captures structural shapes and trace patterns.
  - **Layer 3**: Captures complex layout combinations and spatial relationships.
- Registration hooks are attached to the end of each of these layers (`layer1[-1]`, `layer2[-1]`, `layer3[-1]`) to grab intermediate activations during a single forward pass.

### B. Feature Alignment & Concatenation
Since activations from deeper layers have lower spatial resolution due to pooling and strided convolutions, the feature maps are aligned:
1. Feature maps from `layer2` and `layer3` are spatially upsampled using **bilinear interpolation** to match the size of `layer1`.
2. The aligned feature maps are concatenated along the channel dimension to form a unified multiscale embedding.

### C. Anomaly Distance Map
To compute the differences between the Reference Master ($R$) and Target Sample ($T$):
1. The distance between corresponding spatial embeddings is calculated using the L2 norm (Euclidean distance):
   $$\text{Difference}(x, y) = \sqrt{\sum (R_{x, y} - T_{x, y})^2}$$
2. The resulting difference map is upsampled to the original input resolution ($224 \times 224$).
3. A **Gaussian Filter** ($\sigma = 4$) is applied to smooth out high-frequency noise and model local spatial variations.

### D. Contour Detection & Segmentation
1. The smoothed anomaly map is normalized to a $[0, 255]$ range to create a grayscale heatmap.
2. An adjustable sensitivity threshold is applied to convert the heatmap into a binary mask.
3. OpenCV's contour finding algorithm (`cv2.findContours`) isolates disconnected defect clusters.
4. Bounding boxes are computed around any contours exceeding the minimum area threshold and drawn on the output image.

---

## 2. Tech Stack & Architecture

- **Backend**: 
  - `FastAPI`: High-performance, modern Python web framework.
  - `PyTorch` & `Torchvision`: Core deep learning framework loading Wide ResNet-50-2 weights.
  - `OpenCV` (`cv2`) & `SciPy`: Computer vision algorithms, image normalization, contour mapping, and Gaussian smoothing.
- **Frontend**: 
  - Monochromatic light developer workbench layout using semantic HTML5, clean CSS3, and modern vanilla JavaScript. No bloated dependencies.
  - Dual slider controls supporting dynamic, real-time bounding box adjustments using cached calculations on the server.

---

## 3. Installation & Setup

### Prerequisites
- Python 3.8 or higher installed on your system.

### Steps
1. **Clone or copy the project files** to your system. Ensure the directory looks like this:
   ```
   pcb/
   ├── server.py
   ├── wide_resnet50_2-95faca4d (1).pth
   └── static/
       ├── index.html
       ├── style.css
       └── app.js
   ```

2. **Install the required dependencies**:
   ```bash
   pip install torch torchvision opencv-python fastapi uvicorn python-multipart scipy pillow numpy
   ```

3. **Start the FastAPI server**:
   ```bash
   python server.py
   ```

4. **Access the Web Console**:
   Open your browser and navigate to `http://localhost:8000/` (or `http://localhost:8001/` if port 8000 was changed).

---

## 4. How to Use the Dashboard

1. **Upload Reference Master**: Drag and drop a clean, defect-free PCB image (e.g., from the `PCB_USED` folder) into the left slot.
2. **Upload Target Sample**: Drag and drop a test sample image matching the ID prefix of your master (e.g., from the `images` folder) into the second slot.
3. **Run Analysis**: Click the **Run Analysis** button. A blue laser line scanner animation will indicate features are being processed.
4. **Inspect Results**:
   - Check the **03. Anomaly Heatmap** to see exactly where activation deviations occurred.
   - Look at the **04. Defect Highlights** to view the segmented bounding boxes.
   - Adjust the **Anomaly Threshold** and **Min Defect Size** sliders at the left to update the detections instantly on the page without reload.
