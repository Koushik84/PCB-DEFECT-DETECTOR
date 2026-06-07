import io
import os
import base64
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from scipy.ndimage import gaussian_filter
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI(title="PCB Anomaly Detection System")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
IM_SIZE = 224
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(BASE_DIR, 'wide_resnet50_2-95faca4d (1).pth')

# In-memory cache for fast threshold updating
class InspectionCache:
    def __init__(self):
        self.heatmap_uint8 = None
        self.test_img_resized = None

cache = InspectionCache()

# Feature Extractor using Wide ResNet 50-2
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        # Load backbone without default weights, then load state dict from local path
        self.backbone = models.wide_resnet50_2(weights=None)
        if os.path.exists(WEIGHTS_PATH):
            state_dict = torch.load(WEIGHTS_PATH, map_location='cpu')
            self.backbone.load_state_dict(state_dict)
            print("Loaded local weights successfully.")
        else:
            raise FileNotFoundError(f"Weights file not found at {WEIGHTS_PATH}")
        
        self.backbone.eval()
        self.features = []

        def hook_t(module, input, output):
            self.features.append(output)

        # Register hooks
        self.backbone.layer1[-1].register_forward_hook(hook_t)
        self.backbone.layer2[-1].register_forward_hook(hook_t)
        self.backbone.layer3[-1].register_forward_hook(hook_t)

    def forward(self, x):
        self.features = []
        with torch.no_grad():
            _ = self.backbone(x)
        return self.features

# Instantiate model
try:
    model = FeatureExtractor().to(DEVICE)
except Exception as e:
    print(f"Error initializing model: {e}")
    model = None

# Transformations
data_transform = transforms.Compose([
    transforms.Resize((IM_SIZE, IM_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def encode_img_to_base64(img_rgb):
    """Encodes an RGB image array (uint8) to a base64 png data URL."""
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    is_success, buffer = cv2.imencode(".png", img_bgr)
    if not is_success:
        raise ValueError("Failed to encode image to PNG.")
    encoded = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{encoded}"

def process_contours(heatmap_uint8, test_img_resized, threshold, min_area):
    """Calculates binary mask, bounding boxes and verdict for a given threshold."""
    threshold_value = int(255 * threshold)
    _, binary_mask = cv2.threshold(heatmap_uint8, threshold_value, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bbox_img = test_img_resized.copy()
    anomalies = []
    
    # Sort contours by area descending
    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            valid_contours.append((area, contour))
            
    valid_contours.sort(key=lambda x: x[0], reverse=True)
    
    for idx, (area, contour) in enumerate(valid_contours):
        x, y, w, h = cv2.boundingRect(contour)
        # Draw bold red rectangle on test image
        cv2.rectangle(bbox_img, (x, y), (x + w, y + h), (239, 68, 68), 2)
        # Draw custom text label with background
        label = f"Defect {idx + 1}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.35
        thickness = 1
        label_size, baseline = cv2.getTextSize(label, font, font_scale, thickness)
        label_w, label_h = label_size
        cv2.rectangle(bbox_img, (x, y - label_h - 4), (x + label_w + 4, y), (239, 68, 68), -1)
        cv2.putText(bbox_img, label, (x + 2, y - 2), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        
        anomalies.append({
            "id": idx + 1,
            "x": int(x),
            "y": int(y),
            "w": int(w),
            "h": int(h),
            "area": int(area)
        })

    verdict = "DEFECT DETECTED" if len(anomalies) > 0 else "PASSED"
    return bbox_img, anomalies, verdict

@app.post("/api/detect")
async def detect(
    master: UploadFile = File(...),
    test: UploadFile = File(...),
    threshold: float = Form(0.3),
    min_area: float = Form(20.0)
):
    if model is None:
        raise HTTPException(status_code=500, detail="FeatureExtractor model is not initialized.")

    try:
        # Load files as PIL images
        master_bytes = await master.read()
        test_bytes = await test.read()
        
        master_pil = Image.open(io.BytesIO(master_bytes)).convert('RGB')
        test_pil = Image.open(io.BytesIO(test_bytes)).convert('RGB')
        
        # Prepare inputs
        master_tensor = data_transform(master_pil).unsqueeze(0).to(DEVICE)
        test_tensor = data_transform(test_pil).unsqueeze(0).to(DEVICE)
        
        # Feature extraction
        with torch.no_grad():
            ref_features = model(master_tensor)
            test_features = model(test_tensor)

        def embed_concat(features):
            t_features = [features[0]]
            for i in range(1, len(features)):
                t_features.append(F.interpolate(features[i], size=features[0].shape[-2:], mode='bilinear', align_corners=True))
            return torch.cat(t_features, dim=1)

        ref_emb = embed_concat(ref_features)
        test_emb = embed_concat(test_features)

        # Distance calculation
        diff = (ref_emb - test_emb).pow(2).sum(1).sqrt()
        anomaly_map = F.interpolate(diff.unsqueeze(0), size=(IM_SIZE, IM_SIZE), mode='bilinear', align_corners=True)
        anomaly_map = anomaly_map.squeeze().cpu().numpy()
        
        # Gaussian smoothing
        smoothed_map = gaussian_filter(anomaly_map, sigma=4)
        
        # Normalize heatmap to [0, 255]
        heatmap_uint8 = cv2.normalize(smoothed_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Save base resized images for visualization
        master_resized = np.array(master_pil.resize((IM_SIZE, IM_SIZE)))
        test_resized = np.array(test_pil.resize((IM_SIZE, IM_SIZE)))

        # Update cache
        cache.heatmap_uint8 = heatmap_uint8
        cache.test_img_resized = test_resized

        # Create Heatmap color image (Jet map)
        heatmap_color_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_color_rgb = cv2.cvtColor(heatmap_color_bgr, cv2.COLOR_BGR2RGB)

        # Create overlay image (70% original test image, 30% heatmap)
        overlay_rgb = cv2.addWeighted(test_resized, 0.7, heatmap_color_rgb, 0.3, 0)

        # Calculate bounding boxes
        bbox_img, anomalies, verdict = process_contours(heatmap_uint8, test_resized, threshold, min_area)

        # Encode everything to base64 URLs
        return JSONResponse(content={
            "master": encode_img_to_base64(master_resized),
            "test": encode_img_to_base64(test_resized),
            "heatmap": encode_img_to_base64(heatmap_color_rgb),
            "overlay": encode_img_to_base64(overlay_rgb),
            "bbox_img": encode_img_to_base64(bbox_img),
            "anomalies": anomalies,
            "verdict": verdict,
            "max_score": float(np.max(smoothed_map))
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.post("/api/update_threshold")
async def update_threshold(
    threshold: float = Form(...),
    min_area: float = Form(...)
):
    if cache.heatmap_uint8 is None or cache.test_img_resized is None:
        raise HTTPException(status_code=400, detail="No active inspection session cached. Please run an initial inspection first.")
    
    try:
        # Calculate updated bounding boxes using cached heatmap
        bbox_img, anomalies, verdict = process_contours(cache.heatmap_uint8, cache.test_img_resized, threshold, min_area)
        
        return JSONResponse(content={
            "bbox_img": encode_img_to_base64(bbox_img),
            "anomalies": anomalies,
            "verdict": verdict
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update threshold: {str(e)}")

# Mount static files (will hold index.html, style.css, app.js)
static_dir = os.path.join(BASE_DIR, 'static')
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
async def get_dashboard():
    static_dir = os.path.join(BASE_DIR, 'static')
    html_path = os.path.join(static_dir, 'index.html')
    if os.path.exists(html_path):
        with open(html_path, 'r', encoding='utf-8') as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h3>PCB Anomaly Detection System - Static index.html not found!</h3>")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)