import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "output", "resnext50.pth")
svm_path = os.path.join(BASE_DIR, "output", "svm_rbf.pkl")
"""
Betel Leaf Disease Prediction - Flask Web App
Model: RBF SVM (97% accuracy) with ResNeXt50-32x4d feature extraction
Pipeline mirrors notebook exactly:
  - ResNeXt50_32X4D (IMAGENET1K_V2 weights) -> avgpool forward-hook -> 2048-d vector
  - SVC(kernel='rbf', C=10, gamma='scale', probability=True)

NO model-path input on the web page.
Models are auto-loaded from  ./output/resnext50.pth  and  ./output/svm_rbf.pkl
"""

import os
import io
import numpy as np
import joblib
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request, jsonify, render_template

# ─────────────────────────────────────────────────────────────────────────────
# 1.  CLASS NAMES
#     Must match sorted(os.listdir(TRAIN_PATH)) from your notebook.
#     Look at the output of:  CLASS_NAMES = sorted([d.name for d in TRAIN_PATH.iterdir() if d.is_dir()])
#     in your notebook and paste those exact names here.
# ─────────────────────────────────────────────────────────────────────────────
CLASS_NAMES = [
   "Bacterial Leaf Disease",
   "Dried Leaf",
   "Fungal Brown Spot Disease",
   "Healthy Leaf",
]

# ─────────────────────────────────────────────────────────────────────────────
# 2.  FILE PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR    = os.path.join(BASE_DIR, "output")
RESNEXT_PATH = os.path.join(MODEL_DIR, "resnext50.pth")
SVM_PATH     = os.path.join(MODEL_DIR, "svm_rbf.pkl")

# ─────────────────────────────────────────────────────────────────────────────
# 3.  IMAGE TRANSFORM  (val_transform from notebook)
#     Resize(224,224) -> ToTensor -> Normalize(ImageNet)
# ─────────────────────────────────────────────────────────────────────────────
IMG_SIZE      = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# ─────────────────────────────────────────────────────────────────────────────
# 4.  LOAD MODELS AT STARTUP  (runs once when Flask starts)
# ─────────────────────────────────────────────────────────────────────────────
device = torch.device("cpu")

print("[INFO] Loading ResNeXt50-32x4d feature extractor ...")
resnext_model = models.resnext50_32x4d(
    weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2
)
state_dict = torch.load(RESNEXT_PATH, map_location=device)
resnext_model.load_state_dict(state_dict)
resnext_model.to(device)
resnext_model.eval()

# Grab avgpool layer — exactly as in notebook:
#   extraction_layer = model._modules.get('avgpool')
extraction_layer = resnext_model._modules.get("avgpool")

print("[INFO] Loading RBF-SVM classifier ...")
svm_model = joblib.load(SVM_PATH)

print("[INFO] Both models loaded successfully. Starting Flask ...")

# ─────────────────────────────────────────────────────────────────────────────
# 5.  FEATURE EXTRACTION
#     Mirrors get_vec() from notebook:
#       registers a forward hook on avgpool,
#       runs one forward pass,
#       returns the flattened 2048-d vector.
# ─────────────────────────────────────────────────────────────────────────────
def extract_features(pil_image: Image.Image) -> np.ndarray:
    """
    Input : PIL image (any mode – converted to RGB internally)
    Output: numpy array of shape (2048,)
    """
    holder = {}

    def _hook(module, inp, output):
        holder["vec"] = output.detach()

    hook_handle = extraction_layer.register_forward_hook(_hook)

    rgb_img = pil_image.convert("RGB")
    tensor  = val_transform(rgb_img).unsqueeze(0).to(device)  # [1, 3, 224, 224]

    with torch.no_grad():
        resnext_model(tensor)

    hook_handle.remove()

    # squeeze: [1, 2048, 1, 1] -> (2048,)
    feat_vec = holder["vec"].squeeze().cpu().numpy()
    return feat_vec

# ─────────────────────────────────────────────────────────────────────────────
# 6.  FLASK APPLICATION
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024   # 16 MB upload limit

ALLOWED_EXT = {"png", "jpg", "jpeg", "bmp", "tiff", "webp"}

def _allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts: multipart/form-data  with field 'image'
    Returns: JSON {
               "class":         "Predicted Disease Name",
               "confidence":    97.32,
               "probabilities": {"Class1": 97.32, "Class2": 1.20, ...}
             }
    """
    # --- validate request ---
    if "image" not in request.files:
        return jsonify({"error": "No image field in request"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    if not _allowed(file.filename):
        return jsonify({"error": "Unsupported file type. Use PNG/JPG/JPEG/BMP."}), 400

    try:
        # --- read & extract features ---
        img_bytes = file.read()
        pil_img   = Image.open(io.BytesIO(img_bytes))
        feat_vec  = extract_features(pil_img)           # (2048,)
        X         = feat_vec.reshape(1, -1)              # (1, 2048)

        # --- SVM inference ---
        pred_idx  = int(svm_model.predict(X)[0])
        proba_arr = svm_model.predict_proba(X)[0]        # (n_classes,)

        pred_class  = CLASS_NAMES[pred_idx]
        confidence  = float(proba_arr[pred_idx]) * 100

        proba_dict = {
            CLASS_NAMES[i]: round(float(proba_arr[i]) * 100, 2)
            for i in range(len(CLASS_NAMES))
        }

        return jsonify({
            "class":         pred_class,
            "confidence":    round(confidence, 2),
            "probabilities": proba_dict,
        })

    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# 7.  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
