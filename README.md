# Betel Leaf Disease Predictor

**Model: RBF-SVM (97 % accuracy) + ResNeXt50-32x4d feature extraction**

---

## How the app works

```
Upload image
     │
     ▼
Resize(224,224) → ToTensor → Normalize(ImageNet)
     │
     ▼
ResNeXt50-32x4d  (output/resnext50.pth)
 └─ avgpool forward-hook  →  2048-d feature vector
     │
     ▼
RBF-SVM  (output/svm_rbf.pkl)
 SVC(kernel='rbf', C=10, gamma='scale', probability=True)
     │
     ▼
Predicted class + confidence + class probabilities
```

No model-path input on the web page — models are auto-loaded at startup.

---

## Folder structure

```
betel_leaf_disease_app/
├── app.py                  ← Flask backend
├── requirements.txt        ← Python dependencies
├── README.md               ← this file
├── templates/
│   └── index.html          ← Web UI
└── output/
    ├── resnext50.pth       ← ⚠ copy here from your notebook output/
    └── svm_rbf.pkl         ← ⚠ copy here from your notebook output/
```

---

## Step-by-step setup in PyCharm

### Step 1 — Open the project
Open the `betel_leaf_disease_app/` folder in PyCharm.

### Step 2 — Copy your model files
From wherever your notebook saved them, copy:
- `resnext50.pth`  →  into  `betel_leaf_disease_app/output/`
- `svm_rbf.pkl`    →  into  `betel_leaf_disease_app/output/`

> Both files must be inside the `output/` folder.

### Step 3 — Update CLASS_NAMES (important!)
Open `app.py` and find the `CLASS_NAMES` list near the top.
Replace it with the **exact** sorted folder names from your training dataset.

To check, run this in your notebook or Python:
```python
from pathlib import Path
TRAIN_PATH = Path("path/to/your/train/folder")
print(sorted([d.name for d in TRAIN_PATH.iterdir() if d.is_dir()]))
```
Paste that output as the `CLASS_NAMES` list in `app.py`.

### Step 4 — Open Terminal in PyCharm
Press  **Alt+F12**  (Windows/Linux)  or  **Option+F12**  (Mac).

### Step 5 — Create a virtual environment
```bash
python -m venv venv
```

Activate it:
- **Windows:**  `venv\Scripts\activate`
- **Mac/Linux:** `source venv/bin/activate`

### Step 6 — Install dependencies
```bash
pip install -r requirements.txt
```
> First install may take a few minutes (downloads PyTorch etc.).

### Step 7 — Run the app
```bash
python app.py
```
You should see:
```
[INFO] Loading ResNeXt50-32x4d feature extractor ...
[INFO] Loading RBF-SVM classifier ...
[INFO] Both models loaded successfully. Starting Flask ...
 * Running on http://0.0.0.0:5000
```

### Step 8 — Open in browser
Go to  **http://localhost:5000**

---

## Using the web app

1. Click the upload zone (or drag & drop a betel leaf image).
2. A preview of your image appears.
3. Click **"🔍 Analyze Leaf"**.
4. Results appear below:
   - Predicted disease class
   - Confidence percentage + animated bar
   - Probability breakdown for all classes

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `FileNotFoundError: output/resnext50.pth` | Copy model files into the `output/` folder (Step 2) |
| Wrong class names | Update `CLASS_NAMES` in `app.py` (Step 3) |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` again |
| Port 5000 in use | Change `port=5000` in `app.py` to e.g. `5001` |
| Slow first prediction | Normal — PyTorch JIT warm-up on first inference |

---

## Technical details

| Component | Detail |
|-----------|--------|
| Feature extractor | `resnext50_32x4d` — `IMAGENET1K_V2` weights |
| Extraction layer | `avgpool` (2048-dim output) |
| Hook method | `register_forward_hook` |
| Classifier | `SVC(kernel='rbf', C=10, gamma='scale', probability=True)` |
| Image size | 224 × 224 |
| Normalisation | ImageNet mean/std |
| Test accuracy | **97%** |
