# 🌿 Betel Leaf Disease Predictor

**Model: RBF-SVM (97% accuracy) + ResNeXt50-32x4d Feature Extraction**

---

## 🚀 How the App Works

```
Upload image
     │
     ▼
Resize(224,224) → ToTensor → Normalize(ImageNet)
     │
     ▼
ResNeXt50-32x4d (output/resnext50.pth)
 └─ avgpool forward-hook → 2048-d feature vector
     │
     ▼
RBF-SVM (output/svm_rbf.pkl)
 SVC(kernel='rbf', C=10, gamma='scale', probability=True)
     │
     ▼
Predicted class + confidence + class probabilities
```

👉 Models are automatically loaded at startup (no manual selection needed).

---

## 📁 Folder Structure

```
betel_leaf_disease_app/
├── app.py
├── requirements.txt
├── README.md
├── templates/
│   └── index.html
└── output/
    ├── resnext50.pth
    └── svm_rbf.pkl
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository

```bash
git clone https://github.com/souvikhaldarsouvikhaldar-beep/betel-leaf-disease-prediction.git
cd betel_leaf_disease_app
```

---

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

Activate:

* **Windows**

```bash
venv\Scripts\activate
```

* **Mac/Linux**

```bash
source venv/bin/activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Run Application

```bash
python app.py
```

---

### 5️⃣ Open in Browser

```
http://localhost:5000
```

---

## 🧠 Usage

1. Upload a betel leaf image
2. Click **Analyze Leaf**
3. Get:

   * Predicted disease
   * Confidence score
   * Probability distribution

---

## ⚠️ Troubleshooting

| Issue               | Solution                           |
| ------------------- | ---------------------------------- |
| Model not found     | Ensure files exist in `output/`    |
| Module error        | Reinstall using `requirements.txt` |
| Port already in use | Change port in `app.py`            |
| Slow first run      | Normal (model warm-up)             |

---

## 🛠 Tech Stack

* Python
* Flask
* PyTorch
* Scikit-learn
* HTML, CSS, JavaScript

---

## 🌟 Highlights

* High accuracy (**97%**)
* Deep learning + classical ML hybrid
* Clean UI with real-time prediction
* Ready for real-world agricultural use

---

## 📌 Future Improvements

* 📷 Real-time camera detection
* ☁️ Cloud deployment
* 📱 Mobile-friendly UI

---

## 👨‍💻 Author

**Souvik Haldar**

---

⭐ If you like this project, consider giving it a star!
