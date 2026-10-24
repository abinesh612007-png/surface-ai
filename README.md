# SurfaceAI — Intelligent Product Surface Defect Detection & Analytics Platform

> ResNet-18 transfer learning on the NEU-CLS steel surface defect dataset, wrapped in a Flask web application with a single-page dashboard, inspection history, and date-wise analytics.

---

## Project Structure

```
SurfaceAI/
├── app.py              ← Flask server and API endpoints
├── index.html          ← Entire single-file frontend (HTML + CSS + JS)
├── model.py            ← AI model loading and inference
├── database.py         ← SQLite schema and queries
├── analytics.py        ← Chart.js-ready aggregation helpers
├── train.py            ← Training script (ResNet-18 transfer learning)
├── evaluate.py         ← Standalone evaluation and metrics
├── requirements.txt    ← Python dependencies
├── data/
│   └── README.md       ← Dataset download instructions
├── models/
│   ├── best_model.pth  ← Trained weights (after training)
│   └── README.md
└── tests/
    ├── test_api.py
    └── test_analytics.py
```

---

## Quick Start (Windows)

### 1 — Install dependencies

```powershell
# CPU-only PyTorch (recommended for first setup)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 2 — Run the application (without model)

The application starts and all features work even without a trained model.  
The New Inspection page will show a clear "Model not loaded" message.

```powershell
python app.py
```

Open **https://ai-surface.onrender.com/** in your browser.

---

## Training the Model

### Step 1 — Download NEU-CLS dataset

Download from Kaggle:  
https://www.kaggle.com/datasets/fantacher/neu-metal-surface-defects-data

Extract so that the directory structure matches:

```
data/NEU-CLS/
    Crazing/          (300 images, .bmp or .jpg)
    Inclusion/        (300 images)
    Patches/          (300 images)
    Pitted_surface/   (300 images)
    Rolled-in_scale/  (300 images)
    Scratches/        (300 images)
```

### Step 2 — Train

```powershell
python train.py --data_dir data/NEU-CLS --epochs 30 --batch_size 32
```

Training will:
- Split data 70 / 15 / 15 (train / val / test) with a fixed seed
- Apply augmentation only to training images
- Train the classifier head first, then fine-tune the full network
- Save the best checkpoint to `models/best_model.pth`
- Print a classification report on the held-out test set

### Step 3 — Restart the server

```powershell
python app.py
```

The model loads automatically. The sidebar indicator turns green.

### Step 4 — (Optional) Run standalone evaluation

```powershell
python evaluate.py --data_dir data/NEU-CLS --checkpoint models/best_model.pth
```

---

## Running Tests

```powershell
pip install pytest
python -m pytest tests/ -v
```

Tests do not require a trained model or the NEU-CLS dataset.

---

## API Reference

| Method   | Endpoint                        | Description                          |
|----------|---------------------------------|--------------------------------------|
| GET      | `/api/health`                   | Health check                         |
| GET      | `/api/model-status`             | Model readiness and version          |
| POST     | `/api/inspect`                  | Upload image → run inference         |
| GET      | `/api/summary`                  | Dashboard KPI metrics                |
| GET      | `/api/history`                  | Paginated inspection history         |
| GET      | `/api/history/<id>`             | Single inspection record             |
| DELETE   | `/api/history/<id>`             | Delete inspection record             |
| GET      | `/api/analytics/daily`          | Daily trend data                     |
| GET      | `/api/analytics/monthly`        | Monthly trend data                   |
| GET      | `/api/analytics/yearly`         | Yearly trend data                    |
| GET      | `/api/analytics/categories`     | Category distribution                |
| GET      | `/api/analytics/confidence`     | Confidence bucket distribution       |
| GET      | `/api/export/csv`               | Export filtered inspections as CSV   |

### `/api/inspect` — POST

**Form field:** `image` (multipart/form-data, JPG or PNG, max 16 MB)

**Response (success):**
```json
{
  "inspection_id": 42,
  "filename": "surface_01.jpg",
  "predicted_class": "Crazing",
  "confidence": 0.9312,
  "class_probabilities": { "Crazing": 0.9312, "Inclusion": 0.021, ... },
  "top3": [
    { "class": "Crazing", "confidence": 0.9312 },
    ...
  ],
  "model_version": "resnet18-neu-cls-v1",
  "gradcam": "<base64-jpeg or null>",
  "timestamp": "2024-06-15T10:30:00Z"
}
```

---

## Supported Defect Classes

| Class          | NEU-CLS Label    | Color    |
|----------------|------------------|----------|
| Crazing        | Crazing          | Cyan     |
| Inclusion      | Inclusion        | Purple   |
| Patches        | Patches          | Pink     |
| Pitted Surface | Pitted_surface   | Orange   |
| Rolled-in Scale| Rolled-in_scale  | Green    |
| Scratches      | Scratches        | Yellow   |

---

## Important: Binary Defect / Non-Defect Classification

The six NEU-CLS classes are **all defect categories**. There is no defect-free (normal) class in this dataset.

**Therefore:**
- This system reports the *predicted defect category*, not a binary pass/fail result.
- Defect-versus-non-defect metrics are labelled **unavailable** in the UI.
- Do not interpret a low-confidence prediction as evidence that a surface is defect-free.
- For reliable pass/fail inspection, an anomaly-detection model trained on normal images must be added.

---

## Grad-CAM Explainability

When `opencv-python-headless` is installed, the `/api/inspect` endpoint returns a Grad-CAM heatmap overlay in the response as a base64-encoded JPEG.

This heatmap shows the approximate image regions that influenced the prediction.  
**It is an explanatory visualisation, not a precise defect boundary.**

---

## Notes for College Presentation

1. Start the server: `python app.py`
2. Open https://ai-surface.onrender.com/
3. Navigate to **New Inspection** and upload any steel surface image
4. The result panel shows the predicted class, confidence, top-3 predictions, and (if model is loaded) a Grad-CAM overlay
5. All records are stored in `surfaceai.db` and visible in **Inspection History** and **Analytics**
6. Use **Reports** to export CSV files for your documentation
