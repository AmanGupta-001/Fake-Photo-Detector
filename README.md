<<<<<<< HEAD
# FakePhoto Detector 🔍

A lightweight machine learning tool that classifies whether an image is a **genuine photograph** or a **screen recapture** (a photo taken of a screen) — with no GPU required.

## How It Works

Instead of a neural network, this uses **20 hand-crafted physical features** that exploit the real differences between camera shots and screen photos:

| Feature Group | Signal |
|---|---|
| **FFT Frequency Bands** | Camera + screen pixel grids create moiré → anomalous energy peaks in mid-frequency FFT rings |
| **Hough Straight Edges** | Monitor bezels produce long border-aligned lines that organic scenes don't |
| **Glare Blobs** | Glass screens reflect room lights as large near-white blobs |
| **Colour Statistics** | Backlit RGB subpixels shift hue and bias the blue channel |
| **LBP Texture Entropy** | Organic surfaces (skin, grass) produce richer LBP histograms than screen fills |
| **DCT AC Energy** | Re-photographing a JPEG-compressed screen leaves a double-banding signature |

The model is a **scikit-learn Random Forest** pipeline (StandardScaler + RandomForestClassifier), weighing under 500 KB, achieving **~95% accuracy** on real-world photos.

## Project Structure

```
FakePhoto/
├── app.py                  # Streamlit web UI
├── train.py                # Train & evaluate the model
├── predict.py              # CLI: classify a single image
├── features.py             # Feature extraction (20 features)
├── generate_demo_data.py   # Generate synthetic demo images
├── model.joblib            # Trained model (committed for convenience)
├── report.txt              # Cross-validation metrics
├── NOTE.md                 # Technical design notes
├── .streamlit/
│   └── config.toml         # Dark theme config
└── .gitignore
```

> **Note:** `real/` and `screen/` image folders are excluded from this repo (private data). See [Training](#training) to set up your own.

## Quick Start

### 1. Install dependencies

```bash
pip install opencv-python scikit-learn joblib scipy numpy streamlit plotly pillow pillow-heif
```

### 2. Run the web app (model already included)

```bash
python -m streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) and upload any image to classify it.

### 3. CLI prediction

```bash
python predict.py path/to/your/photo.jpg
# Output: 0.0341   (0.0 = real, 1.0 = screen)
```

## Training

To retrain on your own photos:

1. Create two folders: `real/` and `screen/` and fill them with images (JPG, PNG, WebP, HEIC supported)
2. Run training:

```bash
python train.py
# Outputs: model.joblib, report.txt
```

Training automatically cross-validates 3 models (Logistic Regression, Random Forest, Gradient Boosting) and saves the best one.

### Generate synthetic demo data

If you don't have photos yet:

```bash
python generate_demo_data.py --n 50
python train.py
```

## Model Performance (Real Photos, 5-fold CV)

| Model | Accuracy | AUC-ROC |
|---|---|---|
| **Random Forest** ⭐ | **94.95%** | **0.9955** |
| Logistic Regression | 89.90% | 0.9522 |
| Gradient Boosting | 89.90% | 0.8912 |

## What I'd Improve With More Time

1. **More data** — 200+ images/class with varied angles, screens, lighting
2. **Augmentation** — JPEG re-compression, brightness jitter, random crop
3. **MobileNetV3-Small fine-tuned head** — catches subtle patterns hand-crafted features miss
4. **Calibrated probabilities** — isotonic regression for well-calibrated confidence scores
5. **ROC-curve threshold selection** — tune operating point via business cost matrix

## Latency

~20–60 ms per image on a CPU after one-time model load. No GPU required.

## License

MIT
=======
# Fake-Photo-Detector
>>>>>>> b3c9f0dfdee0adf32a66e987523aaed3ddbb0797
