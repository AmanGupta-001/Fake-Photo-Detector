# Spot the Fake Photo — Approach Note

## How it works

Rather than training a neural network on ~100 images (which would overfit badly), this
solution uses **19 hand-crafted features** that capture the physical fingerprints left by
photographing a screen, fed into a Random Forest classifier.

| Feature group | Physical signal exploited |
|---|---|
| **FFT frequency bands** | Camera pixel grid + screen pixel grid create moiré → anomalous energy peaks in mid-frequency rings of the 2-D FFT |
| **Hough straight edges** | Screens have a hard rectangular bezel → long near-horizontal + near-vertical lines |
| **Glare blobs** | Glass screens reflect room lights as large near-white blobs — real objects rarely do |
| **Colour statistics** | Backlit RGB subpixels shift hue and bias the blue channel relative to natural-light shots |
| **LBP texture entropy** | Organic surfaces (skin, fabric, grass) produce richer LBP histograms than the flat fills and gradients on a screen |
| **DCT AC energy** | Re-photographing an already-JPEG-compressed screen leaves a double-banding signature in 8×8 DCT blocks |

No GPU. No neural network. The model is a scikit-learn `RandomForest` wrapped in a
`StandardScaler` pipeline, weighing under 1 MB.

## Accuracy

Cross-validated (5-fold stratified) on the collected dataset — reported in `report.txt`.
With ~50 images per class and variety in lighting, angles and screen types, target is ≥ 95%.
All numbers in this note are cross-validated, not train-set accuracy.

## Latency

~**20–50 ms per image on a laptop CPU** after the one-time model load.  
Measured on Apple M2, Python 3.11, OpenCV 4.x, with the vectorised batch DCT and LBP.

## Cost per image

| Mode | Cost |
|---|---|
| **On-device (recommended)** | $0 — model runs inside the app, no server round-trip |
| AWS Lambda (128 MB, ~50 ms/call) | ≈ $0.004–$0.008 / 1,000 images |
| Small cloud VM (t3.small, ~40 img/s) | ≈ $0.001 / 1,000 images |

On-device is the right call: the model is tiny, latency is good, and it avoids sending
user photos to a server (privacy win).

## What I'd improve with more time

1. **More data** — 200+ images per class, deliberately varied (different phones, laptop
   screens, printouts, distances, angles, strong/dim lighting). This alone would push
   accuracy to 97–99 %.
2. **Augmentation** — random JPEG re-compression at varying quality, brightness/contrast
   jitter, and random crop/rotate to reduce overfitting on small datasets.
3. **Fine-tuned MobileNetV3-Small head** — freeze backbone, retrain final layer on these
   images. Adds ~5 MB but catches subtle patterns hand-crafted features miss. Export to
   TFLite / CoreML for on-device deployment.
4. **Calibrated probabilities** — apply isotonic regression so the 0–1 output is a
   well-calibrated probability, useful for threshold tuning and A/B testing.
5. **Threshold selection via ROC curve** — choose the operating point from cross-validation
   ROC data, weighted by business cost: a false positive (honest user flagged) hurts
   retention more than a false negative (cheater missed), so bias toward lower FPR.

## Keeping it accurate as cheaters adapt

- Log borderline predictions (score 0.35–0.65) for periodic human review; feed confirmed
  labels back into retraining (active learning loop).
- Monitor a held-out validation set monthly; trigger retraining when accuracy drops > 2 pp.
- As high-DPI screens reduce moiré, shift weight toward glare + colour + DCT features.

## Making it phone-ready

The current pipeline is already phone-friendly: all features use OpenCV (available natively
on iOS / Android). The model weights (~300 shallow trees) can be exported to ONNX (~800 KB)
and run via ONNX Runtime Mobile, or rewritten in Swift/Kotlin with the same OpenCV feature
calls. Alternatively, quantise a MobileNetV3-Small to int8 → ~2.5 MB, ~20 ms on mid-range
phones.
