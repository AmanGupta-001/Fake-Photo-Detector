"""
train.py — Train and evaluate the real-vs-screen classifier.

Usage:
    python train.py                           # looks for ./real/ and ./screen/
    python train.py --real PATH --screen PATH

Outputs:
    model.joblib   — saved pipeline (StandardScaler + classifier)
    report.txt     — honest cross-validated metrics + feature importances
"""

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import extract, NAMES

warnings.filterwarnings("ignore")

REAL   = 0   # label for genuine photos
SCREEN = 1   # label for screen / recapture photos
EXTS   = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".heic"}


# -- data loading -----------------------------------------------------

def load_folder(folder: Path, label: int):
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTS)
    if not paths:
        sys.exit(f"ERROR: no images found in {folder}/")
    X, y = [], []
    for p in paths:
        try:
            X.append(extract(str(p)))
            y.append(label)
        except Exception as e:
            print(f"  skip {p.name}: {e}")
    print(f"  {len(X):3d} images ← {folder}/")
    return np.array(X, dtype=np.float32), np.array(y, dtype=int)


# -- models to try ----------------------------------------------------

def candidates():
    return {
        "logistic": Pipeline([
            ("scale", StandardScaler()),
            ("clf",   LogisticRegression(C=1.0, max_iter=2000,
                                         class_weight="balanced",
                                         random_state=42)),
        ]),
        "random_forest": Pipeline([
            ("scale", StandardScaler()),
            ("clf",   RandomForestClassifier(n_estimators=300,
                                              max_depth=8,
                                              min_samples_leaf=2,
                                              class_weight="balanced",
                                              random_state=42,
                                              n_jobs=-1)),
        ]),
        "grad_boost": Pipeline([
            ("scale", StandardScaler()),
            ("clf",   GradientBoostingClassifier(n_estimators=150,
                                                  max_depth=4,
                                                  learning_rate=0.05,
                                                  random_state=42)),
        ]),
    }


# -- evaluation -------------------------------------------------------

def cv_eval(name, pipe, X, y, cv):
    y_hat   = cross_val_predict(pipe, X, y, cv=cv)
    y_prob  = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
    acc = accuracy_score(y, y_hat)
    auc = roc_auc_score(y, y_prob)
    print(f"  {name:<16}  acc={acc:.3f}   auc={auc:.3f}")
    return dict(name=name, pipe=pipe, acc=acc, auc=auc,cm=confusion_matrix(y, y_hat),report=classification_report(y, y_hat,target_names=["real", "screen"]))


def importances(pipe):
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        imp = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        imp = np.abs(clf.coef_[0])
    else:
        return []
    idx = np.argsort(imp)[::-1]
    return [(NAMES[i], float(imp[i])) for i in idx]


# -- main -------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real",   default="real")
    ap.add_argument("--screen", default="screen")
    ap.add_argument("--out",    default="model.joblib")
    ap.add_argument("--report", default="report.txt")
    args = ap.parse_args()

    rd = Path(args.real)
    sd = Path(args.screen)
    for d in (rd, sd):
        if not d.exists():
            sys.exit(f"ERROR: folder not found: {d}\n"
                     "Create 'real/' and 'screen/' with your phone photos first.\n"
                     "Or run:  python generate_demo_data.py")

    print("\n-- Loading images ------------------------------------------")
    t0 = time.perf_counter()
    Xr, yr = load_folder(rd, REAL)
    Xs, ys = load_folder(sd, SCREEN)
    X = np.vstack([Xr, Xs])
    y = np.concatenate([yr, ys])
    n_total = len(y)
    elapsed = time.perf_counter() - t0
    print(f"  {n_total} images total  ({elapsed*1000:.0f} ms, "
          f"{elapsed/n_total*1000:.1f} ms/image)")

    if n_total < 30:
        print("  [WARN] Very few images — collect more for reliable accuracy.")

    print("\n-- Cross-validating (5-fold stratified) --------------------")
    cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = [cv_eval(n, p, X, y, cv) for n, p in candidates().items()]
    best    = max(results, key=lambda r: r["acc"])
    print(f"\n  Best → {best['name']}  (acc={best['acc']:.4f}  auc={best['auc']:.4f})")

    print("\n-- Retraining on full dataset ------------------------------")
    best["pipe"].fit(X, y)
    joblib.dump(best["pipe"], args.out)
    print(f"  Saved → {args.out}  ({os.path.getsize(args.out)//1024} KB)")

    # -- report --
    imps = importances(best["pipe"])
    lines = []
    sep   = "=" * 58

    lines += [sep, "SPOT THE FAKE PHOTO — Training Report", sep, ""]
    lines += [f"Dataset : {(y==0).sum()} real + {(y==1).sum()} screen "
              f"= {n_total} images"]
    lines += [f"Features: {len(NAMES)}  ({', '.join(NAMES)})", ""]

    for r in results:
        lines += [f"-- {r['name']} --",
                  f"  Accuracy (5-fold CV) : {r['acc']:.4f}",
                  f"  AUC-ROC  (5-fold CV) : {r['auc']:.4f}",
                  f"  Confusion matrix:",
                  f"    [[TN={r['cm'][0,0]}  FP={r['cm'][0,1]}]",
                  f"     [FN={r['cm'][1,0]}  TP={r['cm'][1,1]}]]",
                  r["report"], ""]

    lines += [sep,
              f"Selected : {best['name']}  "
              f"acc={best['acc']:.4f}  auc={best['auc']:.4f}", ""]

    if imps:
        lines += ["Feature importances (top 10):"]
        for name, imp in imps[:10]:
            bar = "█" * max(1, int(imp * 50))
            lines += [f"  {name:<24} {imp:.4f}  {bar}"]

    lines += ["", "What to improve with more time:",
              "  1. More data: 200+ images/class, varied angles/screens/lighting.",
              "  2. Augmentation: JPEG re-compress, brightness jitter, random crop.",
              "  3. MobileNetV3-Small fine-tuned head for subtle patterns.",
              "  4. Isotonic regression for calibrated probability output.",
              "  5. ROC-curve threshold selection via business cost matrix."]

    report = "\n".join(lines)
    print("\n" + report)
    Path(args.report).write_text(report, encoding="utf-8")
    print(f"\nReport → {args.report}")


if __name__ == "__main__":
    main()
