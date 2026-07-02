import os
import sys
import time


def predict(image_path: str) -> float:
    import joblib
    from features import extract

    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"model.joblib")
    if not os.path.exists(model_path):
        sys.exit("ERROR: model.joblib not found — run  python train.py  first.")

    model = joblib.load(model_path)

    t0    = time.perf_counter()
    vec   = extract(image_path)
    score = float(model.predict_proba(vec.reshape(1, -1))[0, 1])
    ms    = (time.perf_counter() - t0) * 1000

    print(f"[latency: {ms:.1f} ms]", file=sys.stderr)
    return score


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python predict.py <image_path>")
    if not os.path.exists(sys.argv[1]):
        sys.exit(f"ERROR: file not found: {sys.argv[1]}")

    print(f"{predict(sys.argv[1]):.4f}")
