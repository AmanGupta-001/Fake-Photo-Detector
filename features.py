"""
features.py — Extract a 20-number feature vector from one image.

Physical signals exploited:
  ┌─────────────────────────────────────────────────────────────┐
  │ Signal          │ Why screens have it                       │
  ├─────────────────┼───────────────────────────────────────────┤
  │ FFT mid-freq    │ Camera grid ↔ pixel grid → moiré peaks    │
  │ Straight edges  │ Bezel / monitor border                    │
  │ Glare blobs     │ Glass reflects room lights                │
  │ Color stats     │ Backlit RGB shifts hue, blue channel      │
  │ LBP texture     │ Organic surfaces richer than screen fills │
  │ DCT AC energy   │ Double-JPEG leaves banding signature      │
  └─────────────────┴───────────────────────────────────────────┘

 Public API:
    vec = extract("photo.jpg")   # → np.ndarray shape (20,) float32
    NAMES                        # list of 20 feature name strings
"""

import cv2
import numpy as np
from PIL import Image as _PILImage
from scipy import stats
from scipy.fft import dctn

# Register HEIC/HEIF support if pillow-heif is available
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

# ── constants ────────────────────────────────────────────────────────
_LONG_SIDE = 512          # resize so shortest side = this before feature work
_LBP_R     = 2
_LBP_N     = 8 * _LBP_R  # 16 neighbours

NAMES = [
    # FFT bands (fraction of total log-magnitude in each ring)
    "fft_lo", "fft_mid", "fft_hi",
    # FFT peak sharpness (moiré → sharp spike above the mean)
    "fft_peak_ratio",
    # Straight-edge / bezel
    "hough_line_density", "hough_rect_score",
    # Glare
    "glare_frac", "glare_blob_frac",
    # Colour
    "hue_mean", "hue_std", "sat_mean", "sat_std", "blue_bias",
    # Texture
    "lap_var", "lbp_entropy", "lbp_uniformity",
    # DCT double-compression
    "dct_ac_mean", "dct_ac_std", "dct_ac_skew",
    # Edge density (screens have crisp hard edges inside)
    "edge_density",
]
assert len(NAMES) == 20


# ── helpers ──────────────────────────────────────────────────────────

def _load(path: str):
    img = cv2.imread(path)
    if img is None:
        # Fallback: use Pillow (handles HEIC, HEIF, WebP, etc.)
        try:
            pil = _PILImage.open(path).convert("RGB")
            img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        except Exception:
            raise FileNotFoundError(f"Cannot read: {path}")
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    h, w = img.shape[:2]
    scale = _LONG_SIDE / min(h, w)
    img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def _fft(gray):
    mag = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(gray.astype(np.float32)))))
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    R = np.sqrt((np.arange(w) - cx) ** 2 + (np.arange(h)[:, None] - cy) ** 2)
    rmax = min(cx, cy)
    total = mag.sum() + 1e-9

    def band(lo, hi):
        return float(mag[(R >= lo * rmax) & (R < hi * rmax)].sum() / total)

    lo  = band(0.00, 0.10)
    mid = band(0.10, 0.30)
    hi  = band(0.30, 0.55)
    peak_ratio = float(mag.max() / (mag.mean() + 1e-9))
    return [lo, mid, hi, peak_ratio]


def _hough(gray):
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60,
                             minLineLength=gray.shape[1] * 0.40,  # long spans only
                             maxLineGap=25)
    if lines is None:
        return [0.0, 0.0]
    # OpenCV 4 returns shape (N, 1, 4); OpenCV 5 returns (N, 4)
    lines_2d = lines[:, 0] if lines.ndim == 3 else lines
    h, w = gray.shape
    border_h = h * 0.25   # lines must sit in the outer 25% vertically (top/bottom bezel)
    border_w = w * 0.25   # or outer 25% horizontally (left/right bezel)
    h_n = v_n = 0
    for x1, y1, x2, y2 in lines_2d:
        ang = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        # Horizontal lines (bezel top/bottom): must be near top or bottom edge
        if ang < 20 or ang > 160:
            if min(y1, y2) < border_h or max(y1, y2) > h - border_h:
                h_n += 1
        # Vertical lines (bezel left/right): must be near left or right edge
        elif 70 < ang < 110:
            if min(x1, x2) < border_w or max(x1, x2) > w - border_w:
                v_n += 1
    density    = float(len(lines) / (gray.size ** 0.5 + 1))
    rect_score = float(min(h_n, 6) / 6 * min(v_n, 6) / 6)  # needs 6 border-hugging lines for score=1.0
    return [density, rect_score]



def _glare(bgr):
    hsv  = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 215), (180, 35, 255))
    px   = mask.size
    gf   = int(mask.sum() // 255) / px

    n, _, st, _ = cv2.connectedComponentsWithStats(mask, 8)
    biggest = int(st[1:, cv2.CC_STAT_AREA].max()) if n > 1 else 0
    return [float(gf), float(biggest / px)]


def _color(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, _ = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    b = bgr[:, :, 0].astype(float)
    r = bgr[:, :, 2].astype(float)
    blue_bias = float((b.mean() - r.mean()) / (r.mean() + 1e-9))
    return [float(h.mean()), float(h.std()),
            float(s.mean()), float(s.std()),
            blue_bias]


def _lbp_map(gray):
    g   = gray.astype(np.int16)
    out = np.zeros(gray.shape, dtype=np.int32)
    for bit in range(_LBP_N):
        angle = 2 * np.pi * bit / _LBP_N
        dy = int(round(_LBP_R * np.sin(angle)))
        dx = int(round(_LBP_R * np.cos(angle)))
        shifted = np.roll(np.roll(g, -dy, axis=0), -dx, axis=1)
        out |= ((shifted >= g).astype(np.int32)) << bit
    return out.astype(np.uint8)


def _texture(gray):
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    lbp  = _lbp_map(gray)
    hist, _ = np.histogram(lbp, bins=_LBP_N + 2,
                           range=(0, _LBP_N + 2), density=True)
    hist += 1e-12
    entropy    = float(stats.entropy(hist))
    uniformity = float((hist ** 2).sum())
    return [lap_var, entropy, uniformity]


def _dct(gray):
    g = gray.astype(np.float32)
    h, w = g.shape
    h8, w8 = (h // 8) * 8, (w // 8) * 8
    g = g[:h8, :w8]
    # reshape to (N, 8, 8) blocks, batch DCT
    blocks = (g.reshape(h8 // 8, 8, w8 // 8, 8)
               .transpose(0, 2, 1, 3)
               .reshape(-1, 8, 8))
    dct_b = dctn(blocks, axes=(1, 2), norm="ortho")
    ac    = np.abs(dct_b.reshape(-1, 64)[:, 1:]).mean(axis=1)
    return [float(ac.mean()), float(ac.std()), float(stats.skew(ac))]


def _edge_density(gray):
    edges = cv2.Canny(gray, 50, 150)
    return [float(edges.sum() // 255) / edges.size]


# ── public API ───────────────────────────────────────────────────────

def extract(path: str) -> np.ndarray:
    """Return a float32 vector of length 20 for the image at `path`."""
    bgr  = _load(path)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    vec  = (_fft(gray) + _hough(gray) + _glare(bgr) +
            _color(bgr) + _texture(gray) + _dct(gray) +
            _edge_density(gray))
    assert len(vec) == 20
    return np.array(vec, dtype=np.float32)


if __name__ == "__main__":
    import sys
    v = extract(sys.argv[1])
    for n, x in zip(NAMES, v):
        print(f"  {n:<22} {x:.5f}")
