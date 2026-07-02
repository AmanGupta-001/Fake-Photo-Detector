"""
generate_demo_data.py — Create synthetic demo images for training.

Generates realistic-enough synthetic images so train.py / predict.py can
be demonstrated without real phone photos:
  real/   — images that look like natural photographs (noise, gradients, texture)
  screen/ — images that look like photos of a screen (grid, moiré, glare, bezel)

Usage:
    python generate_demo_data.py
    python generate_demo_data.py --n 50 --out_dir .
"""

import argparse
import os
import random
from pathlib import Path

import cv2
import numpy as np


def make_real_image(size: int = 512, seed: int = 0) -> np.ndarray:
    """Simulate a natural photo: organic texture, warm/natural colours, no bezel."""
    rng = np.random.default_rng(seed)

    # Base — smooth gradient background (sky / ground)
    img = np.zeros((size, size, 3), dtype=np.float32)
    sky_col  = rng.uniform([80, 120, 160], [160, 200, 255])   # blue-ish sky
    gnd_col  = rng.uniform([60,  80,  30], [120, 150,  80])   # green-ish ground
    for y in range(size):
        t = y / size
        img[y] = (1 - t) * sky_col + t * gnd_col

    # Add organic noise (simulates leaves, grass, fabric)
    noise = rng.standard_normal((size, size, 3)).astype(np.float32) * 18
    img  += noise

    # Add a few soft blobs (rocks, clouds…)
    n_blobs = rng.integers(3, 8)
    for _ in range(n_blobs):
        cx, cy = rng.integers(0, size, 2)
        r      = rng.integers(20, size // 4)
        colour = rng.uniform(50, 220, 3).astype(np.float32)
        mask   = np.zeros((size, size), dtype=np.uint8)
        cv2.circle(mask, (int(cx), int(cy)), int(r), 255, -1)
        mask_f = cv2.GaussianBlur(mask, (0, 0), r // 3 + 1).astype(np.float32) / 255
        img   += mask_f[:, :, None] * (colour - img) * 0.4

    # JPEG-style compression artefacts (real photos have one round of JPEG)
    img = np.clip(img, 0, 255).astype(np.uint8)
    enc_param = [cv2.IMWRITE_JPEG_QUALITY, rng.integers(75, 95)]
    _, buf = cv2.imencode(".jpg", img, enc_param)
    img    = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    # Slight vignette
    Y, X   = np.ogrid[:size, :size]
    dist   = np.sqrt((X - size / 2) ** 2 + (Y - size / 2) ** 2)
    vignette = 1 - 0.4 * (dist / dist.max()) ** 2
    img    = (img.astype(np.float32) * vignette[:, :, None]).clip(0, 255).astype(np.uint8)

    return img


def make_screen_image(size: int = 512, seed: int = 0) -> np.ndarray:
    """Simulate a photo of a screen: moiré, pixel grid, bezel, glare."""
    rng = np.random.default_rng(seed)

    # Rendered content on the screen (app UI / web page feel)
    img = np.zeros((size, size, 3), dtype=np.float32)

    # Background – coolish, backlit
    bg = rng.uniform([30, 40, 60], [80, 100, 130])
    img[:] = bg

    # Horizontal text-like bands
    n_bands = rng.integers(6, 16)
    for _ in range(n_bands):
        y = rng.integers(0, size)
        h = rng.integers(2, 12)
        c = rng.uniform(150, 255, 3)
        img[y:y + h, 40:size - 40] = c

    # Rectangular UI elements
    for _ in range(rng.integers(4, 10)):
        x1, y1 = rng.integers(0, size - 80, 2)
        w       = rng.integers(40, size // 3)
        h       = rng.integers(20, size // 4)
        c       = rng.uniform(80, 220, 3)
        img[y1:y1 + h, x1:x1 + w] = c

    # Screen pixel grid (sub-pixel RGB pattern at ~400 PPI spacing)
    grid_pitch = rng.integers(3, 6)  # pixels per screen pixel
    for y in range(0, size, grid_pitch):
        img[y, :] *= 0.90            # dim every N-th row slightly

    # Moiré pattern — sinusoidal interference in both axes
    freq = rng.uniform(0.04, 0.10)
    X, Y  = np.meshgrid(np.arange(size), np.arange(size))
    moire = (np.sin(freq * X) * np.sin(freq * Y)).astype(np.float32)
    img  += moire[:, :, None] * rng.uniform(5, 18)

    # Double JPEG compression (screen content was JPEG, then recaptured)
    img = np.clip(img, 0, 255).astype(np.uint8)
    for quality in [rng.integers(60, 80), rng.integers(70, 90)]:
        enc_param = [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
        _, buf = cv2.imencode(".jpg", img, enc_param)
        img    = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    # Bezel — black border around the screen
    bezel = rng.integers(20, 50)
    img = img.astype(np.float32)
    img[:bezel, :]            = 10
    img[-bezel:, :]           = 10
    img[:, :bezel]            = 10
    img[:, -bezel:]           = 10

    # Glare blob — large near-white highlight from room lighting
    n_glare = rng.integers(1, 3)
    for _ in range(n_glare):
        cx = rng.integers(bezel, size - bezel)
        cy = rng.integers(bezel, size - bezel)
        r  = rng.integers(30, 90)
        mask  = np.zeros((size, size), dtype=np.uint8)
        cv2.circle(mask, (int(cx), int(cy)), int(r), 255, -1)
        mask_f = cv2.GaussianBlur(mask, (0, 0), r // 2 + 1).astype(np.float32) / 255
        img   += mask_f[:, :, None] * (255 - img) * rng.uniform(0.5, 0.9)

    # Camera shake / slight blur from hand-holding
    ksize = rng.integers(1, 4) * 2 + 1
    img   = cv2.GaussianBlur(img.clip(0, 255).astype(np.uint8), (ksize, ksize), 0)

    return img


# ── main ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",       type=int, default=50, help="images per class")
    ap.add_argument("--size",    type=int, default=512)
    ap.add_argument("--out_dir", default=".")
    args = ap.parse_args()

    real_dir   = Path(args.out_dir) / "real"
    screen_dir = Path(args.out_dir) / "screen"
    real_dir.mkdir(parents=True, exist_ok=True)
    screen_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.n} real images  -> {real_dir}/")
    for i in range(args.n):
        img  = make_real_image(size=args.size, seed=i * 7 + 13)
        path = real_dir / f"real_{i:04d}.jpg"
        cv2.imwrite(str(path), img)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{args.n}")

    print(f"Generating {args.n} screen images -> {screen_dir}/")
    for i in range(args.n):
        img  = make_screen_image(size=args.size, seed=i * 11 + 7)
        path = screen_dir / f"screen_{i:04d}.jpg"
        cv2.imwrite(str(path), img)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{args.n}")

    print(f"\nDone! {args.n} real + {args.n} screen images ready.")
    print("Now run:  python train.py")


if __name__ == "__main__":
    main()
