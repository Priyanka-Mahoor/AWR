#!/usr/bin/env python3
"""Generate air-writing style 28x28 samples (shaky finger strokes, not fonts).

Each character is drawn as 1–3 polylines (like index-finger air writing),
then jittered, smoothed, and cropped the same way as the app's ink.
Saves data/user_chars/<char>/air_*.npy  — real Teach samples are kept.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from awr.learn import USER_DIR, _normalize  # noqa: E402
from awr.recognizer import LABELS  # noqa: E402

# Unit-box strokes, y down. Each char → list of strokes → list of (x, y).
# These mimic typical air-written block letters (separate lifts).
TEMPLATES: dict[str, list[list[tuple[float, float]]]] = {
    "0": [[(0.50, 0.08), (0.18, 0.22), (0.12, 0.50), (0.18, 0.78), (0.50, 0.92),
           (0.82, 0.78), (0.88, 0.50), (0.82, 0.22), (0.50, 0.08)]],
    "1": [[(0.32, 0.22), (0.50, 0.08), (0.50, 0.92)], [(0.28, 0.92), (0.72, 0.92)]],
    "2": [[(0.18, 0.22), (0.35, 0.08), (0.68, 0.08), (0.82, 0.22), (0.82, 0.38),
           (0.18, 0.88), (0.82, 0.88)]],
    "3": [[(0.22, 0.12), (0.72, 0.12), (0.78, 0.28), (0.42, 0.48), (0.78, 0.62),
           (0.72, 0.88), (0.22, 0.88)]],
    "4": [[(0.68, 0.08), (0.68, 0.92)], [(0.22, 0.62), (0.82, 0.62)], [(0.22, 0.62), (0.62, 0.08)]],
    "5": [[(0.78, 0.10), (0.22, 0.10), (0.22, 0.42), (0.62, 0.42), (0.82, 0.58),
           (0.78, 0.82), (0.22, 0.90)]],
    "6": [[(0.72, 0.12), (0.28, 0.22), (0.16, 0.52), (0.28, 0.85), (0.58, 0.92),
           (0.82, 0.72), (0.58, 0.50), (0.28, 0.58)]],
    "7": [[(0.18, 0.10), (0.82, 0.10), (0.38, 0.92)]],
    "8": [[(0.50, 0.08), (0.22, 0.22), (0.22, 0.40), (0.50, 0.50), (0.78, 0.60),
           (0.78, 0.82), (0.50, 0.92), (0.22, 0.82), (0.22, 0.62), (0.50, 0.50),
           (0.78, 0.40), (0.78, 0.22), (0.50, 0.08)]],
    "9": [[(0.78, 0.48), (0.72, 0.18), (0.42, 0.08), (0.18, 0.28), (0.28, 0.48),
           (0.78, 0.48), (0.72, 0.92)]],
    "A": [[(0.12, 0.92), (0.50, 0.08), (0.88, 0.92)], [(0.30, 0.58), (0.70, 0.58)]],
    "B": [[(0.22, 0.08), (0.22, 0.92)], [(0.22, 0.08), (0.68, 0.08), (0.78, 0.22),
           (0.68, 0.42), (0.22, 0.48), (0.70, 0.52), (0.82, 0.70), (0.68, 0.92), (0.22, 0.92)]],
    "C": [[(0.82, 0.18), (0.55, 0.08), (0.22, 0.28), (0.18, 0.50), (0.22, 0.75),
           (0.55, 0.92), (0.82, 0.82)]],
    "D": [[(0.22, 0.08), (0.22, 0.92)], [(0.22, 0.08), (0.62, 0.10), (0.82, 0.32),
           (0.82, 0.68), (0.62, 0.90), (0.22, 0.92)]],
    "E": [[(0.22, 0.08), (0.22, 0.92)], [(0.22, 0.08), (0.82, 0.08)],
          [(0.22, 0.50), (0.68, 0.50)], [(0.22, 0.92), (0.82, 0.92)]],
    "F": [[(0.22, 0.08), (0.22, 0.92)], [(0.22, 0.08), (0.82, 0.08)],
          [(0.22, 0.50), (0.68, 0.50)]],
    "G": [[(0.82, 0.18), (0.50, 0.08), (0.20, 0.28), (0.16, 0.55), (0.28, 0.85),
           (0.58, 0.92), (0.84, 0.72), (0.84, 0.52), (0.52, 0.52)]],
    "H": [[(0.22, 0.08), (0.22, 0.92)], [(0.78, 0.08), (0.78, 0.92)],
          [(0.22, 0.50), (0.78, 0.50)]],
    "I": [[(0.28, 0.08), (0.72, 0.08)], [(0.50, 0.08), (0.50, 0.92)],
          [(0.28, 0.92), (0.72, 0.92)]],
    "J": [[(0.22, 0.08), (0.78, 0.08)], [(0.62, 0.08), (0.62, 0.72), (0.48, 0.92),
           (0.22, 0.82)]],
    "K": [[(0.22, 0.08), (0.22, 0.92)], [(0.78, 0.08), (0.22, 0.52), (0.82, 0.92)]],
    "L": [[(0.22, 0.08), (0.22, 0.92), (0.82, 0.92)]],
    "M": [[(0.12, 0.92), (0.12, 0.08), (0.50, 0.55), (0.88, 0.08), (0.88, 0.92)]],
    "N": [[(0.18, 0.92), (0.18, 0.08), (0.82, 0.92), (0.82, 0.08)]],
    "O": [[(0.50, 0.08), (0.18, 0.25), (0.12, 0.50), (0.18, 0.78), (0.50, 0.92),
           (0.82, 0.78), (0.88, 0.50), (0.82, 0.25), (0.50, 0.08)]],
    "P": [[(0.22, 0.92), (0.22, 0.08), (0.68, 0.08), (0.80, 0.22), (0.68, 0.42), (0.22, 0.48)]],
    "Q": [[(0.50, 0.08), (0.18, 0.25), (0.12, 0.50), (0.18, 0.75), (0.50, 0.88),
           (0.82, 0.75), (0.88, 0.50), (0.82, 0.25), (0.50, 0.08)],
          [(0.58, 0.68), (0.88, 0.95)]],
    "R": [[(0.22, 0.92), (0.22, 0.08), (0.68, 0.08), (0.80, 0.22), (0.68, 0.42),
           (0.22, 0.48)], [(0.48, 0.48), (0.82, 0.92)]],
    "S": [[(0.78, 0.18), (0.50, 0.08), (0.22, 0.18), (0.22, 0.38), (0.50, 0.50),
           (0.78, 0.62), (0.78, 0.82), (0.50, 0.92), (0.22, 0.82)]],
    "T": [[(0.12, 0.10), (0.88, 0.10)], [(0.50, 0.10), (0.50, 0.92)]],
    "U": [[(0.18, 0.08), (0.18, 0.68), (0.32, 0.90), (0.50, 0.94), (0.68, 0.90),
           (0.82, 0.68), (0.82, 0.08)]],
    "V": [[(0.12, 0.08), (0.50, 0.92), (0.88, 0.08)]],
    "W": [[(0.08, 0.08), (0.28, 0.92), (0.50, 0.40), (0.72, 0.92), (0.92, 0.08)]],
    "X": [[(0.18, 0.08), (0.82, 0.92)], [(0.82, 0.08), (0.18, 0.92)]],
    "Y": [[(0.15, 0.08), (0.50, 0.48), (0.85, 0.08)], [(0.50, 0.48), (0.50, 0.92)]],
    "Z": [[(0.18, 0.10), (0.82, 0.10), (0.18, 0.90), (0.82, 0.90)]],
}


def _resample(pts: list[tuple[float, float]], spacing: float = 0.02) -> np.ndarray:
    if len(pts) < 2:
        return np.asarray(pts, dtype=np.float32)
    out = [pts[0]]
    for i in range(1, len(pts)):
        x0, y0 = out[-1] if i == 1 else pts[i - 1]
        x0, y0 = pts[i - 1]
        x1, y1 = pts[i]
        d = math.hypot(x1 - x0, y1 - y0)
        n = max(1, int(d / spacing))
        for k in range(1, n + 1):
            t = k / n
            out.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t))
    return np.asarray(out, dtype=np.float32)


def _wobble(pts: np.ndarray, rng: np.random.Generator, amp: float) -> np.ndarray:
    n = len(pts)
    if n < 3:
        return pts
    # Smooth 1/f-ish noise along the stroke (hand tremor).
    noise = rng.normal(0, amp, size=(n, 2)).astype(np.float32)
    kernel = np.array([0.15, 0.2, 0.3, 0.2, 0.15], dtype=np.float32)
    pad = np.pad(noise, ((2, 2), (0, 0)), mode="edge")
    smooth = np.stack(
        [np.convolve(pad[:, 0], kernel, mode="valid"), np.convolve(pad[:, 1], kernel, mode="valid")],
        axis=1,
    )[:n]
    # Keep endpoints closer to the template so the letter stays readable.
    fade = np.linspace(0.25, 1.0, n, dtype=np.float32)
    fade = np.minimum(fade, fade[::-1])[:, None]
    return pts + smooth * fade


def _ema(pts: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    out = np.empty_like(pts)
    out[0] = pts[0]
    for i in range(1, len(pts)):
        out[i] = alpha * pts[i] + (1 - alpha) * out[i - 1]
    return out


def _render(
    strokes: list[np.ndarray], rng: np.random.Generator, size: int = 160
) -> np.ndarray:
    canvas = np.zeros((size, size), dtype=np.uint8)
    # Random pose like a hand writing in front of the camera.
    rot = float(rng.uniform(-14, 14))
    scale = float(rng.uniform(0.72, 1.12))
    ox, oy = float(rng.uniform(-0.06, 0.06)), float(rng.uniform(-0.06, 0.06))
    thick = int(rng.choice([2, 3, 3, 4]))
    rad = math.radians(rot)
    c, s = math.cos(rad), math.sin(rad)
    for stroke in strokes:
        pts = []
        for x, y in stroke:
            x, y = (x - 0.5) * scale + 0.5 + ox, (y - 0.5) * scale + 0.5 + oy
            xr = (x - 0.5) * c - (y - 0.5) * s + 0.5
            yr = (x - 0.5) * s + (y - 0.5) * c + 0.5
            px = int(np.clip(xr, 0, 1) * (size - 1))
            py = int(np.clip(yr, 0, 1) * (size - 1))
            pts.append((px, py))
        if len(pts) < 2:
            continue
        arr = np.asarray(pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [arr], False, 255, thick, cv2.LINE_8)
    return canvas


def _to_emnist(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    pad = 10
    h, w = mask.shape
    x0, y0 = max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad)
    x1, y1 = min(w, int(xs.max()) + pad + 1), min(h, int(ys.max()) + pad + 1)
    crop = mask[y0:y1, x0:x1]
    if min(crop.shape) < 6:
        return None
    side = int(max(crop.shape) * 1.28)
    side = max(side, 28)
    square = np.zeros((side, side), dtype=np.uint8)
    oy, ox = (side - crop.shape[0]) // 2, (side - crop.shape[1]) // 2
    square[oy : oy + crop.shape[0], ox : ox + crop.shape[1]] = crop
    resized = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)
    _, resized = cv2.threshold(resized, 18, 255, cv2.THRESH_BINARY)
    return resized.astype(np.float32) / 255.0


def generate_one(ch: str, rng: np.random.Generator) -> np.ndarray | None:
    tmpl = TEMPLATES[ch]
    amp = float(rng.uniform(0.012, 0.035))
    strokes = []
    for raw in tmpl:
        pts = _resample(raw)
        pts = _wobble(pts, rng, amp)
        pts = _ema(pts, alpha=float(rng.uniform(0.28, 0.5)))
        strokes.append(pts)
    canvas = _render(strokes, rng)
    return _to_emnist(canvas)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=20)
    args = parser.parse_args()

    rng = np.random.default_rng(20260822)
    stamp = int(time.time() * 1000)
    saved = 0
    for ci, ch in enumerate(LABELS):
        folder = USER_DIR / ch
        folder.mkdir(parents=True, exist_ok=True)
        n_ok = 0
        attempts = 0
        while n_ok < args.per_class and attempts < args.per_class * 10:
            attempts += 1
            img = generate_one(ch, rng)
            if img is None or float(img.sum()) < 6:
                continue
            np.save(folder / f"air_{stamp}_{ci}_{n_ok}.npy", _normalize(img))
            n_ok += 1
            saved += 1
        print(f"  {ch}: {n_ok}")
    print(f"Saved {saved} air-writing samples under {USER_DIR}")


if __name__ == "__main__":
    main()
