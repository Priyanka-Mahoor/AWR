#!/usr/bin/env python3
"""Generate synthetic 28x28 air-writing-like samples for 0-9 and A-Z.

Saves into data/user_chars/<char>/*.npy (same format as Teach).
Existing Teach samples are kept.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from awr.learn import USER_DIR, _normalize  # noqa: E402
from awr.recognizer import LABELS  # noqa: E402

WIN_FONTS = Path(r"C:\Windows\Fonts")
FONT_FILES = [
    "arial.ttf",
    "arialbd.ttf",
    "calibri.ttf",
    "calibrib.ttf",
    "consola.ttf",
    "cour.ttf",
    "segoeui.ttf",
    "segoeuib.ttf",
    "tahoma.ttf",
    "verdana.ttf",
]


def _to_emnist(mask: np.ndarray) -> np.ndarray | None:
    """Crop, pad, resize to 28x28 white-on-black float32 — same idea as InkCanvas."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    pad = 4
    h, w = mask.shape
    x0 = max(0, int(xs.min()) - pad)
    y0 = max(0, int(ys.min()) - pad)
    x1 = min(w, int(xs.max()) + pad + 1)
    y1 = min(h, int(ys.max()) + pad + 1)
    crop = mask[y0:y1, x0:x1]
    if crop.shape[0] < 4 or crop.shape[1] < 4:
        return None
    side = int(max(crop.shape[0], crop.shape[1]) * 1.25)
    side = max(side, 28)
    square = np.zeros((side, side), dtype=np.uint8)
    oy = (side - crop.shape[0]) // 2
    ox = (side - crop.shape[1]) // 2
    square[oy : oy + crop.shape[0], ox : ox + crop.shape[1]] = crop
    resized = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)
    _, resized = cv2.threshold(resized, 20, 255, cv2.THRESH_BINARY)
    return resized.astype(np.float32) / 255.0


def _load_fonts() -> list[Path]:
    found = [WIN_FONTS / name for name in FONT_FILES if (WIN_FONTS / name).is_file()]
    return found or []


def _draw_pil(ch: str, font_path: Path, size: int, rng: np.random.Generator) -> np.ndarray:
    canvas = Image.new("L", (96, 96), 0)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(str(font_path), size)
    bbox = draw.textbbox((0, 0), ch, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (96 - tw) // 2 - bbox[0]
    y = (96 - th) // 2 - bbox[1]
    draw.text((x, y), ch, fill=255, font=font)
    img = np.array(canvas)
    if rng.random() < 0.55:
        k = 2 if rng.random() < 0.5 else 3
        img = cv2.erode(img, np.ones((k, k), np.uint8), iterations=1)
    return img


def _draw_hershey(ch: str, rng: np.random.Generator) -> np.ndarray:
    img = np.zeros((96, 96), dtype=np.uint8)
    fonts = [
        cv2.FONT_HERSHEY_SIMPLEX,
        cv2.FONT_HERSHEY_PLAIN,
        cv2.FONT_HERSHEY_DUPLEX,
        cv2.FONT_HERSHEY_COMPLEX,
        cv2.FONT_HERSHEY_TRIPLEX,
        cv2.FONT_HERSHEY_COMPLEX_SMALL,
    ]
    font = fonts[int(rng.integers(0, len(fonts)))]
    scale = float(rng.uniform(1.4, 2.4))
    thick = int(rng.choice([1, 1, 1, 2]))
    (tw, th), _ = cv2.getTextSize(ch, font, scale, thick)
    x = max(2, (96 - tw) // 2)
    y = min(94, (96 + th) // 2)
    cv2.putText(img, ch, (x, y), font, scale, 255, thick, cv2.LINE_AA)
    if rng.random() < 0.4:
        img = cv2.erode(img, np.ones((2, 2), np.uint8), iterations=1)
    return img


def _jitter(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    angle = float(rng.uniform(-16, 16))
    tx, ty = float(rng.uniform(-5, 5)), float(rng.uniform(-5, 5))
    scale = float(rng.uniform(0.78, 1.18))
    h, w = img.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    m[0, 2] += tx
    m[1, 2] += ty
    out = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    if rng.random() < 0.25:
        noise = rng.integers(0, 40, out.shape, dtype=np.uint8)
        out = np.where(out > 0, np.clip(out.astype(np.int16) - noise, 0, 255).astype(np.uint8), out)
    if rng.random() < 0.2:
        ksize = int(rng.choice([3, 5]))
        out = cv2.GaussianBlur(out, (ksize, ksize), 0)
        _, out = cv2.threshold(out, 40, 255, cv2.THRESH_BINARY)
    return out


def generate_one(ch: str, fonts: list[Path], rng: np.random.Generator) -> np.ndarray | None:
    if fonts and rng.random() < 0.7:
        font = fonts[int(rng.integers(0, len(fonts)))]
        size = int(rng.integers(36, 58))
        raw = _draw_pil(ch, font, size, rng)
    else:
        raw = _draw_hershey(ch, rng)
    raw = _jitter(raw, rng)
    return _to_emnist(raw)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=20)
    args = parser.parse_args()

    fonts = _load_fonts()
    rng = np.random.default_rng(20260822)
    stamp = int(time.time() * 1000)
    saved = 0
    failed: list[str] = []

    for ci, ch in enumerate(LABELS):
        folder = USER_DIR / ch
        folder.mkdir(parents=True, exist_ok=True)
        n_ok = 0
        attempts = 0
        while n_ok < args.per_class and attempts < args.per_class * 8:
            attempts += 1
            img = generate_one(ch, fonts, rng)
            if img is None or float(img.sum()) < 8:
                continue
            np.save(folder / f"syn_{stamp}_{ci}_{n_ok}.npy", _normalize(img))
            n_ok += 1
            saved += 1
        if n_ok < args.per_class:
            failed.append(f"{ch}:{n_ok}")
        print(f"  {ch}: {n_ok}")

    print(f"Saved {saved} synthetic samples under {USER_DIR}")
    if failed:
        print("Short classes:", ", ".join(failed))


if __name__ == "__main__":
    main()
