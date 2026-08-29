#!/usr/bin/env python3
"""Fine-tune the EMNIST CNN on your air-writing samples.

Collect samples in the app with Teach / Fix, then run this script.
Uses models/char_cnn.keras as a starting point (EMNIST) and oversamples
your letters so the model learns your stroke style.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from awr.learn import load_samples, write_finetune_info  # noqa: E402
from awr.recognizer import LABELS  # noqa: E402

KERAS = ROOT / "models" / "char_cnn.keras"
NPZ = ROOT / "models" / "char_cnn_weights.npz"


def augment(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Make extra variants that look like shaky, thin air writing."""
    x = img.copy()
    angle = float(rng.uniform(-18, 18))
    tx, ty = float(rng.uniform(-3, 3)), float(rng.uniform(-3, 3))
    scale = float(rng.uniform(0.75, 1.15))
    m = cv2.getRotationMatrix2D((14, 14), angle, scale)
    m[0, 2] += tx
    m[1, 2] += ty
    x = cv2.warpAffine(x, m, (28, 28), flags=cv2.INTER_LINEAR, borderValue=0)
    k = int(rng.integers(0, 3))
    if k == 1:
        x = cv2.dilate(x, np.ones((2, 2), np.uint8))
    elif k == 2:
        x = cv2.erode((x * 255).astype(np.uint8), np.ones((2, 2), np.uint8))
        x = x.astype(np.float32) / 255.0
    if rng.random() < 0.3:
        x = np.clip(x + rng.normal(0, 0.08, x.shape).astype(np.float32), 0, 1)
    return x


def expand_user(x: np.ndarray, y: np.ndarray, repeats: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    images = []
    labels = []
    for i in range(len(x)):
        images.append(x[i])
        labels.append(y[i])
        for _ in range(repeats):
            images.append(augment(x[i], rng))
            labels.append(y[i])
    return np.stack(images), np.array(labels, dtype=np.int32)


def export_npz(model, out: Path) -> None:
    arrays: list[np.ndarray] = []
    for layer in model.layers:
        arrays.extend(np.asarray(w, dtype=np.float32) for w in layer.get_weights())
    if len(arrays) != 8:
        raise SystemExit(f"Expected 8 weight arrays, got {len(arrays)}")
    np.savez(out, **{f"w{i}": a for i, a in enumerate(arrays)})
    print(f"Wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    x, y = load_samples()
    if len(x) < 5:
        raise SystemExit(
            f"Only {len(x)} user sample(s) in data/user_chars.\n"
            "In the app: write a letter, click Teach, type the letter. "
            "Do this ~10 times per letter you care about, then rerun."
        )
    print(f"User samples: {len(x)}")
    n_user = len(x)
    counts = {LABELS[i]: int(np.sum(y == i)) for i in range(len(LABELS)) if np.any(y == i)}
    print("Per letter:", counts)

    x, y = expand_user(x, y, args.repeats)
    x = x[..., np.newaxis]
    print(f"After augmentation: {len(x)}")

    from tensorflow import keras

    if not KERAS.is_file():
        raise SystemExit(
            f"Missing {KERAS}. Train EMNIST first:\n"
            "  python scripts/train_emnist.py --epochs 8"
        )
    model = keras.models.load_model(KERAS)
    model.compile(
        optimizer=keras.optimizers.Adam(args.lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(x))
    x, y = x[idx], y[idx]
    model.fit(x, y, epochs=args.epochs, batch_size=32, validation_split=0.1)
    KERAS.parent.mkdir(parents=True, exist_ok=True)
    model.save(KERAS)
    print(f"Saved {KERAS}")
    export_npz(model, NPZ)
    write_finetune_info(n_user, counts)
    print("Restart the app to load the updated model.")


if __name__ == "__main__":
    main()
