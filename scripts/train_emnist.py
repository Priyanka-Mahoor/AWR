#!/usr/bin/env python3
"""Train a 36-class CNN on EMNIST digits + uppercase letters.

Stroke convention: white ink on black background, 28x28, values in [0, 1].
EMNIST byclass labels 0-9 = digits, 10-35 = A-Z (we drop lowercase 36-61).
Images are transposed to upright orientation (EMNIST quirk).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

LABELS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
NUM_CLASSES = 36
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load_emnist_numpy():
    import tensorflow_datasets as tfds

    ds = tfds.load("emnist/byclass", split="train", as_supervised=True, shuffle_files=True)
    images = []
    labels = []
    for img, label in tfds.as_numpy(ds):
        y = int(label)
        if y > 35:
            continue
        # EMNIST is transposed relative to visual upright
        x = np.transpose(img.squeeze())
        images.append(x)
        labels.append(y)
        if len(images) >= 200_000:  # cap for reasonable train time
            break
    x = np.stack(images).astype(np.float32) / 255.0
    y = np.array(labels, dtype=np.int32)
    return x, y


def build_model():
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential(
        [
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(NUM_CLASSES, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _air_writing_augment(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Shift/rotate/thin EMNIST so it looks closer to webcam air writing."""
    import cv2

    out = np.empty_like(x)
    for i in range(len(x)):
        img = x[i, ..., 0]
        angle = float(rng.uniform(-12, 12))
        tx, ty = float(rng.uniform(-2.5, 2.5)), float(rng.uniform(-2.5, 2.5))
        scale = float(rng.uniform(0.85, 1.12))
        m = cv2.getRotationMatrix2D((14, 14), angle, scale)
        m[0, 2] += tx
        m[1, 2] += ty
        img = cv2.warpAffine(img, m, (28, 28), flags=cv2.INTER_LINEAR, borderValue=0)
        if rng.random() < 0.4:
            img = cv2.erode((img * 255).astype(np.uint8), np.ones((2, 2), np.uint8))
            img = img.astype(np.float32) / 255.0
        out[i, ..., 0] = img
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "models" / "char_cnn.keras",
    )
    parser.add_argument(
        "--air-aug",
        action="store_true",
        help="Augment EMNIST toward thin/wobbly air writing",
    )
    parser.add_argument(
        "--mix-user",
        action="store_true",
        help="Mix in data/user_chars samples (from Teach/Fix in the app)",
    )
    parser.add_argument("--user-repeats", type=int, default=50)
    args = parser.parse_args()

    print("Loading EMNIST byclass (digits + uppercase)...")
    x, y = load_emnist_numpy()
    x = x[..., np.newaxis]
    print(f"Samples: {len(x)}")
    if args.air_aug:
        print("Applying air-writing augmentation to a copy of the set…")
        rng_aug = np.random.default_rng(1)
        extra = _air_writing_augment(x, rng_aug)
        x = np.concatenate([x, extra], axis=0)
        y = np.concatenate([y, y], axis=0)
        print(f"Samples after aug: {len(x)}")
    if args.mix_user:
        from awr.learn import load_samples

        ux, uy = load_samples()
        if len(ux) == 0:
            print("No user samples in data/user_chars — skipping mix.")
        else:
            reps = [ux] * args.user_repeats
            rlab = [uy] * args.user_repeats
            ux = np.concatenate(reps, axis=0)[..., np.newaxis]
            uy = np.concatenate(rlab, axis=0)
            x = np.concatenate([x, ux], axis=0)
            y = np.concatenate([y, uy], axis=0)
            print(f"Mixed {len(ux)} user (repeated) samples. Total: {len(x)}")

    # Shuffle and split
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(x))
    x, y = x[idx], y[idx]
    n_val = max(1000, len(x) // 10)
    x_val, y_val = x[:n_val], y[:n_val]
    x_train, y_train = x[n_val:], y[n_val:]

    model = build_model()
    model.summary()
    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out)
    print(f"Saved {args.out}")
    # Quick sanity
    probs = model.predict(x_val[:8], verbose=0)
    for i, p in enumerate(probs):
        print(f"  val[{i}] -> {LABELS[int(p.argmax())]} ({p.max():.2f})")


if __name__ == "__main__":
    main()
