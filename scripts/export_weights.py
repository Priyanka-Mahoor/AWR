#!/usr/bin/env python3
"""Export trained Keras CNN weights to NumPy .npz for runtime inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from tensorflow import keras

ROOT = Path(__file__).resolve().parents[1]
KERAS = ROOT / "models" / "char_cnn.keras"
OUT = ROOT / "models" / "char_cnn_weights.npz"


def main() -> None:
    model = keras.models.load_model(KERAS)
    arrays: list[np.ndarray] = []
    for layer in model.layers:
        for w in layer.get_weights():
            arrays.append(np.asarray(w, dtype=np.float32))
            print(layer.name, w.shape)
    if len(arrays) != 8:
        raise SystemExit(f"Expected 8 weight arrays, got {len(arrays)}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, **{f"w{i}": a for i, a in enumerate(arrays)})
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
