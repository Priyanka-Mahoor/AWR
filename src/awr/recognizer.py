"""EMNIST character CNN inference (NumPy — no TensorFlow at runtime)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

LABELS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    e = np.exp(z)
    return e / np.sum(e)


def _conv2d(x: np.ndarray, kernel: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """x (H,W,Cin), kernel (kh,kw,Cin,Cout) → valid convolution."""
    kh, kw, _cin, cout = kernel.shape
    h, w, _ = x.shape
    out_h, out_w = h - kh + 1, w - kw + 1
    cols = np.lib.stride_tricks.as_strided(
        x,
        shape=(out_h, out_w, kh, kw, x.shape[2]),
        strides=x.strides[:2] + x.strides,
        writeable=False,
    ).reshape(out_h * out_w, -1)
    flat_k = kernel.reshape(-1, cout)
    out = (cols @ flat_k + bias).reshape(out_h, out_w, cout)
    return out.astype(np.float32, copy=False)


def _maxpool2(x: np.ndarray) -> np.ndarray:
    h, w, c = x.shape
    h2, w2 = h // 2, w // 2
    cropped = x[: h2 * 2, : w2 * 2, :]
    return cropped.reshape(h2, 2, w2, 2, c).max(axis=(1, 3))


def _disambiguate(probs: np.ndarray, geometry: dict | None) -> tuple[str, float]:
    """Fix common air-writing confusions using stroke geometry."""
    order = np.argsort(-probs)
    top_char = LABELS[int(order[0])]
    top_p = float(probs[order[0]])

    def prob_of(ch: str) -> float:
        return float(probs[LABELS.index(ch)])

    def pick(ch: str) -> tuple[str, float]:
        return ch, max(prob_of(ch), top_p * 0.9)

    if geometry is None:
        return top_char, top_p

    v = geometry.get("vertical", 0)
    h = geometry.get("horizontal", 0)
    n = geometry.get("n", 0)

    # H: two uprights + crossbar. CNN often says I/1/4 on air-written H.
    looks_like_h = v >= 2 and h >= 1
    if looks_like_h and top_char in {"4", "I", "1", "L", "U", "N"}:
        return pick("H")

    # Single vertical stem misread as H → keep I/1/4
    if top_char == "H" and (n <= 2 or (v <= 1 and h == 0)):
        for alt in ("I", "1", "L", "4"):
            if prob_of(alt) > 0.15:
                return pick(alt)

    # 0 vs O: round single blob / one loop stroke → prefer 0 if digit-like confidence close
    if top_char in {"0", "O"} and n <= 2:
        # keep model choice unless nearly tied the other way
        other = "O" if top_char == "0" else "0"
        if abs(prob_of("0") - prob_of("O")) < 0.08:
            return top_char, top_p

    # 1 vs I / L: one vertical stroke
    if top_char in {"1", "I", "L"} and v >= 1 and h == 0 and n <= 2:
        # Prefer letter I when writing words (user asked for H words); keep model top
        pass

    return top_char, top_p


class CharRecognizer:
    def __init__(self, model_path: str | Path | None = None) -> None:
        root = Path(__file__).resolve().parents[2]
        path = Path(model_path) if model_path else root / "models" / "char_cnn_weights.npz"
        if path.suffix in {".keras", ".h5"}:
            path = path.with_name("char_cnn_weights.npz")
        self.weights_path = path
        self.available = False
        self.error: str | None = None
        self._w: list[np.ndarray] = []

        if not self.weights_path.is_file():
            self.error = (
                f"Weights not found: {self.weights_path}. "
                "Run: python scripts/train_emnist.py && python scripts/export_weights.py"
            )
            return
        try:
            data = np.load(self.weights_path)
            self._w = [data[f"w{i}"] for i in range(8)]
            self.available = True
        except Exception as exc:  # noqa: BLE001
            self.error = f"Failed to load weights: {exc}"

    def _probs(self, emnist_28: np.ndarray) -> np.ndarray | None:
        if not self.available or emnist_28 is None:
            return None
        x = emnist_28.astype(np.float32)
        if x.ndim == 2:
            x = x[..., np.newaxis]
        k0, b0, k1, b1, k2, b2, k3, b3 = self._w
        x = _relu(_conv2d(x, k0, b0))
        x = _maxpool2(x)
        x = _relu(_conv2d(x, k1, b1))
        x = _maxpool2(x)
        x = _relu(x.reshape(-1) @ k2 + b2)
        return _softmax(x @ k3 + b3)

    def predict(
        self, emnist_28: np.ndarray, geometry: dict | None = None
    ) -> tuple[str, float] | None:
        probs = self._probs(emnist_28)
        if probs is None:
            return None
        return _disambiguate(probs, geometry)
