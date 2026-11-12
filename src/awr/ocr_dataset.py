"""Personal air-writing dataset: capture ink and match later writes."""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import cv2
import numpy as np

from awr.ocr_space import clean_ocr_text
from awr.recognizer import LABELS

ROOT = Path(__file__).resolve().parents[2]
OCR_DIR = ROOT / "data" / "ocr_captures"
SIZE = 64
MATCH_MIN = 0.84
_INDEX = "index.jsonl"


def _dataset_root(root: Path | None) -> Path:
    return Path(root) if root is not None else OCR_DIR


def folder_name(label: str) -> str:
    cleaned = clean_ocr_text(label) or "UNLABELED"
    safe = "".join(ch if ch.isalnum() else "_" for ch in cleaned)
    return (safe or "UNLABELED")[:48]


def fingerprint(png: bytes) -> np.ndarray | None:
    arr = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if arr is None or arr.size == 0:
        return None
    if float(arr.mean()) > 127:
        arr = 255 - arr
    arr = cv2.resize(arr, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    _, arr = cv2.threshold(arr, 20, 255, cv2.THRESH_BINARY)
    vec = arr.astype(np.float32).ravel()
    vec -= float(vec.mean())
    norm = float(np.linalg.norm(vec))
    if norm < 1.0:
        return None
    return vec / norm


def save_capture(png: bytes, label: str, root: Path | None = None) -> Path | None:
    """Store an air-written sample labeled with recognized (or taught) text."""
    text = clean_ocr_text(label)
    if not text or text.upper() == "(UNCLEAR)" or png is None:
        return None
    fp = fingerprint(png)
    if fp is None:
        return None
    base = _dataset_root(root)
    folder = base / folder_name(text)
    folder.mkdir(parents=True, exist_ok=True)
    stamp = int(time.time() * 1000)
    path = folder / f"{stamp}.png"
    path.write_bytes(png)
    row = {"label": text, "file": str(path.relative_to(base)), "when": stamp}
    with (base / _INDEX).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")
    return path


def _iter_samples(root: Path) -> list[tuple[str, np.ndarray]]:
    samples: list[tuple[str, np.ndarray]] = []
    if not root.is_dir():
        return samples
    for png_path in root.glob("*/*.png"):
        label = clean_ocr_text(png_path.parent.name.replace("_", " "))
        if not label:
            continue
        fp = fingerprint(png_path.read_bytes())
        if fp is not None:
            samples.append((label, fp))
    return samples


def match_capture(png: bytes, root: Path | None = None) -> tuple[str, float] | None:
    """Return (label, score) when the writing matches a stored sample."""
    query = fingerprint(png)
    if query is None:
        return None
    best_label = ""
    best = -1.0
    for label, fp in _iter_samples(_dataset_root(root)):
        score = float(np.dot(query, fp))
        if score > best:
            best = score
            best_label = label
    if best >= MATCH_MIN and best_label:
        return best_label, best
    return None


def _templates() -> dict[str, list[list[tuple[float, float]]]]:
    path = ROOT / "scripts" / "generate_air_writing.py"
    spec = importlib.util.spec_from_file_location("awr_air_templates", path)
    if spec is None or spec.loader is None:
        return {}
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, "TEMPLATES", {})


def render_letter_png(ch: str) -> bytes:
    """Black ink on white PNG from the built-in air-writing stroke template."""
    tmpl = _templates().get(ch.upper())
    if not tmpl:
        raise ValueError(f"No template for {ch!r}")
    size = 160
    canvas = np.zeros((size, size), dtype=np.uint8)
    for stroke in tmpl:
        pts = []
        for x, y in stroke:
            pts.append((int(x * (size - 1)), int(y * (size - 1))))
        if len(pts) < 2:
            continue
        arr = np.asarray(pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [arr], False, 255, 4, cv2.LINE_8)
    page = np.full_like(canvas, 255)
    page[canvas > 0] = 0
    rgb = cv2.cvtColor(page, cv2.COLOR_GRAY2BGR)
    ok, buf = cv2.imencode(".png", rgb)
    if not ok:
        raise RuntimeError("Could not encode letter PNG")
    return buf.tobytes()


def ensure_letter_dataset(root: Path | None = None) -> int:
    """Write A–Z / 0–9 template captures once so first-time matching has data."""
    base = _dataset_root(root)
    saved = 0
    for ch in LABELS:
        folder = base / folder_name(ch)
        if folder.exists() and any(folder.glob("*.png")):
            saved += 1
            continue
        try:
            png = render_letter_png(ch)
        except (ValueError, RuntimeError):
            continue
        if save_capture(png, ch, root=base) is not None:
            saved += 1
    return saved
