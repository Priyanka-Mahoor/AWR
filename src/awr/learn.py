"""Save and load user-corrected 28x28 air-writing samples."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from awr.recognizer import LABELS

ROOT = Path(__file__).resolve().parents[2]
USER_DIR = ROOT / "data" / "user_chars"


def _normalize(img: np.ndarray) -> np.ndarray:
    x = np.asarray(img, dtype=np.float32)
    if x.ndim == 3:
        x = x.squeeze()
    if x.max() > 1.0:
        x = x / 255.0
    return np.clip(x, 0.0, 1.0)


def save_samples(images: list[np.ndarray], labels: str) -> int:
    """Save one 28x28 image per character in `labels`. Returns count saved."""
    labels = labels.strip().upper()
    if len(images) != len(labels):
        raise ValueError(f"Need {len(images)} character(s), got {len(labels)!r}")
    saved = 0
    stamp = int(time.time() * 1000)
    for i, (img, ch) in enumerate(zip(images, labels)):
        if ch not in LABELS:
            continue
        folder = USER_DIR / ch
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / f"{stamp}_{i}.npy", _normalize(img))
        saved += 1
    return saved


def load_samples() -> tuple[np.ndarray, np.ndarray]:
    """Return (N, 28, 28) images and int labels."""
    images: list[np.ndarray] = []
    labels: list[int] = []
    if not USER_DIR.is_dir():
        return np.zeros((0, 28, 28), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    for ch in LABELS:
        folder = USER_DIR / ch
        if not folder.is_dir():
            continue
        y = LABELS.index(ch)
        for path in folder.glob("*.npy"):
            x = _normalize(np.load(path))
            if x.shape != (28, 28):
                continue
            images.append(x)
            labels.append(y)
    if not images:
        return np.zeros((0, 28, 28), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    return np.stack(images), np.array(labels, dtype=np.int32)


def sample_count() -> int:
    x, _ = load_samples()
    return int(x.shape[0])


INFO_PATH = ROOT / "models" / "user_finetune.json"


def write_finetune_info(n_samples: int, letters: dict[str, int]) -> None:
    import json
    from datetime import datetime, timezone

    INFO_PATH.parent.mkdir(parents=True, exist_ok=True)
    INFO_PATH.write_text(
        json.dumps(
            {
                "n_samples": n_samples,
                "letters": letters,
                "when": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def finetune_info() -> dict | None:
    import json

    if not INFO_PATH.is_file():
        return None
    try:
        return json.loads(INFO_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def model_status_line() -> str:
    n = sample_count()
    info = finetune_info()
    if info:
        return (
            f"Model: fine-tuned {info.get('when', '?')} "
            f"on {info.get('n_samples', '?')} of your letters"
        )
    if n:
        return (
            f"CNN ready. {n} Teach sample(s) saved but NOT in the model yet — "
            "run: python scripts/finetune_user.py"
        )
    return "CNN ready (EMNIST only). Use Teach to save your letters."

