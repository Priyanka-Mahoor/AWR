"""Finger pose heuristics for air-writing."""

from __future__ import annotations

import math
from typing import Sequence

# MediaPipe hand landmark indices
WRIST = 0
INDEX_PIP, INDEX_TIP = 6, 8
MIDDLE_PIP, MIDDLE_TIP = 10, 12
RING_PIP, RING_TIP = 14, 16
PINKY_PIP, PINKY_TIP = 18, 20

EXTENDED_RATIO = 1.12
CURLED_RATIO = 1.05


def _dist(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _xy(landmark) -> tuple[float, float]:
    return float(landmark.x), float(landmark.y)


def finger_state(landmarks, tip_idx: int, pip_idx: int) -> str:
    """Return 'extended', 'curled', or 'unknown' for one finger."""
    wrist = _xy(landmarks[WRIST])
    tip = _xy(landmarks[tip_idx])
    pip = _xy(landmarks[pip_idx])
    d_pip = _dist(pip, wrist)
    if d_pip < 1e-6:
        return "unknown"
    ratio = _dist(tip, wrist) / d_pip
    if ratio > EXTENDED_RATIO:
        return "extended"
    if ratio < CURLED_RATIO:
        return "curled"
    return "unknown"


class PointingGate:
    """Index-pointing writing pose. Pen-up is immediate when the hand closes."""

    def __init__(self) -> None:
        self._pointing = False

    @property
    def pointing(self) -> bool:
        return self._pointing

    def reset(self) -> None:
        self._pointing = False

    def update(self, landmarks) -> bool:
        index = finger_state(landmarks, INDEX_TIP, INDEX_PIP)
        middle = finger_state(landmarks, MIDDLE_TIP, MIDDLE_PIP)
        ring = finger_state(landmarks, RING_TIP, RING_PIP)
        pinky = finger_state(landmarks, PINKY_TIP, PINKY_PIP)

        # Writing: index extended, other fingers not extended
        definite_on = index == "extended" and all(
            s != "extended" for s in (middle, ring, pinky)
        )
        # Stop immediately on closed fist / curled index / opened other fingers
        definite_off = (
            index == "curled"
            or index != "extended"
            or any(s == "extended" for s in (middle, ring, pinky))
        )

        if definite_on:
            self._pointing = True
        elif definite_off:
            self._pointing = False
        # unknown mid-zone: keep previous only if still vaguely pointing
        return self._pointing
