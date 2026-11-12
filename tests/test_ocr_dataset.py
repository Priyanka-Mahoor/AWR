"""Captured air-writing OCR dataset: save samples and match later writes."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from awr.ocr_dataset import (  # noqa: E402
    ensure_letter_dataset,
    match_capture,
    save_capture,
)


def _letter_png(text: str) -> bytes:
    canvas = np.full((120, 220), 255, dtype=np.uint8)
    cv2.putText(canvas, text, (12, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.0, 0, 4)
    rgb = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    ok, buf = cv2.imencode(".png", rgb)
    assert ok
    return buf.tobytes()


class OcrDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_saved_sample_matches_the_same_writing(self) -> None:
        png = _letter_png("HI")
        path = save_capture(png, "HI", root=self.root)
        self.assertIsNotNone(path)
        hit = match_capture(png, root=self.root)
        self.assertIsNotNone(hit)
        label, score = hit
        self.assertEqual(label, "HI")
        self.assertGreater(score, 0.9)

    def test_different_letter_does_not_steal_a_weak_match(self) -> None:
        save_capture(_letter_png("A"), "A", root=self.root)
        hit = match_capture(_letter_png("Q"), root=self.root)
        if hit is not None:
            self.assertNotEqual(hit[0], "A")

    def test_builtin_letter_dataset_can_match_a_template_letter(self) -> None:
        n = ensure_letter_dataset(root=self.root)
        self.assertGreaterEqual(n, 36)
        from awr.ocr_dataset import render_letter_png

        png = render_letter_png("H")
        hit = match_capture(png, root=self.root)
        self.assertIsNotNone(hit)
        self.assertEqual(hit[0], "H")
