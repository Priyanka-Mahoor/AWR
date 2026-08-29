"""Shape classifier: geometric ink → Circle, Square, Triangle, etc."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from awr.shapes import classify_mask  # noqa: E402


def _blank(size: int = 240) -> np.ndarray:
    return np.zeros((size, size), dtype=np.uint8)


class ClassifyMaskTests(unittest.TestCase):
    def test_empty_mask_returns_none(self) -> None:
        self.assertIsNone(classify_mask(_blank()))

    def test_circle_is_circle(self) -> None:
        mask = _blank()
        cv2.circle(mask, (120, 120), 55, 255, 4)
        hit = classify_mask(mask)
        self.assertIsNotNone(hit)
        self.assertEqual(hit.name, "Circle")

    def test_square_is_square(self) -> None:
        mask = _blank()
        cv2.rectangle(mask, (50, 50), (190, 190), 255, 4)
        hit = classify_mask(mask)
        self.assertIsNotNone(hit)
        self.assertEqual(hit.name, "Square")

    def test_wide_rectangle_is_rectangle(self) -> None:
        mask = _blank()
        cv2.rectangle(mask, (20, 80), (220, 150), 255, 4)
        hit = classify_mask(mask)
        self.assertIsNotNone(hit)
        self.assertEqual(hit.name, "Rectangle")

    def test_triangle_is_triangle(self) -> None:
        mask = _blank()
        pts = np.array([[120, 30], [30, 200], [210, 200]], dtype=np.int32)
        cv2.polylines(mask, [pts], True, 255, 4)
        hit = classify_mask(mask)
        self.assertIsNotNone(hit)
        self.assertEqual(hit.name, "Triangle")

    def test_gapped_three_stroke_triangle_is_triangle_not_square(self) -> None:
        # Air-drawn triangle: three sides that miss at the corners.
        mask = _blank()
        cv2.line(mask, (120, 40), (40, 190), 255, 4)
        cv2.line(mask, (50, 200), (200, 200), 255, 4)
        cv2.line(mask, (200, 190), (130, 40), 255, 4)
        hit = classify_mask(mask)
        self.assertIsNotNone(hit)
        self.assertEqual(hit.name, "Triangle")
        self.assertNotEqual(hit.name, "Square")

    def test_triangle_open_at_apex_is_triangle_not_square(self) -> None:
        mask = _blank()
        cv2.line(mask, (115, 50), (40, 200), 255, 8)
        cv2.line(mask, (40, 200), (200, 200), 255, 8)
        cv2.line(mask, (200, 200), (130, 55), 255, 8)
        hit = classify_mask(mask)
        self.assertIsNotNone(hit)
        self.assertEqual(hit.name, "Triangle")
        self.assertNotEqual(hit.name, "Square")

    def test_right_triangle_is_triangle(self) -> None:
        mask = _blank()
        pts = np.array([[40, 40], [40, 200], [200, 200]], dtype=np.int32)
        cv2.polylines(mask, [pts], True, 255, 4)
        hit = classify_mask(mask)
        self.assertIsNotNone(hit)
        self.assertEqual(hit.name, "Triangle")

    def test_horizontal_line_is_line(self) -> None:
        mask = _blank()
        cv2.line(mask, (20, 120), (220, 125), 255, 4)
        hit = classify_mask(mask)
        self.assertIsNotNone(hit)
        self.assertEqual(hit.name, "Line")

    def test_wide_ellipse_is_circle_not_oval(self) -> None:
        mask = _blank()
        cv2.ellipse(mask, (120, 120), (80, 36), 0, 0, 360, 255, 4)
        hit = classify_mask(mask)
        self.assertIsNotNone(hit)
        self.assertEqual(hit.name, "Circle")
        self.assertNotEqual(hit.name, "Oval")

    def test_star_is_star(self) -> None:
        mask = _blank()
        cv2.polylines(mask, [_star_pts(120, 120, 70, 28)], True, 255, 3)
        hit = classify_mask(mask)
        self.assertIsNotNone(hit)
        self.assertEqual(hit.name, "Star")

    def test_heart_is_heart(self) -> None:
        mask = _blank()
        cv2.polylines(mask, [_heart_pts(120, 120, 5.2)], True, 255, 3)
        hit = classify_mask(mask)
        self.assertIsNotNone(hit)
        self.assertEqual(hit.name, "Heart")

    def test_star_drawn_as_separate_points_is_star(self) -> None:
        mask = _blank()
        pts = _star_pts(120, 120, 70, 28)
        for i in range(5):
            a = tuple(pts[i * 2])
            tip = tuple(pts[i * 2 + 1])
            b = tuple(pts[(i * 2 + 2) % 10])
            cv2.line(mask, a, tip, 255, 3)
            cv2.line(mask, tip, b, 255, 3)
        hit = classify_mask(mask)
        self.assertIsNotNone(hit)
        self.assertEqual(hit.name, "Star")

    def test_two_triangle_star_is_star_not_heart(self) -> None:
        mask = _blank()
        up = np.array([[120, 40], [50, 180], [190, 180]], dtype=np.int32)
        down = np.array([[120, 200], [50, 60], [190, 60]], dtype=np.int32)
        cv2.polylines(mask, [up], True, 255, 6)
        cv2.polylines(mask, [down], True, 255, 6)
        hit = classify_mask(mask)
        self.assertIsNotNone(hit)
        self.assertEqual(hit.name, "Star")
        self.assertNotEqual(hit.name, "Heart")

    def test_hexagon_is_hexagon_not_circle(self) -> None:
        mask = _blank()
        pts = np.array(
            [
                [120 + int(70 * np.cos(a)), 120 + int(70 * np.sin(a))]
                for a in np.linspace(0, 2 * np.pi, 7)[:-1]
            ],
            dtype=np.int32,
        )
        cv2.polylines(mask, [pts], True, 255, 4)
        hit = classify_mask(mask)
        self.assertIsNotNone(hit)
        self.assertEqual(hit.name, "Hexagon")
        self.assertNotEqual(hit.name, "Circle")

    def test_thick_rectangle_is_rectangle_not_circle(self) -> None:
        mask = _blank()
        cv2.rectangle(mask, (20, 80), (220, 150), 255, 10)
        hit = classify_mask(mask)
        self.assertIsNotNone(hit)
        self.assertEqual(hit.name, "Rectangle")
        self.assertNotEqual(hit.name, "Circle")


def _star_pts(cx: int, cy: int, r_out: float, r_in: float, n: int = 5) -> np.ndarray:
    pts = []
    for i in range(n * 2):
        r = r_out if i % 2 == 0 else r_in
        a = -np.pi / 2 + i * np.pi / n
        pts.append([cx + int(r * np.cos(a)), cy + int(r * np.sin(a))])
    return np.asarray(pts, dtype=np.int32)


def _heart_pts(cx: int, cy: int, scale: float) -> np.ndarray:
    pts = []
    for t in np.linspace(0, 2 * np.pi, 90):
        x = 16 * np.sin(t) ** 3
        y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
        pts.append([cx + int(scale * x), cy - int(scale * y)])
    return np.asarray(pts, dtype=np.int32)


if __name__ == "__main__":
    unittest.main()
