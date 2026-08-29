"""Recognize basic geometric shapes from a binary ink mask."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ShapeHit:
    name: str
    kind: str
    confidence: float
    overlay: dict


def classify_mask(mask: np.ndarray) -> ShapeHit | None:
    """Return the best shape for a white-on-black uint8 mask, or None if empty."""
    if mask is None or mask.size == 0:
        return None
    work = mask
    if work.ndim == 3:
        work = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    work = (work > 0).astype(np.uint8) * 255
    if cv2.countNonZero(work) < 40:
        return None

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    closed = cv2.morphologyEx(work, cv2.MORPH_CLOSE, kernel, iterations=3)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    raw = max(contours, key=cv2.contourArea)
    total_area = sum(cv2.contourArea(c) for c in contours)
    # Fragmented squares/rects: several strokes. One complete star/heart: one blob.
    if len(contours) > 1 and cv2.contourArea(raw) < 0.55 * max(total_area, 1):
        raw = cv2.convexHull(np.vstack(contours))

    area = float(cv2.contourArea(raw))
    if area < 30:
        return None
    peri = float(cv2.arcLength(raw, True))
    if peri < 8:
        return None

    hull = cv2.convexHull(raw)
    hull_area = float(cv2.contourArea(hull)) or 1.0
    solidity = area / hull_area
    circularity = float(4.0 * np.pi * area / (peri * peri)) if peri > 0 else 0.0
    bx, by, bw, bh = cv2.boundingRect(raw)
    box_aspect = max(bw, bh) / max(min(bw, bh), 1)

    rect = cv2.minAreaRect(raw)
    rw, rh = float(rect[1][0]), float(rect[1][1])
    thin = max(rw, rh) / max(min(rw, rh), 1.0)
    if thin >= 5.5 or (area < 400 and thin >= 4.0):
        box = cv2.boxPoints(rect).astype(np.int32)
        if rw >= rh:
            p1, p2 = (box[0] + box[3]) // 2, (box[1] + box[2]) // 2
        else:
            p1, p2 = (box[0] + box[1]) // 2, (box[2] + box[3]) // 2
        return ShapeHit(
            name="Line",
            kind="line",
            confidence=min(0.99, 0.7 + thin / 20.0),
            overlay={"kind": "line", "p1": (int(p1[0]), int(p1[1])), "p2": (int(p2[0]), int(p2[1]))},
        )

    raw_approx = cv2.approxPolyDP(raw, 0.02 * peri, True)
    hull_peri = float(cv2.arcLength(hull, True))
    hull_approx = cv2.approxPolyDP(hull, 0.03 * hull_peri, True)
    if len(hull_approx) > 6:
        hull_approx = cv2.approxPolyDP(hull, 0.045 * hull_peri, True)
    hull_verts = len(hull_approx)
    deep_defects = _deep_defect_count(raw, peri)

    if _is_star(solidity, len(raw_approx), deep_defects, hull_verts):
        return ShapeHit(
            name="Star",
            kind="star",
            confidence=0.9,
            overlay={"kind": "polygon", "points": raw_approx.reshape(-1, 2)},
        )

    if _is_heart(closed, raw, solidity, box_aspect, bx, by, bw, bh, deep_defects):
        return ShapeHit(
            name="Heart",
            kind="heart",
            confidence=0.88,
            overlay={"kind": "polygon", "points": raw_approx.reshape(-1, 2)},
        )

    (cx, cy), radius = cv2.minEnclosingCircle(raw)
    circle_area = float(np.pi * radius * radius)
    fill = area / max(circle_area, 1.0)
    raw_verts = len(raw_approx)
    # Hull can collapse an ellipse to 4–6 sides; a true polygon has few raw verts.
    smooth_round = raw_verts >= 8 and circularity >= 0.68

    # Polygons before circles: a regular hexagon is ~0.83 circular and was
    # labeled Circle, and thick rectangles were caught by the oval rule.
    # Air-drawn triangles often miss a corner, so the hull is 4–5 verts.
    if not smooth_round and (hull_verts == 3 or _is_gapped_triangle(hull, hull_verts)):
        return ShapeHit(
            name="Triangle",
            kind="triangle",
            confidence=0.9,
            overlay={"kind": "polygon", "points": _triangle_points(hull, hull_approx)},
        )
    if not smooth_round and hull_verts == 4:
        name, kind = _quad_name(hull_approx, box_aspect)
        return ShapeHit(
            name=name,
            kind=kind,
            confidence=0.88,
            overlay={"kind": "polygon", "points": hull_approx.reshape(-1, 2)},
        )
    if not smooth_round and hull_verts == 5:
        return ShapeHit(
            name="Pentagon",
            kind="pentagon",
            confidence=0.82,
            overlay={"kind": "polygon", "points": hull_approx.reshape(-1, 2)},
        )
    if not smooth_round and hull_verts == 6:
        return ShapeHit(
            name="Hexagon",
            kind="hexagon",
            confidence=0.82,
            overlay={"kind": "polygon", "points": hull_approx.reshape(-1, 2)},
        )

    # Tight round blobs → Circle. Do not name elongated rounds "Oval".
    if circularity >= 0.82 and box_aspect < 1.28 and fill >= 0.45:
        return ShapeHit(
            name="Circle",
            kind="circle",
            confidence=float(min(0.99, circularity)),
            overlay={"kind": "circle", "center": (int(cx), int(cy)), "radius": int(radius)},
        )
    if circularity >= 0.50 and box_aspect >= 1.25 and hull_verts >= 7:
        return ShapeHit(
            name="Circle",
            kind="circle",
            confidence=float(min(0.99, circularity)),
            overlay={"kind": "circle", "center": (int(cx), int(cy)), "radius": int(radius)},
        )
    if circularity >= 0.50 and fill >= 0.35:
        return ShapeHit(
            name="Circle",
            kind="circle",
            confidence=float(circularity),
            overlay={"kind": "circle", "center": (int(cx), int(cy)), "radius": int(radius)},
        )

    if hull_verts >= 3:
        return ShapeHit(
            name="Unknown",
            kind="polygon",
            confidence=0.4,
            overlay={"kind": "polygon", "points": hull_approx.reshape(-1, 2)},
        )
    return None


def classify_drawing(mask: np.ndarray, strokes: list | None = None) -> ShapeHit | None:
    """Classify air-drawn ink, closing near stroke endpoints first."""
    if mask is None:
        return None
    work = mask.copy()
    if strokes:
        for stroke in strokes:
            pts = getattr(stroke, "points", None)
            if not pts or len(pts) < 2:
                continue
            a, b = pts[0], pts[-1]
            dist = float(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)
            if dist <= 90:
                cv2.line(work, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), 255, 8)
    return classify_mask(work)


def _deep_defect_count(cnt: np.ndarray, peri: float) -> int:
    if len(cnt) < 4:
        return 0
    hull_idx = cv2.convexHull(cnt, returnPoints=False)
    if hull_idx is None or len(hull_idx) < 3:
        return 0
    try:
        defects = cv2.convexityDefects(cnt, hull_idx)
    except cv2.error:
        return 0
    if defects is None:
        return 0
    min_depth = max(6.0, 0.04 * peri)
    n = 0
    for d in defects:
        row = d[0] if getattr(d, "ndim", 1) > 1 else d
        if float(row[3]) / 256.0 >= min_depth:
            n += 1
    return n


def _is_star(solidity: float, raw_verts: int, defects: int, hull_verts: int) -> bool:
    if solidity >= 0.82:
        return False
    if defects >= 4 and solidity <= 0.75:
        return True
    if raw_verts >= 8 and solidity <= 0.76:
        return True
    if hull_verts >= 5 and solidity <= 0.70 and defects >= 3:
        return True
    # Two overlapping triangles (a common air-drawn star) have only ~2 dents.
    if defects >= 2 and raw_verts >= 8 and solidity <= 0.80:
        return True
    return False


def _is_heart(
    mask: np.ndarray,
    cnt: np.ndarray,
    solidity: float,
    box_aspect: float,
    bx: int,
    by: int,
    bw: int,
    bh: int,
    _defects: int,
) -> bool:
    if bw < 12 or bh < 12:
        return False
    if solidity < 0.62 or solidity > 0.97:
        return False
    if box_aspect > 1.85:
        return False

    # Bottom tip near the horizontal center.
    pts = cnt.reshape(-1, 2)
    bottom = pts[np.argmax(pts[:, 1])]
    cx = bx + bw / 2.0
    if abs(bottom[0] - cx) > 0.32 * bw:
        return False
    if bottom[1] < by + 0.62 * bh:
        return False

    # Top cleft: ink dips in the middle of the upper band.
    y1 = by + max(4, int(bh * 0.38))
    band = mask[by:y1, bx : bx + bw]
    if band.size == 0:
        return False
    col = np.sum(band > 0, axis=0).astype(np.float32)
    if col.max() < 1:
        return False
    third = max(2, bw // 3)
    left = float(col[:third].max())
    mid = float(col[third : 2 * third].min()) if 2 * third <= len(col) else 0.0
    right = float(col[2 * third :].max()) if 2 * third < len(col) else 0.0
    cleft = left > 2 and right > 2 and mid <= 0.55 * min(left, right)
    # Require the top cleft. The old 1–3-defect fallback labeled stars as hearts.
    return cleft


def _is_gapped_triangle(hull: np.ndarray, hull_verts: int) -> bool:
    """True when a 4–5-gon hull is a triangle with a chopped / open corner."""
    if hull_verts not in (4, 5):
        return False
    hull_area = float(cv2.contourArea(hull))
    if hull_area < 30:
        return False
    try:
        tri_area, _ = cv2.minEnclosingTriangle(np.asarray(hull, dtype=np.float32))
    except cv2.error:
        return False
    return hull_area / max(float(tri_area), 1.0) >= 0.88


def _triangle_points(hull: np.ndarray, hull_approx: np.ndarray) -> np.ndarray:
    if len(hull_approx) == 3:
        return hull_approx.reshape(-1, 2)
    try:
        _, tri = cv2.minEnclosingTriangle(np.asarray(hull, dtype=np.float32))
    except cv2.error:
        return hull_approx.reshape(-1, 2)
    return np.asarray(tri, dtype=np.int32).reshape(-1, 2)


def _quad_name(approx: np.ndarray, box_aspect: float) -> tuple[str, str]:
    pts = approx.reshape(-1, 2).astype(np.float32)
    sides = [float(np.linalg.norm(pts[i] - pts[(i + 1) % 4])) for i in range(4)]
    mean = sum(sides) / 4.0
    similar_sides = all(abs(s - mean) / max(mean, 1.0) < 0.22 for s in sides)

    right = 0
    for i in range(4):
        v1 = pts[i] - pts[(i - 1) % 4]
        v2 = pts[(i + 1) % 4] - pts[i]
        n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
        if n1 < 1 or n2 < 1:
            continue
        cos = abs(float(np.dot(v1, v2) / (n1 * n2)))
        if cos < 0.35:
            right += 1

    if box_aspect < 1.22 and (similar_sides or right >= 3):
        return "Square", "square"
    if right >= 2:
        return "Rectangle", "rectangle"
    if similar_sides:
        return "Diamond", "diamond"
    if box_aspect < 1.22:
        return "Square", "square"
    return "Rectangle", "rectangle"


def draw_shape_overlay(frame: np.ndarray, hit: ShapeHit, color: tuple[int, int, int] = (0, 220, 255)) -> None:
    """Draw a cleaned shape + name on the camera frame (in-place)."""
    ov = hit.overlay
    kind = ov.get("kind")
    if kind == "circle":
        cv2.circle(frame, ov["center"], max(2, ov["radius"]), color, 2, cv2.LINE_AA)
        origin = ov["center"]
    elif kind == "ellipse":
        cv2.ellipse(frame, ov["ellipse"], color, 2, cv2.LINE_AA)
        origin = (int(ov["ellipse"][0][0]), int(ov["ellipse"][0][1]))
    elif kind == "line":
        cv2.line(frame, ov["p1"], ov["p2"], color, 2, cv2.LINE_AA)
        origin = ov["p1"]
    elif kind == "polygon":
        pts = np.asarray(ov["points"], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts], True, color, 2, cv2.LINE_AA)
        xs, ys = pts[:, 0, 0], pts[:, 0, 1]
        origin = (int(xs.min()), int(ys.min()))
    else:
        origin = (20, 40)

    x, y = origin[0], max(28, origin[1] - 8)
    cv2.putText(
        frame,
        hit.name,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        color,
        2,
        cv2.LINE_AA,
    )
