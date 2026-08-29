"""Stroke ink buffer for text and drawing modes."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Stroke:
    points: list[tuple[int, int]] = field(default_factory=list)
    color: tuple[int, int, int] = (0, 0, 0)  # BGR
    width: int = 3


@dataclass
class LetterGlyph:
    """One letter region cropped for CNN + optional stroke geometry."""

    image: np.ndarray  # (28, 28) float32 [0, 1]
    geometry: dict
    x0: int = 0
    x1: int = 0


class InkCanvas:
    def __init__(self, width: int = 960, height: int = 540) -> None:
        self.width = width
        self.height = height
        self.color: tuple[int, int, int] = (0, 0, 0)
        # Thin strokes match EMNIST ink better after 28x28 downscale.
        self.width_px = 3
        self.strokes: list[Stroke] = []
        self._current: Stroke | None = None
        self._smooth: tuple[float, float] | None = None
        self.alpha = 0.4
        self.min_move = 3.0

    def set_color(self, bgr: tuple[int, int, int]) -> None:
        self.color = bgr

    def set_size(self, width: int, height: int) -> None:
        self.width = width
        self.height = height

    def on_event(self, event: dict) -> None:
        kind = event.get("kind")
        if kind == "draw":
            x, y = int(event["x"]), int(event["y"])
            if self._smooth is None:
                sx, sy = float(x), float(y)
            else:
                sx = self.alpha * x + (1 - self.alpha) * self._smooth[0]
                sy = self.alpha * y + (1 - self.alpha) * self._smooth[1]
            if self._smooth is not None:
                if abs(sx - self._smooth[0]) < self.min_move and abs(sy - self._smooth[1]) < self.min_move:
                    return
            self._smooth = (sx, sy)
            pt = (int(sx), int(sy))
            if self._current is None:
                self._current = Stroke(points=[pt], color=self.color, width=self.width_px)
                self.strokes.append(self._current)
            else:
                self._current.points.append(pt)
        else:
            self._current = None
            self._smooth = None

    def has_ink(self) -> bool:
        return any(len(s.points) >= 2 for s in self.strokes)

    def to_mask(self) -> np.ndarray | None:
        """White-on-black ink mask for shape recognition (thicker so gaps close)."""
        return self._raster(thickness=max(8, self.width_px + 5), dilate=True)

    def crop_png(self) -> bytes | None:
        """PNG of the drawing cropped to ink. Light ink uses a dark background."""
        mask = self._raster(thickness=max(4, self.width_px), dilate=False)
        if mask is None:
            return None
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return None
        pad = 18
        h, w = mask.shape
        x0 = max(0, int(xs.min()) - pad)
        y0 = max(0, int(ys.min()) - pad)
        x1 = min(w, int(xs.max()) + pad + 1)
        y1 = min(h, int(ys.max()) + pad + 1)
        colors = [s.color for s in self.strokes if len(s.points) >= 2]
        if colors:
            mean_bgr = np.mean(np.asarray(colors, dtype=np.float32), axis=0)
            luma = 0.114 * mean_bgr[0] + 0.587 * mean_bgr[1] + 0.299 * mean_bgr[2]
            bg = (18, 22, 32) if luma >= 200 else (255, 255, 255)
        else:
            bg = (255, 255, 255)
        canvas = self.render_drawing(bg)
        crop = canvas[y0:y1, x0:x1]
        if crop.size == 0:
            return None
        ok, buf = cv2.imencode(".png", crop)
        return buf.tobytes() if ok else None

    def undo(self) -> bool:
        if not self.strokes:
            return False
        self.strokes.pop()
        self._current = None
        self._smooth = None
        return True

    def clear(self) -> None:
        self.strokes.clear()
        self._current = None
        self._smooth = None

    def render_on(self, frame: np.ndarray, cursor: tuple[int, int] | None = None) -> np.ndarray:
        out = frame  # draw in-place; camera loop already owns this buffer
        for stroke in self.strokes:
            pts = stroke.points
            if len(pts) < 2:
                continue
            arr = np.asarray(pts, dtype=np.int32).reshape(-1, 1, 2)
            # Light halo so dark ink stays visible on the camera.
            cv2.polylines(out, [arr], False, (255, 255, 255), stroke.width + 2, cv2.LINE_8)
            cv2.polylines(out, [arr], False, stroke.color, stroke.width, cv2.LINE_8)
        if cursor is not None:
            cv2.circle(out, cursor, 6, (255, 255, 255), 1)
        return out

    def render_drawing(self, bg: tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
        canvas = np.full((self.height, self.width, 3), bg, dtype=np.uint8)
        for stroke in self.strokes:
            pts = stroke.points
            for i in range(1, len(pts)):
                cv2.line(canvas, pts[i - 1], pts[i], stroke.color, stroke.width, cv2.LINE_AA)
        return canvas

    def to_ocr_png(self) -> bytes | None:
        """Black ink on white PNG for OCR.space (upscaled crop)."""
        mask = self._raster(thickness=max(10, self.width_px + 6), dilate=True)
        if mask is None:
            return None
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return None
        pad = 24
        h, w = mask.shape
        x0 = max(0, int(xs.min()) - pad)
        y0 = max(0, int(ys.min()) - pad)
        x1 = min(w, int(xs.max()) + pad + 1)
        y1 = min(h, int(ys.max()) + pad + 1)
        crop = mask[y0:y1, x0:x1]
        # OCR wants dark text on a light page
        page = np.full_like(crop, 255)
        page[crop > 0] = 0
        # Upscale small air-writing crops so the API has enough pixels
        ph, pw = page.shape
        scale = max(3.0, 480 / max(pw, 1), 240 / max(ph, 1))
        nw, nh = max(64, int(pw * scale)), max(64, int(ph * scale))
        page = cv2.resize(page, (nw, nh), interpolation=cv2.INTER_NEAREST)
        rgb = cv2.cvtColor(page, cv2.COLOR_GRAY2BGR)
        ok, buf = cv2.imencode(".png", rgb)
        if not ok:
            return None
        return buf.tobytes()

    def stroke_geometry(self, strokes: list[Stroke] | None = None) -> dict:
        """Classify strokes for OCR disambiguation (e.g. H vs 4)."""
        vertical = horizontal = other = 0
        usable = 0
        for stroke in strokes if strokes is not None else self.strokes:
            if len(stroke.points) < 2:
                continue
            usable += 1
            xs = [p[0] for p in stroke.points]
            ys = [p[1] for p in stroke.points]
            dx = max(xs) - min(xs)
            dy = max(ys) - min(ys)
            if dy >= dx * 1.25:
                vertical += 1
            elif dx >= dy * 1.25:
                horizontal += 1
            else:
                other += 1
        return {
            "n": usable,
            "vertical": vertical,
            "horizontal": horizontal,
            "other": other,
        }

    def _raster(self, thickness: int | None = None, dilate: bool = True) -> np.ndarray | None:
        if not self.has_ink():
            return None
        canvas = np.zeros((self.height, self.width), dtype=np.uint8)
        # Keep recognition ink close to EMNIST stroke weight (avoid blob fills).
        t = thickness if thickness is not None else max(3, self.width_px)
        for stroke in self.strokes:
            pts = stroke.points
            for i in range(1, len(pts)):
                cv2.line(canvas, pts[i - 1], pts[i], 255, t, cv2.LINE_AA)
        if dilate:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            canvas = cv2.dilate(canvas, kernel, iterations=1)
        return canvas

    @staticmethod
    def _mask_to_emnist(mask: np.ndarray) -> np.ndarray | None:
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return None
        pad = 12
        h, w = mask.shape
        x0 = max(0, int(xs.min()) - pad)
        y0 = max(0, int(ys.min()) - pad)
        x1 = min(w, int(xs.max()) + pad + 1)
        y1 = min(h, int(ys.max()) + pad + 1)
        crop = mask[y0:y1, x0:x1]
        if crop.shape[0] < 8 or crop.shape[1] < 8:
            return None
        side = int(max(crop.shape[0], crop.shape[1]) * 1.25)
        side = max(side, 28)
        square = np.zeros((side, side), dtype=np.uint8)
        oy = (side - crop.shape[0]) // 2
        ox = (side - crop.shape[1]) // 2
        square[oy : oy + crop.shape[0], ox : ox + crop.shape[1]] = crop
        resized = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)
        _, resized = cv2.threshold(resized, 30, 255, cv2.THRESH_BINARY)
        return resized.astype(np.float32) / 255.0

    @staticmethod
    def _column_ink_runs(canvas: np.ndarray) -> list[tuple[int, int]]:
        """Return [x0, x1) ranges where any row has ink."""
        ink = np.any(canvas > 0, axis=0)
        runs: list[tuple[int, int]] = []
        i = 0
        w = ink.shape[0]
        while i < w:
            if not ink[i]:
                i += 1
                continue
            j = i + 1
            while j < w and ink[j]:
                j += 1
            runs.append((i, j))
            i = j
        return runs

    def _merge_x_runs(self, runs: list[tuple[int, int]], glue: int = 4) -> list[tuple[int, int]]:
        """Glue tiny gaps (broken strokes); keep real letter spaces."""
        if not runs:
            return []
        merged: list[list[int]] = [[runs[0][0], runs[0][1]]]
        for x0, x1 in runs[1:]:
            gap = x0 - merged[-1][1]
            if gap <= glue:
                merged[-1][1] = x1
            else:
                merged.append([x0, x1])
        return [(a, b) for a, b in merged]

    def _stroke_bboxes(self) -> list[tuple[int, int, int, int, Stroke]]:
        out: list[tuple[int, int, int, int, Stroke]] = []
        for stroke in self.strokes:
            if len(stroke.points) < 2:
                continue
            xs = [p[0] for p in stroke.points]
            ys = [p[1] for p in stroke.points]
            out.append((min(xs), min(ys), max(xs) + 1, max(ys) + 1, stroke))
        out.sort(key=lambda t: (t[0] + t[2]) / 2)
        return out

    def _ranges_from_valleys(self, thin: np.ndarray) -> list[tuple[int, int]]:
        """Split on low-ink columns between letters (works with small gaps)."""
        col = np.sum(thin > 0, axis=0).astype(np.float32)
        if float(col.max()) < 1:
            return []
        # Smooth so single-pixel noise doesn't create fake letters
        ksize = 9
        kernel = np.ones(ksize, dtype=np.float32) / ksize
        smooth = np.convolve(col, kernel, mode="same")
        peak = float(smooth.max())
        # Near-empty valley between letters
        is_ink = smooth >= max(1.5, 0.10 * peak)
        runs: list[tuple[int, int]] = []
        i = 0
        w = len(is_ink)
        while i < w:
            if not is_ink[i]:
                i += 1
                continue
            j = i + 1
            while j < w and is_ink[j]:
                j += 1
            if j - i >= 4:
                runs.append((i, j))
            i = j
        return self._merge_x_runs(runs, glue=4)

    def _letter_x_ranges(self, merge_h: bool = True) -> list[tuple[int, int]]:
        """Split word into letter x-ranges (supports 3+ letters like BAT)."""
        geom = self.stroke_geometry()
        clusters = self._ranges_from_overlap_clusters()
        if (
            merge_h
            and len(clusters) == 1
            and geom["vertical"] >= 2
            and geom["horizontal"] >= 1
            and geom["n"] <= 4
        ):
            return clusters

        candidates: list[list[tuple[int, int]]] = []

        thin = self._raster(thickness=2, dilate=False)
        if thin is not None:
            candidates.append(self._merge_x_runs(self._column_ink_runs(thin), glue=6))
            candidates.append(self._ranges_from_valleys(thin))

        candidates.append(self._ranges_from_stroke_gaps())
        candidates.append(self._ranges_from_overlap_clusters())

        usable = [c for c in candidates if c]
        if not usable:
            return []
        # Prefer more letter splits; only merge true H halves afterward.
        best = max(usable, key=len)
        return self._merge_h_like_splits(best) if merge_h else best

    def _midband_bridge(self, x0: int, x1: int) -> bool:
        """True if mid-height ink connects across the combined letter width."""
        canvas = self._raster(thickness=3, dilate=False)
        if canvas is None:
            return False
        h, w = canvas.shape
        y0, y1 = int(h * 0.28), int(h * 0.72)
        x0 = max(0, x0)
        x1 = min(w, x1)
        if x1 - x0 < 12:
            return False
        band = canvas[y0:y1, x0:x1]
        cols = np.any(band > 0, axis=0)
        if cols.size < 8:
            return False
        return float(np.mean(cols)) >= 0.35

    def _has_h_crossbar(self, x0: int, x1: int) -> bool:
        """True if a horizontal stroke spans both stems of [x0, x1] (an H bar)."""
        width = max(1, x1 - x0)
        y_lo, y_hi = int(self.height * 0.22), int(self.height * 0.78)
        for stroke in self.strokes:
            if len(stroke.points) < 2:
                continue
            xs = [p[0] for p in stroke.points]
            ys = [p[1] for p in stroke.points]
            dx = max(xs) - min(xs)
            dy = max(ys) - min(ys)
            if dx < dy * 1.15:
                continue
            overlap = min(max(xs), x1) - max(min(xs), x0)
            if overlap < 0.35 * width:
                continue
            mid_y = sum(ys) / len(ys)
            if y_lo <= mid_y <= y_hi:
                return True
        return False

    def _vertical_stem_in(self, x0: int, x1: int) -> bool:
        """True if a vertical stroke's center x lies in [x0, x1]."""
        for stroke in self.strokes:
            if len(stroke.points) < 2:
                continue
            xs = [p[0] for p in stroke.points]
            ys = [p[1] for p in stroke.points]
            cx = (min(xs) + max(xs)) / 2.0
            if not (x0 <= cx <= x1):
                continue
            dx = max(xs) - min(xs)
            dy = max(ys) - min(ys)
            if dy >= dx * 1.25:
                return True
        return False

    def _merge_h_like_splits(self, ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Merge two nearby vertical stems when an H crossbar connects them."""
        if len(ranges) < 2:
            return ranges
        max_gap = max(120, int(self.width * 0.22))
        out: list[list[int]] = [[ranges[0][0], ranges[0][1]]]
        for x0, x1 in ranges[1:]:
            lx0, lx1 = out[-1]
            gap = x0 - lx1
            left_w = lx1 - lx0
            right_w = x1 - x0
            stem_like = left_w < max(90, int(self.width * 0.12)) and right_w < max(
                90, int(self.width * 0.12)
            )
            close = 0 <= gap <= max_gap
            left_stem = self._vertical_stem_in(lx0, lx1)
            right_stem = self._vertical_stem_in(x0, x1)
            cross = self._has_h_crossbar(lx0, x1) or self._midband_bridge(lx0, x1)
            if stem_like and close and left_stem and right_stem and cross:
                out[-1][1] = x1
            else:
                out.append([x0, x1])
        return [(a, b) for a, b in out]

    def _ranges_from_stroke_gaps(self) -> list[tuple[int, int]]:
        boxes = self._stroke_bboxes()
        if not boxes:
            return []
        if len(boxes) == 1:
            return [(boxes[0][0], boxes[0][2])]

        # Keep letter gaps workable; H stems still merge via _merge_h_like_splits.
        split_gap = 14

        clusters: list[list[int]] = [[0]]
        for i in range(1, len(boxes)):
            prev_right = max(boxes[j][2] for j in clusters[-1])
            gap = boxes[i][0] - prev_right
            if gap >= split_gap:
                clusters.append([i])
            else:
                clusters[-1].append(i)

        return [
            (min(boxes[j][0] for j in idxs), max(boxes[j][2] for j in idxs))
            for idxs in clusters
        ]

    def _ranges_from_overlap_clusters(self) -> list[tuple[int, int]]:
        """One cluster per non-overlapping letter blob (B | A | T)."""
        boxes = self._stroke_bboxes()
        if not boxes:
            return []

        pad = 6
        clusters: list[list[int]] = []
        for i, (x0, _y0, x1, _y1, _s) in enumerate(boxes):
            matched = None
            for ci, cl in enumerate(clusters):
                cx0 = min(boxes[j][0] for j in cl)
                cx1 = max(boxes[j][2] for j in cl)
                if x0 <= cx1 + pad and x1 >= cx0 - pad:
                    matched = ci
                    break
            if matched is None:
                clusters.append([i])
            else:
                clusters[matched].append(i)

        clusters.sort(key=lambda cl: min(boxes[j][0] for j in cl))
        # Merge only hairline separations
        merged: list[list[int]] = [clusters[0]]
        for cl in clusters[1:]:
            prev_r = max(boxes[j][2] for j in merged[-1])
            cur_l = min(boxes[j][0] for j in cl)
            if cur_l - prev_r <= 8:
                merged[-1].extend(cl)
            else:
                merged.append(cl)

        return [
            (min(boxes[j][0] for j in cl), max(boxes[j][2] for j in cl))
            for cl in merged
        ]

    def _strokes_overlapping_x(self, x0: int, x1: int) -> list[Stroke]:
        """Strokes that intersect [x0, x1] (includes H crossbars spanning both stems)."""
        pad = 10
        out: list[Stroke] = []
        for stroke in self.strokes:
            if len(stroke.points) < 2:
                continue
            xs = [p[0] for p in stroke.points]
            if max(xs) >= (x0 - pad) and min(xs) <= (x1 + pad):
                out.append(stroke)
        return out

    def _strokes_in_x_range(self, x0: int, x1: int) -> list[Stroke]:
        return self._strokes_overlapping_x(x0, x1)

    def letter_glyphs(self, merge_h: bool = True) -> list[LetterGlyph]:
        """Split word ink into left→right letter images using spatial gaps."""
        thick = self._raster(dilate=True)
        if thick is None:
            return []

        ranges = self._letter_x_ranges(merge_h=merge_h)
        if not ranges:
            # Whole ink as one letter
            img = self._mask_to_emnist(thick)
            if img is None:
                return []
            return [LetterGlyph(image=img, geometry=self.stroke_geometry(), x0=0, x1=self.width)]

        glyphs: list[LetterGlyph] = []
        for x0, x1 in ranges:
            px0 = max(0, x0 - 6)
            px1 = min(self.width, x1 + 6)
            mask = np.zeros_like(thick)
            mask[:, px0:px1] = thick[:, px0:px1]
            # Drop empty / tiny slices
            if np.count_nonzero(mask) < 40:
                continue
            img = self._mask_to_emnist(mask)
            if img is None:
                continue
            geom = self.stroke_geometry(self._strokes_overlapping_x(x0, x1))
            glyphs.append(LetterGlyph(image=img, geometry=geom, x0=x0, x1=x1))
        if merge_h:
            return self._collapse_split_h_glyphs(glyphs)
        return glyphs

    def _collapse_split_h_glyphs(self, glyphs: list[LetterGlyph]) -> list[LetterGlyph]:
        """If two I-like slices are an H (crossbar), keep one glyph."""
        if len(glyphs) < 2:
            return glyphs
        thick = self._raster(dilate=True)
        if thick is None:
            return glyphs
        stem_w = max(90, int(self.width * 0.12))

        def is_stem(g: LetterGlyph) -> bool:
            if (g.x1 - g.x0) > stem_w:
                return False
            n = g.geometry.get("n", 0)
            v = g.geometry.get("vertical", 0)
            return v >= 1 and n <= 2

        out: list[LetterGlyph] = []
        i = 0
        while i < len(glyphs):
            if i + 1 < len(glyphs) and is_stem(glyphs[i]) and is_stem(glyphs[i + 1]):
                a, b = glyphs[i], glyphs[i + 1]
                geom = self.stroke_geometry(self._strokes_overlapping_x(a.x0, b.x1))
                if geom["vertical"] >= 2 and geom["horizontal"] >= 1 and (
                    self._has_h_crossbar(a.x0, b.x1) or self._midband_bridge(a.x0, b.x1)
                ):
                    x0, x1 = a.x0, b.x1
                    px0 = max(0, x0 - 6)
                    px1 = min(self.width, x1 + 6)
                    mask = np.zeros_like(thick)
                    mask[:, px0:px1] = thick[:, px0:px1]
                    img = self._mask_to_emnist(mask)
                    if img is not None:
                        out.append(LetterGlyph(image=img, geometry=geom, x0=x0, x1=x1))
                        i += 2
                        continue
            out.append(glyphs[i])
            i += 1
        return out

    def to_emnist(self) -> np.ndarray | None:
        """Return (28, 28) float32 white-on-black in [0, 1], or None if empty."""
        canvas = self._raster(dilate=True)
        if canvas is None:
            return None
        return self._mask_to_emnist(canvas)
