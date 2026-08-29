"""Save recognized output as UTF-8 text or a simple PDF."""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image


HistoryRow = tuple[str, bytes | None]


def write_txt(path: str | Path, body: str) -> None:
    Path(path).write_text(body, encoding="utf-8")


def write_pdf(path: str | Path, rows: list[HistoryRow], title: str = "AWR Output") -> None:
    """Write a letter-size PDF. Each row is (label, optional PNG bytes)."""
    pages = _layout_pages(rows, title)
    objects: list[bytes] = []

    font_obj = 3
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    # pages object filled later
    objects.append(b"<< /Type /Pages /Kids [] /Count 0 >>")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    page_ids: list[int] = []
    image_ids: dict[int, int] = {}

    for page in pages:
        content_parts: list[bytes] = []
        xobjects: dict[str, int] = {}
        content_parts.append(_begin_text())
        content_parts.append(_tf(16 if page is pages[0] and page.lines and page.lines[0][0] == "title" else 12))
        for kind, y, text, img_jpeg in page.lines:
            if kind == "title":
                content_parts.append(_tf(16))
                content_parts.append(_td(72, y, text))
                content_parts.append(_tf(12))
            elif kind == "text":
                content_parts.append(_td(72, y, text))
            elif kind == "image" and img_jpeg is not None:
                img_id = len(objects) + 1
                objects.append(_image_xobject(img_jpeg[0], img_jpeg[1], img_jpeg[2]))
                image_ids[img_id] = img_id
                name = f"Im{img_id}"
                xobjects[name] = img_id
                w, h = img_jpeg[0], img_jpeg[1]
                content_parts.append(_end_text())
                content_parts.append(
                    f"q {w:.1f} 0 0 {h:.1f} 72 {y:.1f} cm /{name} Do Q\n".encode("ascii")
                )
                content_parts.append(_begin_text())
                content_parts.append(_tf(12))
        content_parts.append(_end_text())
        stream = b"".join(content_parts)
        content_id = len(objects) + 1
        objects.append(
            b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream"
        )
        page_id = len(objects) + 1
        xobj_dict = ""
        if xobjects:
            parts = " ".join(f"/{n} {i} 0 R" for n, i in xobjects.items())
            xobj_dict = f"/XObject << {parts} >>"
        objects.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Contents {content_id} 0 R /Resources << /Font << /F1 {font_obj} 0 R >> "
                f"{xobj_dict} >> >>"
            ).encode("ascii")
        )
        page_ids.append(page_id)

    kids = " ".join(f"{i} 0 R" for i in page_ids)
    objects[1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode("ascii")

    # Object 1 is catalog; we already have it. Rebuild file.
    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out.extend(f"{i} 0 obj\n".encode("ascii"))
        out.extend(obj)
        out.extend(b"\nendobj\n")
    xref = len(out)
    out.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    out.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        out.extend(f"{off:010d} 00000 n \n".encode("ascii"))
    out.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode(
            "ascii"
        )
    )
    Path(path).write_bytes(bytes(out))


class _Page:
    def __init__(self) -> None:
        self.lines: list[tuple[str, float, str, tuple[int, int, bytes] | None]] = []


def _layout_pages(rows: list[HistoryRow], title: str) -> list[_Page]:
    pages: list[_Page] = []
    page = _Page()
    y = 742.0
    page.lines.append(("title", y, title, None))
    y -= 28
    if not rows:
        page.lines.append(("text", y, "(empty)", None))
        pages.append(page)
        return pages

    for idx, (label, png) in enumerate(rows, start=1):
        text = f"{idx}. {label}"
        need = 18
        jpeg = None
        draw_h = 0.0
        if png:
            jpeg = _png_to_jpeg_fit(png, max_w=400, max_h=220)
            if jpeg:
                draw_h = float(jpeg[1]) + 10
                need += draw_h
        if y - need < 60:
            pages.append(page)
            page = _Page()
            y = 742.0
        page.lines.append(("text", y, text, None))
        y -= 18
        if jpeg:
            y -= draw_h
            page.lines.append(("image", y + 8, "", jpeg))
    pages.append(page)
    return pages


def _png_to_jpeg_fit(png: bytes, max_w: int, max_h: int) -> tuple[int, int, bytes] | None:
    try:
        im = Image.open(io.BytesIO(png)).convert("RGB")
    except Exception:
        return None
    w, h = im.size
    if w < 1 or h < 1:
        return None
    scale = min(max_w / w, max_h / h, 1.0)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    if (nw, nh) != (w, h):
        im = im.resize((nw, nh), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=85)
    return nw, nh, buf.getvalue()


def _pdf_escape(text: str) -> str:
    cleaned = "".join(ch if 32 <= ord(ch) < 127 else "?" for ch in text)
    return cleaned.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _begin_text() -> bytes:
    return b"BT\n"


def _end_text() -> bytes:
    return b"ET\n"


def _tf(size: int) -> bytes:
    return f"/F1 {size} Tf\n".encode("ascii")


def _td(x: float, y: float, text: str) -> bytes:
    return f"1 0 0 1 {x:.1f} {y:.1f} Tm ({_pdf_escape(text)}) Tj\n".encode("ascii")


def _image_xobject(w: int, h: int, jpeg: bytes) -> bytes:
    header = (
        f"<< /Type /XObject /Subtype /Image /Width {w} /Height {h} "
        f"/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode "
        f"/Length {len(jpeg)} >>\nstream\n"
    ).encode("ascii")
    return header + jpeg + b"\nendstream"
