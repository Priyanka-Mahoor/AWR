"""Optional OCR.space cloud OCR (free tier). Off unless enabled in the UI.

Get a free key at https://ocr.space/ocrapi — or set OCR_SPACE_API_KEY.
Without a key, the public demo key ``helloworld`` is used (rate-limited).
Engine 3 is preferred (handwriting); falls back to engine 2.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request

ENDPOINT = "https://api.ocr.space/parse/image"
DEFAULT_KEY = "helloworld"


def api_key() -> str:
    return os.environ.get("OCR_SPACE_API_KEY", DEFAULT_KEY).strip() or DEFAULT_KEY


def _multipart(fields: dict[str, str]) -> tuple[bytes, str]:
    boundary = "----AWROcrSpaceBoundary"
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.append(f"--{boundary}\r\n".encode())
        chunks.append(
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
        )
        chunks.append(value.encode("utf-8"))
        chunks.append(b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode())
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def _parse_text(payload: dict) -> str:
    if payload.get("IsErroredOnProcessing"):
        err = payload.get("ErrorMessage") or payload.get("ErrorDetails") or "OCR error"
        if isinstance(err, list):
            err = "; ".join(str(x) for x in err)
        raise RuntimeError(str(err))
    results = payload.get("ParsedResults") or []
    if not results:
        return ""
    return str(results[0].get("ParsedText") or "").strip()


def recognize_png(png: bytes, engine: str = "3") -> str:
    """Send a PNG (black ink on white) to OCR.space; return raw parsed text."""
    b64 = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
    body, content_type = _multipart(
        {
            "apikey": api_key(),
            "base64Image": b64,
            "language": "eng",
            "isOverlayRequired": "false",
            "OCREngine": engine,
            "scale": "true",
            "detectOrientation": "true",
            "filetype": "PNG",
        }
    )
    req = urllib.request.Request(
        ENDPOINT,
        data=body,
        method="POST",
        headers={"Content-Type": content_type, "apikey": api_key()},
    )
    try:
        with urllib.request.urlopen(req, timeout=18) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"OCR.space HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"OCR.space unreachable: {exc.reason}") from exc
    return _parse_text(payload)


def recognize_png_with_fallback(png: bytes) -> tuple[str, str]:
    """Try handwriting engine 3, then engine 2. Returns (text, engine_used)."""
    last_err: Exception | None = None
    for engine in ("3", "2"):
        try:
            text = recognize_png(png, engine=engine)
            return text, engine
        except Exception as exc:  # noqa: BLE001
            last_err = exc
    raise RuntimeError(str(last_err) if last_err else "OCR.space failed")
