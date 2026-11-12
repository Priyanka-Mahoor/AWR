"""OCR.space cloud OCR for English air writing.

Get a free key at https://ocr.space/ocrapi — or set OCR_SPACE_API_KEY.
Without a key, the public demo key ``helloworld`` is used (rate-limited).
Tries print Engine 2, handwriting Engine 3, then Tesseract Engine 1.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request

ENDPOINT = "https://api.ocr.space/parse/image"
DEFAULT_KEY = "helloworld"
ENGLISH_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
KEEP_PUNCT = ".,!?'-:;\"()"
OCR_JUNK = {
    "NO TEXT DETECTED",
    "NO TEXT",
    "ERROR",
    "N/A",
    "NULL",
    "UNDEFINED",
}


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


def clean_ocr_text(raw: str, language: str = "eng") -> str:
    """Keep letters, digits, spaces, and punctuation, preserving written case."""
    text = (raw or "").replace("\r\n", "\n").replace("\r", "\n").replace("\n", " ")
    kept: list[str] = []
    for ch in text:
        if ch.isspace():
            kept.append(" ")
        elif ch.upper() in ENGLISH_CHARS:
            kept.append(ch)
        elif ch in KEEP_PUNCT:
            kept.append(ch)
    return " ".join("".join(kept).split())


def is_ocr_junk(raw: str) -> bool:
    """True for OCR.space placeholders that are not air-written words."""
    compact = " ".join((raw or "").upper().split())
    if not compact:
        return True
    if compact in OCR_JUNK:
        return True
    return compact.startswith("NO TEXT")


def ocr_reading_score(raw: str) -> int:
    """Rank an engine reading: reject junk, prefer real letters and mixed case."""
    if is_ocr_junk(raw):
        return -1
    text = clean_ocr_text(raw)
    if is_ocr_junk(text):
        return -1
    alnum = [ch for ch in text if ch.isalnum()]
    if not alnum:
        return -1
    letters = sum(ch.isalpha() for ch in alnum)
    mixed = int(any(ch.islower() for ch in alnum) and any(ch.isupper() for ch in alnum))
    return letters * 10 + mixed * 3 + len(alnum)


# Short letters vs the tallest in the word (x-height vs cap height).
_SHORT_OF_TALLEST = 0.72
# Tallest ink vs camera canvas: small air writing → lowercase.
_SMALL_ON_CANVAS = 0.28
_LARGE_ON_CANVAS = 0.35


def apply_written_case(
    letters: list[str],
    heights: list[float] | None = None,
    canvas_height: float = 0,
) -> list[str]:
    """Map CNN A–Z to upper/lower from how large each letter was written."""
    if not letters:
        return []
    if not heights or len(heights) != len(letters):
        return list(letters)
    usable = [float(h) for h in heights]
    tallest = max(usable) if usable else 0.0
    if tallest <= 0:
        return list(letters)
    if canvas_height > 0 and tallest < _SMALL_ON_CANVAS * canvas_height:
        return [ch.lower() if ch.isalpha() else ch for ch in letters]
    short_cut = _SHORT_OF_TALLEST * tallest
    out: list[str] = []
    for ch, height in zip(letters, usable):
        if ch.isalpha() and height < short_cut:
            out.append(ch.lower())
        else:
            out.append(ch)
    return out


def apply_written_case_to_text(
    text: str,
    heights: list[float] | None = None,
    canvas_height: float = 0,
) -> str:
    """Re-case a recognized line from air-written letter heights."""
    if not text:
        return text
    alphas = [ch for ch in text if ch.isalpha()]
    mixed = any(ch.islower() for ch in alphas) and any(ch.isupper() for ch in alphas)
    if mixed:
        return text
    if heights and len(alphas) == len(heights):
        cased = apply_written_case(alphas, heights, canvas_height)
        it = iter(cased)
        return "".join(next(it) if ch.isalpha() else ch for ch in text)
    if not heights or canvas_height <= 0:
        return text
    tallest = max(float(h) for h in heights)
    if tallest < _SMALL_ON_CANVAS * canvas_height:
        return "".join(ch.lower() if ch.isalpha() else ch for ch in text)
    if tallest >= _LARGE_ON_CANVAS * canvas_height:
        return "".join(ch.upper() if ch.isalpha() else ch for ch in text)
    return text


def ocr_words(raw: str, language: str = "eng") -> list[str]:
    """Split OCR output into English words (every word, no cap)."""
    cleaned = clean_ocr_text(raw, language)
    return cleaned.split() if cleaned else []


def pick_fuller_text(ocr_raw: str, cnn_letters: list[str] | None = None) -> str:
    """Prefer a real OCR word/phrase; use CNN only if OCR is empty or a stub."""
    ocr_line = "" if is_ocr_junk(ocr_raw) else clean_ocr_text(ocr_raw)
    if is_ocr_junk(ocr_line):
        ocr_line = ""
    cnn_word = "".join(cnn_letters or [])
    ocr_alnum = "".join(ch for ch in ocr_line if ch.upper() in ENGLISH_CHARS)
    if not ocr_alnum:
        return cnn_word
    if " " in ocr_line or len(ocr_alnum) >= 3:
        return ocr_line
    if cnn_word and len(cnn_word) > len(ocr_alnum):
        return cnn_word
    return ocr_line


def ocr_attempts(language: str) -> list[tuple[str, str]]:
    """English: handwriting engine, print engine, then Tesseract."""
    if language == "eng":
        return [("3", "eng"), ("2", "eng"), ("1", "eng")]
    return [("3", "auto")]


def ocr_request_fields(
    png: bytes,
    engine: str = "3",
    language: str = "eng",
    detect_orientation: bool | None = None,
) -> dict[str, str]:
    b64 = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
    if detect_orientation is None:
        detect_orientation = False
    return {
        "apikey": api_key(),
        "base64Image": b64,
        "language": language,
        "isOverlayRequired": "false",
        "OCREngine": engine,
        "scale": "true",
        "detectOrientation": "true" if detect_orientation else "false",
        "isTable": "false",
        "filetype": "PNG",
    }


def ocr_payload_text(payload: dict) -> str:
    """Extract text, raising on OCR.space parse failures that used to look empty."""
    if payload.get("IsErroredOnProcessing"):
        err = payload.get("ErrorMessage") or payload.get("ErrorDetails") or "OCR error"
        if isinstance(err, list):
            err = "; ".join(str(x) for x in err)
        raise RuntimeError(str(err))
    results = payload.get("ParsedResults") or []
    if not results:
        return ""
    parts: list[str] = []
    errors: list[str] = []
    for item in results:
        code = item.get("FileParseExitCode")
        text = str(item.get("ParsedText") or "").strip()
        if text and not is_ocr_junk(text):
            parts.append(text)
            continue
        if text and is_ocr_junk(text):
            continue
        if code not in (1, "1", None, 0, "0"):
            msg = item.get("ErrorMessage") or f"OCR parse code {code}"
            if isinstance(msg, list):
                msg = "; ".join(str(x) for x in msg)
            errors.append(str(msg))
    if parts:
        return "\n".join(parts)
    if errors:
        raise RuntimeError(errors[0])
    return ""


def _parse_text(payload: dict) -> str:
    return ocr_payload_text(payload)


def recognize_png(png: bytes, engine: str = "3", language: str = "eng") -> str:
    """Send a PNG (black ink on white) to OCR.space; return raw parsed text."""
    body, content_type = _multipart(ocr_request_fields(png, engine=engine, language=language))
    req = urllib.request.Request(
        ENDPOINT,
        data=body,
        method="POST",
        headers={"Content-Type": content_type, "apikey": api_key()},
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"OCR.space HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"OCR.space unreachable: {exc.reason}") from exc
    return _parse_text(payload)


def recognize_png_with_fallback(
    png: bytes, language: str = "eng"
) -> tuple[str, str]:
    """Try Engine 3 then 2; keep the longest reading of the air writing."""
    last_err: Exception | None = None
    attempts = ocr_attempts(language)
    last_empty_tag = f"{attempts[0][0]}/{attempts[0][1]}"
    saw_response = False
    best_text = ""
    best_tag = last_empty_tag
    best_score = -1
    for engine, lang in attempts:
        tag = f"{engine}/{lang}"
        try:
            text = recognize_png(png, engine=engine, language=lang)
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            continue
        saw_response = True
        score = ocr_reading_score(text)
        if score > best_score:
            best_score = score
            best_text = text
            best_tag = tag
        last_empty_tag = tag
    if best_score > 0:
        return best_text, best_tag
    if saw_response:
        return "", last_empty_tag
    raise RuntimeError(str(last_err) if last_err else "OCR.space failed")
