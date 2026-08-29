"""Export history to .txt / .pdf (no extra PDF library)."""

from __future__ import annotations

import io
import textwrap
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from awr.export import write_pdf, write_txt  # noqa: E402


class ExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = Path(__file__).resolve().parent / "_tmp_export"
        self.dir.mkdir(exist_ok=True)

    def test_write_txt_saves_body(self) -> None:
        path = self.dir / "out.txt"
        write_txt(path, "HELLO\nWORLD")
        self.assertEqual(path.read_text(encoding="utf-8"), "HELLO\nWORLD")

    def test_write_pdf_is_pdf_and_contains_text(self) -> None:
        path = self.dir / "out.pdf"
        write_pdf(path, [("HELLO", None), ("Circle", None)])
        data = path.read_bytes()
        self.assertTrue(data.startswith(b"%PDF"))
        self.assertIn(b"HELLO", data)
        self.assertIn(b"Circle", data)

    def test_write_pdf_embeds_drawing_image(self) -> None:
        path = self.dir / "draw.pdf"
        img = Image.fromarray(np.full((40, 60, 3), 200, dtype=np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        write_pdf(path, [("Square", buf.getvalue())])
        data = path.read_bytes()
        self.assertTrue(data.startswith(b"%PDF"))
        self.assertIn(b"/DCTDecode", data)
        self.assertIn(b"Square", data)


if __name__ == "__main__":
    unittest.main()
