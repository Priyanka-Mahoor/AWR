"""English OCR.space client and ink PNG size."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from awr.ink import InkCanvas, Stroke  # noqa: E402
from awr.ocr_space import (  # noqa: E402
    apply_written_case,
    apply_written_case_to_text,
    clean_ocr_text,
    ocr_attempts,
    ocr_payload_text,
    ocr_reading_score,
    ocr_request_fields,
    ocr_words,
    pick_fuller_text,
)
from awr.session import Session, WORD_IDLE_SEC  # noqa: E402


class CleanOcrTextTests(unittest.TestCase):
    def test_english_keeps_word_spaces(self) -> None:
        self.assertEqual(clean_ocr_text("Hi cat 12!", "eng"), "Hi cat 12!")

    def test_english_keeps_punctuation_in_the_phrase(self) -> None:
        self.assertEqual(clean_ocr_text("Hello, world.", "eng"), "Hello, world.")

    def test_english_keeps_capital_and_small_letters(self) -> None:
        self.assertEqual(clean_ocr_text("HeLLo World"), "HeLLo World")

    def test_ocr_words_splits_english(self) -> None:
        self.assertEqual(ocr_words("HELLO\nworld  12"), ["HELLO", "world", "12"])

    def test_ocr_words_keeps_every_word(self) -> None:
        raw = "one two three four five six seven eight nine ten"
        self.assertEqual(len(ocr_words(raw)), 10)


class OcrAttemptTests(unittest.TestCase):
    def test_english_ocr_tries_print_then_handwriting_engines(self) -> None:
        self.assertEqual(
            ocr_attempts("eng"),
            [("3", "eng"), ("2", "eng"), ("1", "eng")],
        )


class OcrRequestFieldsTests(unittest.TestCase):
    def test_english_uses_eng_and_engine_3(self) -> None:
        fields = ocr_request_fields(b"fake-png", engine="3", language="eng")
        self.assertEqual(fields["language"], "eng")
        self.assertEqual(fields["OCREngine"], "3")
        self.assertEqual(fields["detectOrientation"], "false")
        self.assertEqual(fields["isTable"], "false")
        self.assertIn("base64Image", fields)


class OcrPayloadTextTests(unittest.TestCase):
    def test_parse_exit_code_failure_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            ocr_payload_text(
                {
                    "IsErroredOnProcessing": False,
                    "ParsedResults": [
                        {
                            "ParsedText": "",
                            "FileParseExitCode": -10,
                            "ErrorMessage": "Timed out",
                        }
                    ],
                }
            )

    def test_parse_success_keeps_text(self) -> None:
        text = ocr_payload_text(
            {
                "IsErroredOnProcessing": False,
                "ParsedResults": [{"ParsedText": "HELLO", "FileParseExitCode": 1}],
            }
        )
        self.assertEqual(text, "HELLO")

    def test_no_text_detected_is_treated_as_empty(self) -> None:
        text = ocr_payload_text(
            {
                "IsErroredOnProcessing": False,
                "ParsedResults": [
                    {"ParsedText": "NO TEXT DETECTED", "FileParseExitCode": 1}
                ],
            }
        )
        self.assertEqual(text, "")


class RecognizeEnglishPngTests(unittest.TestCase):
    def test_english_fallback_tries_eng_then_engine2(self) -> None:
        calls: list[bytes] = []

        class _Resp:
            def __init__(self, payload: bytes) -> None:
                self._payload = payload

            def read(self) -> bytes:
                return self._payload

            def __enter__(self):
                return self

            def __exit__(self, *_a):
                return False

        def fake_urlopen(req, timeout=18):
            calls.append(req.data)
            if len(calls) == 1:
                return _Resp(
                    b'{"ParsedResults":[{"ParsedText":""}],"IsErroredOnProcessing":false}'
                )
            return _Resp(
                b'{"ParsedResults":[{"ParsedText":"HELLO WORLD"}],'
                b'"IsErroredOnProcessing":false}'
            )

        from awr import ocr_space

        with patch.object(ocr_space.urllib.request, "urlopen", fake_urlopen):
            text, engine = ocr_space.recognize_png_with_fallback(b"\x89PNG", language="eng")
        self.assertEqual(text, "HELLO WORLD")
        self.assertEqual(len(calls), 3)
        self.assertIn(b"eng", calls[0])
        self.assertIn("2", engine)

    def test_english_keeps_the_longest_engine_result(self) -> None:
        calls: list[bytes] = []

        class _Resp:
            def __init__(self, payload: bytes) -> None:
                self._payload = payload

            def read(self) -> bytes:
                return self._payload

            def __enter__(self):
                return self

            def __exit__(self, *_a):
                return False

        def fake_urlopen(req, timeout=20):
            calls.append(req.data)
            if len(calls) == 1:
                return _Resp(
                    b'{"ParsedResults":[{"ParsedText":"HE"}],"IsErroredOnProcessing":false}'
                )
            return _Resp(
                b'{"ParsedResults":[{"ParsedText":"HELLO WORLD"}],'
                b'"IsErroredOnProcessing":false}'
            )

        from awr import ocr_space

        with patch.object(ocr_space.urllib.request, "urlopen", fake_urlopen):
            text, engine = ocr_space.recognize_png_with_fallback(b"\x89PNG", language="eng")
        self.assertEqual(text, "HELLO WORLD")
        self.assertEqual(len(calls), 3)
        self.assertIn("2", engine)

    def test_skips_no_text_detected_for_a_real_engine_word(self) -> None:
        calls: list[bytes] = []

        class _Resp:
            def __init__(self, payload: bytes) -> None:
                self._payload = payload

            def read(self) -> bytes:
                return self._payload

            def __enter__(self):
                return self

            def __exit__(self, *_a):
                return False

        def fake_urlopen(req, timeout=20):
            calls.append(req.data)
            if len(calls) == 1:
                return _Resp(
                    b'{"ParsedResults":[{"ParsedText":"NO TEXT DETECTED"}],'
                    b'"IsErroredOnProcessing":false}'
                )
            return _Resp(
                b'{"ParsedResults":[{"ParsedText":"Aishu"}],'
                b'"IsErroredOnProcessing":false}'
            )

        from awr import ocr_space

        with patch.object(ocr_space.urllib.request, "urlopen", fake_urlopen):
            text, engine = ocr_space.recognize_png_with_fallback(b"\x89PNG", language="eng")
        self.assertEqual(text, "Aishu")
        self.assertIn("2", engine)

    def test_prefers_mixed_case_word_over_same_length_caps(self) -> None:
        calls: list[bytes] = []

        class _Resp:
            def __init__(self, payload: bytes) -> None:
                self._payload = payload

            def read(self) -> bytes:
                return self._payload

            def __enter__(self):
                return self

            def __exit__(self, *_a):
                return False

        def fake_urlopen(req, timeout=20):
            calls.append(req.data)
            if len(calls) == 1:
                return _Resp(
                    b'{"ParsedResults":[{"ParsedText":"AISTU"}],'
                    b'"IsErroredOnProcessing":false}'
                )
            return _Resp(
                b'{"ParsedResults":[{"ParsedText":"Aishu"}],'
                b'"IsErroredOnProcessing":false}'
            )

        from awr import ocr_space

        with patch.object(ocr_space.urllib.request, "urlopen", fake_urlopen):
            text, _engine = ocr_space.recognize_png_with_fallback(b"\x89PNG", language="eng")
        self.assertEqual(text, "Aishu")


class PickFullerTextTests(unittest.TestCase):
    def test_keeps_ocr_phrase_over_longer_cnn_noise(self) -> None:
        from awr.ocr_space import pick_fuller_text

        self.assertEqual(pick_fuller_text("HELLO", ["I"] * 8), "HELLO")

    def test_uses_cnn_when_ocr_is_much_shorter(self) -> None:
        from awr.ocr_space import pick_fuller_text

        self.assertEqual(pick_fuller_text("HE", ["H", "E", "L", "L", "O"]), "HELLO")

    def test_uses_cnn_only_when_ocr_empty(self) -> None:
        from awr.ocr_space import pick_fuller_text

        self.assertEqual(pick_fuller_text("", ["H", "E", "L", "L", "O"]), "HELLO")

    def test_keeps_full_ocr_phrase(self) -> None:
        from awr.ocr_space import pick_fuller_text

        self.assertEqual(pick_fuller_text("HELLO WORLD", ["H"]), "HELLO WORLD")

    def test_keeps_ocr_mixed_case_over_cnn_caps(self) -> None:
        self.assertEqual(pick_fuller_text("Hello", ["H", "E", "L", "L", "O"]), "Hello")

    def test_ocr_junk_falls_back_to_cnn(self) -> None:
        self.assertEqual(
            pick_fuller_text("NO TEXT DETECTED", ["P", "R", "I", "Y", "A"]),
            "PRIYA",
        )


class OcrReadingScoreTests(unittest.TestCase):
    def test_junk_scores_below_a_real_word(self) -> None:
        self.assertLess(ocr_reading_score("NO TEXT DETECTED"), 0)
        self.assertGreater(ocr_reading_score("Aishu"), ocr_reading_score("NO TEXT DETECTED"))

    def test_mixed_case_handwriting_beats_all_caps_same_length(self) -> None:
        self.assertGreater(ocr_reading_score("Aishu"), ocr_reading_score("AISTU"))


class WrittenCaseTests(unittest.TestCase):
    def test_shorter_letters_become_lowercase(self) -> None:
        self.assertEqual(apply_written_case(["H", "I"], [80, 35]), ["H", "i"])

    def test_similar_tall_letters_stay_uppercase(self) -> None:
        self.assertEqual(
            apply_written_case(["H", "E", "L", "L", "O"], [80, 78, 82, 81, 79], canvas_height=200),
            ["H", "E", "L", "L", "O"],
        )

    def test_small_writing_on_the_canvas_is_lowercase(self) -> None:
        self.assertEqual(
            apply_written_case(["H", "I"], [40, 38], canvas_height=200),
            ["h", "i"],
        )

    def test_ocr_all_caps_follow_small_air_writing(self) -> None:
        self.assertEqual(
            apply_written_case_to_text("HELLO", [40, 38, 41, 39, 42], canvas_height=200),
            "hello",
        )

    def test_ocr_mixed_case_is_kept(self) -> None:
        self.assertEqual(
            apply_written_case_to_text("Hello", [80, 40, 42, 41, 79], canvas_height=200),
            "Hello",
        )

    def test_per_letter_heights_recase_ocr_caps(self) -> None:
        self.assertEqual(
            apply_written_case_to_text("HI", [80, 35], canvas_height=200),
            "Hi",
        )


class LetterGlyphHeightTests(unittest.TestCase):
    def test_letter_glyphs_keep_ink_height(self) -> None:
        ink = InkCanvas(width=200, height=200)
        ink.strokes.append(Stroke(points=[(20, 20), (20, 160)], width=3))
        ink.strokes.append(Stroke(points=[(90, 90), (90, 120)], width=3))
        glyphs = ink.letter_glyphs(merge_h=False)
        self.assertGreaterEqual(len(glyphs), 2)
        tall = glyphs[0].y1 - glyphs[0].y0
        short = glyphs[1].y1 - glyphs[1].y0
        self.assertGreater(tall, short * 1.5)


class OcrPngSizeTests(unittest.TestCase):
    def test_wide_air_word_fits_free_ocr_limit(self) -> None:
        import cv2
        import numpy as np

        ink = InkCanvas(1280, 720)
        ink.strokes.append(Stroke(points=[(30, 240), (1240, 280)], width=3))
        png = ink.to_ocr_png()
        self.assertIsNotNone(png)
        img = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertIsNotNone(img)
        height, width = img.shape[:2]
        self.assertLessEqual(width, 1000, f"OCR PNG width {width} exceeds free API limit")
        self.assertLessEqual(height, 1000, f"OCR PNG height {height} exceeds free API limit")


class SessionIdleTests(unittest.TestCase):
    def test_word_idle_is_two_seconds(self) -> None:
        self.assertEqual(WORD_IDLE_SEC, 2.0)

    def test_commit_recognized_words_stores_each_word(self) -> None:
        session = Session()
        phrase = session.commit_recognized_words(["HELLO", "WORLD"], 1.0)
        self.assertEqual(phrase, "HELLO WORLD")
        names = [row[0] for row in session.history_rows_for_mode("text")]
        self.assertEqual(names, ["HELLO", "WORLD"])

    def test_later_ocr_keeps_earlier_rows_and_ink_png(self) -> None:
        session = Session()
        session.commit_recognized_words(["HELLO"], 1.0, image_png=b"PNG1")
        session.commit_recognized_words(["WORLD"], 2.0, image_png=b"PNG2")
        rows = session.history_rows_for_mode("text")
        self.assertEqual([row[0] for row in rows], ["HELLO", "WORLD"])
        self.assertEqual(rows[0][2], b"PNG1")
        self.assertEqual(rows[1][2], b"PNG2")

    def test_empty_ocr_still_stores_the_ink_png(self) -> None:
        session = Session()
        phrase = session.commit_recognized_words([], 1.0, image_png=b"PNG")
        self.assertEqual(phrase, "(unclear)")
        rows = session.history_rows_for_mode("text")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], "(unclear)")
        self.assertEqual(rows[0][2], b"PNG")
