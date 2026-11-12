"""Stored-output panel must stay on screen after a result is committed."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import tkinter as tk  # noqa: E402

from awr.session import Session  # noqa: E402
from awr.ui import AppUI  # noqa: E402


def _noop(*_a, **_k) -> None:
    return None


class StoredOutputVisibleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ui = AppUI(
            on_mode=_noop,
            on_color=_noop,
            on_undo=_noop,
            on_clear=_noop,
            on_clear_transcript=_noop,
            on_teach=_noop,
            on_fix=_noop,
            on_toggle_auto=_noop,
            on_read=_noop,
            on_toggle_ocr=_noop,
            on_save_txt=_noop,
            on_save_pdf=_noop,
        )
        self.ui.root.geometry("1440x900")
        self.ui.root.update_idletasks()
        self.ui.root.update()

    def tearDown(self) -> None:
        self.ui.root.destroy()

    def test_stored_output_text_has_usable_height(self) -> None:
        height = int(self.ui._out.winfo_height())
        self.assertGreater(
            height,
            90,
            f"stored output widget is only {height}px tall — results look invisible",
        )

    def test_undo_and_clear_canvas_are_on_the_right_rail(self) -> None:
        self.ui.root.update_idletasks()
        self.ui.root.update()
        self.assertEqual(str(self.ui._undo_btn.cget("text")), "Undo Last")
        self.assertEqual(str(self.ui._clear_canvas_btn.cget("text")), "Clear")
        self.assertIs(self.ui._undo_btn.master, self.ui._ink_tools)
        self.assertIs(self.ui._clear_canvas_btn.master, self.ui._ink_tools)
        labels = [
            str(w.cget("text"))
            for w in self.ui._write_tools.winfo_children()
            if str(w.winfo_class()) == "Button"
        ]
        self.assertNotIn("Undo Last", labels)
        self.assertNotIn("Clear Canvas", labels)
        self.assertNotIn("Clear", labels)
        self.assertIn("OCR: OFF", labels)
        for btn in (self.ui._undo_btn, self.ui._clear_canvas_btn):
            self.assertTrue(bool(btn.winfo_ismapped()), f"{btn.cget('text')} is not mapped")
            bottom_y = int(btn.winfo_rooty()) + int(btn.winfo_height())
            window_bottom = int(self.ui.root.winfo_rooty()) + int(self.ui.root.winfo_height())
            self.assertLessEqual(bottom_y, window_bottom + 4)

    def test_writing_stage_is_taller_than_results_strip(self) -> None:
        stage_h = int(self.ui._stage.winfo_height())
        bottom_h = int(self.ui._bottom.winfo_height())
        self.assertGreater(stage_h, bottom_h)
        self.assertLess(bottom_h, 340)

    def test_stored_output_stays_readable_after_commit(self) -> None:
        self.ui.set_history([("Star", "shape", None), ("Heart", "shape", None)])
        state = str(self.ui._out.cget("state"))
        fg = str(self.ui._out.cget("fg")).lower()
        self.assertNotEqual(
            state,
            "disabled",
            "disabled Text on Windows uses a dark system color, so stored results vanish",
        )
        self.assertNotIn(fg, ("#111827", "black", "systemdisabledtext"))

    def test_committed_shapes_appear_in_stored_output(self) -> None:
        self.ui._select_mode("draw")
        self.ui.set_history(
            [
                ("Circle", "shape", None),
                ("Star", "shape", None),
                ("Heart", "shape", None),
            ]
        )
        body = self.ui._draw_out.get("1.0", tk.END)
        self.assertIn("Circle", body)
        self.assertIn("Star", body)
        self.assertIn("Heart", body)

    def test_stored_output_omits_text_and_shape_tags(self) -> None:
        self.ui.set_history(
            [
                ("HELLO", "word", None),
                ("Circle", "shape", None),
            ]
        )
        body = self.ui._out.get("1.0", tk.END)
        self.assertIn("HELLO", body)
        self.assertNotIn("Circle", body)
        self.assertNotIn("[Text]", body)
        self.assertNotIn("[Shape]", body)

    def test_text_mode_output_is_text_only(self) -> None:
        import cv2
        import numpy as np

        rgb = np.full((40, 80, 3), 255, dtype=np.uint8)
        ok, buf = cv2.imencode(".png", rgb)
        self.assertTrue(ok)
        self.ui.set_history([("HELLO WORLD", "word", buf.tobytes())])
        body = self.ui._out.get("1.0", tk.END)
        self.assertIn("HELLO WORLD", body)
        self.assertFalse(self.ui._out.image_names())

    def test_text_draw_and_voice_panels_do_not_mix(self) -> None:
        self.ui.set_history([("HELLO", "word", None)])
        self.assertIn("HELLO", self.ui._out.get("1.0", tk.END))
        self.ui._select_mode("draw")
        self.ui.set_history([("Circle", "shape", None)])
        self.assertIn("Circle", self.ui._draw_out.get("1.0", tk.END))
        self.assertNotIn("HELLO", self.ui._draw_out.get("1.0", tk.END))
        self.assertNotIn("Circle", self.ui._out.get("1.0", tk.END))
        self.ui._select_mode("voice")
        self.ui.set_history([("hello there", "voice", None)])
        self.assertIn("hello there", self.ui._voice_out.get("1.0", tk.END))
        self.assertNotIn("HELLO", self.ui._voice_out.get("1.0", tk.END))
        self.assertNotIn("Circle", self.ui._voice_out.get("1.0", tk.END))

    def test_canvas_title_is_air_writing(self) -> None:
        self.assertEqual(str(self.ui._canvas_title.cget("text")), "AIR WRITING")

    def test_text_mode_hides_index_finger_tip(self) -> None:
        self.ui.root.update_idletasks()
        self.assertFalse(
            bool(self.ui._canvas_tip.winfo_ismapped()),
            "writing tip overlay should be hidden in Text mode",
        )

    def test_live_tracking_pill_is_removed(self) -> None:
        self.assertIsNone(getattr(self.ui, "_live_pill", None))


class SessionHistoryFieldsTests(unittest.TestCase):
    def test_preview_fields_always_include_every_stored_shape(self) -> None:
        session = Session()
        session.commit_recognized_shape("Circle", 1.0)
        session.commit_recognized_shape("Star", 2.0)
        session.commit_recognized_shape("Heart", 3.0)
        session.set_mode("draw")
        fields = session.preview_fields()
        self.assertIn("history_rows", fields)
        names = [row[0] for row in fields["history_rows"]]
        self.assertEqual(names, ["Circle", "Star", "Heart"])
        self.assertEqual(fields["word"], "Heart")

    def test_history_display_omits_text_and_shape_tags(self) -> None:
        session = Session()
        session.commit_recognized_word([("H", 0.9), ("I", 0.9)], 1.0)
        session.commit_recognized_shape("Circle", 2.0)
        body = session.history_display
        self.assertIn("HI", body)
        self.assertIn("Circle", body)
        self.assertNotIn("[Text]", body)
        self.assertNotIn("[Shape]", body)
        self.assertNotRegex(body, r"\bText\b")
        self.assertNotRegex(body, r"\bShape\b")

    def test_each_mode_has_its_own_history_section(self) -> None:
        session = Session()
        session.commit_recognized_word([("H", 0.9), ("I", 0.9)], 1.0)
        session.commit_recognized_shape("Circle", 2.0)
        session.commit_voice_text("hello there", 3.0)
        session.set_mode("text")
        self.assertEqual([row[0] for row in session.history_rows_for_mode()], ["HI"])
        session.set_mode("draw")
        self.assertEqual([row[0] for row in session.history_rows_for_mode()], ["Circle"])
        session.set_mode("voice")
        self.assertEqual([row[0] for row in session.history_rows_for_mode()], ["hello there"])
        self.assertEqual(session.status.startswith("Voice mode"), True)

    def test_preview_fields_follow_the_active_mode_section(self) -> None:
        session = Session()
        session.commit_recognized_word([("H", 0.9), ("I", 0.9)], 1.0)
        session.commit_recognized_shape("Circle", 2.0)
        session.commit_recognized_shape("Star", 3.0)
        session.set_mode("text")
        fields = session.preview_fields()
        self.assertEqual(fields["word"], "HI")
        self.assertEqual([row[0] for row in fields["history_rows"]], ["HI"])
        session.set_mode("draw")
        fields = session.preview_fields()
        self.assertEqual(fields["word"], "Star")
        self.assertEqual([row[0] for row in fields["history_rows"]], ["Circle", "Star"])


class VoiceControlsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ui = AppUI(
            on_mode=_noop,
            on_color=_noop,
            on_undo=_noop,
            on_clear=_noop,
            on_clear_transcript=_noop,
            on_teach=_noop,
            on_fix=_noop,
            on_toggle_auto=_noop,
            on_read=_noop,
            on_toggle_ocr=_noop,
            on_save_txt=_noop,
            on_save_pdf=_noop,
        )
        self.ui.root.update_idletasks()
        self.ui.root.update()

    def tearDown(self) -> None:
        self.ui.root.destroy()

    def test_mic_and_pause_buttons_exist(self) -> None:
        self.assertEqual(str(self.ui._mic_btn.cget("text")), "Mic: OFF")
        self.assertEqual(str(self.ui._pause_btn.cget("text")), "Pause")

    def test_voice_mode_button_sits_with_text_and_draw(self) -> None:
        self.assertIn("voice", self.ui._mode_btns)
        self.assertEqual(str(self.ui._mode_btns["voice"].cget("text")), "Voice")
        self.assertEqual(str(self.ui._mode_btns["text"].cget("text")), "Text")
        self.assertEqual(str(self.ui._mode_btns["draw"].cget("text")), "Draw")

    def test_voice_section_uses_its_own_output_panel(self) -> None:
        self.ui._select_mode("voice")
        self.ui.root.update_idletasks()
        self.assertEqual(str(self.ui._voice_title.cget("text")), "Voice")
        self.ui._select_mode("text")
        self.assertEqual(str(self.ui._out_title.cget("text")), "TEXT OUTPUT")
        self.ui._select_mode("draw")
        self.assertEqual(str(self.ui._draw_title.cget("text")), "DRAWING OUTPUT")

    def test_header_shows_product_name_and_tagline(self) -> None:
        self.assertEqual(str(self.ui.root.title()), "AIR WRITING RECOGNITION")
        self.assertEqual(str(self.ui._brand_title.cget("text")), "AIR WRITING RECOGNITION")
        self.assertEqual(
            str(self.ui._brand_tagline.cget("text")),
            "Write in the air. We recognize your ideas.",
        )

    def test_voice_mode_hides_camera(self) -> None:
        self.ui._select_mode("voice")
        self.ui.root.update_idletasks()
        self.ui.root.update()
        self.assertFalse(bool(self.ui._stage.winfo_ismapped()))
        self.assertTrue(bool(self.ui._voice_page.winfo_ismapped()))
        self.ui._select_mode("text")
        self.ui.root.update_idletasks()
        self.ui.root.update()
        self.assertTrue(bool(self.ui._stage.winfo_ismapped()))
        self.assertFalse(bool(self.ui._voice_page.winfo_ismapped()))

    def test_live_voice_stays_in_the_voice_section(self) -> None:
        self.ui.set_history([("HELLO", "word", None)])
        self.ui.set_voice_live("hello there friend", "listening")
        body = self.ui._out.get("1.0", tk.END)
        self.assertIn("HELLO", body)
        self.assertNotIn("hello there friend", body)
        self.ui._select_mode("voice")
        self.ui.set_history([("hello there friend", "voice", None)])
        self.ui.set_voice_live("hello there friend", "listening")
        body = self.ui._panel().get("1.0", tk.END)
        self.assertIn("hello there friend", body)
        self.assertNotIn("HELLO", body)

    def test_listening_state_shows_mic_on_and_enables_pause(self) -> None:
        self.ui.set_voice_state("listening")
        self.assertEqual(str(self.ui._mic_btn.cget("text")), "Mic: ON")
        self.assertEqual(str(self.ui._pause_btn.cget("text")), "Pause")
        self.assertEqual(str(self.ui._pause_btn.cget("state")), "normal")

    def test_paused_state_shows_start_to_resume_while_speaking(self) -> None:
        self.ui.set_voice_state("paused")
        self.assertEqual(str(self.ui._mic_btn.cget("text")), "Mic: PAUSED")
        self.assertEqual(str(self.ui._pause_btn.cget("text")), "Start")

    def test_voice_output_can_scroll(self) -> None:
        self.ui._select_mode("voice")
        self.ui.root.geometry("1440x900")
        self.ui.root.update_idletasks()
        self.ui.root.update()
        rows = [(f"stored speech line {i} " * 6, "voice", None) for i in range(40)]
        self.ui.set_history(rows)
        self.ui.root.update_idletasks()
        first, last = self.ui._voice_out.yview()
        self.assertLess(
            float(last) - float(first),
            1.0,
            "voice output should overflow so the scrollbar can move",
        )
        self.ui._voice_out.yview_moveto(0.0)
        self.ui.root.update_idletasks()
        self.assertEqual(self.ui._voice_out.yview()[0], 0.0)
        self.ui.set_voice_live("stored speech line 0", "stopped")
        self.ui.root.update_idletasks()
        self.assertEqual(
            self.ui._voice_out.yview()[0],
            0.0,
            "scrolling up should not jump back to the bottom",
        )
        self.ui._voice_out.yview_scroll(4, "units")
        self.assertGreater(self.ui._voice_out.yview()[0], 0.0)


if __name__ == "__main__":
    unittest.main()
