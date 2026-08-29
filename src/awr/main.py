"""Entry point: camera thread + Tkinter UI."""

from __future__ import annotations

import queue
import threading
import time
from pathlib import Path

from awr.export import write_pdf, write_txt
from awr.ink import InkCanvas
from awr.learn import save_samples, sample_count, model_status_line
from awr.ocr_space import recognize_png_with_fallback
from awr.recognizer import CharRecognizer, LABELS
from awr.session import SHAPE_IDLE_SEC, WORD_IDLE_SEC, Session
from awr.shapes import ShapeHit, classify_drawing, draw_shape_overlay
from awr.tracker import HandTracker
from awr.ui import AppUI
from awr.voice import VoiceListener

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "char_cnn_weights.npz"


class App:
    def __init__(self) -> None:
        self.session = Session()
        self.text_ink = InkCanvas()
        self.draw_ink = InkCanvas()
        self.recognizer = CharRecognizer(MODEL_PATH)
        self._q: queue.Queue = queue.Queue(maxsize=2)
        self._stop = threading.Event()
        self._was_drawing = False
        self._pen_up_since: float | None = None
        self._lock = threading.Lock()
        self._pending_images: list = []
        self._pending_pred = ""
        self._pending_whole = None
        self._shape_hit: ShapeHit | None = None
        self._draw_committed = False
        self._sent_hist: tuple | None = None
        self._hist_seq = 0
        self._voice_lock = threading.Lock()
        self._voice_ui: tuple[str, str] | None = None
        self._voice_commits: list[str] = []
        self._voice_error: str | None = None
        self._voice_status_msg: str | None = None

        self.ui = AppUI(
            on_mode=self.set_mode,
            on_color=self.set_color,
            on_undo=self.undo,
            on_clear=self.clear,
            on_clear_transcript=self.clear_transcript,
            on_teach=self.teach,
            on_fix=self.fix,
            on_toggle_auto=self.toggle_auto,
            on_read=self.read_now,
            on_toggle_ocr=self.toggle_ocr,
            on_save_txt=self.save_txt,
            on_save_pdf=self.save_pdf,
            on_toggle_mic=self.toggle_mic,
            on_pause_voice=self.pause_voice,
        )
        self.ui.set_auto_recognize(True)
        self.ui.set_ocr_space(False)
        self.voice = VoiceListener(
            on_live=self._on_voice_live,
            on_committed=self._on_voice_committed,
            on_error=self._on_voice_error,
            on_status=self._on_voice_status,
        )

        if not self.recognizer.available:
            self.ui.set_status(
                self.recognizer.error
                or "CNN missing — Drawing Mode works; train with scripts/train_emnist.py"
            )
        else:
            self.ui.set_status(model_status_line())
            self.session.status = model_status_line()

        try:
            self.tracker = HandTracker()
        except RuntimeError as exc:
            self.tracker = None
            self.ui.alert("Camera", str(exc))
            self.ui.set_status(str(exc))

    @property
    def active_ink(self) -> InkCanvas:
        return self.draw_ink if self.session.mode == "draw" else self.text_ink

    def set_mode(self, mode: str) -> None:
        leaving_voice = self.session.mode == "voice" and mode != "voice"
        with self._lock:
            self.session.set_mode(mode)
            self._was_drawing = False
            self._pen_up_since = None
            self._shape_hit = None
            self._draw_committed = False
        if leaving_voice and self.voice.state != "stopped":
            self.voice.stop()
        self.ui.set_status(self.session.status)
        self._hist_seq = self.session.history_seq
        self.ui.set_history(self._history_rows())
        if mode == "voice":
            self.ui.set_voice_live(self.voice.transcript.display(), self.voice.state)

    def set_color(self, bgr: tuple[int, int, int]) -> None:
        with self._lock:
            self.draw_ink.set_color(bgr)
            self.text_ink.set_color(bgr)
            self.session.status = "Color set"
        self.ui.set_status("Color set")

    def undo(self) -> None:
        with self._lock:
            self.active_ink.undo()
            self._pen_up_since = None
            self._shape_hit = None
            self._draw_committed = False
            if self.session.mode == "text" and self.text_ink.has_ink():
                self.session.status = "Writing word…"

    def clear(self) -> None:
        with self._lock:
            self.active_ink.clear()
            self._shape_hit = None
            self._draw_committed = False
            self.session.clear_word()
            self._was_drawing = False
            self._pen_up_since = None
            status = self.session.status
        self.ui.set_drawing_thumb(None)
        self.ui.set_word(" ")
        self.ui.set_status(status)

    def clear_transcript(self) -> None:
        kind = self.session.history_kind_for_mode()
        with self._lock:
            self.session.clear_transcript(kind)
            self._shape_hit = None
            self._draw_committed = False
            self._sent_hist = None
            self._hist_seq = self.session.history_seq
        if kind == "voice":
            paused = self.voice.state == "paused"
            self.voice.transcript.clear()
            if paused:
                self.voice.transcript.pause()
            self.ui.set_voice_live("", self.voice.state)
        self.ui.set_word(" ")
        self.ui.set_history([])
        self.ui.set_drawing_thumb(None)
        self.ui.set_status(self.session.status)

    def toggle_auto(self) -> None:
        with self._lock:
            self.session.auto_recognize = not self.session.auto_recognize
            on = self.session.auto_recognize
            self._pen_up_since = None
            if on:
                self.session.status = "Auto-recognize ON — pen-up 3s to read"
            else:
                self.session.status = "Auto-recognize OFF — click Teach or Read"
        self.ui.set_auto_recognize(on)
        self.ui.set_status(self.session.status)

    def toggle_ocr(self) -> None:
        with self._lock:
            self.session.use_ocr_space = not self.session.use_ocr_space
            on = self.session.use_ocr_space
            if on:
                self.session.status = (
                    "OCR.space ON — cloud OCR (free). Set OCR_SPACE_API_KEY for your key."
                )
            else:
                self.session.status = "OCR.space OFF — local CNN"
        self.ui.set_ocr_space(on)
        self.ui.set_status(self.session.status)

    def toggle_mic(self) -> None:
        if self.voice.state == "stopped":
            self.ui.set_status("Starting microphone…")
            self.voice.start()
        else:
            self.voice.stop()
        self.ui.set_voice_state(self.voice.state)

    def pause_voice(self) -> None:
        if self.voice.state == "listening":
            self.voice.pause()
        elif self.voice.state == "paused":
            self.voice.start()
        self.ui.set_voice_state(self.voice.state)

    def _on_voice_live(self, text: str, state: str) -> None:
        with self._voice_lock:
            self._voice_ui = (text, state)

    def _on_voice_committed(self, text: str) -> None:
        with self._voice_lock:
            self._voice_commits.append(text)

    def _on_voice_error(self, message: str) -> None:
        with self._voice_lock:
            self._voice_error = message

    def _on_voice_status(self, message: str) -> None:
        with self._voice_lock:
            self._voice_status_msg = message

    def read_now(self) -> None:
        with self._lock:
            if self.session.mode == "draw":
                if not self.draw_ink.has_ink():
                    self.session.status = "Nothing to read — draw a shape first"
                else:
                    self._finalize_shape(time.monotonic())
                status = self.session.status
            elif not self.text_ink.has_ink():
                self.session.status = "Nothing to read — write first"
                status = self.session.status
            else:
                self._finalize_word(time.monotonic())
                status = self.session.status
        self.ui.set_status(status)
        self.ui.set_word(self.session.current_word)
        self._hist_seq = self.session.history_seq
        self.ui.set_history(self._history_rows())
        last_png = next(
            (item.image_png for item in reversed(self.session.history) if item.image_png),
            None,
        )
        self.ui.set_drawing_thumb(last_png)

    def _history_rows(self) -> list[tuple[str, str, bytes | None]]:
        return self.session.history_rows_for_mode()

    def save_txt(self) -> None:
        body = self.session.history_display_for_mode()
        if not body:
            self.ui.info("Save TXT", "Nothing to save yet. Recognize some text first.")
            return
        path = self.ui.ask_save_path(
            "Save output as text",
            ".txt",
            [("Text", "*.txt"), ("All files", "*.*")],
        )
        if not path:
            return
        write_txt(path, body)
        self.ui.set_status(f"Saved {path}")

    def save_pdf(self) -> None:
        rows = self.session.export_rows()
        if not rows:
            self.ui.info("Save PDF", "Nothing to save yet. Recognize some text first.")
            return
        path = self.ui.ask_save_path(
            "Save output as PDF",
            ".pdf",
            [("PDF", "*.pdf"), ("All files", "*.*")],
        )
        if not path:
            return
        write_pdf(path, rows)
        self.ui.set_status(f"Saved {path}")

    def _snapshot_ink(self) -> tuple[list, object]:
        glyphs = self.text_ink.letter_glyphs()
        whole = self.text_ink.to_emnist()
        return [g.image.copy() for g in glyphs], whole

    def teach(self) -> None:
        """Label what is on screen (or last recognized letters) and save for training."""
        with self._lock:
            images, whole = self._snapshot_ink()
            if not images and self._pending_images:
                images = [img.copy() for img in self._pending_images]
            if whole is None:
                whole = None if self._pending_whole is None else self._pending_whole.copy()
        if not images and whole is None:
            self.ui.info("Teach", "Write a letter first, then click Teach.")
            return
        label = self.ui.ask_text(
            "Teach",
            "What did you write?\nOne letter (H) or a whole word matching the splits.",
        )
        if not label:
            return
        label = label.strip().upper()
        if any(ch not in LABELS for ch in label):
            self.ui.alert("Teach", "Use only 0-9 and A-Z.")
            return
        try:
            if len(label) == 1:
                img = whole if whole is not None else (images[0] if len(images) == 1 else None)
                if img is None:
                    # Split failed (H → two stems): train on the full ink crop.
                    self.ui.alert("Teach", "Could not crop a letter. Write one letter and try again.")
                    return
                n = save_samples([img], label)
            else:
                n = save_samples(images, label)
        except ValueError as exc:
            self.ui.alert("Teach", str(exc))
            return
        with self._lock:
            self.text_ink.clear()
            self._pen_up_since = None
        self.ui.set_status(f"Saved {n} sample(s). Total: {sample_count()}. Then run scripts/finetune_user.py")

    def fix(self) -> None:
        """Correct the last recognized word and save those letter crops."""
        with self._lock:
            images = [img.copy() for img in self._pending_images]
            pred = self._pending_pred
            whole = None if self._pending_whole is None else self._pending_whole.copy()
        if not images and whole is None:
            self.ui.info("Fix", "Recognize a word first, then click Fix.")
            return
        label = self.ui.ask_text(
            "Fix",
            f"App read {len(images)} letter(s) as: {pred or '?'}\n"
            "Type the correct word (same letter count), or one letter if it was a single character.",
            initial=pred,
        )
        if not label:
            return
        label = label.strip().upper()
        if any(ch not in LABELS for ch in label):
            self.ui.alert("Fix", "Use only 0-9 and A-Z.")
            return
        try:
            if len(label) == 1:
                img = whole if whole is not None else (images[0] if len(images) == 1 else None)
                if img is None:
                    self.ui.alert("Fix", "No full-letter crop. Write the letter and click Teach.")
                    return
                n = save_samples([img], label)
            else:
                n = save_samples(images, label)
        except ValueError as exc:
            self.ui.alert("Fix", str(exc))
            return
        with self._lock:
            if self.session.history and pred:
                self.session.replace_last(label)
            else:
                self.session.current_word = label
            self.session.status = f"Corrected to {label} — saved {n} sample(s)"
        self.ui.set_word(label)
        self._hist_seq = self.session.history_seq
        self.ui.set_history(self._history_rows())
        self.ui.set_status(self.session.status)

    def _finalize_word(self, now: float) -> None:
        results: list[tuple[str, float]] = []
        glyphs = self.text_ink.letter_glyphs()
        images = [g.image.copy() for g in glyphs]
        ocr_note = ""
        if self.session.use_ocr_space:
            png = self.text_ink.to_ocr_png()
            if png is not None:
                try:
                    raw, engine = recognize_png_with_fallback(png)
                    cleaned = "".join(ch for ch in raw.upper() if ch in LABELS)
                    if cleaned:
                        results = [(ch, 0.95) for ch in cleaned]
                        ocr_note = f" via OCR.space (engine {engine})"
                    else:
                        ocr_note = " OCR.space empty — using CNN"
                except Exception as exc:  # noqa: BLE001
                    ocr_note = f" OCR.space failed ({exc}) — using CNN"
        if not results and self.recognizer.available:
            for glyph in glyphs:
                pred = self.recognizer.predict(glyph.image, geometry=glyph.geometry)
                if pred is not None:
                    results.append(pred)
        word = self.session.commit_recognized_word(results, now)
        if word is not None:
            n_strokes = sum(1 for s in self.text_ink.strokes if len(s.points) >= 2)
            if len(glyphs) <= 1 and n_strokes >= 3 and not self.session.use_ocr_space:
                self.session.status += " — tip: wider gap between letters"
            if ocr_note:
                self.session.status += ocr_note
            self._pending_images = images
            self._pending_pred = word
            self._pending_whole = self.text_ink.to_emnist()
            self.text_ink.clear()
        elif ocr_note:
            self.session.status = (self.session.status or "unclear") + ocr_note
        self._pen_up_since = None

    def _finalize_shape(self, now: float) -> None:
        mask = self.draw_ink.to_mask()
        hit = (
            classify_drawing(mask, self.draw_ink.strokes) if mask is not None else None
        )
        png = self.draw_ink.crop_png()
        saved = False
        if hit is None:
            if png is not None:
                self.session.commit_recognized_shape("Drawing", now, png)
                self.session.status = "Saved your drawing (shape was unclear)"
                saved = True
            else:
                self.session.status = (
                    "unclear shape — try a larger star, heart, circle, square, triangle, or line"
                )
            self._shape_hit = None
        else:
            self.session.commit_recognized_shape(hit.name, now, png)
            self._shape_hit = hit
            saved = True
        self._draw_committed = saved
        self._pen_up_since = None

    def _camera_loop(self) -> None:
        if self.tracker is None:
            return
        while not self._stop.is_set():
            with self._lock:
                voice_mode = self.session.mode == "voice"
            if voice_mode:
                time.sleep(0.05)
                with self._lock:
                    fields = self.session.preview_fields()
                    state = {
                        "word": fields["word"],
                        "status": self.session.status,
                        "history_rows": fields["history_rows"],
                        "history_seq": fields["history_seq"],
                    }
                self._put(state)
                continue
            try:
                frame, event = self.tracker.read()
            except Exception as exc:  # noqa: BLE001
                self._put({"error": str(exc)})
                break

            now = time.monotonic()
            with self._lock:
                h, w = frame.shape[:2]
                self.active_ink.set_size(w, h)
                kind = event.get("kind")
                drawing = kind == "draw"

                if self.session.mode == "text":
                    self.text_ink.on_event(event)

                    if drawing:
                        self.session.note_activity(now)
                        self._pen_up_since = None
                    elif self.text_ink.has_ink():
                        if not self.session.auto_recognize:
                            self._pen_up_since = None
                        elif self._pen_up_since is None:
                            self._pen_up_since = now
                        elif now - self._pen_up_since >= WORD_IDLE_SEC:
                            self._finalize_word(now)
                    else:
                        self._pen_up_since = None
                elif self.session.mode == "draw":
                    self.draw_ink.on_event(event)
                    if drawing:
                        self.session.note_activity(now)
                        self._pen_up_since = None
                        self._shape_hit = None
                        self._draw_committed = False
                    elif self.draw_ink.has_ink():
                        if not self.session.auto_recognize or self._draw_committed:
                            self._pen_up_since = None
                        elif self._pen_up_since is None:
                            self._pen_up_since = now
                        elif now - self._pen_up_since >= SHAPE_IDLE_SEC:
                            self._finalize_shape(now)
                    else:
                        self._pen_up_since = None

                self._was_drawing = drawing
                cursor = (
                    (event["x"], event["y"])
                    if kind in ("draw", "penUp") and "x" in event and "y" in event
                    else None
                )
                preview = self.active_ink.render_on(frame, cursor)
                if self.session.mode == "draw" and self._shape_hit is not None:
                    draw_shape_overlay(preview, self._shape_hit)
                status = self.session.status
                if self.session.mode == "text":
                    if drawing and self.text_ink.has_ink():
                        status = "Writing word…"
                    elif self._pen_up_since is not None and self.text_ink.has_ink():
                        left = WORD_IDLE_SEC - (now - self._pen_up_since)
                        if left > 0:
                            status = f"Hold… recognizing word in {left:.1f}s"
                        else:
                            status = "Recognizing…"
                    elif not self.session.auto_recognize and self.text_ink.has_ink():
                        status = "Auto OFF — click Teach or Read"
                elif drawing and self.draw_ink.has_ink():
                    status = "Drawing shape…"
                elif self._pen_up_since is not None and self.draw_ink.has_ink():
                    left = SHAPE_IDLE_SEC - (now - self._pen_up_since)
                    if left > 0:
                        status = f"Hold… recognizing shape in {left:.1f}s"
                    else:
                        status = "Recognizing shape…"
                elif not self.session.auto_recognize and self.draw_ink.has_ink():
                    status = "Auto OFF — click Read to name the shape"

                fields = self.session.preview_fields()
                state = {
                    "preview": preview,
                    "word": fields["word"],
                    "status": status,
                    "drawing_png": fields["drawing_png"],
                    # Always include history. The queue keeps only the latest
                    # frame; sending it once means stored results never appear.
                    "history_rows": fields["history_rows"],
                    "history_seq": fields["history_seq"],
                }
                if not self.recognizer.available and self.session.mode == "text":
                    state["status"] = (
                        self.recognizer.error
                        or "Train model: python scripts/train_emnist.py"
                    )

            self._put(state)

        if self.tracker is not None:
            self.tracker.release()

    def _put(self, item: dict) -> None:
        try:
            self._q.put_nowait(item)
        except queue.Full:
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(item)
            except queue.Full:
                pass

    def _pump(self) -> None:
        item = None
        try:
            while True:
                item = self._q.get_nowait()
        except queue.Empty:
            pass
        if item is not None:
            if "error" in item:
                self.ui.set_status(item["error"])
            else:
                if "preview" in item:
                    self.ui.show_preview(item["preview"])
                if "word" in item:
                    self.ui.set_word(item["word"])
                if "history_rows" in item:
                    seq = int(item.get("history_seq") or 0)
                    if seq > self._hist_seq:
                        self._hist_seq = seq
                        self.ui.set_history(item["history_rows"])
                elif "transcript" in item:
                    self.ui.set_transcript(item["transcript"])
                if "status" in item and self.voice.state == "stopped":
                    self.ui.set_status(item["status"])
                if "drawing_png" in item:
                    self.ui.set_drawing_thumb(item["drawing_png"])
            if self.voice.state == "paused":
                self.ui.set_status("Microphone: paused — click Start to continue")
        with self._voice_lock:
            live = self._voice_ui
            commits = list(self._voice_commits)
            self._voice_commits.clear()
            err = self._voice_error
            self._voice_error = None
            voice_status = self._voice_status_msg
            self._voice_status_msg = None
        for text in commits:
            with self._lock:
                self.session.commit_voice_text(text, time.monotonic())
                self._hist_seq = self.session.history_seq
            self.ui.set_word(text)
            self.ui.set_history(self._history_rows())
        if live is not None:
            self.ui.set_voice_live(live[0], live[1])
            if live[1] == "listening":
                self.ui.set_status("Microphone: listening — speak now")
            elif live[1] == "paused":
                self.ui.set_status("Microphone: paused — click Start to continue")
        if err:
            self.ui.set_voice_state("stopped")
            self.ui.set_voice_live("", "stopped")
            self.ui.set_status(err)
            self.ui.alert("Voice recognition", err)
        elif voice_status and self.voice.state != "stopped":
            self.ui.set_status(voice_status)
        if not self._stop.is_set():
            self.ui.root.after(16, self._pump)

    def run(self) -> None:
        if self.tracker is not None:
            t = threading.Thread(target=self._camera_loop, daemon=True)
            t.start()
        self.ui.root.after(16, self._pump)

        def on_close() -> None:
            self._stop.set()
            self.voice.stop()
            self.ui.root.destroy()

        self.ui.root.protocol("WM_DELETE_WINDOW", on_close)
        self.ui.run()


def main() -> None:
    App().run()


if __name__ == "__main__":
    main()
