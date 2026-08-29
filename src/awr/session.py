"""Writing session: mode, word commit, persistent output history."""

from __future__ import annotations

from dataclasses import dataclass, field


CONFIDENCE_MIN = 0.55
# Pen-up this long after finishing the word → segment + OCR + commit
WORD_IDLE_SEC = 3.0
# Pen-up this long after finishing a drawing → classify the shape
SHAPE_IDLE_SEC = 2.0


@dataclass
class HistoryItem:
    text: str
    kind: str  # "word" | "shape" | "voice"
    image_png: bytes | None = None


@dataclass
class Session:
    mode: str = "text"  # "text" | "draw" | "voice"
    current_word: str = ""
    history: list[HistoryItem] = field(default_factory=list)
    status: str = "Write a word; leave gaps between letters; pen-up 3s to recognize"
    last_activity: float = 0.0
    auto_recognize: bool = True
    use_ocr_space: bool = False
    history_seq: int = 0

    def set_mode(self, mode: str) -> None:
        if mode not in ("text", "draw", "voice"):
            return
        self.mode = mode
        if mode == "text":
            self.status = (
                "Text mode — write word, pen-up 3s to recognize. Output stays until Clr text."
            )
        elif mode == "draw":
            self.status = (
                "Drawing mode — star, heart, circle, square, triangle, line. Hold 2s. Results stay listed."
            )
        else:
            self.status = (
                "Voice mode — camera is off. Click Mic to listen. Speech stays in this section."
            )

    def history_kind_for_mode(self, mode: str | None = None) -> str:
        name = mode or self.mode
        if name == "draw":
            return "shape"
        if name == "voice":
            return "voice"
        return "word"

    def history_rows_for_mode(
        self, mode: str | None = None
    ) -> list[tuple[str, str, bytes | None]]:
        kind = self.history_kind_for_mode(mode)
        return [
            (item.text, item.kind, item.image_png)
            for item in self.history
            if item.kind == kind
        ]

    def note_activity(self, now: float) -> None:
        self.last_activity = now
        if self.mode == "text":
            self.status = "Writing word…"
        elif self.mode == "draw":
            self.status = "Drawing shape…"
        else:
            self.status = "Voice mode — listening for speech…"

    def commit_recognized_word(
        self, results: list[tuple[str, float]], now: float
    ) -> str | None:
        """Build word from every detected letter's top guess."""
        self.last_activity = now
        n = len(results)
        if not results:
            self.status = "unclear / empty stroke"
            return None

        word = "".join(c for c, _ in results)
        weak = sum(1 for _, p in results if p < CONFIDENCE_MIN)
        conf_bits = " ".join(f"{c}:{p:.0%}" for c, p in results)
        if weak:
            self.status = f"Committed: {word} ({n} letters, {conf_bits})"
        else:
            self.status = f"Committed: {word} ({n} letters)"
        self.history.append(HistoryItem(text=word, kind="word"))
        self.current_word = word
        self.history_seq += 1
        return word

    def commit_recognized_shape(
        self, name: str, now: float, image_png: bytes | None = None
    ) -> str:
        self.last_activity = now
        self.current_word = name
        self.history.append(HistoryItem(text=name, kind="shape", image_png=image_png))
        self.status = f"Shape: {name}"
        self.history_seq += 1
        return name

    def commit_voice_text(self, text: str, now: float) -> str | None:
        cleaned = " ".join((text or "").split())
        if not cleaned:
            return None
        self.last_activity = now
        self.current_word = cleaned
        self.history.append(HistoryItem(text=cleaned, kind="voice"))
        self.status = f"Voice: {cleaned}"
        self.history_seq += 1
        return cleaned

    def replace_last(self, text: str) -> None:
        if not self.history:
            return
        self.history[-1].text = text
        self.current_word = text
        self.history_seq += 1

    def clear_word(self) -> None:
        self.status = "Canvas cleared"

    def clear_transcript(self, kind: str | None = None) -> None:
        if kind is None:
            self.history.clear()
        else:
            self.history = [item for item in self.history if item.kind != kind]
        self.current_word = ""
        self.status = "Output cleared"
        self.history_seq += 1

    @property
    def transcript_text(self) -> str:
        return " ".join(item.text for item in self.history)

    @property
    def history_display(self) -> str:
        if not self.history:
            return ""
        lines: list[str] = []
        for i, item in enumerate(self.history, 1):
            lines.append(f"{i}. {item.text}")
        return "\n".join(lines)

    def history_display_for_mode(self, mode: str | None = None) -> str:
        rows = self.history_rows_for_mode(mode)
        if not rows:
            return ""
        return "\n".join(f"{i}. {text}" for i, (text, _k, _p) in enumerate(rows, 1))

    def export_rows(self, mode: str | None = None) -> list[tuple[str, bytes | None]]:
        return [(text, png) for text, _k, png in self.history_rows_for_mode(mode)]

    def preview_fields(self) -> dict:
        """Fields the camera loop should send every frame so history cannot be dropped."""
        last_label = ""
        last_png = None
        rows = self.history_rows_for_mode()
        if rows:
            last_label, _, last_png = rows[-1]
        return {
            "word": last_label,
            "history_rows": rows,
            "drawing_png": last_png if self.mode == "draw" else None,
            "history_seq": self.history_seq,
        }
