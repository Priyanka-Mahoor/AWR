"""Tkinter UI for Air Writing Recognition."""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from typing import Callable

import cv2
import numpy as np
from PIL import Image, ImageTk


COLORS = [
    ("Black", (0, 0, 0), "#111111"),
    ("Red", (0, 0, 220), "#dc0000"),
    ("Green", (0, 180, 0), "#00b400"),
    ("Blue", (220, 0, 0), "#0070dc"),
    ("Yellow", (0, 220, 220), "#dcdc00"),
    ("White", (255, 255, 255), "#f2f2f2"),
]

BG = "#07091a"
STRIP = "#0c1024"
OVERLAY = "#12162c"
ACCENT = "#7c5cff"
ACCENT_HOVER = "#6d4eff"
CYAN = "#22d3ee"
MAGENTA = "#e879f9"
LIVE = "#22c55e"
TEXT = "#f1f5ff"
MUTED = "#8b93b3"
BTN = "#1a1f38"
BTN_HOVER = "#252b48"
BORDER = "#2a3158"
CARD = "#101428"
VOICE_BG = "#07091a"
VOICE_CARD = "#101428"
VOICE_PILL = "#12162c"
PILL_BG = "#101428"


class AppUI:
    def __init__(
        self,
        on_mode: Callable[[str], None],
        on_color: Callable[[tuple[int, int, int]], None],
        on_undo: Callable[[], None],
        on_clear: Callable[[], None],
        on_clear_transcript: Callable[[], None],
        on_teach: Callable[[], None],
        on_fix: Callable[[], None],
        on_toggle_auto: Callable[[], None],
        on_read: Callable[[], None],
        on_toggle_ocr: Callable[[], None],
        on_save_txt: Callable[[], None],
        on_save_pdf: Callable[[], None],
        on_toggle_mic: Callable[[], None] | None = None,
        on_pause_voice: Callable[[], None] | None = None,
    ) -> None:
        self.root = tk.Tk()
        self.root.title("AIR WRITING RECOGNITION")
        self.root.geometry("1440x900")
        self.root.minsize(960, 600)
        self.root.configure(bg=BG)

        self._photo = None
        self._mode = "text"
        self._color_bgr = COLORS[0][1]
        self._on_mode = on_mode
        self._on_color = on_color
        self._mode_btns: dict[str, tk.Button] = {}
        self._color_btns: dict[tuple[int, int, int], tk.Button] = {}
        self._view_size = (1280, 720)
        self._wraplength = 1200
        self._last_transcript = ""
        self._last_word = " "
        self._last_status = "Starting…"
        self._thumb_photo = None
        self._hist_photos: list = []
        self._hist_photos_by_mode: dict[str, list] = {"text": [], "draw": [], "voice": []}
        self._last_history_key = None
        self._voice_live = ""
        self._voice_state = "stopped"
        self._wave_job = None
        self._wave_phase = 0

        self._build_header()

        # Results strip: tall enough to read stored items; canvas still gets the rest.
        bottom = tk.Frame(
            self.root,
            bg=STRIP,
            padx=18,
            pady=8,
            highlightbackground=BORDER,
            highlightthickness=1,
            height=280,
        )
        self._bottom = bottom
        bottom.pack(side=tk.BOTTOM, fill=tk.X)
        bottom.pack_propagate(False)

        actions = tk.Frame(bottom, bg=STRIP)
        actions.pack(fill=tk.X, pady=(0, 8))
        self._mode_slot = tk.Frame(actions, bg=STRIP)
        self._mode_slot.pack(side=tk.LEFT)
        self._write_tools = tk.Frame(self._mode_slot, bg=STRIP)
        self._auto_btn = self._rail_btn(self._write_tools, "Auto: ON", on_toggle_auto)
        self._auto_btn.pack(side=tk.LEFT, padx=(0, 6))
        self._rail_btn(self._write_tools, "Read", on_read).pack(side=tk.LEFT, padx=6)
        self._ocr_btn = self._rail_btn(self._write_tools, "OCR: OFF", on_toggle_ocr)
        self._ocr_btn.pack(side=tk.LEFT, padx=6)
        self._rail_btn(self._write_tools, "Teach", on_teach).pack(side=tk.LEFT, padx=6)
        self._rail_btn(self._write_tools, "Fix", on_fix).pack(side=tk.LEFT, padx=6)
        self._write_tools.pack(side=tk.LEFT)
        self._rail_btn(actions, "Save TXT", on_save_txt).pack(side=tk.LEFT, padx=6)
        self._rail_btn(actions, "Save PDF", on_save_pdf).pack(side=tk.LEFT, padx=6)
        clr = self._primary_btn(actions, "Clear text", on_clear_transcript)
        clr.pack(side=tk.RIGHT)

        self.status_var = tk.StringVar(value="Starting…")
        tk.Label(
            bottom,
            textvariable=self.status_var,
            fg=MUTED,
            bg=STRIP,
            font=("Segoe UI", 11),
            anchor="w",
        ).pack(fill=tk.X)

        body = tk.Frame(bottom, bg=STRIP)
        body.pack(fill=tk.BOTH, expand=True)

        latest_row = tk.Frame(body, bg=STRIP)
        latest_row.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 12))
        self.word_var = tk.StringVar(value=" ")
        tk.Label(
            latest_row,
            textvariable=self.word_var,
            fg=CYAN,
            bg=STRIP,
            font=("Segoe UI", 28, "bold"),
            anchor="w",
        ).pack(anchor="w")
        self.thumb = tk.Label(
            latest_row,
            bg=CARD,
            bd=0,
            highlightbackground=CYAN,
            highlightthickness=1,
        )
        # Shown only in Draw mode so text results stay in the text panel.

        out_wrap = tk.Frame(body, bg=STRIP)
        out_wrap.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._text_wrap = tk.Frame(out_wrap, bg=STRIP)
        self._text_wrap.pack(fill=tk.BOTH, expand=True)
        self._out_title = tk.Label(
            self._text_wrap,
            text="TEXT OUTPUT",
            fg=CYAN,
            bg=STRIP,
            font=("Segoe UI", 11, "bold"),
            anchor="w",
        )
        self._out_title.pack(fill=tk.X, pady=(0, 6))
        self._out = self._make_output_text(self._text_wrap)
        self._out.insert("1.0", "No stored text yet. Write a word — it stays listed here.", "hint")

        self._draw_wrap = tk.Frame(out_wrap, bg=STRIP)
        self._draw_title = tk.Label(
            self._draw_wrap,
            text="DRAWING OUTPUT",
            fg=MAGENTA,
            bg=STRIP,
            font=("Segoe UI", 11, "bold"),
            anchor="w",
        )
        self._draw_title.pack(fill=tk.X, pady=(0, 6))
        self._draw_out = self._make_output_text(self._draw_wrap)
        self._draw_out.insert(
            "1.0",
            "No stored drawings yet. Draw a shape — it stays listed here.",
            "hint",
        )
        bottom.bind("<Configure>", self._on_bottom_resize)

        # Camera fills everything above the strip.
        self._stage = tk.Frame(self.root, bg=BG)
        self._stage.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.preview = tk.Label(self._stage, bg="#050714", bd=0, highlightthickness=0)
        self.preview.place(relx=0, rely=0, relwidth=1, relheight=1)
        self._stage.bind("<Configure>", self._on_stage_resize)

        self._canvas_title = tk.Label(
            self._stage,
            text="AIR WRITING",
            fg=CYAN,
            bg="#0b1024",
            font=("Segoe UI", 10, "bold"),
            padx=12,
            pady=6,
        )
        self._canvas_title.place(x=16, y=12, anchor="nw")
        self._canvas_tip = tk.Label(
            self._stage,
            text="Tip: Raise your index finger to write. Lift your hand to stop.",
            fg=TEXT,
            bg="#152044",
            font=("Segoe UI", 10),
            padx=14,
            pady=7,
        )
        self._canvas_tip.place(relx=0.5, rely=1.0, x=0, y=-14, anchor="s")

        # Tool rail floats on the camera (hidden in Voice — modes live in the header).
        self._rail = tk.Frame(
            self._stage,
            bg=OVERLAY,
            padx=10,
            pady=10,
            highlightbackground=BORDER,
            highlightthickness=1,
        )
        self._rail.place(relx=1.0, rely=0.0, x=-16, y=48, anchor="ne")

        tools_lbl = tk.Label(
            self._rail,
            text="TOOLS",
            fg=CYAN,
            bg=OVERLAY,
            font=("Segoe UI", 9, "bold"),
            anchor="w",
        )
        tools_lbl.pack(fill=tk.X, pady=(0, 6))

        self._ink_tools = tk.Frame(self._rail, bg=OVERLAY)
        self._ink_tools.pack(fill=tk.X)
        self._undo_btn = self._rail_btn(self._ink_tools, "Undo Last", on_undo)
        self._undo_btn.pack(fill=tk.X, pady=2)
        self._clear_canvas_btn = self._primary_btn(self._ink_tools, "Clear", on_clear)
        self._clear_canvas_btn.pack(fill=tk.X, pady=2)

        self._rail_div1 = tk.Frame(self._rail, bg=BORDER, height=1)
        self._rail_div1.pack(fill=tk.X, pady=(6, 6))

        self._swatches = tk.Frame(self._rail, bg=OVERLAY)
        self._swatches.pack()
        tk.Label(
            self._swatches,
            text="Pen Color",
            fg=MUTED,
            bg=OVERLAY,
            font=("Segoe UI", 8, "bold"),
        ).pack(pady=(0, 4))
        swatch_row = tk.Frame(self._swatches, bg=OVERLAY)
        swatch_row.pack()
        for i, (_name, bgr, hex_color) in enumerate(COLORS):
            if i % 3 == 0:
                swatch_row = tk.Frame(self._swatches, bg=OVERLAY)
                swatch_row.pack()
            btn = tk.Button(
                swatch_row,
                text=" ",
                bg=hex_color,
                activebackground=hex_color,
                fg=hex_color,
                width=2,
                height=1,
                bd=0,
                relief=tk.FLAT,
                cursor="hand2",
                highlightthickness=2,
                highlightbackground=hex_color,
                command=lambda c=bgr: self._select_color(c),
            )
            btn.pack(side=tk.LEFT, padx=2, pady=2)
            self._color_btns[bgr] = btn

        self._rail_div2 = tk.Frame(self._rail, bg=BORDER, height=1)
        self._rail_div2.pack(fill=tk.X, pady=(6, 0))

        self._build_voice_page(on_toggle_mic, on_pause_voice, on_save_txt, on_save_pdf, on_clear_transcript)

        self._refresh_mode_buttons()
        self._refresh_color_buttons()
        self.set_auto_recognize(True)
        self._set_live_pill()

    def _rail_btn(self, parent: tk.Widget, text: str, command: Callable[[], None]) -> tk.Button:
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=BTN,
            fg=TEXT,
            activebackground=BTN_HOVER,
            activeforeground=TEXT,
            relief=tk.FLAT,
            bd=0,
            padx=12,
            pady=7,
            font=("Segoe UI", 10, "bold"),
            cursor="hand2",
            highlightthickness=1,
            highlightbackground=BORDER,
        )

    def _primary_btn(self, parent: tk.Widget, text: str, command: Callable[[], None]) -> tk.Button:
        btn = self._rail_btn(parent, text, command)
        btn.configure(
            bg=ACCENT,
            fg="#ffffff",
            activebackground=ACCENT_HOVER,
            activeforeground="#ffffff",
            highlightbackground=ACCENT,
            font=("Segoe UI", 11, "bold"),
            padx=16,
            pady=8,
        )
        return btn

    def _chip_btn(self, parent: tk.Widget, text: str, command: Callable[[], None]) -> tk.Button:
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=BTN,
            fg=TEXT,
            activebackground=BTN_HOVER,
            activeforeground=TEXT,
            relief=tk.FLAT,
            bd=0,
            padx=16,
            pady=8,
            font=("Segoe UI", 10, "bold"),
            cursor="hand2",
            highlightthickness=1,
            highlightbackground=BORDER,
        )

    def _build_header(self) -> None:
        header = tk.Frame(self.root, bg=BG, padx=20, pady=10)
        self._header = header
        header.pack(side=tk.TOP, fill=tk.X)
        header.columnconfigure(0, weight=1)
        header.columnconfigure(1, weight=0)

        brand = tk.Frame(header, bg=BG)
        brand.grid(row=0, column=0, sticky="w")
        title_row = tk.Frame(brand, bg=BG)
        title_row.pack(anchor="w")
        tk.Label(
            title_row,
            text="✎",
            fg=CYAN,
            bg=BG,
            font=("Segoe UI", 18, "bold"),
        ).pack(side=tk.LEFT, padx=(0, 8))
        self._brand_title = tk.Label(
            title_row,
            text="AIR WRITING RECOGNITION",
            fg=TEXT,
            bg=BG,
            font=("Segoe UI", 16, "bold"),
        )
        self._brand_title.pack(side=tk.LEFT)
        self._brand_tagline = tk.Label(
            brand,
            text="Write in the air. We recognize your ideas.",
            fg=MUTED,
            bg=BG,
            font=("Segoe UI", 10),
        )
        self._brand_tagline.pack(anchor="w", padx=(30, 0), pady=(2, 0))

        modes = tk.Frame(header, bg=BG)
        modes.grid(row=0, column=1, sticky="e")
        self._mode_btns["text"] = self._chip_btn(
            modes, "Text", lambda: self._select_mode("text")
        )
        self._mode_btns["text"].pack(side=tk.LEFT, padx=4)
        self._mode_btns["draw"] = self._chip_btn(
            modes, "Draw", lambda: self._select_mode("draw")
        )
        self._mode_btns["draw"].pack(side=tk.LEFT, padx=4)
        self._mode_btns["voice"] = self._chip_btn(
            modes, "Voice", lambda: self._select_mode("voice")
        )
        self._mode_btns["voice"].pack(side=tk.LEFT, padx=4)

    def _hero_btn(self, parent: tk.Widget, text: str, command: Callable[[], None]) -> tk.Button:
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=BTN,
            fg=TEXT,
            activebackground=BTN_HOVER,
            activeforeground=TEXT,
            relief=tk.FLAT,
            bd=0,
            padx=28,
            pady=14,
            font=("Segoe UI", 15, "bold"),
            cursor="hand2",
            highlightthickness=1,
            highlightbackground=BORDER,
        )

    def _make_output_text(self, parent: tk.Widget) -> tk.Text:
        out_frame = tk.Frame(parent, bg=CARD, highlightbackground=BORDER, highlightthickness=1)
        out_frame.pack(fill=tk.BOTH, expand=True)
        scroll = tk.Scrollbar(out_frame, bg=STRIP, troughcolor=CARD, highlightthickness=0)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        widget = tk.Text(
            out_frame,
            bg=CARD,
            fg=TEXT,
            insertbackground=CARD,
            font=("Segoe UI", 18, "bold"),
            wrap=tk.WORD,
            height=6,
            bd=0,
            highlightthickness=0,
            yscrollcommand=scroll.set,
            padx=12,
            pady=10,
            takefocus=0,
        )
        widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.configure(command=widget.yview)
        self._style_transcript(widget)
        return widget

    def _style_transcript(self, widget: tk.Text) -> None:
        widget.tag_configure("item", foreground=TEXT)
        widget.tag_configure("shape", foreground=MAGENTA)
        widget.tag_configure("voice", foreground=CYAN)
        widget.tag_configure("voice_live", foreground=CYAN)
        widget.tag_configure("hint", foreground=MUTED)
        widget.bind("<Key>", lambda e: "break")
        widget.bind("<<Paste>>", lambda e: "break")
        widget.bind("<<Cut>>", lambda e: "break")

    def _bind_mousewheel(self, widget: tk.Text) -> None:
        def on_wheel(event: tk.Event) -> str:
            steps = int(-event.delta / 120) if event.delta else 0
            if steps:
                widget.yview_scroll(steps, "units")
            return "break"

        widget.bind("<MouseWheel>", on_wheel)
        widget.bind("<Button-4>", lambda e: widget.yview_scroll(-1, "units") or "break")
        widget.bind("<Button-5>", lambda e: widget.yview_scroll(1, "units") or "break")

    def _scroll_if_pinned(self, widget: tk.Text) -> None:
        try:
            _first, last = widget.yview()
        except tk.TclError:
            return
        if float(last) >= 0.995:
            widget.see(tk.END)

    def _build_voice_page(
        self,
        on_toggle_mic: Callable[[], None] | None,
        on_pause_voice: Callable[[], None] | None,
        on_save_txt: Callable[[], None],
        on_save_pdf: Callable[[], None],
        on_clear_transcript: Callable[[], None],
    ) -> None:
        self._voice_page = tk.Frame(self.root, bg=VOICE_BG)
        inner = tk.Frame(self._voice_page, bg=VOICE_BG)
        inner.pack(fill=tk.BOTH, expand=True, padx=48, pady=28)

        heading = tk.Frame(inner, bg=VOICE_BG)
        heading.pack(fill=tk.X, pady=(0, 8))
        self._voice_title = tk.Label(
            heading,
            text="Voice",
            fg=TEXT,
            bg=VOICE_BG,
            font=("Segoe UI", 26, "bold"),
            anchor="w",
        )
        self._voice_title.pack(side=tk.LEFT)
        tk.Label(
            inner,
            text="Camera is off. Speak into your microphone — words appear below as you talk.",
            fg=MUTED,
            bg=VOICE_BG,
            font=("Segoe UI", 12),
            anchor="w",
            wraplength=820,
            justify=tk.LEFT,
        ).pack(anchor="w", pady=(0, 16))

        pill = tk.Frame(
            inner,
            bg=VOICE_PILL,
            padx=16,
            pady=10,
            highlightbackground=BORDER,
            highlightthickness=1,
        )
        pill.pack(anchor="w", pady=(0, 14))
        self._voice_status_dot = tk.Label(
            pill, text="●", fg=MUTED, bg=VOICE_PILL, font=("Segoe UI", 12)
        )
        self._voice_status_dot.pack(side=tk.LEFT, padx=(0, 8))
        self._voice_status_var = tk.StringVar(value="Ready to listen")
        tk.Label(
            pill,
            textvariable=self._voice_status_var,
            fg=TEXT,
            bg=VOICE_PILL,
            font=("Segoe UI", 13),
        ).pack(side=tk.LEFT)

        self._wave = tk.Canvas(
            inner,
            height=64,
            bg=VOICE_BG,
            highlightthickness=0,
            bd=0,
        )
        self._wave.pack(fill=tk.X, pady=(0, 16))
        self._wave.bind("<Configure>", lambda _e: self._draw_wave())

        self._voice_tools = tk.Frame(inner, bg=VOICE_BG)
        self._voice_tools.pack(anchor="w", pady=(0, 18))
        self._mic_btn = self._hero_btn(
            self._voice_tools, "Mic: OFF", on_toggle_mic or (lambda: None)
        )
        self._mic_btn.pack(side=tk.LEFT, padx=(0, 12))
        self._pause_btn = self._hero_btn(
            self._voice_tools, "Pause", on_pause_voice or (lambda: None)
        )
        self._pause_btn.configure(state=tk.DISABLED, fg=MUTED)
        self._pause_btn.pack(side=tk.LEFT)

        card = tk.Frame(
            inner,
            bg=VOICE_CARD,
            highlightbackground=CYAN,
            highlightthickness=1,
        )
        card.pack(fill=tk.BOTH, expand=True)
        tk.Label(
            card,
            text="RECOGNIZED TEXT",
            fg=CYAN,
            bg=VOICE_CARD,
            font=("Segoe UI", 11, "bold"),
            anchor="w",
        ).pack(fill=tk.X, padx=18, pady=(16, 6))

        out_frame = tk.Frame(card, bg=VOICE_CARD)
        out_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))
        self._voice_scroll = tk.Scrollbar(
            out_frame,
            orient=tk.VERTICAL,
            width=18,
            bg=BTN,
            troughcolor=VOICE_PILL,
            activebackground=CYAN,
            highlightthickness=0,
        )
        self._voice_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._voice_out = tk.Text(
            out_frame,
            bg=VOICE_CARD,
            fg=TEXT,
            insertbackground=VOICE_CARD,
            font=("Segoe UI", 20),
            wrap=tk.WORD,
            height=12,
            bd=0,
            highlightthickness=0,
            yscrollcommand=self._voice_scroll.set,
            padx=8,
            pady=8,
            takefocus=1,
        )
        self._voice_out.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._voice_scroll.configure(command=self._voice_out.yview)
        self._style_transcript(self._voice_out)
        self._bind_mousewheel(self._voice_out)
        self._voice_out.insert(
            "1.0",
            "No speech yet. Click Mic and speak — text appears here live.",
            "hint",
        )

        footer = tk.Frame(inner, bg=VOICE_BG)
        footer.pack(fill=tk.X, pady=(16, 0))
        self._rail_btn(footer, "Save TXT", on_save_txt).pack(side=tk.LEFT)
        self._rail_btn(footer, "Save PDF", on_save_pdf).pack(side=tk.LEFT, padx=8)
        self._primary_btn(footer, "Clear text", on_clear_transcript).pack(side=tk.RIGHT)
        quote = tk.Label(
            footer,
            text="Every idea starts with a single stroke in the air.",
            fg=MUTED,
            bg=VOICE_BG,
            font=("Segoe UI", 9, "italic"),
        )
        quote.pack(side=tk.RIGHT, padx=(0, 16))

    def _panel(self) -> tk.Text:
        if self._mode == "voice":
            return self._voice_out
        if self._mode == "draw":
            return self._draw_out
        return self._out

    def _history_kind(self) -> str:
        if self._mode == "draw":
            return "shape"
        if self._mode == "voice":
            return "voice"
        return "word"

    def _select_mode(self, mode: str) -> None:
        self._mode = mode
        self._refresh_mode_buttons()
        self._apply_mode_chrome()
        self._last_history_key = None
        self._on_mode(mode)

    def _select_color(self, bgr: tuple[int, int, int]) -> None:
        self._color_bgr = bgr
        self._refresh_color_buttons()
        self._on_color(bgr)

    def _refresh_mode_buttons(self) -> None:
        for mode, btn in self._mode_btns.items():
            if mode == self._mode:
                btn.configure(
                    bg=ACCENT,
                    activebackground=ACCENT_HOVER,
                    fg="#ffffff",
                    highlightbackground=ACCENT,
                )
            else:
                btn.configure(
                    bg=BTN,
                    activebackground=BTN_HOVER,
                    fg=TEXT,
                    highlightbackground=BORDER,
                )

    def _set_live_pill(self) -> None:
        return

    def _lift_overlays(self) -> None:
        self._canvas_title.lift()
        self._canvas_tip.lift()
        self._rail.lift()

    def _tick_wave(self) -> None:
        self._wave_job = None
        try:
            if not self.root.winfo_exists():
                return
            if self._mode != "voice" or self._voice_state != "listening":
                self._draw_wave()
                return
            self._wave_phase = (self._wave_phase + 1) % 24
            self._draw_wave()
            self._wave_job = self.root.after(90, self._tick_wave)
        except tk.TclError:
            return

    def _draw_wave(self) -> None:
        canvas = getattr(self, "_wave", None)
        if canvas is None:
            return
        try:
            canvas.delete("all")
            width = max(int(canvas.winfo_width()), 120)
            height = max(int(canvas.winfo_height()), 40)
        except tk.TclError:
            return
        bars = 36
        gap = 4
        bar_w = max(3, int((width - gap * (bars + 1)) / bars))
        active = self._mode == "voice" and self._voice_state == "listening"
        paused = self._mode == "voice" and self._voice_state == "paused"
        for i in range(bars):
            x = gap + i * (bar_w + gap)
            if active:
                wave = abs(((i + self._wave_phase) % 12) - 6) / 6.0
                h = int(10 + wave * (height - 18))
                color = CYAN if i % 3 else ACCENT
            elif paused:
                h = 14 + (i % 4) * 4
                color = "#fbbf24"
            else:
                h = 8 + (i % 5) * 3
                color = BORDER
            y0 = (height - h) // 2
            canvas.create_rectangle(x, y0, x + bar_w, y0 + h, fill=color, outline="")

    def _apply_mode_chrome(self) -> None:
        if self._mode == "voice":
            self._stage.pack_forget()
            self._bottom.pack_forget()
            self._voice_page.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            self._draw_wave()
            if self._voice_state == "listening" and self._wave_job is None:
                self._tick_wave()
            return
        if self._wave_job is not None:
            self.root.after_cancel(self._wave_job)
            self._wave_job = None
        self._voice_page.pack_forget()
        self._bottom.pack(side=tk.BOTTOM, fill=tk.X)
        self._stage.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self._write_tools.pack(side=tk.LEFT)
        self._rail.place(relx=1.0, rely=0.0, x=-16, y=48, anchor="ne")
        if self._mode == "draw":
            self._text_wrap.pack_forget()
            self._draw_wrap.pack(fill=tk.BOTH, expand=True)
            self.thumb.pack(anchor="w", pady=(8, 0))
            self._canvas_tip.configure(
                text="Tip: Draw a star, heart, circle, square, triangle, or line. Hold 2s."
            )
        else:
            self._draw_wrap.pack_forget()
            self._text_wrap.pack(fill=tk.BOTH, expand=True)
            self.thumb.pack_forget()
            self._canvas_tip.configure(
                text="Tip: Raise your index finger to write. Lift your hand to stop."
            )
        self._lift_overlays()

    def _refresh_color_buttons(self) -> None:
        for bgr, btn in self._color_btns.items():
            hex_color = next(h for _n, c, h in COLORS if c == bgr)
            if bgr == self._color_bgr:
                btn.configure(
                    relief=tk.SUNKEN,
                    highlightbackground="#ffffff",
                    highlightcolor="#ffffff",
                    highlightthickness=3,
                    bd=4,
                )
            else:
                btn.configure(
                    relief=tk.RAISED,
                    highlightbackground=hex_color,
                    highlightcolor=hex_color,
                    highlightthickness=1,
                    bd=2,
                )

    def _on_stage_resize(self, event: tk.Event) -> None:
        if event.widget is not self._stage or event.width <= 1 or event.height <= 1:
            return
        self._view_size = (event.width, event.height)
        if hasattr(self, "_canvas_title"):
            self._lift_overlays()

    def _on_bottom_resize(self, event: tk.Event) -> None:
        wrap = max(320, event.width - 36)
        if wrap != self._wraplength:
            self._wraplength = wrap

    def set_status(self, text: str) -> None:
        if text == self._last_status:
            return
        self._last_status = text
        self.status_var.set(text)
        if (
            self._mode == "voice"
            and self._voice_state == "stopped"
            and not text.startswith("Voice mode")
        ):
            self._voice_status_var.set(text)

    def set_word(self, text: str) -> None:
        value = text if text else " "
        if value == self._last_word:
            return
        self._last_word = value
        self.word_var.set(value)

    def set_transcript(self, text: str) -> None:
        rows: list[tuple[str, str, bytes | None]] = []
        if text:
            rows = [(line, "word", None) for line in text.splitlines() if line.strip()]
        self.set_history(rows)

    def set_history(self, rows: list[tuple[str, str, bytes | None]]) -> None:
        kind = self._history_kind()
        rows = [(t, k, p) for t, k, p in rows if k == kind]
        key = (self._mode, tuple((t, k, None if p is None else len(p)) for t, k, p in rows))
        if key == self._last_history_key:
            return
        self._last_history_key = key
        photos = self._hist_photos_by_mode.setdefault(self._mode, [])
        photos.clear()
        self._hist_photos = photos
        panel = self._panel()
        panel.delete("1.0", tk.END)
        if not rows:
            if self._mode == "voice":
                hint = "No speech yet. Click Mic and speak — text appears here live."
            elif self._mode == "draw":
                hint = "No stored drawings yet. Draw a shape — it stays listed here."
            else:
                hint = "No stored text yet. Write a word — it stays listed here."
            panel.insert("1.0", hint, "hint")
        else:
            for i, (label, item_kind, png) in enumerate(rows, 1):
                color_tag = "shape" if item_kind == "shape" else (
                    "voice" if item_kind == "voice" else "item"
                )
                panel.insert(tk.END, f"{i}. {label}\n", color_tag)
                if png:
                    photo = self._png_to_photo(png, max_h=96)
                    if photo is not None:
                        photos.append(photo)
                        panel.image_create(tk.END, image=photo)
                        panel.insert(tk.END, "\n")
        self._render_voice_live()
        if self._mode == "voice":
            self._scroll_if_pinned(panel)
        else:
            panel.see(tk.END)
        self._last_transcript = key

    def _png_to_photo(self, png: bytes, max_h: int) -> ImageTk.PhotoImage | None:
        arr = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_COLOR)
        if arr is None:
            return None
        h, w = arr.shape[:2]
        if h > max_h:
            s = max_h / h
            arr = cv2.resize(arr, (max(1, int(w * s)), max_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        return ImageTk.PhotoImage(Image.fromarray(rgb))

    def set_drawing_thumb(self, png: bytes | None) -> None:
        if png == getattr(self, "_last_thumb_png", object()):
            return
        self._last_thumb_png = png
        if not png:
            self.thumb.configure(image="", text="")
            self._thumb_photo = None
            return
        photo = self._png_to_photo(png, max_h=110)
        if photo is None:
            return
        self._thumb_photo = photo
        self.thumb.configure(image=self._thumb_photo)

    def ask_save_path(
        self, title: str, defaultextension: str, filetypes: list[tuple[str, str]]
    ) -> str | None:
        path = filedialog.asksaveasfilename(
            parent=self.root,
            title=title,
            defaultextension=defaultextension,
            filetypes=filetypes,
        )
        return path or None

    def show_preview(self, frame_bgr: np.ndarray) -> None:
        if self._mode == "voice":
            return
        tw, th = self._view_size
        tw, th = max(tw, 320), max(th, 240)
        # Cap PhotoImage size — huge Tk bitmaps stall the UI when tracking is busy.
        max_w, max_h = 1280, 720
        if tw > max_w or th > max_h:
            s = min(max_w / tw, max_h / th)
            tw, th = max(1, int(tw * s)), max(1, int(th * s))
        h, w = frame_bgr.shape[:2]
        scale = max(tw / w, th / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(frame_bgr, (nw, nh), interpolation=interp)
        left = max(0, (nw - tw) // 2)
        top = max(0, (nh - th) // 2)
        crop = resized[top : top + th, left : left + tw]
        if not crop.flags["C_CONTIGUOUS"]:
            crop = np.ascontiguousarray(crop)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.preview.configure(image=self._photo)
        self._lift_overlays()

    def show_drawing(self, frame_bgr: np.ndarray) -> None:
        return

    def alert(self, title: str, message: str) -> None:
        messagebox.showerror(title, message)

    def info(self, title: str, message: str) -> None:
        messagebox.showinfo(title, message)

    def set_auto_recognize(self, enabled: bool) -> None:
        if enabled:
            self._auto_btn.configure(
                text="Auto: ON",
                bg=ACCENT,
                activebackground=ACCENT_HOVER,
                fg="#ffffff",
                highlightbackground=ACCENT,
            )
        else:
            self._auto_btn.configure(
                text="Auto: OFF",
                bg=BTN,
                activebackground=BTN_HOVER,
                fg=TEXT,
                highlightbackground=BORDER,
            )

    def set_ocr_space(self, enabled: bool) -> None:
        if enabled:
            self._ocr_btn.configure(
                text="OCR: ON",
                bg=ACCENT,
                activebackground=ACCENT_HOVER,
                fg="#ffffff",
                highlightbackground=ACCENT,
            )
        else:
            self._ocr_btn.configure(
                text="OCR: OFF",
                bg=BTN,
                activebackground=BTN_HOVER,
                fg=TEXT,
                highlightbackground=BORDER,
            )

    def set_voice_state(self, state: str) -> None:
        self._voice_state = state
        hero = {"font": ("Segoe UI", 15, "bold"), "padx": 28, "pady": 14}
        if state == "listening":
            self._mic_btn.configure(
                text="Mic: ON",
                bg="#dc2626",
                activebackground="#b91c1c",
                fg="#ffffff",
                **hero,
            )
            self._pause_btn.configure(
                text="Pause",
                state=tk.NORMAL,
                bg=BTN,
                fg=TEXT,
                activebackground=BTN_HOVER,
                **hero,
            )
            self._voice_status_var.set("Listening — speak now")
            self._voice_status_dot.configure(fg=LIVE)
        elif state == "paused":
            self._mic_btn.configure(
                text="Mic: PAUSED",
                bg="#d97706",
                activebackground="#b45309",
                fg="#ffffff",
                **hero,
            )
            self._pause_btn.configure(
                text="Start",
                state=tk.NORMAL,
                bg=ACCENT,
                activebackground=ACCENT_HOVER,
                fg="#ffffff",
                **hero,
            )
            self._voice_status_var.set("Paused — tap Start to continue")
            self._voice_status_dot.configure(fg="#fbbf24")
        else:
            self._mic_btn.configure(
                text="Mic: OFF",
                bg=BTN,
                activebackground=BTN_HOVER,
                fg=TEXT,
                **hero,
            )
            self._pause_btn.configure(
                text="Pause",
                state=tk.DISABLED,
                bg=BTN,
                fg=MUTED,
                activebackground=BTN_HOVER,
                **hero,
            )
            self._voice_status_var.set("Ready to listen")
            self._voice_status_dot.configure(fg=MUTED)
        self._set_live_pill()
        if state == "listening" and self._mode == "voice":
            if self._wave_job is None:
                self._tick_wave()
        else:
            if self._wave_job is not None:
                self.root.after_cancel(self._wave_job)
                self._wave_job = None
            self._draw_wave()

    def set_voice_live(self, text: str, state: str) -> None:
        text = text or ""
        if text == self._voice_live and state == self._voice_state:
            return
        self._voice_live = text
        self.set_voice_state(state)
        self._render_voice_live()
        if self._mode == "voice":
            self._scroll_if_pinned(self._panel())
        else:
            self._panel().see(tk.END)

    def _render_voice_live(self) -> None:
        panel = self._panel()
        ranges = panel.tag_ranges("voice_live")
        if ranges:
            panel.delete(ranges[0], ranges[1])
        if self._voice_state == "stopped" and not self._voice_live:
            return
        if self._mode != "voice":
            return
        if self._voice_state == "listening":
            prefix = "Listening: "
        elif self._voice_state == "paused":
            prefix = "Paused: "
        else:
            prefix = "Voice: "
        body = self._voice_live if self._voice_live else (
            "speak now…" if self._voice_state == "listening" else ""
        )
        if not body and self._voice_state != "listening":
            return
        current = panel.get("1.0", "end-1c")
        if current.startswith("No stored results yet") or current.startswith("No speech yet"):
            panel.delete("1.0", tk.END)
        line = f"{prefix}{body}\n"
        panel.insert(tk.END, line, ("voice_live", "voice"))

    def ask_text(self, title: str, prompt: str, initial: str = "") -> str | None:
        return simpledialog.askstring(
            title, prompt, initialvalue=initial, parent=self.root
        )

    def run(self) -> None:
        self.root.mainloop()
