"""Speech-to-text via SpeechRecognition + PyAudio."""

from __future__ import annotations

import re
import threading
import time
from typing import Callable, Protocol

LiveCallback = Callable[[str, str], None]
TextCallback = Callable[[str], None]

# End a phrase only after a real pause — not between words, not mid-sentence.
PAUSE_THRESHOLD = 1.2
NON_SPEAKING_DURATION = 0.5
# Google Web Speech is reliable on ~8s clips. Longer sentences continue in the
# next listen and are joined in VoiceTranscript.
PHRASE_TIME_LIMIT = 8.0
GOOGLE_CHUNK_SEC = 8.0
GOOGLE_CHUNK_OVERLAP_SEC = 0.7


def _normalize(text: str) -> str:
    return " ".join((text or "").split())


def audio_duration_sec(audio) -> float:
    rate = max(1, int(getattr(audio, "sample_rate", 1) or 1))
    width = max(1, int(getattr(audio, "sample_width", 1) or 1))
    data = getattr(audio, "frame_data", b"") or b""
    return len(data) / (rate * width)


def split_audio_for_google(
    audio, chunk_sec: float = GOOGLE_CHUNK_SEC, overlap_sec: float = GOOGLE_CHUNK_OVERLAP_SEC
):
    """Split long utterances so Google Web Speech can transcribe a full sentence."""
    duration = audio_duration_sec(audio)
    if duration <= chunk_sec + 0.3:
        return [audio]
    rate = max(1, int(getattr(audio, "sample_rate", 1) or 1))
    width = max(1, int(getattr(audio, "sample_width", 1) or 1))
    data = getattr(audio, "frame_data", b"") or b""
    if data:
        step = max(width, int(chunk_sec * rate) * width)
        overlap = int(overlap_sec * rate) * width
        chunks = []
        start = 0
        n = len(data)
        while start < n:
            end = min(n, start + step)
            piece = data[start:end]
            if len(piece) < width * max(1, rate // 10):
                break
            chunk = _audio_from_frames(audio, piece, rate, width)
            chunks.append(chunk if chunk is not None else audio)
            if end >= n:
                break
            start = max(0, end - overlap)
            if start >= end:
                break
        return chunks or [audio]
    if hasattr(audio, "get_segment"):
        chunks = []
        start = 0.0
        while start < duration - 0.05:
            end = min(duration, start + chunk_sec)
            chunks.append(audio.get_segment(start * 1000.0, end * 1000.0))
            if end >= duration - 0.02:
                break
            start = max(0.0, end - overlap_sec)
            if start >= end:
                break
        return chunks or [audio]
    return [audio]


def _audio_from_frames(audio, piece: bytes, rate: int, width: int):
    cls = type(audio)
    try:
        chunk = cls(piece, rate, width)
        if getattr(chunk, "frame_data", None) == piece:
            return chunk
    except Exception:
        chunk = None
    try:
        part = cls(0)
    except Exception:
        return None
    part.sample_rate = rate
    part.sample_width = width
    part.frame_data = piece
    return part


def join_chunk_transcripts(parts: list[str]) -> str:
    cleaned = [_normalize(p) for p in parts if _normalize(p)]
    if not cleaned:
        return ""
    merged = cleaned[0]
    for nxt in cleaned[1:]:
        left = merged.split()
        right = nxt.split()
        overlap = 0
        max_k = min(len(left), len(right), 8)
        for k in range(max_k, 0, -1):
            if left[-k:] == right[:k]:
                overlap = k
                break
        merged = " ".join(left + right[overlap:])
    return merged


def _product_key(name: str) -> str:
    text = name or ""
    found = re.search(r"\(([^)]*)", text)
    raw = found.group(1) if found else text
    return " ".join(raw.lower().replace(",", " ").split())


def _api_rank(device: dict) -> int:
    api = str(device.get("hostapi_name") or "").lower()
    name = str(device.get("name") or "").lower()
    blob = f"{api} {name}"
    if "wdm-ks" in blob or "mapper" in blob or blob.startswith("primary sound"):
        return 90
    if "wasapi" in blob:
        return 0
    if "mme" in blob:
        return 1
    if "directsound" in blob:
        return 2
    return 5


def choose_input_device(
    devices: list[dict], default_in: int, default_out: int | None
) -> int:
    """Prefer the headset mic when headphones/earbuds are the output device."""

    def usable(index: int) -> bool:
        if index < 0 or index >= len(devices):
            return False
        d = devices[index]
        name = str(d.get("name") or "").lower()
        if "mapper" in name or name.startswith("primary sound"):
            return False
        return int(d.get("max_input_channels") or 0) >= 1

    def best_sibling(index: int) -> int:
        if not usable(index):
            return index
        prod = _product_key(str(devices[index].get("name") or ""))
        best = index
        best_rank = _api_rank(devices[index])
        for i, d in enumerate(devices):
            if not usable(i):
                continue
            if _product_key(str(d.get("name") or "")) != prod:
                continue
            rank = _api_rank(d)
            if rank < best_rank:
                best, best_rank = i, rank
        return best

    chosen = best_sibling(int(default_in))
    if default_out is None or not (0 <= default_out < len(devices)):
        return chosen
    out_name = str(devices[default_out].get("name") or "").lower()
    headset_out = any(
        token in out_name
        for token in ("headphone", "headset", "earbuds", "airpod", "airbass", "bluetooth", "buds")
    )
    if not headset_out:
        return chosen
    out_prod = _product_key(str(devices[default_out].get("name") or ""))
    matches: list[tuple[int, int]] = []
    for i, d in enumerate(devices):
        if not usable(i):
            continue
        prod = _product_key(str(d.get("name") or ""))
        name = str(d.get("name") or "").lower()
        if out_prod and (out_prod in prod or prod in out_prod):
            matches.append((_api_rank(d), i))
        elif "headset" in name or "headphone" in name:
            matches.append((_api_rank(d) + 10, i))
    if matches:
        matches.sort()
        return matches[0][1]
    return chosen


def pyaudio_device_table(pa) -> tuple[list[dict], int, int | None]:
    devices: list[dict] = []
    default_in = int(pa.get_default_input_device_info()["index"])
    try:
        default_out = int(pa.get_default_output_device_info()["index"])
    except Exception:
        default_out = None
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        host = ""
        try:
            host = str(pa.get_host_api_info_by_index(int(info["hostApi"])).get("name") or "")
        except Exception:
            host = ""
        devices.append(
            {
                "name": str(info.get("name") or ""),
                "max_input_channels": int(info.get("maxInputChannels") or 0),
                "hostapi_name": host,
            }
        )
    return devices, default_in, default_out


class VoiceTranscript:
    """Assemble live speech with spaces. Pauses freeze text; they do not clear it."""

    def __init__(self) -> None:
        self._finals: list[str] = []
        self._partial: str = ""
        self._paused: bool = False

    @property
    def paused(self) -> bool:
        return self._paused

    def apply_partial(self, text: str) -> None:
        if self._paused:
            return
        self._partial = _normalize(text)

    def apply_final(self, text: str) -> None:
        if self._paused:
            return
        cleaned = _normalize(text)
        if cleaned:
            self._finals.append(cleaned)
        self._partial = ""

    def display(self) -> str:
        parts = list(self._finals)
        if self._partial:
            parts.append(self._partial)
        return " ".join(parts)

    def pause(self) -> None:
        if self._partial:
            self._finals.append(self._partial)
            self._partial = ""
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def take_finals(self) -> str:
        text = " ".join(self._finals)
        self._finals = []
        return text

    def clear(self) -> None:
        self._finals.clear()
        self._partial = ""
        self._paused = False


class StreamingEngine(Protocol):
    def start(
        self,
        on_partial: TextCallback,
        on_final: TextCallback,
        on_error: TextCallback,
    ) -> None: ...

    def pause(self) -> None: ...

    def resume(self) -> None: ...

    def stop(self) -> None: ...


class SpeechMicEngine:
    """PyAudio mic → SpeechRecognition. Starts immediately (no wait-for-silence)."""

    def __init__(self, on_status: TextCallback | None = None) -> None:
        self._on_status = on_status
        self._on_partial: TextCallback | None = None
        self._on_final: TextCallback | None = None
        self._on_error: TextCallback | None = None
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._device_name = "microphone"

    def start(
        self,
        on_partial: TextCallback,
        on_final: TextCallback,
        on_error: TextCallback,
    ) -> None:
        self._on_partial = on_partial
        self._on_final = on_final
        self._on_error = on_error
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                self._paused.clear()
                return
            self._stop.clear()
            self._paused.clear()
            self._thread = threading.Thread(target=self._run, daemon=True, name="awr-voice")
            self._thread.start()

    def pause(self) -> None:
        self._paused.set()

    def resume(self) -> None:
        self._paused.clear()

    def stop(self) -> None:
        self._stop.set()
        self._paused.clear()
        thread = self._thread
        if thread is not None and thread.is_alive() and threading.current_thread() is not thread:
            thread.join(timeout=3.0)
        self._thread = None

    def _emit_status(self, text: str) -> None:
        if self._on_status:
            self._on_status(text)

    def _emit_error(self, text: str) -> None:
        if self._on_error:
            self._on_error(text)

    def _run(self) -> None:
        try:
            import pyaudio
            import speech_recognition as sr
        except ImportError:
            self._emit_error(
                "Voice recognition needs SpeechRecognition and PyAudio. "
                "In the project venv run: pip install SpeechRecognition PyAudio"
            )
            return

        pa = pyaudio.PyAudio()
        try:
            devices, default_in, default_out = pyaudio_device_table(pa)
            device_idx = choose_input_device(devices, default_in, default_out)
            self._device_name = str(devices[device_idx].get("name") or "microphone")
            recognizer = sr.Recognizer()
            recognizer.dynamic_energy_threshold = True
            recognizer.dynamic_energy_adjustment_damping = 0.15
            recognizer.dynamic_energy_ratio = 1.5
            recognizer.pause_threshold = PAUSE_THRESHOLD
            recognizer.non_speaking_duration = NON_SPEAKING_DURATION
            recognizer.phrase_threshold = 0.2
            mic = sr.Microphone(device_index=device_idx)
            with mic as source:
                try:
                    recognizer.adjust_for_ambient_noise(source, duration=0.25)
                except Exception:
                    pass
                # Ambient calibration often sets the gate too high and drops words.
                recognizer.energy_threshold = min(float(recognizer.energy_threshold), 300)
                self._emit_status(f"Microphone: listening ({self._device_name})")
                while not self._stop.is_set():
                    if self._paused.is_set():
                        time.sleep(0.05)
                        continue
                    try:
                        audio = recognizer.listen(
                            source,
                            timeout=1.0,
                            phrase_time_limit=PHRASE_TIME_LIMIT,
                        )
                    except sr.WaitTimeoutError:
                        continue
                    except Exception as exc:  # noqa: BLE001
                        if self._stop.is_set() or self._paused.is_set():
                            continue
                        self._emit_error(f"Could not read microphone: {exc}")
                        return
                    if self._stop.is_set() or self._paused.is_set():
                        continue
                    # Transcribe in the background so listening does not stop
                    # while Google is processing the last sentence.
                    threading.Thread(
                        target=self._transcribe,
                        args=(audio,),
                        daemon=True,
                        name="awr-voice-stt",
                    ).start()
        except Exception as exc:  # noqa: BLE001
            self._emit_error(
                f"Could not open microphone: {exc}. "
                "Allow mic access and check the default input device."
            )
        finally:
            try:
                pa.terminate()
            except Exception:
                pass

    def _transcribe(self, audio) -> None:
        if self._stop.is_set():
            return
        try:
            import speech_recognition as sr
        except ImportError:
            return

        def recognize(chunk) -> str:
            try:
                rec = sr.Recognizer()
                return rec.recognize_google(chunk, language="en-US")
            except sr.UnknownValueError:
                return ""
            except sr.RequestError as exc:
                self._emit_error(
                    "Google speech recognition needs an internet connection. "
                    f"({exc})"
                )
                raise
            except Exception:
                return ""

        pieces: list[str] = []
        try:
            for chunk in split_audio_for_google(audio):
                if self._stop.is_set():
                    return
                text = recognize(chunk)
                cleaned = _normalize(text)
                if cleaned:
                    pieces.append(cleaned)
                    # Show each piece live as the long sentence is decoded.
                    if self._on_partial:
                        self._on_partial(join_chunk_transcripts(pieces))
        except sr.RequestError:
            return
        if self._stop.is_set():
            return
        full = join_chunk_transcripts(pieces)
        if full and self._on_final:
            self._on_final(full)


class VoiceListener:
    """Start / pause / resume / stop around a streaming engine + transcript."""

    def __init__(
        self,
        engine: StreamingEngine | None = None,
        on_live: LiveCallback | None = None,
        on_committed: TextCallback | None = None,
        on_error: TextCallback | None = None,
        on_status: TextCallback | None = None,
    ) -> None:
        self.engine = engine
        self.on_live = on_live
        self.on_committed = on_committed
        self.on_error = on_error
        self.on_status = on_status
        self.transcript = VoiceTranscript()
        self.state = "stopped"

    def start(self) -> None:
        if self.state == "listening":
            return
        if self.state == "paused":
            self.transcript.resume()
            self.state = "listening"
            if self.engine is not None:
                self.engine.resume()
            self._emit_live()
            if self.on_status:
                self.on_status("Microphone: listening…")
            return
        if self.engine is None:
            self.engine = SpeechMicEngine(on_status=self.on_status)
        self.state = "listening"
        if self.on_status:
            self.on_status("Microphone: listening…")
        self.engine.start(self._on_partial, self._on_final, self._on_engine_error)
        self._emit_live()

    def pause(self) -> None:
        if self.state != "listening":
            return
        if self.engine is not None:
            self.engine.pause()
        self.transcript.pause()
        self.state = "paused"
        self._emit_live()
        if self.on_status:
            self.on_status("Microphone: paused — click Start to continue")

    def stop(self) -> None:
        if self.state == "stopped":
            return
        if self.engine is not None:
            self.engine.stop()
        self.transcript.pause()
        remaining = self.transcript.take_finals()
        self.transcript.clear()
        self.state = "stopped"
        if remaining and self.on_committed:
            self.on_committed(remaining)
        self._emit_live()

    def _on_partial(self, text: str) -> None:
        self.transcript.apply_partial(text)
        self._emit_live()

    def _on_final(self, text: str) -> None:
        self.transcript.apply_final(text)
        self._emit_live()

    def _on_engine_error(self, message: str) -> None:
        self.transcript.pause()
        remaining = self.transcript.take_finals()
        self.transcript.clear()
        self.state = "stopped"
        if remaining and self.on_committed:
            self.on_committed(remaining)
        if self.engine is not None:
            try:
                self.engine.stop()
            except Exception:
                pass
        if self.on_error:
            self.on_error(message)
        self._emit_live()

    def _emit_live(self) -> None:
        if self.on_live:
            self.on_live(self.transcript.display(), self.state)
