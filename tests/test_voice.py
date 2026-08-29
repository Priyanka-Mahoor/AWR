"""Live speech-to-text buffer and listener state (no microphone required)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from awr.session import Session  # noqa: E402
from awr.voice import VoiceListener, VoiceTranscript  # noqa: E402


class VoiceTranscriptTests(unittest.TestCase):
    def test_partial_words_appear_live_with_spaces(self) -> None:
        t = VoiceTranscript()
        t.apply_partial("hello")
        self.assertEqual(t.display(), "hello")
        t.apply_partial("hello world")
        self.assertEqual(t.display(), "hello world")

    def test_final_keeps_text_and_clears_only_the_partial(self) -> None:
        t = VoiceTranscript()
        t.apply_partial("hello world")
        t.apply_final("hello world")
        t.apply_partial("how are you")
        self.assertEqual(t.display(), "hello world how are you")

    def test_long_sentence_grows_across_pauses_in_speech(self) -> None:
        t = VoiceTranscript()
        t.apply_final("this is a long sentence")
        t.apply_final("that continues after a pause")
        t.apply_partial("and still listening")
        self.assertEqual(
            t.display(),
            "this is a long sentence that continues after a pause and still listening",
        )

    def test_pause_freezes_live_text_including_in_progress_words(self) -> None:
        t = VoiceTranscript()
        t.apply_partial("please hold")
        t.pause()
        t.apply_partial("this must be ignored")
        t.apply_final("ignored too")
        self.assertEqual(t.display(), "please hold")
        self.assertTrue(t.paused)

    def test_resume_appends_new_speech_with_a_space(self) -> None:
        t = VoiceTranscript()
        t.apply_final("hello")
        t.pause()
        t.resume()
        t.apply_partial("there")
        self.assertEqual(t.display(), "hello there")
        self.assertFalse(t.paused)

    def test_single_word_and_empty_partials(self) -> None:
        t = VoiceTranscript()
        t.apply_partial("")
        self.assertEqual(t.display(), "")
        t.apply_final("yes")
        t.apply_partial("  ")
        self.assertEqual(t.display(), "yes")

    def test_take_committed_leaves_in_progress_partial(self) -> None:
        t = VoiceTranscript()
        t.apply_final("one")
        t.apply_partial("two")
        taken = t.take_finals()
        self.assertEqual(taken, "one")
        self.assertEqual(t.display(), "two")

    def test_clear(self) -> None:
        t = VoiceTranscript()
        t.apply_final("keep")
        t.apply_partial("going")
        t.clear()
        self.assertEqual(t.display(), "")


class FakeEngine:
    def __init__(self) -> None:
        self.started = 0
        self.paused = 0
        self.resumed = 0
        self.stopped = 0
        self.waited_for_silence = False

    def start(self, on_partial, on_final, on_error) -> None:
        self.started += 1
        self.on_partial = on_partial
        self.on_final = on_final
        self.on_error = on_error
        # Start-while-speaking: emit immediately, no silence wait.
        on_partial("already talking")

    def pause(self) -> None:
        self.paused += 1

    def resume(self) -> None:
        self.resumed += 1
        self.on_partial("still talking")

    def stop(self) -> None:
        self.stopped += 1


class VoiceListenerTests(unittest.TestCase):
    def test_start_listens_immediately_while_already_speaking(self) -> None:
        events: list[tuple] = []
        engine = FakeEngine()
        listener = VoiceListener(
            engine=engine,
            on_live=lambda text, state: events.append(("live", text, state)),
            on_committed=lambda text: events.append(("commit", text)),
        )
        listener.start()
        self.assertEqual(listener.state, "listening")
        self.assertEqual(engine.started, 1)
        self.assertFalse(engine.waited_for_silence)
        self.assertEqual(listener.transcript.display(), "already talking")

    def test_pause_then_start_resumes_without_clearing_text(self) -> None:
        engine = FakeEngine()
        listener = VoiceListener(engine=engine)
        listener.start()
        listener.transcript.apply_final("hello")
        listener.pause()
        self.assertEqual(listener.state, "paused")
        listener.start()
        self.assertEqual(listener.state, "listening")
        self.assertEqual(engine.resumed, 1)
        self.assertIn("hello", listener.transcript.display())

    def test_stop_commits_remaining_text(self) -> None:
        committed: list[str] = []
        engine = FakeEngine()
        listener = VoiceListener(
            engine=engine,
            on_committed=committed.append,
        )
        listener.start()
        listener.transcript.apply_final("saved speech")
        listener.stop()
        self.assertEqual(listener.state, "stopped")
        self.assertEqual(committed, ["saved speech"])
        self.assertEqual(engine.stopped, 1)

    def test_stop_while_stopped_is_idle(self) -> None:
        engine = FakeEngine()
        listener = VoiceListener(engine=engine)
        listener.stop()
        self.assertEqual(engine.stopped, 0)
        self.assertEqual(listener.state, "stopped")

    def test_finals_stay_visible_while_still_listening(self) -> None:
        engine = FakeEngine()
        listener = VoiceListener(engine=engine)
        listener.start()
        listener._on_final("hello world")
        listener._on_partial("how are you")
        self.assertEqual(listener.transcript.display(), "hello world how are you")


class AudioPrepTests(unittest.TestCase):
    def test_listen_settings_keep_going_through_a_sentence(self) -> None:
        from awr.voice import PAUSE_THRESHOLD, PHRASE_TIME_LIMIT

        self.assertGreaterEqual(
            PAUSE_THRESHOLD,
            0.8,
            "short pause_threshold treats gaps between words as the end of speech",
        )
        self.assertGreaterEqual(
            PHRASE_TIME_LIMIT,
            8,
            "phrase_time_limit must keep enough of a long sentence in one listen",
        )
        self.assertLessEqual(
            PHRASE_TIME_LIMIT,
            12,
            "phrase_time_limit above ~12s delays live text and Google often drops the clip",
        )

    def test_long_audio_is_split_for_google(self) -> None:
        from awr.voice import audio_duration_sec, split_audio_for_google

        class FakeAudio:
            def __init__(self, seconds: float, rate: int = 16000, width: int = 2) -> None:
                self.sample_rate = rate
                self.sample_width = width
                self.frame_data = b"\x00" * int(seconds * rate * width)

            def get_segment(self, start_ms=None, end_ms=None):
                start = 0 if start_ms is None else int(start_ms)
                end = (
                    len(self.frame_data) * 1000 // (self.sample_rate * self.sample_width)
                    if end_ms is None
                    else int(end_ms)
                )
                s = int(start * self.sample_rate * self.sample_width / 1000)
                e = int(end * self.sample_rate * self.sample_width / 1000)
                part = FakeAudio(0)
                part.sample_rate = self.sample_rate
                part.sample_width = self.sample_width
                part.frame_data = self.frame_data[s:e]
                return part

        short = FakeAudio(5)
        self.assertEqual(len(split_audio_for_google(short)), 1)
        long = FakeAudio(25)
        chunks = split_audio_for_google(long)
        self.assertGreaterEqual(len(chunks), 3)
        self.assertLessEqual(max(audio_duration_sec(c) for c in chunks), 8.5)

    def test_join_chunk_transcripts_merges_overlap(self) -> None:
        from awr.voice import join_chunk_transcripts

        self.assertEqual(
            join_chunk_transcripts(
                ["this is a long sentence", "long sentence that continues"]
            ),
            "this is a long sentence that continues",
        )

    def test_choose_headset_mic_when_headphones_are_the_output(self) -> None:
        from awr.voice import choose_input_device

        devices = [
            {"name": "Microphone Array (AMD Audio Device)", "max_input_channels": 2, "hostapi_name": "MME"},
            {"name": "Headset (Boult Audio Airbass)", "max_input_channels": 1, "hostapi_name": "MME"},
            {"name": "Headphones (Boult Audio Airbass)", "max_input_channels": 0, "hostapi_name": "MME"},
            {"name": "Headset (Boult Audio Airbass)", "max_input_channels": 1, "hostapi_name": "Windows WASAPI"},
        ]
        chosen = choose_input_device(devices, default_in=0, default_out=2)
        self.assertEqual(devices[chosen]["name"], "Headset (Boult Audio Airbass)")
        self.assertEqual(devices[chosen]["hostapi_name"], "Windows WASAPI")


class SessionVoiceTests(unittest.TestCase):
    def test_commit_voice_text_stores_a_history_item(self) -> None:
        session = Session()
        session.commit_recognized_word([("H", 0.9), ("I", 0.9)], 1.0)
        got = session.commit_voice_text("hello there", 2.0)
        self.assertEqual(got, "hello there")
        self.assertEqual([item.kind for item in session.history], ["word", "voice"])
        self.assertEqual(session.history[-1].text, "hello there")
        self.assertIn("hello there", session.history_display)

    def test_commit_voice_text_skips_blank(self) -> None:
        session = Session()
        self.assertIsNone(session.commit_voice_text("   ", 1.0))
        self.assertEqual(session.history, [])


if __name__ == "__main__":
    unittest.main()
