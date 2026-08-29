# Air Writing Recognition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a modular Python desktop app (OpenCV + MediaPipe + Tkinter + EMNIST CNN) for air writing words (A–Z, 0–9) and freehand drawing per `docs/superpowers/specs/2026-08-21-air-writing-recognition-design.md`.

**Architecture:** Camera thread runs MediaPipe tracking and ink updates; Tkinter main thread shows preview (`PhotoImage`), controls, transcript, and drawing panel. Text Mode recognizes one character per pen-up via Keras CNN; ~1s idle commits the character buffer as a word.

**Tech Stack:** Python 3.10–3.12, OpenCV, MediaPipe, NumPy, TensorFlow/Keras, Tkinter

## Global Constraints

- No automated tests (user decision)
- Alphabet: 36 classes `0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ`
- Confidence threshold: 0.55
- Word commit idle: 1.0 second
- EMNIST stroke convention: white strokes on black, 28×28, values in [0, 1]
- Do not commit unless the user asks
- Prefer Python 3.12 if 3.14 lacks TensorFlow/MediaPipe wheels

---

## File structure

```
open-cv/
  requirements.txt
  README.md
  .gitignore
  models/.gitkeep
  src/awr/
    __init__.py
    gestures.py      # finger extended/curled + pointing
    tracker.py
    ink.py
    recognizer.py
    session.py
    ui.py
    main.py
  scripts/train_emnist.py
```

---

### Task 1: Project scaffold

**Files:**
- Create: `requirements.txt`, `.gitignore`, `README.md`, `models/.gitkeep`, `src/awr/__init__.py`

- [ ] **Step 1: Write requirements and ignore rules**

`requirements.txt`:
```
opencv-python>=4.8
mediapipe>=0.10.9
numpy>=1.24
tensorflow>=2.15
Pillow>=10.0
```

`.gitignore`:
```
.venv/
__pycache__/
*.pyc
.DS_Store
models/*.keras
models/*.h5
```

- [ ] **Step 2: Create venv with Python 3.12 if available and install deps**

```bash
cd /Users/kaskichandrakant/pet-projects/pri/open-cv
python3.12 -m venv .venv || python3 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -r requirements.txt
```

---

### Task 2: Gestures + tracker

**Files:**
- Create: `src/awr/gestures.py`, `src/awr/tracker.py`

**Produces:**
- `is_pointing(landmarks) -> bool` with hysteresis state object `PointingGate`
- `HandTracker` with `read() -> tuple[frame_bgr, event]` where event is `{"kind":"draw","x":int,"y":int}` | `{"kind":"penUp"}` | `{"kind":"noHand"}`

- [ ] **Step 1: Implement `gestures.py`**

Finger indices (MediaPipe): wrist=0, index tip=8 pip=6, middle tip=12 pip=10, ring tip=16 pip=14, pinky tip=20 pip=18.

Extended if `dist(tip,wrist)/dist(pip,wrist) > 1.15`; curled if `< 1.08`. Pointing = index extended and middle/ring/pinky curled. `PointingGate` keeps last boolean when ratios are between thresholds.

- [ ] **Step 2: Implement `tracker.py`**

OpenCV `VideoCapture(0)`, MediaPipe `Hands(max_num_hands=1, ...)`. Mirror X for display coords: `x = int((1 - lm.x) * w)`. Map index tip to pixel coords. Return annotated frame (optional landmarks) and event.

---

### Task 3: Ink canvas

**Files:**
- Create: `src/awr/ink.py`

**Produces:**
- `InkCanvas` with `on_event(event)`, `set_color(bgr)`, `undo()`, `clear()`, `has_ink()`, `to_image() -> np.ndarray` (uint8 HxW, white bg dark strokes for display crop; also `to_emnist() -> (28,28) float32` white-on-black matching training)

Smoothing alpha 0.4; ignore moves < 3px. Separate stroke lists. `to_emnist` crops with padding, square pad, resize 28×28, invert to white stroke on black, `/255.0`.

---

### Task 4: Session (word commit)

**Files:**
- Create: `src/awr/session.py`

**Produces:**
- `Session` dataclass-like: `mode` (`"text"|"draw"`), `current_word`, `transcript` (list[str]), `status`, `pending_recognize` flag
- Methods: `set_mode`, `on_pen_up_text(char|None)`, `tick(now) -> committed_word|None`, `undo_char`, `clear_word`, `clear_transcript`, `append_char`

Word commit: after `on_pen_up_text`, record `last_activity = now`. On `tick`, if `now - last_activity >= 1.0` and `current_word`, push to transcript and clear word; return the committed word.

---

### Task 5: Recognizer + EMNIST train script

**Files:**
- Create: `src/awr/recognizer.py`, `scripts/train_emnist.py`

**Produces:**
- `CharRecognizer(model_path).predict(emnist_28x28) -> tuple[str, float] | None`
- Training script downloads EMNIST byclass via `tensorflow_datasets` or torchvision fallback, filters labels to 0–35 (digits + uppercase), trains small CNN, saves `models/char_cnn.keras`

CNN sketch:
```python
Sequential([
  Input((28,28,1)),
  Conv2D(32,3,activation="relu"), MaxPool2D(),
  Conv2D(64,3,activation="relu"), MaxPool2D(),
  Flatten(), Dense(128,activation="relu"), Dropout(0.3),
  Dense(36,activation="softmax"),
])
```

LABELS = `"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"`

If weights missing, `CharRecognizer.available` is False.

- [ ] **Step 1: Implement train script and run a short training** (enough epochs for a usable demo, e.g. 3–5 on a subset if full train is too slow; prefer full EMNIST with 2–3 epochs)

- [ ] **Step 2: Implement recognizer loading the saved model**

---

### Task 6: Tkinter UI + main wiring

**Files:**
- Create: `src/awr/ui.py`, `src/awr/main.py`

**Behavior:**
- Left: live preview
- Right: current word, transcript, drawing canvas (PIL/Tk), status
- Buttons: Text / Draw, colors (red/green/blue/yellow/white/black), Undo, Clear, Clear transcript
- Camera loop thread: read tracker → ink → on penUp in text mode call recognizer → session; push UI state to queue
- `root.after(16, pump)` drains queue and updates labels/images

Entry: `python -m awr.main` with `PYTHONPATH=src` or package install.

- [ ] **Step 1: Implement UI**
- [ ] **Step 2: Implement main**
- [ ] **Step 3: Write README with run + train instructions**
- [ ] **Step 4: Manual smoke — launch app, confirm window and camera (or clear error)**

---

## Spec coverage check

| Spec item | Task |
| --- | --- |
| Tracker / gestures | 2 |
| Ink / colors / undo / clear | 3 |
| Word commit session | 4 |
| EMNIST CNN | 5 |
| Tkinter hybrid UI | 6 |
| Drawing mode | 3+6 |
| Error handling (camera/model) | 6 |
| No tests | all |
