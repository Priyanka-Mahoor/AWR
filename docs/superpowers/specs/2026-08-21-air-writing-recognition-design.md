# Air Writing Recognition — Design

Date: 2026-08-21

Source: project report `AWR reprot.pdf` (GEC Raichur, Air Writing Recognition).

## Goal

A desktop Python app where a user writes uppercase letters and digits in the air with an index finger. Each stroke is recognized as a character via a CNN; a short pause commits the buffered characters as a **word** into a live transcript. A second mode supports freehand air drawing with color selection. Interaction uses an OpenCV camera preview plus a Tkinter control/output panel.

Success: write words such as `HI`, `CAT`, and `A1` in the air and see them appear in the transcript without a keyboard (aside from granting camera access). Drawing mode produces colored strokes with undo/clear.

## Non-goals

- Automated unit/integration tests
- Lowercase or cursive handwriting
- Cloud OCR or any backend server
- Multi-hand tracking or mobile-first UI
- Wearable / IMU sensors
- Large-scale deployment

## Decisions locked with the user

| Topic | Choice |
| --- | --- |
| UI | Hybrid: OpenCV capture/processing + Tkinter app (embedded live preview + buttons + output) |
| Recognition | Character CNN; pause commits a word from the char buffer |
| Alphabet | `A–Z` + `0–9` (36 classes) |
| Model data | Train on EMNIST; ship a saved Keras model |
| Structure | Modular Python package (not a single script) |
| Testing | None required |

## Architecture

```
Webcam → OpenCV frames
           ↓
     Hand Tracker (MediaPipe)
           ↓  draw / penUp / noHand
     Ink Canvas (strokes + colors)
           ↓
     ┌─────┴──────┐
 Text Mode              Drawing Mode
 trajectory → image        colored strokes
      ↓                    on canvas / panel
 EMNIST CNN (A–Z, 0–9)
      ↓
 char buffer → pause → commit WORD
           ↓
 Live output panel (Tkinter)
 + Mode / Color / Undo / Clear
```

One process: OpenCV capture/processing off the Tk main thread; UI updates via a thread-safe queue and `after()`. Shared state lives in a small `session` object (mode, strokes, current word, transcript, drawing surface).

## Components

| Module | Responsibility |
| --- | --- |
| `tracker` | `getUserMedia`-equivalent via OpenCV; MediaPipe Hands; emits `draw(x,y)` / `penUp` / `noHand` |
| `ink` | Stroke list, smoothing, active color; `to_image()` for CNN; undo last stroke; clear |
| `recognizer` | Load `models/char_cnn.keras`; preprocess to 28×28 grayscale; return char + confidence |
| `session` | Mode, char buffer, word-commit timer, transcript, drawing buffer |
| `ui` | Single Tkinter window: live preview (`PhotoImage` from OpenCV frames), mode toggle, color swatches, Undo, Clear, Clear transcript; text + drawing output panels |
| `train_emnist` | Offline script: fetch EMNIST balanced/byclass letters+digits subset, train CNN, save weights |
| `main` | Wire modules, start camera thread and Tk mainloop |

### Writing pose

Index finger extended; middle, ring, and pinky curled → ink on (`draw`). Otherwise → `penUp`. Thumb ignored. Single hand only.

A finger is **extended** when `distance(tip, wrist) / distance(pip, wrist) > 1.15`, **curled** when that ratio is `< 1.08`, unchanged between (hysteresis).

## Data flow and timing

1. Each frame: `tracker` → event → `ink` updates strokes and cursor on a mirrored preview.
2. **Text Mode**, on pen-up with ink: build cropped white-background / dark-stroke image → `recognizer`. If confidence ≥ `0.55`, append character to **current word**; else status “unclear” and do not append.
3. If no new ink for **~1.0 s** and current word is non-empty → append word to transcript (space-separated), clear letter ink and char buffer.
4. **Drawing Mode:** no CNN; strokes render in the selected color on the drawing output. Same undo/clear semantics for strokes.
5. **Undo:** remove last stroke; if Text Mode and that stroke already produced a character, remove that character from the current word.
6. **Clear:** wipe active mode’s ink / pending char. **Clear transcript** clears committed words only.

## CNN (EMNIST)

- Framework: TensorFlow / Keras (as in the report).
- Input: 28×28 grayscale, normalized.
- Output: 36-way softmax (`0–9`, `A–Z`).
- Training: EMNIST (letters + digits mapped to the 36 labels); script writes `models/char_cnn.keras`.
- Runtime: load once at startup; inference on CPU is acceptable for one image per pen-up.

Trajectory preprocessing before inference: crop with padding, keep aspect ratio on a square canvas, resize to 28×28, invert if needed so strokes are bright-on-dark or match EMNIST convention used at train time (document the chosen convention in the train script and reuse identically in `recognizer`).

## Error handling

- Camera missing or permission denied → Tkinter message; app remains open for retry.
- MediaPipe assets fail to load → status error; do not spin a broken loop silently.
- CNN weights missing → Text Mode disabled with “train model first” guidance; Drawing Mode still available.
- Empty or tiny scribble → skip recognition.
- Low confidence → do not append character.

## Stack

- Python 3.10+
- OpenCV, MediaPipe, NumPy, TensorFlow/Keras
- Tkinter (stdlib) for controls and output panels
- Optional Matplotlib only inside training/debug if useful

## Project layout (planned)

```
open-cv/
  docs/superpowers/specs/2026-08-21-air-writing-recognition-design.md
  requirements.txt
  README.md
  models/                 # char_cnn.keras (generated)
  src/awr/
    __init__.py
    tracker.py
    ink.py
    recognizer.py
    session.py
    ui.py
    main.py
  scripts/train_emnist.py
```

## Alignment with the report

| Report requirement | Design coverage |
| --- | --- |
| Camera + MediaPipe/OpenCV tracking | `tracker` |
| Text Mode + CNN | `recognizer` + EMNIST train script |
| Drawing Mode + color | `ink` + Tkinter color swatches |
| Continuous writing | char buffer → word commit |
| Live output panel | Tkinter transcript + drawing panel |
| Undo, clear, mode selection | Tkinter controls |
| Python / OpenCV / MediaPipe / TensorFlow | stack above |

Word assembly via pause-after-characters is an explicit product choice on top of the report’s character CNN (user-selected option A).
