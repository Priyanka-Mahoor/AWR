# Air Writing Recognition

Real-time air writing and drawing with a webcam (OpenCV + MediaPipe + TensorFlow CNN + Tkinter).

## Setup

On macOS with Homebrew Python, install Tk support first (needed for the UI):

```bash
brew install python-tk@3.12
```

```bash
cd open-cv
/opt/homebrew/bin/python3.12 -m venv .venv312
source .venv312/bin/activate
pip install -r requirements.txt
```

Place `models/hand_landmarker.task` is optional (Hands solutions API is used). Train the character CNN (needs TensorFlow in a separate install or temporary install):

```bash
source .venv312/bin/activate
pip install tensorflow tensorflow-datasets importlib_resources
python scripts/train_emnist.py --epochs 3
python scripts/export_weights.py
# runtime does not need TensorFlow — uses models/char_cnn_weights.npz
```

`models/char_cnn_weights.npz` is already included if training was run once.

## Run

```bash
source .venv312/bin/activate
PYTHONPATH=src python -m awr.main
```

## How to use

1. Allow camera access.
2. **Text mode:** point with index finger (other fingers curled) to write a **whole word**, leaving a clear **gap between letters**. Stay pen-up ~**3s** to recognize letter-by-letter and commit the word.
3. **Draw mode:** same pose draws in the selected color.
4. Buttons: Text / Draw, colors, Undo, Clear, Clear transcript.

Alphabet: `0–9` and `A–Z`.
