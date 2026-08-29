# Whole-word air writing — Design

Date: 2026-08-21

Extends: `2026-08-21-air-writing-recognition-design.md`

## Goal

In Text mode, the user writes an **entire word** in the air (letters separated by visible space). Recognition runs **only after** a ~3s pen-up pause: ink is split left→right into letter blobs, each is classified with the existing EMNIST CNN, then the word is committed to the transcript.

## Non-goals

- Cursive / connected script without gaps
- True whole-word neural models (TrOCR, etc.)
- Changing Drawing mode
- Retraining the character CNN

## User flow

1. Point and write letter 1; optionally lift between strokes of the same letter.
2. Leave a horizontal **gap**, write letter 2, and so on until the word is done.
3. Stay **pen-up** for **`WORD_IDLE_SEC` (3.0s)** → word finished.
4. App segments ink into letter regions, runs CNN per region, appends the word to the transcript, clears text ink.
5. Status shows writing / countdown / result (e.g. `HI`).

Short pen-ups mid-word **do not** trigger recognition. There is no live “current word” character buffer during writing; the word appears only after the 3s commit.

## Segmentation

On commit:

1. Rasterize all text-mode strokes (same style as today’s glyph prep: thick white-on-black).
2. Find connected components (or column-projection gaps) and merge components that overlap in **x** enough to belong to one letter (e.g. the crossbar and uprights of **H**).
3. Sort letter groups by leftmost x.
4. Reject tiny noise blobs.
5. For each group: crop → square pad → 28×28 → `CharRecognizer.predict` (reuse H/4 geometry hints when stroke subsets are available).
6. Concatenate characters with confidence ≥ threshold; skip or mark unclear glyphs without aborting the whole word if others succeed.

Preferred split signal: **horizontal gaps between letter blobs** (matches “spaces between letters”). Pen-up alone does not define letter boundaries.

## Timing

| Event | Behavior |
| --- | --- |
| `draw` | Add ink; cancel any commit countdown |
| Short `penUp` with ink | Keep ink; start / refresh 3s commit timer |
| `penUp` held ≥ 3s with ink | Segment → OCR → commit word → clear ink |
| No ink | No commit |

Remove the old **per-letter 2s settle → single-char OCR** path in Text mode.

## Session / UI

- `current_word` during writing may stay empty or show `(writing…)` until commit.
- After commit: update transcript; status `Committed: WORD`.
- Undo: last stroke only (no per-char undo from OCR until after commit).
- Clear: wipe pending word ink.
- Drawing mode unchanged.

## Errors

- Empty / one tiny blob → status unclear; clear or keep ink per existing Clear UX (prefer clear after failed commit attempt only if nothing usable).
- All letters below confidence → do not append word; keep ink so user can Undo/Clear/retry.
- Partial success: append only confident chars **or** reject whole word — choose **append confident chars** and status note if any skipped.

## Success criteria

Write `HI` or `CAT` with clear gaps, hold pen-up 3s, see the word in the transcript without recognizing letter-by-letter mid-word.
