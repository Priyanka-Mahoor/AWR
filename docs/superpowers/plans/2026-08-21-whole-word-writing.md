# Whole-word air writing — Implementation Plan

Date: 2026-08-21

Spec: `docs/superpowers/specs/2026-08-21-whole-word-writing-design.md`

## Files

| File | Change |
| --- | --- |
| `src/awr/ink.py` | Rasterize; segment letter glyphs by connected components + x-gap merge; `to_emnist` from crop |
| `src/awr/session.py` | Drop per-char settle; `commit_recognized_word` |
| `src/awr/main.py` | 3s pen-up → segment → OCR → commit; status countdown |
| `README.md` | Update Text mode instructions |

## Tasks

1. Ink segmentation API  
2. Session word commit API  
3. Wire main loop  
4. README tip about letter gaps  
