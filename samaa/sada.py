"""Loading SADA from the Hugging Face mirror, with column auto-detection.

Column names differ between SADA copies (the Kaggle release uses names like
ProcessedText / GroundTruthText / SpeakerDialect). Run data/inspect_sada.py
first and pass explicit --text_col / --dialect_col if detection picks wrong.
"""
from __future__ import annotations

from datasets import Audio, load_dataset

DATASET_ID = "MohamedRashad/SADA22"

TEXT_CANDIDATES = ["ProcessedText", "processed_text", "text", "transcript",
                   "sentence", "GroundTruthText", "ground_truth_text"]
DIALECT_CANDIDATES = ["SpeakerDialect", "speaker_dialect", "dialect", "Dialect"]


def load_sada(split: str, streaming: bool = True):
    ds = load_dataset(DATASET_ID, split=split, streaming=streaming)
    audio_col = find_audio_col(ds)
    if audio_col:
        ds = ds.cast_column(audio_col, Audio(decode=False))
    return ds


def column_names(ds) -> list[str]:
    feats = getattr(ds, "features", None)
    return list(feats.keys()) if feats else []


def find_audio_col(ds) -> str | None:
    feats = getattr(ds, "features", None) or {}
    for name, feat in feats.items():
        if isinstance(feat, Audio):
            return name
    return "audio" if "audio" in feats else None


def pick_col(ds, candidates: list[str], override: str | None = None) -> str | None:
    if override:
        return override
    cols = column_names(ds)
    for c in candidates:
        if c in cols:
            return c
    return None
