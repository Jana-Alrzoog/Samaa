"""Audio loading helpers.

Datasets are loaded with Audio(decode=False) and decoded here, which avoids
differences in how `datasets` versions decode audio.
"""
from __future__ import annotations

import io

import numpy as np

TARGET_SR = 16_000


def load_audio_bytes(data: bytes, target_sr: int = TARGET_SR) -> np.ndarray:
    """Decode raw audio bytes to mono float32 at target_sr."""
    try:
        import soundfile as sf
        audio, sr = sf.read(io.BytesIO(data), dtype="float32")
    except Exception:
        import librosa  # fallback for formats libsndfile can't read
        audio, sr = librosa.load(io.BytesIO(data), sr=None, mono=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1) if audio.shape[-1] <= 8 else audio.mean(axis=0)
    if sr != target_sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio.astype(np.float32)


def load_example_audio(audio_field: dict, target_sr: int = TARGET_SR) -> np.ndarray:
    """Handle a non-decoded datasets Audio field: {"bytes": ..., "path": ...}."""
    if audio_field.get("bytes"):
        return load_audio_bytes(audio_field["bytes"], target_sr)
    import librosa
    audio, _ = librosa.load(audio_field["path"], sr=target_sr, mono=True)
    return audio.astype(np.float32)
