"""Telephone-channel simulation for Whisper training and evaluation.

Chain: band-limit 300-3400 Hz -> 8 kHz -> narrowband codec -> decode ->
additive noise at a random SNR -> random gain -> back to 16 kHz (Whisper input).

Codecs are applied with the ffmpeg CLI because library codec wrappers change
between versions. Availability depends on your ffmpeg build, so check with:
  python data/telephony.py --list-codecs

Hold one codec out of training (e.g. AMR-NB) and use it only at test time,
to separate real robustness from memorizing the augmentation.

Usage:
  python data/telephony.py --in clip.wav --out clip_tel.wav --codec mulaw --snr 15
"""
from __future__ import annotations

import argparse
import random
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf

TARGET_SR = 16_000
BANDPASS = "highpass=f=300,lowpass=f=3400"

# name -> (ffmpeg encoder, container extension, extra encoder args)
CODECS = {
    "mulaw": ("pcm_mulaw", "wav", []),
    "gsm": ("libgsm", "gsm", []),
    "amr": ("libopencore_amrnb", "amr", ["-b:a", "12.2k"]),
    "opus": ("libopus", "ogg", ["-b:a", "8k", "-application", "voip"]),
}


@lru_cache(maxsize=1)
def available_codecs() -> list[str]:
    out = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True, check=True
    ).stdout
    return [name for name, (enc, _, _) in CODECS.items() if f" {enc} " in out]


def _run(cmd: list[str]) -> None:
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {' '.join(cmd)}\n{res.stderr[-500:]}")


def apply_codec(in_path: str, out_path: str, codec: str) -> None:
    """Band-limit, resample to 8 kHz, encode with `codec`, decode back to 16 kHz WAV."""
    if codec not in available_codecs():
        raise ValueError(f"Codec '{codec}' not in this ffmpeg build. Available: {available_codecs()}")
    enc, ext, extra = CODECS[codec]
    with tempfile.TemporaryDirectory() as tmp:
        encoded = str(Path(tmp) / f"enc.{ext}")
        _run(["ffmpeg", "-y", "-loglevel", "error", "-i", in_path, "-af", BANDPASS,
              "-ac", "1", "-ar", "8000", "-c:a", enc, *extra, encoded])
        _run(["ffmpeg", "-y", "-loglevel", "error", "-i", encoded,
              "-ac", "1", "-ar", str(TARGET_SR), "-c:a", "pcm_s16le", out_path])


def add_noise(audio: np.ndarray, snr_db: float, rng: np.random.Generator,
              noise: np.ndarray | None = None) -> np.ndarray:
    """Mix noise at the given SNR. Uses a real noise clip if given, else white noise."""
    if noise is None:
        noise = rng.standard_normal(len(audio))
    else:
        if len(noise) < len(audio):
            noise = np.tile(noise, int(np.ceil(len(audio) / len(noise))))
        start = rng.integers(0, len(noise) - len(audio) + 1)
        noise = noise[start:start + len(audio)]
    sig_pow = np.mean(audio ** 2) + 1e-10
    noise_pow = np.mean(noise ** 2) + 1e-10
    scale = np.sqrt(sig_pow / (noise_pow * 10 ** (snr_db / 10)))
    return audio + scale * noise


def degrade(in_path: str, out_path: str, codec: str, snr_db: float | None,
            gain_db: float = 0.0, seed: int = 0, noise_path: str | None = None) -> dict:
    """Full telephone chain. Returns the parameters used, for logging."""
    rng = np.random.default_rng(seed)
    with tempfile.TemporaryDirectory() as tmp:
        coded = str(Path(tmp) / "coded.wav")
        apply_codec(in_path, coded, codec)
        audio, sr = sf.read(coded, dtype="float32")
    assert sr == TARGET_SR
    if snr_db is not None:
        noise = sf.read(noise_path, dtype="float32")[0] if noise_path else None
        if noise is not None and noise.ndim > 1:
            noise = noise.mean(axis=1)
        audio = add_noise(audio, snr_db, rng, noise)
    audio = audio * (10 ** (gain_db / 20))
    peak = np.max(np.abs(audio)) + 1e-10
    if peak > 1.0:  # avoid clipping after gain
        audio = audio / peak * 0.99
    sf.write(out_path, audio, TARGET_SR, subtype="PCM_16")
    return {"codec": codec, "snr_db": snr_db, "gain_db": gain_db, "seed": seed, "noise": noise_path}


def random_degrade(in_path: str, out_path: str, seed: int, train_codecs: list[str],
                   noise_files: list[str] | None = None) -> dict:
    """Random telephone condition for training (deterministic given the seed)."""
    r = random.Random(seed)
    return degrade(
        in_path, out_path,
        codec=r.choice(train_codecs),
        snr_db=r.uniform(5, 20),
        gain_db=r.uniform(-6, 6),
        seed=seed,
        noise_path=r.choice(noise_files) if noise_files else None,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-codecs", action="store_true")
    ap.add_argument("--in", dest="inp")
    ap.add_argument("--out")
    ap.add_argument("--codec", default="mulaw")
    ap.add_argument("--snr", type=float, default=15.0)
    ap.add_argument("--gain", type=float, default=0.0)
    ap.add_argument("--noise")
    args = ap.parse_args()

    if args.list_codecs:
        print("Available telephone codecs:", available_codecs())
    else:
        print(degrade(args.inp, args.out, args.codec, args.snr, args.gain, noise_path=args.noise))
