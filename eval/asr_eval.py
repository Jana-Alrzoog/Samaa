"""Evaluate a Whisper model on SADA (clean or telephone-degraded) with WER/CER.

Same normalization and the same clips for every model, so results are comparable.
The clip selection is deterministic: the first N clips passing the filters.

Usage (Day 1 baselines):
  python eval/asr_eval.py --model openai/whisper-small --n 200
  python eval/asr_eval.py --model openai/whisper-small --n 200 --codec mulaw --snr 15
  python eval/asr_eval.py --model openai/whisper-large-v3-turbo --n 200
  python eval/asr_eval.py --model VohoAI/voho-saudi-stt-small --n 200

Writes results/asr/<model>__<condition>.json and a per-utterance CSV.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
import time
from pathlib import Path

import jiwer
import soundfile as sf
import torch
from transformers import pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.telephony import degrade  # noqa: E402
from samaa.audio import TARGET_SR, load_example_audio  # noqa: E402
from samaa.normalize import normalize_arabic  # noqa: E402
from samaa.sada import (DIALECT_CANDIDATES, TEXT_CANDIDATES, find_audio_col,  # noqa: E402
                        load_sada, pick_col)

MAX_SECONDS = 30.0


def iter_clips(args):
    ds = load_sada(args.split)
    text_col = pick_col(ds, TEXT_CANDIDATES, args.text_col)
    dialect_col = pick_col(ds, DIALECT_CANDIDATES, args.dialect_col)
    audio_col = find_audio_col(ds)
    keep = {d.strip().lower() for d in args.dialects.split(",")} if args.dialects else None
    taken = 0
    for i, ex in enumerate(ds):
        if taken >= args.n:
            break
        ref = (ex.get(text_col) or "").strip()
        if not ref:
            continue
        if keep and dialect_col and str(ex.get(dialect_col)).strip().lower() not in keep:
            continue
        audio = load_example_audio(ex[audio_col])
        if len(audio) / TARGET_SR > MAX_SECONDS or len(audio) < TARGET_SR * 0.5:
            continue
        taken += 1
        yield i, ref, audio


def maybe_degrade(audio, idx: int, args):
    if not args.codec:
        return audio
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = f"{tmp}/in.wav", f"{tmp}/out.wav"
        sf.write(src, audio, TARGET_SR, subtype="PCM_16")
        degrade(src, dst, codec=args.codec, snr_db=args.snr, seed=idx)  # fixed per clip
        out, _ = sf.read(dst, dtype="float32")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--dialects", help="comma-separated dialect labels to keep (see inspect_sada.py)")
    ap.add_argument("--text_col")
    ap.add_argument("--dialect_col")
    ap.add_argument("--codec", help="telephone codec (mulaw, gsm, amr, opus); omit for clean")
    ap.add_argument("--snr", type=float, default=15.0)
    ap.add_argument("--out_dir", default="results/asr")
    args = ap.parse_args()

    device = 0 if torch.cuda.is_available() else -1
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    try:  # newer transformers use `dtype`, older ones `torch_dtype`
        asr = pipeline("automatic-speech-recognition", model=args.model, device=device, dtype=dtype)
    except TypeError:
        asr = pipeline("automatic-speech-recognition", model=args.model, device=device, torch_dtype=dtype)
    gen = {"language": "arabic", "task": "transcribe"}

    condition = f"{args.codec}_snr{int(args.snr)}" if args.codec else "clean"
    refs, hyps, rows = [], [], []
    t0 = time.time()
    for idx, ref, audio in iter_clips(args):
        audio = maybe_degrade(audio, idx, args)
        hyp = asr({"raw": audio, "sampling_rate": TARGET_SR}, generate_kwargs=gen)["text"]
        r, h = normalize_arabic(ref), normalize_arabic(hyp)
        if not r:  # reference empty after normalization: nothing to score
            continue
        refs.append(r)
        hyps.append(h)
        rows.append({"idx": idx, "ref": ref, "hyp": hyp, "ref_norm": r, "hyp_norm": h})
        if len(rows) % 25 == 0:
            print(f"{len(rows)}/{args.n} clips, running WER {jiwer.wer(refs, hyps):.3f}")
    elapsed = time.time() - t0

    # Guard against empty normalized hypotheses breaking jiwer
    hyps_safe = [h if h else "<empty>" for h in hyps]
    result = {
        "model": args.model,
        "split": args.split,
        "condition": condition,
        "dialects": args.dialects,
        "n": len(rows),
        "wer": round(jiwer.wer(refs, hyps_safe), 4),
        "cer": round(jiwer.cer(refs, hyps_safe), 4),
        "seconds_total": round(elapsed, 1),
        "seconds_per_clip": round(elapsed / max(len(rows), 1), 3),
        "device": "cuda" if device == 0 else "cpu",
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.model.replace('/', '_')}__{args.split}__{condition}"
    (out / f"{stem}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(out / f"{stem}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
