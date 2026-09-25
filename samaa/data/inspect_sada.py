"""Day 1: inspect SADA before building anything on top of it.

Answers three questions:
  1. What are the columns? (text, audio, dialect names)
  2. What dialect labels exist and how common are they?
  3. Are numbers written as digits or as words? (decides how numbers.py works)

Usage:
  python data/inspect_sada.py --split test --n 3000
Writes results/sada_inspection.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from samaa.sada import (DIALECT_CANDIDATES, TEXT_CANDIDATES, column_names,  # noqa: E402
                        find_audio_col, load_sada, pick_col)

DIGITS = re.compile(r"[0-9\u0660-\u0669]")
NUMBER_WORDS = re.compile(
    r"\b(صفر|واحد|وحده|اثنين|ثنين|ثلاث|ثلاثه|ثلاثة|اربع|أربع|اربعه|خمس|خمسه|خمسة|ست|سته|ستة|"
    r"سبع|سبعه|ثمان|ثمانيه|تسع|تسعه|عشر|عشره|عشرة|احدعش|اطعش|ثنعش|عشرين|ثلاثين|اربعين|خمسين|"
    r"ستين|سبعين|ثمانين|تسعين|مية|ميه|مئة|مائة|ميتين|الف|ألف|الفين|ألفين|آلاف|الاف|مليون)\b"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--text_col")
    ap.add_argument("--dialect_col")
    ap.add_argument("--out", default="results/sada_inspection.json")
    args = ap.parse_args()

    ds = load_sada(args.split)
    cols = column_names(ds)
    text_col = pick_col(ds, TEXT_CANDIDATES, args.text_col)
    dialect_col = pick_col(ds, DIALECT_CANDIDATES, args.dialect_col)
    audio_col = find_audio_col(ds)
    print("Columns:", cols)
    print(f"Detected -> text: {text_col} | dialect: {dialect_col} | audio: {audio_col}")
    if not text_col:
        sys.exit("Could not detect the text column. Pass --text_col explicitly.")

    dialects = Counter()
    n = with_digits = with_words = 0
    samples_digits, samples_words, first_rows = [], [], []

    for i, ex in enumerate(ds):
        if i >= args.n:
            break
        n += 1
        text = ex.get(text_col) or ""
        if dialect_col:
            dialects[str(ex.get(dialect_col))] += 1
        if i < 3:
            first_rows.append({k: v for k, v in ex.items() if k != audio_col})
        if DIGITS.search(text):
            with_digits += 1
            if len(samples_digits) < 15:
                samples_digits.append(text)
        if NUMBER_WORDS.search(text):
            with_words += 1
            if len(samples_words) < 15:
                samples_words.append(text)

    report = {
        "split": args.split,
        "n_scanned": n,
        "columns": cols,
        "text_col": text_col,
        "dialect_col": dialect_col,
        "dialect_counts": dialects.most_common(),
        "pct_with_digits": round(100 * with_digits / max(n, 1), 2),
        "pct_with_number_words": round(100 * with_words / max(n, 1), 2),
        "samples_with_digits": samples_digits,
        "samples_with_number_words": samples_words,
        "first_rows": first_rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print(f"\nScanned {n} rows")
    print("Dialects:", dialects.most_common(10))
    print(f"Transcripts with digits: {report['pct_with_digits']}%")
    print(f"Transcripts with number words: {report['pct_with_number_words']}%")
    print("\nExamples with digits:", *samples_digits[:5], sep="\n  ")
    print("\nExamples with number words:", *samples_words[:5], sep="\n  ")
    print(f"\nFull report: {args.out}")


if __name__ == "__main__":
    main()
