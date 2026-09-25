"""Generate recording scripts with SYNTHETIC values (never real numbers or IDs).

For each speaker: 10 mobile, 10 national ID, 10 amount prompts, including
self-corrections (a false start followed by the full correct value).

Outputs (in --out_dir):
  manifest.csv          one row per utterance with the gold normalized value
  speaker_XX.md         a readable Arabic sheet to send to each speaker

Speakers are split by speaker into dev/test (tune on dev, report on test).

Usage:
  python data/make_speaker_scripts.py --n_speakers 10 --n_dev 3
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from samaa.validate import luhn_check_digit, validate_mobile, validate_national_id  # noqa: E402

MOBILE_LEADS = ["رقم جوالي", "رقمي", "الرقم هو", "سجل عندك الرقم", "جوالي"]
ID_LEADS = ["رقم هويتي", "الهوية", "رقم الهوية", "هويتي رقمها"]  # citizen IDs start with 1
IQAMA_LEADS = ["رقم إقامتي", "الإقامة", "رقم الإقامة"]  # iqama numbers start with 2
AMOUNT_LEADS = ["المبلغ", "انخصم مني", "حولت", "المبلغ اللي انسحب", "دفعت"]
CORRECTIONS = ["لا لحظة،", "لا لا، أقصد", "عفوًا،", "لا غلطت،"]

# Amounts chosen to exercise Arabic number grammar (11-19, hundreds, thousands, compounds)
AMOUNT_POOL = [
    11, 12, 15, 19, 250, 300, 450, 800, 1000, 1350, 2000, 2500, 3200,
    4750, 5000, 8600, 11000, 12500, 15250, 20000, 36400, 100000,
]


def fake_mobile(rng: random.Random) -> str:
    v = "05" + "".join(rng.choices("0123456789", k=8))
    assert validate_mobile(v).valid
    return v


def fake_id(rng: random.Random) -> str:
    first9 = rng.choice("12") + "".join(rng.choices("0123456789", k=8))
    v = first9 + luhn_check_digit(first9)
    assert validate_national_id(v).valid
    return v


def group_mobile(v: str) -> str:
    return f"{v[:3]} {v[3:6]} {v[6:]}"


def group_id(v: str) -> str:
    return f"{v[:1]} {v[1:4]} {v[4:7]} {v[7:]}"


def fmt_amount(v: int) -> str:
    return f"{v:,}"


def make_prompts(rng: random.Random) -> list[dict]:
    rows = []

    # Mobiles: 6 plain, 4 self-corrections
    for i in range(10):
        gold = fake_mobile(rng)
        lead = rng.choice(MOBILE_LEADS)
        if i < 6:
            text, style = f"{lead} {group_mobile(gold)}", "plain"
        else:
            wrong_start = gold[:2] + rng.choice([d for d in "0123456789" if d != gold[2]])
            text = f"{lead} {wrong_start}... {rng.choice(CORRECTIONS)} {group_mobile(gold)}"
            style = "self_correction"
        rows.append({"entity_type": "mobile", "style": style, "prompt": text, "gold": gold})

    # National IDs: 7 plain, 3 self-corrections
    for i in range(10):
        gold = fake_id(rng)
        lead = rng.choice(ID_LEADS if gold[0] == "1" else IQAMA_LEADS)
        if i < 7:
            text, style = f"{lead} {group_id(gold)}", "plain"
        else:
            wrong_start = gold[:3] + rng.choice([d for d in "0123456789" if d != gold[3]])
            text = f"{lead} {wrong_start}... {rng.choice(CORRECTIONS)} {group_id(gold)}"
            style = "self_correction"
        rows.append({"entity_type": "national_id", "style": style, "prompt": text, "gold": gold})

    # Amounts: 7 plain, 3 self-corrections
    amounts = rng.sample(AMOUNT_POOL, 10)
    for i, gold in enumerate(amounts):
        lead = rng.choice(AMOUNT_LEADS)
        if i < 7:
            text, style = f"{lead} {fmt_amount(gold)} ريال", "plain"
        else:
            wrong = gold + rng.choice([-100, 100, 1000, -50, 50])
            wrong = max(wrong, 1)
            text = f"{lead} {fmt_amount(wrong)}... {rng.choice(CORRECTIONS)} {fmt_amount(gold)} ريال"
            style = "self_correction"
        rows.append({"entity_type": "amount", "style": style, "prompt": text, "gold": str(gold)})

    rng.shuffle(rows)  # avoid reading all of one type in a row
    return rows


SHEET_HEADER = """# نص التسجيل — المتحدث {sid}

شكرًا لمساعدتك! كل الأرقام هنا **وهمية**، لا تستخدم أرقامك الحقيقية أبدًا.

- اقرأ كل جملة **بطريقتك الطبيعية** كأنك تكلم خدمة عملاء على الجوال.
- قل الأرقام زي ما تقولها عادةً (مثلًا "صفر خمسة خمسة" أو "خمسة وخمسين" — اللي يطلع طبيعي معك).
- في الجمل اللي فيها تصحيح (…لا لحظة)، مثّل إنك غلطت وصححت.
- سجّل كل جملة في ملف مستقل، واحفظه باسم الملف المكتوب جنبها.

| # | اسم الملف | الجملة |
|---|---|---|
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_speakers", type=int, default=10)
    ap.add_argument("--n_dev", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="data/speaker_scripts")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest = []
    for s in range(1, args.n_speakers + 1):
        sid = f"S{s:02d}"
        split = "dev" if s <= args.n_dev else "test"
        rows = make_prompts(rng)
        lines = [SHEET_HEADER.format(sid=sid)]
        for i, r in enumerate(rows, 1):
            utt_id = f"{sid}_{i:02d}"
            lines.append(f"| {i} | `{utt_id}` | {r['prompt']} |\n")
            manifest.append({"utt_id": utt_id, "speaker": sid, "split": split, **r})
        (out / f"speaker_{sid}.md").write_text("".join(lines), encoding="utf-8")

    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["utt_id", "speaker", "split", "entity_type", "style", "prompt", "gold"])
        w.writeheader()
        w.writerows(manifest)

    print(f"Wrote {len(manifest)} prompts for {args.n_speakers} speakers to {out}/")
    print(f"Dev speakers: S01..S{args.n_dev:02d}  |  Test speakers: the rest")


if __name__ == "__main__":
    main()
