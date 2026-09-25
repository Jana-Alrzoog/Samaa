"""Arabic text normalization applied identically to references and hypotheses
before WER/CER. Every model is scored with the same function.

Standard steps: remove diacritics and tatweel, unify alef forms,
ta marbuta -> ha, alef maqsura -> ya, Arabic-Indic digits -> Western digits,
strip punctuation, collapse whitespace.

Numbers are NOT converted between words and digits here. That is done by
samaa/numbers.py (Day 2), and WER is reported both with and without it.
"""
from __future__ import annotations

import re
import unicodedata

_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
_TATWEEL = "\u0640"
_ALEF = re.compile(r"[\u0622\u0623\u0625\u0671]")  # آ أ إ ٱ -> ا
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)
_SPACES = re.compile(r"\s+")


def normalize_arabic(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _DIACRITICS.sub("", text)
    text = text.replace(_TATWEEL, "")
    text = _ALEF.sub("\u0627", text)
    text = text.replace("\u0629", "\u0647")  # ة -> ه
    text = text.replace("\u0649", "\u064A")  # ى -> ي
    text = text.translate(_DIGITS)
    text = _PUNCT.sub(" ", text)
    text = text.replace("_", " ")
    return _SPACES.sub(" ", text).strip().lower()
