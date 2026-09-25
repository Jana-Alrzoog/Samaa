"""Deterministic validators for the three MVP entity types.

Each validator returns a ValidationResult so the decision policy can tell
*why* something failed (wrong length vs failed checksum vs out of range).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

AMOUNT_MIN = 1
AMOUNT_MAX = 1_000_000  # plausibility ceiling for the demo; tune per use case


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    reason: str  # "ok" | "empty" | "non_digit" | "bad_length" | "bad_prefix" | "bad_checksum" | "out_of_range"


def luhn_check_digit(first_digits: str) -> str:
    """Check digit that makes `first_digits + d` pass the Luhn check.

    For a 10-digit Saudi ID, the 1st, 3rd, 5th, 7th and 9th digits are doubled.
    """
    total = 0
    for i, ch in enumerate(reversed(first_digits)):
        n = int(ch)
        if i % 2 == 0:  # rightmost payload digit is doubled
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return str((10 - total % 10) % 10)


def luhn_valid(number: str) -> bool:
    return number[-1] == luhn_check_digit(number[:-1])


def validate_national_id(value: str | None) -> ValidationResult:
    """Saudi national ID (starts with 1) or Iqama (starts with 2): 10 digits + Luhn."""
    if not value:
        return ValidationResult(False, "empty")
    if not value.isdigit():
        return ValidationResult(False, "non_digit")
    if len(value) != 10:
        return ValidationResult(False, "bad_length")
    if value[0] not in "12":
        return ValidationResult(False, "bad_prefix")
    if not luhn_valid(value):
        return ValidationResult(False, "bad_checksum")
    return ValidationResult(True, "ok")


_MOBILE_RE = re.compile(r"^05\d{8}$")


def validate_mobile(value: str | None) -> ValidationResult:
    """Saudi mobile in local format: 05 + 8 digits. No checksum exists."""
    if not value:
        return ValidationResult(False, "empty")
    if not value.isdigit():
        return ValidationResult(False, "non_digit")
    if len(value) != 10:
        return ValidationResult(False, "bad_length")
    if not _MOBILE_RE.match(value):
        return ValidationResult(False, "bad_prefix")
    return ValidationResult(True, "ok")


def validate_amount(value: int | float | None) -> ValidationResult:
    """Amounts have no format rule, only a plausibility range."""
    if value is None:
        return ValidationResult(False, "empty")
    if not (AMOUNT_MIN <= value <= AMOUNT_MAX):
        return ValidationResult(False, "out_of_range")
    return ValidationResult(True, "ok")


VALIDATORS = {
    "national_id": validate_national_id,
    "mobile": validate_mobile,
    "amount": validate_amount,
}
