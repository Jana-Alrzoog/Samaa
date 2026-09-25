import random

import pytest

from samaa.normalize import normalize_arabic
from samaa.validate import (
    luhn_check_digit,
    validate_amount,
    validate_mobile,
    validate_national_id,
)

# Publicly documented examples from Saudi ID validator packages
KNOWN_VALID_IDS = ["1101798278", "1012345672"]
KNOWN_INVALID_IDS = ["1071724369"]


@pytest.mark.parametrize("value", KNOWN_VALID_IDS)
def test_known_valid_ids(value):
    assert validate_national_id(value).valid


@pytest.mark.parametrize("value", KNOWN_INVALID_IDS)
def test_known_invalid_ids(value):
    assert validate_national_id(value).reason == "bad_checksum"


def test_luhn_catches_every_single_digit_substitution():
    """The core claim behind the ID results: no 1-digit error passes the checksum."""
    rng = random.Random(0)
    for _ in range(200):
        first9 = rng.choice("12") + "".join(rng.choices("0123456789", k=8))
        good = first9 + luhn_check_digit(first9)
        assert validate_national_id(good).valid
        for pos in range(1, 10):  # position 0 changes the prefix instead
            for d in "0123456789":
                if d == good[pos]:
                    continue
                bad = good[:pos] + d + good[pos + 1:]
                assert not validate_national_id(bad).valid


@pytest.mark.parametrize(
    "value,reason",
    [("", "empty"), ("12345", "bad_length"), ("3101798278", "bad_prefix"), ("11017982a8", "non_digit")],
)
def test_id_failure_reasons(value, reason):
    assert validate_national_id(value).reason == reason


@pytest.mark.parametrize("value", ["0551234567", "0569999999"])
def test_valid_mobiles(value):
    assert validate_mobile(value).valid


@pytest.mark.parametrize(
    "value,reason",
    [("055123456", "bad_length"), ("0451234567", "bad_prefix"), ("", "empty")],
)
def test_invalid_mobiles(value, reason):
    assert validate_mobile(value).reason == reason


def test_amount_range():
    assert validate_amount(1350).valid
    assert validate_amount(0).reason == "out_of_range"
    assert validate_amount(None).reason == "empty"


def test_normalize_arabic():
    assert normalize_arabic("إِنَّ الأرقامَ ٠٥٥ مكتوبة!") == "ان الارقام 055 مكتوبه"
    assert normalize_arabic("مستشفى") == "مستشفي"
