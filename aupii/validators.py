"""Level 1 - validators for Australian identifiers.

Every validator takes a string the way it might appear in a document
("123 456 782", "51-824-753-556") and returns True or False.

There are two kinds of check:
  * format checks   - right length, right characters, right prefix
  * checksum checks - the weighted-sum maths that catches typos

Identifiers without a public checksum (BSB, passport, licence, phone)
only get format checks, so a pass means "plausible", not "real".
That difference matters later: checksums are what let a detector
reject random 9-digit numbers that merely *look* like a TFN.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Callable, Optional

# ----------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------

# Separators people put inside numbers: spaces, hyphens, dots, slashes.
_SEPARATORS = re.compile(r"[\s\-./]")

# ASCII 0-9 only. str.isdigit() would also accept "²" or fullwidth "１".
_ASCII_DIGITS = re.compile(r"[0-9]+")


def normalise(value: str) -> str:
    """Strip common separators: '123 456-782' -> '123456782'."""
    return _SEPARATORS.sub("", value)


def to_digits(value: str, lengths: set[int]) -> Optional[list[int]]:
    """Return the digits as a list of ints, or None if the shape is wrong."""
    cleaned = normalise(value)
    if not _ASCII_DIGITS.fullmatch(cleaned) or len(cleaned) not in lengths:
        return None
    return [int(ch) for ch in cleaned]


def weighted_sum(digits, weights) -> int:
    """sum(d_i * w_i) - the core of every checksum below."""
    return sum(d * w for d, w in zip(digits, weights))


# ----------------------------------------------------------------------
# Checksum identifiers
# ----------------------------------------------------------------------

TFN_WEIGHTS = (1, 4, 3, 7, 5, 8, 6, 9, 10)


def is_valid_tfn(value: str) -> bool:
    """Tax File Number: 9 digits whose weighted sum is divisible by 11.

    Worked example, 123 456 782:
        1*1 + 2*4 + 3*3 + 4*7 + 5*5 + 6*8 + 7*6 + 8*9 + 2*10 = 253
        253 = 11 * 23  -> valid
    Older 8-digit TFNs exist but are out of scope for this project.
    """
    digits = to_digits(value, {9})
    if digits is None or not any(digits):      # reject 000 000 000
        return False
    return weighted_sum(digits, TFN_WEIGHTS) % 11 == 0


ABN_WEIGHTS = (10, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19)


def abn_remainder(digits: list[int]) -> int:
    """ABN rule: subtract 1 from the FIRST digit, weight, then mod 89."""
    adjusted = [digits[0] - 1] + list(digits[1:])
    return weighted_sum(adjusted, ABN_WEIGHTS) % 89


def is_valid_abn(value: str) -> bool:
    """Australian Business Number: 11 digits, remainder 0 under mod 89.

    The first two digits are the check digits (10-99), so the first
    digit can never be 0.
    """
    digits = to_digits(value, {11})
    if digits is None or digits[0] == 0:
        return False
    return abn_remainder(digits) == 0


ACN_WEIGHTS = (8, 7, 6, 5, 4, 3, 2, 1)


def acn_check_digit(first_eight: list[int]) -> int:
    """Check digit = (10 - weighted_sum mod 10) mod 10."""
    return (10 - weighted_sum(first_eight, ACN_WEIGHTS) % 10) % 10


def is_valid_acn(value: str) -> bool:
    """Australian Company Number: 8 digits + 1 check digit.

    Note: mod 10 is weaker than mod 11 or mod 89. Some single-digit
    typos still pass (weight 2, 4, 6 or 8 combined with a change of 5).
    test_generators.py shows why we must re-check corrupted values.
    """
    digits = to_digits(value, {9})
    if digits is None or not any(digits):
        return False
    return acn_check_digit(digits[:8]) == digits[8]


MEDICARE_WEIGHTS = (1, 3, 7, 9, 1, 3, 7, 9)


def medicare_check_digit(first_eight: list[int]) -> int:
    """Check digit = weighted sum of the first 8 digits, mod 10."""
    return weighted_sum(first_eight, MEDICARE_WEIGHTS) % 10


def is_valid_medicare(value: str) -> bool:
    """Medicare card number.

    Layout (10 digits, optionally 11):
        d1      : 2-6
        d2-d8   : identifier
        d9      : check digit
        d10     : issue number (1-9)
        d11     : individual reference number / IRN (1-9), optional
    """
    digits = to_digits(value, {10, 11})
    if digits is None:
        return False
    if not 2 <= digits[0] <= 6:
        return False
    if medicare_check_digit(digits[:8]) != digits[8]:
        return False
    if digits[9] == 0:
        return False
    if len(digits) == 11 and digits[10] == 0:
        return False
    return True


# ----------------------------------------------------------------------
# Format-only identifiers
# ----------------------------------------------------------------------

def is_valid_bsb(value: str) -> bool:
    """BSB: 6 digits, usually written XXX-XXX. No checksum."""
    return to_digits(value, {6}) is not None


def is_valid_bank_account(value: str) -> bool:
    """Bank account: 6-10 digits. No checksum, very ambiguous on its own."""
    return to_digits(value, set(range(6, 11))) is not None


_PASSPORT = re.compile(r"[A-Z]{1,2}[0-9]{7}")


def is_valid_passport(value: str) -> bool:
    """Australian passport: 1-2 letters then 7 digits, e.g. PA1234567."""
    return bool(_PASSPORT.fullmatch(normalise(value).upper()))


# Best-effort patterns. There is no public checksum and formats have
# changed over the years, so treat these as approximations to refine.
DRIVER_LICENCE_PATTERNS: dict[str, str] = {
    "NSW": r"[0-9]{6,8}",
    "VIC": r"[0-9]{8,10}",
    "QLD": r"[0-9]{8,9}",
    "WA": r"[0-9]{7}",
    "SA": r"[A-Z][0-9]{5}",
    "TAS": r"[A-Z][0-9]{5}",
    "NT": r"[0-9]{5,10}",
    "ACT": r"[0-9]{6,10}",
}


def is_valid_driver_licence(value: str, state: Optional[str] = None) -> bool:
    """Format check against one state's pattern, or any state if none given."""
    cleaned = normalise(value).upper()
    states = [state] if state else list(DRIVER_LICENCE_PATTERNS)
    return any(re.fullmatch(DRIVER_LICENCE_PATTERNS[s], cleaned) for s in states)


# After stripping spaces/brackets/hyphens:
#   0412345678, +61412345678, 61412345678, +610412345678 (from "+61 (0)4...")
#   0398765432, +61398765432
_PHONE = re.compile(r"(?:\+?610?|0)(?:4[0-9]{8}|[2378][0-9]{8})")


def is_valid_phone(value: str) -> bool:
    """Australian mobile (04...) or geographic landline (02/03/07/08).

    13/1300/1800 numbers are business numbers, not personal, so excluded.
    """
    cleaned = re.sub(r"[\s\-().]", "", value)
    return bool(_PHONE.fullmatch(cleaned))


POSTCODE_RANGES: dict[str, list[tuple[int, int]]] = {
    "NSW": [(1000, 2599), (2619, 2899), (2921, 2999)],
    "ACT": [(200, 299), (2600, 2618), (2900, 2920)],
    "VIC": [(3000, 3999), (8000, 8999)],
    "QLD": [(4000, 4999), (9000, 9999)],
    "SA": [(5000, 5999)],
    "WA": [(6000, 6797), (6800, 6999)],
    "TAS": [(7000, 7999)],
    "NT": [(800, 999)],
}


def postcode_state(postcode: str) -> Optional[str]:
    """Which state a 4-digit postcode belongs to, or None."""
    if not re.fullmatch(r"[0-9]{4}", postcode):
        return None
    pc = int(postcode)
    for state, ranges in POSTCODE_RANGES.items():
        if any(lo <= pc <= hi for lo, hi in ranges):
            return state
    return None


def is_valid_postcode(postcode: str, state: Optional[str] = None) -> bool:
    found = postcode_state(postcode)
    return found is not None and (state is None or found == state)


_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")


def is_valid_email(value: str) -> bool:
    """Practical email check. Regex can't fully implement RFC 5322."""
    return bool(_EMAIL.fullmatch(value)) and ".." not in value


# Australian order is DAY first: 03/04/1990 is 3 April, not 4 March.
DATE_FORMATS = (
    "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%Y-%m-%d", "%d %b %Y", "%d %B %Y",
)


def parse_date(value: str) -> Optional[date]:
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(value.strip(), fmt).date()
        except ValueError:
            continue
    return None


def is_valid_dob(value: str, today: Optional[date] = None) -> bool:
    """A real calendar date, not in the future, age at most 120."""
    parsed = parse_date(value)
    if parsed is None:
        return False
    today = today or date.today()
    return parsed <= today and (today.year - parsed.year) <= 120


# ----------------------------------------------------------------------
# Registry: label -> validator. PERSON and AU_ADDRESS have none,
# because a name or an address can't be verified from text alone.
# ----------------------------------------------------------------------

VALIDATORS: dict[str, Callable[[str], bool]] = {
    "AU_TFN": is_valid_tfn,
    "AU_ABN": is_valid_abn,
    "AU_ACN": is_valid_acn,
    "AU_MEDICARE": is_valid_medicare,
    "AU_BSB": is_valid_bsb,
    "AU_BANK_ACCOUNT": is_valid_bank_account,
    "AU_PASSPORT": is_valid_passport,
    "AU_DRIVER_LICENCE": is_valid_driver_licence,
    "AU_PHONE": is_valid_phone,
    "EMAIL": is_valid_email,
    "DATE_OF_BIRTH": is_valid_dob,
}
