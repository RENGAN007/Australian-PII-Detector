"""Level 1 - generators for synthetic Australian identifiers.

Design ideas worth noticing:

1. Every generator uses its own seeded random.Random, never the global
   `random`. Same seed -> same dataset, which makes experiments
   reproducible.
2. Valid checksum numbers are built by choosing the free digits at random
   and then SOLVING for the check digit(s).
3. The same number appears in several surface styles ("123456782",
   "123 456 782", "123-456-782"), because real documents are messy and the
   model must learn all of them.
4. `invalid()` produces hard negatives: right format, wrong checksum.
   These teach the model that "9 digits" is not the same as "a TFN".

Ethics note: a randomly generated number with a valid checksum can
coincide with a real person's identifier. Treat the dataset as
sensitive-looking data. Don't publish it as if it were "real", and say
in the README that it is synthetic.
"""

from __future__ import annotations

import random
from datetime import date, timedelta
from typing import Callable, Optional, Sequence

from . import validators as v

# ----------------------------------------------------------------------
# Small vocabularies (extend these in Level 2 for more diversity)
# ----------------------------------------------------------------------

FIRST_NAMES = (
    "Olivia", "Jack", "Charlotte", "Noah", "Amelia", "William", "Isla", "Oliver",
    "Mia", "Lucas", "Priya", "Arjun", "Wei", "Mei", "Minh", "Linh", "Ahmed",
    "Fatima", "Giorgos", "Eleni", "Marco", "Sofia", "Kiri", "Tane", "Aisha",
    "Omar", "Hannah", "Liam", "Grace", "Ethan", "Chloe", "Ravi", "Anjali",
    "Daniel", "Zara", "Hiroshi", "Yuki", "Mateo", "Lucia", "Kwame",
)

LAST_NAMES = (
    "Smith", "Jones", "Williams", "Brown", "Wilson", "Taylor", "Nguyen", "Tran",
    "Le", "Chen", "Wang", "Li", "Zhang", "Singh", "Kaur", "Patel", "Sharma",
    "Kumar", "Papadopoulos", "Rossi", "Russo", "Ali", "Khan", "Hassan",
    "Murphy", "O'Brien", "Kelly", "Anderson", "Thompson", "White", "Martin",
    "Walker", "Harris", "Lee", "Kim", "Park", "Tanaka", "Garcia", "Mensah",
    "Rahman",
)

EMAIL_DOMAINS = (
    "gmail.com", "outlook.com", "hotmail.com", "yahoo.com.au", "bigpond.com",
    "icloud.com", "example.com", "example.org",
)

# (suburb, state, postcode) - real suburb/postcode pairs.
SUBURBS = (
    ("Melbourne", "VIC", "3000"), ("Richmond", "VIC", "3121"),
    ("Fitzroy", "VIC", "3065"), ("Box Hill", "VIC", "3128"),
    ("Geelong", "VIC", "3220"), ("Sydney", "NSW", "2000"),
    ("Parramatta", "NSW", "2150"), ("Bondi", "NSW", "2026"),
    ("Newtown", "NSW", "2042"), ("Brisbane City", "QLD", "4000"),
    ("Southport", "QLD", "4215"), ("Toowong", "QLD", "4066"),
    ("Adelaide", "SA", "5000"), ("Glenelg", "SA", "5045"),
    ("Perth", "WA", "6000"), ("Fremantle", "WA", "6160"),
    ("Hobart", "TAS", "7000"), ("Launceston", "TAS", "7250"),
    ("Darwin", "NT", "0800"), ("Belconnen", "ACT", "2617"),
)

STREET_NAMES = (
    "High", "Church", "Station", "Victoria", "Park", "George", "King", "Queen",
    "Bridge", "Railway", "Albert", "Elizabeth", "Wattle", "Banksia", "Acacia",
    "Hill", "Main", "Smith", "Ocean", "River",
)

# (full, abbreviated)
STREET_TYPES = (
    ("Street", "St"), ("Road", "Rd"), ("Avenue", "Ave"), ("Drive", "Dr"),
    ("Court", "Ct"), ("Place", "Pl"), ("Crescent", "Cres"), ("Lane", "Ln"),
    ("Parade", "Pde"), ("Way", "Way"),
)

MONTHS = ("January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December")

# Two-digit institution codes used by several Australian banks (first
# two digits of a BSB). The third digit is the state: 2 NSW/ACT, 3 VIC,
# 4 QLD, 5 SA/NT, 6 WA, 7 TAS.
BSB_INSTITUTION_CODES = ("01", "03", "06", "08", "12", "18", "48", "63", "73", "76")

# Letter prefixes seen on Australian passports (approximate).
PASSPORT_PREFIXES = ("N", "E", "M", "L", "P", "R", "PA", "PB", "PC", "PD", "PE", "RA")

# Digit-count ranges for the numeric licence formats (see validators.py).
LICENCE_LENGTHS = {"NSW": (6, 8), "VIC": (8, 10), "QLD": (8, 9), "WA": (7, 7),
                   "NT": (5, 10), "ACT": (6, 10)}

# Medicare issue numbers are mostly low (most people are on card 1-3).
_ISSUE_WEIGHTS = (30, 25, 15, 10, 7, 5, 4, 2, 2)


DATE_STYLES = ("dmy_slash", "dmy_slash", "dmy_short", "iso",
               "dmy_dash", "dmy_dot", "long", "abbrev")


def format_date(d: date, style: str) -> str:
    """Format a date in one of the styles common in Australian documents."""
    return {
        "dmy_slash": f"{d.day:02d}/{d.month:02d}/{d.year}",
        "dmy_short": f"{d.day}/{d.month}/{d.year}",
        "iso": d.isoformat(),
        "dmy_dash": f"{d.day:02d}-{d.month:02d}-{d.year}",
        "dmy_dot": f"{d.day:02d}.{d.month:02d}.{d.year}",
        "long": f"{d.day} {MONTHS[d.month - 1]} {d.year}",
        "abbrev": f"{d.day} {MONTHS[d.month - 1][:3]} {d.year}",
    }[style]


def _join(digits: Sequence[int]) -> str:
    return "".join(str(d) for d in digits)


def _group(text: str, sizes: Sequence[int], sep: str) -> str:
    """'123456782', (3,3,3), ' ' -> '123 456 782'."""
    parts, i = [], 0
    for size in sizes:
        parts.append(text[i:i + size])
        i += size
    if i < len(text):
        parts.append(text[i:])
    return sep.join(p for p in parts if p)


class PIIGenerator:
    """Seeded generator for every entity type in labels.py."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _digits(self, n: int) -> list[int]:
        return [self.rng.randint(0, 9) for _ in range(n)]

    def _style(self, style: Optional[str], allowed: Sequence[str]) -> str:
        if style is None:
            return self.rng.choice(allowed)
        if style not in allowed:
            raise ValueError(f"style must be one of {allowed}, got {style!r}")
        return style

    @staticmethod
    def _format(number: str, groups: Sequence[int], style: str) -> str:
        if style == "plain":
            return number
        sep = {"spaced": " ", "hyphen": "-"}[style]
        return _group(number, groups, sep)

    # ------------------------------------------------------------------
    # checksum identifiers
    # ------------------------------------------------------------------

    def tfn(self, style: Optional[str] = None) -> str:
        """e.g. '123 456 782'. Styles: plain, spaced, hyphen."""
        style = self._style(style, ("plain", "spaced", "hyphen"))
        while True:
            body = self._digits(8)
            # Need (S + 10*c) % 11 == 0. Since 10 = -1 (mod 11), c = S mod 11.
            check = v.weighted_sum(body, v.TFN_WEIGHTS[:8]) % 11
            if check < 10 and any(body + [check]):   # 10 isn't a digit -> retry
                break
        return self._format(_join(body + [check]), (3, 3, 3), style)

    def abn(self, style: Optional[str] = None) -> str:
        """e.g. '51 824 753 556'. Styles: plain, spaced."""
        style = self._style(style, ("plain", "spaced"))
        tail = self._digits(9)
        tail_sum = v.weighted_sum(tail, v.ABN_WEIGHTS[2:])
        # The first two digits contribute 10*(a-1) + b, which covers 0..89,
        # so there is always at least one pair that makes the total % 89 == 0.
        pairs = [(a, b) for a in range(1, 10) for b in range(10)
                 if (tail_sum + 10 * (a - 1) + b) % 89 == 0]
        a, b = self.rng.choice(pairs)
        return self._format(_join([a, b] + tail), (2, 3, 3, 3), style)

    def acn(self, style: Optional[str] = None) -> str:
        """e.g. '004 085 616'. Styles: plain, spaced, hyphen."""
        style = self._style(style, ("plain", "spaced", "hyphen"))
        while True:
            body = self._digits(8)
            if any(body):
                break
        return self._format(_join(body + [v.acn_check_digit(body)]), (3, 3, 3), style)

    def medicare(self, style: Optional[str] = None, include_irn: bool = False) -> str:
        """e.g. '2123 45670 1'. Styles: plain, spaced, hyphen.

        With include_irn=True a 1-digit IRN is appended: '2123 45670 1/2'.
        """
        style = self._style(style, ("plain", "spaced", "hyphen"))
        body = [self.rng.randint(2, 6)] + self._digits(7)
        check = v.medicare_check_digit(body)
        issue = self.rng.choices(range(1, 10), weights=_ISSUE_WEIGHTS)[0]
        text = self._format(_join(body + [check, issue]), (4, 5, 1), style)
        if include_irn:
            irn = str(self.rng.randint(1, 9))
            sep = "" if style == "plain" else self.rng.choice(["/", " ", "-"])
            text += sep + irn
        return text

    # ------------------------------------------------------------------
    # format-only identifiers
    # ------------------------------------------------------------------

    def bsb(self, style: Optional[str] = None) -> str:
        """e.g. '063-123'. Styles: hyphen (most common), plain, spaced."""
        style = self._style(style, ("hyphen", "hyphen", "plain", "spaced"))
        code = self.rng.choice(BSB_INSTITUTION_CODES)
        number = code + str(self.rng.randint(2, 7)) + _join(self._digits(3))
        return self._format(number, (3, 3), style)

    def bank_account(self, style: Optional[str] = None) -> str:
        """6-9 digits. Styles: plain, spaced (groups of 4)."""
        style = self._style(style, ("plain", "plain", "spaced"))
        number = _join(self._digits(self.rng.randint(6, 9)))
        groups = {8: (4, 4), 9: (3, 3, 3), 6: (3, 3)}.get(len(number))
        if groups is None:          # 7 digits: no natural grouping
            return number
        return self._format(number, groups, style)

    def passport(self) -> str:
        """e.g. 'PA1234567'."""
        return self.rng.choice(PASSPORT_PREFIXES) + _join(self._digits(7))

    def driver_licence(self, state: Optional[str] = None) -> str:
        """Licence number in one state's (approximate) format."""
        state = state or self.rng.choice(list(v.DRIVER_LICENCE_PATTERNS))
        if state in ("SA", "TAS"):
            return self.rng.choice("ABCDEFGHJKLMNPRSTUVWXYZ") + _join(self._digits(5))
        lo, hi = LICENCE_LENGTHS[state]
        return _join(self._digits(self.rng.randint(lo, hi)))

    def phone(self, kind: Optional[str] = None, style: Optional[str] = None,
              fictional: bool = False) -> str:
        """Mobile or landline.

        kind: 'mobile' or 'landline' (random if None, 60/40)
        mobile styles:   plain '0412345678', spaced '0412 345 678',
                         intl '+61 412 345 678'
        landline styles: plain '0398765432', spaced '03 9876 5432',
                         paren '(03) 9876 5432', intl '+61 3 9876 5432'
        fictional=True (landline only) uses the 5550 xxxx block that ACMA
        sets aside for use in fiction, so it can never ring a real person.
        """
        if kind is None:
            kind = "mobile" if self.rng.random() < 0.6 else "landline"

        if kind == "mobile":
            style = self._style(style, ("plain", "spaced", "intl"))
            rest = _join(self._digits(8))                     # 4 + 8 digits
            if style == "plain":
                return "04" + rest
            if style == "spaced":
                return f"04{rest[:2]} {rest[2:5]} {rest[5:]}"
            return f"+61 4{rest[:2]} {rest[2:5]} {rest[5:]}"

        if kind != "landline":
            raise ValueError("kind must be 'mobile' or 'landline'")
        style = self._style(style, ("plain", "spaced", "paren", "intl"))
        area = self.rng.choice("2378")
        if fictional:
            local = "5550" + _join(self._digits(4))
        else:
            local = str(self.rng.randint(2, 9)) + _join(self._digits(7))
        if style == "plain":
            return f"0{area}{local}"
        if style == "spaced":
            return f"0{area} {local[:4]} {local[4:]}"
        if style == "paren":
            return f"(0{area}) {local[:4]} {local[4:]}"
        return f"+61 {area} {local[:4]} {local[4:]}"

    # ------------------------------------------------------------------
    # general PII
    # ------------------------------------------------------------------

    def name_parts(self) -> tuple[str, str]:
        """A random (first, last) pair. Reuse it to keep a document consistent."""
        return self.rng.choice(FIRST_NAMES), self.rng.choice(LAST_NAMES)

    def person_name(self, style: Optional[str] = None,
                    first: Optional[str] = None, last: Optional[str] = None) -> str:
        """Styles: full 'Priya Nguyen', initial 'Priya K. Nguyen',
        surname_first 'NGUYEN, Priya'. Pass first/last to name a known person."""
        style = self._style(style, ("full", "full", "initial", "surname_first"))
        if first is None or last is None:
            first, last = self.name_parts()
        if style == "full":
            return f"{first} {last}"
        if style == "initial":
            return f"{first} {self.rng.choice('ABCDEFGHJKLMNPRSTW')}. {last}"
        return f"{last.upper()}, {first}"

    def email(self, first: Optional[str] = None, last: Optional[str] = None) -> str:
        """Built from a name so it can match a PERSON in the same document."""
        if first is None or last is None:
            first, last = self.name_parts()
        f = first.lower()
        l = last.lower().replace("'", "")
        local = self.rng.choice([f"{f}.{l}", f"{f}{l}", f"{f[0]}.{l}", f"{f}_{l}", f"{l}{f[0]}"])
        if self.rng.random() < 0.3:
            local += str(self.rng.randint(1, 99))
        return f"{local}@{self.rng.choice(EMAIL_DOMAINS)}"

    def date_of_birth(self, style: Optional[str] = None) -> str:
        """Random DOB 1940-2008. Styles: dmy_slash '03/04/1990',
        dmy_short '3/4/1990', iso '1990-04-03', dmy_dash '03-04-1990',
        dmy_dot '03.04.1990', long '3 April 1990', abbrev '3 Apr 1990'."""
        return format_date(self.random_date(date(1940, 1, 1), date(2008, 12, 31)),
                           self._style(style, DATE_STYLES))

    def random_date(self, start: date, end: date) -> date:
        return start + timedelta(days=self.rng.randint(0, (end - start).days))

    def address(self, style: Optional[str] = None) -> str:
        """Styles: one_line '12 Smith Street, Richmond VIC 3121',
        multi_line (newline before suburb), comma '12 Smith St, Richmond, VIC, 3121'."""
        style = self._style(style, ("one_line", "one_line", "multi_line", "comma"))
        suburb, state, postcode = self.rng.choice(SUBURBS)
        full, abbrev = self.rng.choice(STREET_TYPES)
        street_type = full if self.rng.random() < 0.6 else abbrev
        street = f"{self.rng.randint(1, 250)} {self.rng.choice(STREET_NAMES)} {street_type}"
        if self.rng.random() < 0.25:                                  # unit
            unit = self.rng.randint(1, 40)
            street = f"Unit {unit}, {street}" if self.rng.random() < 0.5 else f"{unit}/{street}"
        if style == "one_line":
            return f"{street}, {suburb} {state} {postcode}"
        if style == "multi_line":
            return f"{street}\n{suburb} {state} {postcode}"
        return f"{street}, {suburb}, {state}, {postcode}"

    # ------------------------------------------------------------------
    # hard negatives
    # ------------------------------------------------------------------

    def corrupt(self, value: str, validator: Callable[[str], bool],
                max_tries: int = 100) -> str:
        """Keep the formatting, break the number, and confirm it now fails.

        70%: change one digit (a typo). 30%: swap two neighbouring digits
        (a transposition). We re-check every candidate because weak
        checksums (like the ACN's mod 10) can accidentally still pass.
        """
        positions = [i for i, ch in enumerate(value) if ch in "0123456789"]
        for _ in range(max_tries):
            chars = list(value)
            if self.rng.random() < 0.7 or len(positions) < 2:
                i = self.rng.choice(positions)
                chars[i] = self.rng.choice([d for d in "0123456789" if d != chars[i]])
            else:
                k = self.rng.randrange(len(positions) - 1)
                i, j = positions[k], positions[k + 1]
                chars[i], chars[j] = chars[j], chars[i]
            candidate = "".join(chars)
            if not validator(candidate):
                return candidate
        raise RuntimeError(f"could not corrupt {value!r} into an invalid value")

    # ------------------------------------------------------------------
    # label-based access (used by Level 2)
    # ------------------------------------------------------------------

    def generate(self, label: str, **kwargs) -> str:
        """Generate one valid value for a label from labels.py."""
        dispatch: dict[str, Callable[..., str]] = {
            "AU_TFN": self.tfn,
            "AU_ABN": self.abn,
            "AU_ACN": self.acn,
            "AU_MEDICARE": self.medicare,
            "AU_BSB": self.bsb,
            "AU_BANK_ACCOUNT": self.bank_account,
            "AU_PASSPORT": self.passport,
            "AU_DRIVER_LICENCE": self.driver_licence,
            "AU_PHONE": self.phone,
            "AU_ADDRESS": self.address,
            "PERSON": self.person_name,
            "EMAIL": self.email,
            "DATE_OF_BIRTH": self.date_of_birth,
        }
        if label not in dispatch:
            raise KeyError(f"unknown label {label!r}")
        return dispatch[label](**kwargs)

    def invalid(self, label: str, **kwargs) -> str:
        """A checksum-failing lookalike for AU_TFN/ABN/ACN/MEDICARE."""
        if label not in ("AU_TFN", "AU_ABN", "AU_ACN", "AU_MEDICARE"):
            raise ValueError("hard negatives only make sense for checksum labels")
        return self.corrupt(self.generate(label, **kwargs), v.VALIDATORS[label])
