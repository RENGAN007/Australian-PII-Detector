"""Level 2 - turn templates into annotated documents.

The key idea: we record each entity's character offsets *while* we build
the text. Searching for the values afterwards breaks easily (the same
number can appear twice, a name can appear inside an email, and noise
can change the text around it).

Output of render():
    {"text": "...", "entities": [{"start": 15, "end": 26,
                                  "label": "AU_TFN", "text": "123 456 782"}]}
and text[start:end] == entity["text"] for every entity.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from datetime import date
from typing import Callable, Optional

from . import validators as v
from .generators import DATE_STYLES, PIIGenerator, format_date
from .labels import LABELS
from .templates import Template

SLOT = re.compile(r"\{([A-Z0-9_]+)(?::([A-Za-z_]+))?(?:#(\d))?\}")

# Validators that must NOT accept an unlabelled number. Format-only types
# (BSB, bank account) can't be excluded: any 6-10 digit run "looks like" an
# account number. That ambiguity is real, and context is what resolves it.
_LOOKALIKE_CHECKS = (v.is_valid_tfn, v.is_valid_abn, v.is_valid_acn,
                     v.is_valid_medicare, v.is_valid_phone)


# ----------------------------------------------------------------------
# Distractors: realistic non-PII values that sit next to the PII
# ----------------------------------------------------------------------

COMPANY_WORDS = ("Wattle", "Southern Cross", "Bluegum", "Harbour", "Red Gum", "Coastal",
                 "Kookaburra", "Outback", "Bayside", "Summit", "Ironbark", "Riverina")
COMPANY_TRADES = ("Plumbing", "Logistics", "Electrical", "Accounting", "Cafe", "Constructions",
                  "Health", "Removals", "Landscaping", "IT Services", "Dental", "Motors")


def safe_number(rng: random.Random, n_digits: int) -> str:
    """Random digits that do NOT pass any checksum (or phone) check.

    Without this, about 1 in 11 random 9-digit order numbers would be a
    valid TFN, and we'd be silently putting unlabelled TFNs in the data.
    """
    while True:
        s = "".join(str(rng.randint(0, 9)) for _ in range(n_digits))
        if not any(check(s) for check in _LOOKALIKE_CHECKS):
            return s


def _company(gen: PIIGenerator) -> str:
    r = gen.rng
    name = f"{r.choice(COMPANY_WORDS)} {r.choice(COMPANY_TRADES)}"
    return name + (" Pty Ltd" if r.random() < 0.5 else "")


def _amount(gen: PIIGenerator) -> str:
    return f"${gen.rng.uniform(15, 9500):,.2f}"


def _date(gen: PIIGenerator) -> str:
    # Recent/upcoming dates. Unlabelled, so the model learns that not every
    # date is a date of birth - context ("DOB", "born") decides.
    d = gen.random_date(date(2024, 1, 1), date(2027, 12, 31))
    return format_date(d, gen.rng.choice(DATE_STYLES))


def _time(gen: PIIGenerator) -> str:
    h, m = gen.rng.randint(7, 19), gen.rng.choice((0, 15, 30, 45))
    return f"{h:02d}:{m:02d}" if gen.rng.random() < 0.6 else f"{(h - 1) % 12 + 1}:{m:02d}{'am' if h < 12 else 'pm'}"


def _invoice_no(gen: PIIGenerator) -> str:
    r = gen.rng
    if r.random() < 0.5:
        return f"INV-{r.randint(10000, 999999)}"
    return safe_number(r, 9)                 # a bare 9-digit lookalike


def _order_id(gen: PIIGenerator) -> str:
    n = safe_number(gen.rng, gen.rng.randint(8, 11))
    return ("#" + n) if gen.rng.random() < 0.3 else n


def _ref_no(gen: PIIGenerator) -> str:
    r = gen.rng
    kind = r.random()
    if kind < 0.4:
        return safe_number(r, r.randint(6, 10))
    if kind < 0.7:
        return f"REF{r.randint(100000, 9999999)}"
    return "".join(r.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(8))


def _customer_no(gen: PIIGenerator) -> str:
    return ("C" if gen.rng.random() < 0.5 else "") + safe_number(gen.rng, gen.rng.randint(7, 9))


def _tracking(gen: PIIGenerator) -> str:
    r = gen.rng
    return f"{r.choice('ABCDEFGHJKLMNPRSTUVWXYZ')}{r.choice('ABCDEFGHJKLMNPRSTUVWXYZ')}{safe_number(r, 9)}AU"


def _fake(label: str) -> Callable[[PIIGenerator], str]:
    """A lookalike that fails EVERY checksum, not just its own.

    A corrupted TFN fails the TFN check, but about 1 in 10 of them happens
    to be a valid ACN (both are 9 digits). Left unlabelled, that would
    teach the model that a valid ACN isn't PII. So keep trying until no
    validator accepts it.
    """
    def make(gen: PIIGenerator) -> str:
        while True:
            value = gen.invalid(label)
            if not any(check(value) for check in _LOOKALIKE_CHECKS):
                return value
    return make


DISTRACTORS: dict[str, Callable[[PIIGenerator], str]] = {
    "COMPANY": _company,
    "AMOUNT": _amount,
    "DATE": _date,
    "TIME": _time,
    "INVOICE_NO": _invoice_no,
    "ORDER_ID": _order_id,
    "REF_NO": _ref_no,
    "CUSTOMER_NO": _customer_no,
    "TRACKING": _tracking,
    "HOURS": lambda g: g.rng.choice(["38", "40", "76", "22.5", "15"]),
    "CARD_LAST4": lambda g: f"{g.rng.randint(0, 9999):04d}",
    # Checksum-failing lookalikes from Level 1, deliberately left unlabelled.
    "FAKE_TFN": _fake("AU_TFN"),
    "FAKE_ABN": _fake("AU_ABN"),
    "FAKE_MEDICARE": _fake("AU_MEDICARE"),
}


# ----------------------------------------------------------------------
# Noise
# ----------------------------------------------------------------------

@dataclass
class Noise:
    """Per-document noise settings.

    Casing is applied to the whole document, entities included (it never
    changes a checksum). Extra spaces and typos only touch the template's
    own words, never an entity, so every labelled identifier stays valid.
    """
    casing: str = "none"        # none | lower | upper
    extra_spaces: bool = False
    typo_rate: float = 0.0

    @classmethod
    def sample(cls, rng: random.Random) -> "Noise":
        r = rng.random()
        casing = "lower" if r < 0.10 else "upper" if r < 0.15 else "none"
        return cls(casing=casing,
                   extra_spaces=rng.random() < 0.10,
                   typo_rate=0.05 if rng.random() < 0.15 else 0.0)

    def case(self, s: str) -> str:
        if self.casing == "lower":
            return s.lower()
        if self.casing == "upper":
            return s.upper()
        return s

    def literal(self, s: str, rng: random.Random) -> str:
        if self.typo_rate:
            s = re.sub(r"[A-Za-z]{4,}", lambda m: _typo(m.group(), rng, self.typo_rate), s)
        if self.extra_spaces:
            s = re.sub(r" ", lambda m: "  " if rng.random() < 0.2 else " ", s)
        return self.case(s)


def _typo(word: str, rng: random.Random, rate: float) -> str:
    """Swap two neighbouring inner letters: 'Number' -> 'Nubmer'."""
    if rng.random() >= rate:
        return word
    i = rng.randint(1, len(word) - 3)
    return word[:i] + word[i + 1] + word[i] + word[i + 2:]


# ----------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------

class Renderer:
    def __init__(self, seed: Optional[int] = None):
        self.gen = PIIGenerator(seed)
        self.rng = self.gen.rng           # one RNG -> fully reproducible

    def _fill(self, name: str, option: Optional[str], person: tuple[str, str]
              ) -> tuple[str, Optional[str]]:
        """Return (value, label). label is None for distractors."""
        g = self.gen
        first, last = person
        if name == "FIRST_NAME":
            return first, "PERSON"
        if name == "PERSON":
            return g.person_name(style=option, first=first, last=last), "PERSON"
        if name == "EMAIL":
            return g.email(first, last), "EMAIL"
        if name == "AU_DRIVER_LICENCE":
            return g.driver_licence(state=option), name
        if name == "AU_PHONE":
            return g.phone(kind=option), name
        if name == "AU_MEDICARE":
            return g.medicare(include_irn=(option == "irn")), name
        if name == "AU_ADDRESS":
            return g.address(style=option), name
        if name in LABELS:
            return g.generate(name), name
        if name in DISTRACTORS:
            return DISTRACTORS[name](g), None
        raise KeyError(f"unknown slot {{{name}}}")

    def render(self, template: Template, noise: bool = True) -> dict:
        noise_cfg = Noise.sample(self.rng) if noise else Noise()
        people: dict[str, tuple[str, str]] = {}
        cache: dict[tuple, tuple[str, Optional[str]]] = {}

        parts: list[str] = []
        entities: list[dict] = []
        pos, last = 0, 0

        def emit(s: str) -> None:
            nonlocal pos
            parts.append(s)
            pos += len(s)

        for m in SLOT.finditer(template.text):
            emit(noise_cfg.literal(template.text[last:m.start()], self.rng))
            name, option, idx = m.group(1), m.group(2), m.group(3) or "1"

            key = (name, option, idx)
            if key not in cache:
                if idx not in people:
                    people[idx] = self.gen.name_parts()
                cache[key] = self._fill(name, option, people[idx])
            value, label = cache[key]
            value = noise_cfg.case(value)

            if label is not None:
                entities.append({"start": pos, "end": pos + len(value),
                                 "label": label, "text": value})
            emit(value)
            last = m.end()
        emit(noise_cfg.literal(template.text[last:], self.rng))

        text = "".join(parts)
        for e in entities:   # cheap guard: catch any offset bug immediately
            assert text[e["start"]:e["end"]] == e["text"], (template.id, e)
        return {"text": text, "entities": entities,
                "template_id": template.id, "doc_type": template.doc_type,
                "cue": template.cue,
                "noise": {"casing": noise_cfg.casing, "extra_spaces": noise_cfg.extra_spaces,
                          "typos": noise_cfg.typo_rate > 0}}


def template_labels(template: Template) -> set[str]:
    """Which entity labels a template can produce (used for split coverage)."""
    out = set()
    for m in SLOT.finditer(template.text):
        name = m.group(1)
        if name == "FIRST_NAME":
            out.add("PERSON")
        elif name in LABELS:
            out.add(name)
    return out
