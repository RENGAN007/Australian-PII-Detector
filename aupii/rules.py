"""Level 4 - rule-based PII detector (the baseline every model must beat).

How it works, per document:

  1. Each recogniser finds CANDIDATES with a regex.
  2. Checksum types (TFN, ABN, ACN, Medicare) and format types (phone,
     email, DOB, address) must pass their Level 1 validator, or the
     candidate is thrown away.
  3. Each surviving candidate gets a SCORE:
         base score of the recogniser (how specific its pattern is)
       + bonus  if one of its OWN cue words is in the 60 chars before it
       - 0.15   if the NEAREST cue belongs to a different label
       - 0.40   if the nearest cue is a NEGATIVE cue (invoice, order, ref...)
       + 0.45   for a BSB followed by an account number (both get it)
       - 0.50   for a "name" that sits before a state or after a street number
  4. Candidates below their label's threshold are dropped (thresholds are
     tuned on the validation set by run_rules.py).
  5. Overlapping candidates: highest score wins, then the longest.

`RuleDetector.explain(text)` prints every candidate with its score and
reasons, which is the best way to understand a mistake.

PERSON uses a gazetteer (a name list). We build it from TRAIN names only:
a real rule system's name list never covers every name, and our val/test
names are held out, so this shows honestly where gazetteers break.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

from . import validators as v
from .generators import STREET_TYPES

WINDOW = 60              # how far back (in characters) to look for cue words
OTHER_CUE_PENALTY = 0.15
NEG_CUE_PENALTY = 0.40
DEFAULT_THRESHOLD = 0.5


# ----------------------------------------------------------------------
# Cue words
# ----------------------------------------------------------------------

NEG = "NEG"

CUES: dict[str, tuple[str, ...]] = {
    "AU_TFN": ("tfn", "tax file number", "tax file no", "tax file"),
    "AU_ABN": ("abn", "australian business number", "business number"),
    "AU_ACN": ("acn", "australian company number", "company number"),
    "AU_MEDICARE": ("medicare", "medicare number", "medicare no", "card number", "card"),
    "AU_BSB": ("bsb",),
    "AU_BANK_ACCOUNT": ("account", "acc", "acct", "a/c", "account number", "account no"),
    "AU_PASSPORT": ("passport", "passport number", "passport no"),
    "AU_DRIVER_LICENCE": ("licence", "license", "driver licence", "drivers licence",
                          "licence number", "licence no", "dl"),
    "AU_PHONE": ("phone", "mobile", "mob", "ph", "tel", "call", "contact", "number",
                 "reach", "text", "sms"),
    "DATE_OF_BIRTH": ("dob", "d.o.b", "date of birth", "born", "birth", "birthday"),
    "PERSON": ("name", "full name", "name's", "employee", "applicant", "patient", "director",
               "dear", "hi", "hey", "hello", "mr", "mrs", "ms", "miss", "dr", "member",
               "regards", "thanks", "cheers", "customer", "agent", "i'm", "im",
               "account holder", "contact person", "emergency contact", "bill to",
               "sole trader"),
    NEG: ("invoice", "inv", "order", "ref", "receipt", "tracking", "transaction",
          "ticket", "quote", "booking ref", "parcel", "due", "expiry", "expires", "date",
          "service", "period", "valid until", "code", "case", "customer number",
          "company", "start date", "payslip", "hours", "total", "amount"),
}

_WORD = re.compile(r"[A-Za-z][A-Za-z./'’]*")


def osa_distance(a: str, b: str) -> int:
    """Edit distance where a swap of two neighbouring letters costs 1.

    (Optimal string alignment, a restricted Damerau-Levenshtein.)
    'nubmer' -> 'number' = 1, where plain Levenshtein would say 2.
    """
    d = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        d[i][0] = i
    for j in range(len(b) + 1):
        d[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = a[i - 1] != b[j - 1]
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[-1][-1]


@dataclass(frozen=True)
class CueHit:
    start: int
    end: int
    label: str
    cue: str


class CueIndex:
    """Finds every cue word/phrase in a text, once per document.

    Single words of 5+ letters also match with one typo ('Mdeicare'), so
    the noise added in Level 2 doesn't blind the rules completely.
    Phrases must match exactly (apart from case and spacing).
    """

    def __init__(self, cues: dict[str, tuple[str, ...]] = CUES):
        self.words: dict[str, str] = {}
        self.fuzzy: list[tuple[str, str]] = []
        phrases: list[tuple[str, str]] = []
        for label, items in cues.items():
            for cue in items:
                if " " in cue:
                    phrases.append((cue, label))
                else:
                    self.words[cue] = label
                    if len(cue) >= 5:
                        self.fuzzy.append((cue, label))
        self.phrases = [(re.compile(r"(?<![A-Za-z])" + r"\s+".join(map(re.escape, p.split()))
                                    + r"(?![A-Za-z])", re.IGNORECASE), p, label)
                        for p, label in phrases]

    def _fuzzy_label(self, w: str) -> Optional[tuple[str, str]]:
        if len(w) < 5:
            return None
        for cue, label in self.fuzzy:
            # A swap typo keeps the first letter and the length.
            if cue[0] == w[0] and abs(len(cue) - len(w)) <= 1 and osa_distance(w, cue) <= 1:
                return cue, label
        return None

    def find(self, text: str) -> list[CueHit]:
        hits = []
        for m in _WORD.finditer(text):
            w = m.group().lower().rstrip(".")
            if w in self.words:
                hits.append(CueHit(m.start(), m.start() + len(w), self.words[w], w))
            else:
                fz = self._fuzzy_label(w)
                if fz:
                    hits.append(CueHit(m.start(), m.start() + len(w), fz[1], fz[0]))
        for rx, phrase, label in self.phrases:
            hits.extend(CueHit(m.start(), m.end(), label, phrase) for m in rx.finditer(text))
        return hits


# ----------------------------------------------------------------------
# Candidates
# ----------------------------------------------------------------------

@dataclass
class Candidate:
    start: int
    end: int
    label: str
    score: float
    reasons: list[str] = field(default_factory=list)

    def entity(self) -> dict:
        return {"start": self.start, "end": self.end, "label": self.label}


@dataclass(frozen=True)
class Recogniser:
    label: str
    pattern: re.Pattern
    base: float
    cue_bonus: float
    validator: Optional[Callable[[re.Match], bool]] = None
    reason: str = "pattern"


_B, _E = r"(?<!\d)", r"(?!\d)"          # not glued to more digits


def _v(fn: Callable[[str], bool]) -> Callable[[re.Match], bool]:
    return lambda m: fn(m.group())


def _address_ok(m: re.Match) -> bool:
    return v.postcode_state(m.group("pc")) == m.group("state").upper()


_STREET_TYPE = "|".join(sorted({t for pair in STREET_TYPES for t in pair}, key=len, reverse=True))
_AW = r"[A-Za-z][A-Za-z'’-]*"
_ADDRESS = re.compile(rf"""
    (?<![\w/])
    (?:(?:Unit|Apt|Level)[ ]\d+,[ ]|\d+/)?          # optional unit / level
    \d+[A-C]?[ ]                                     # street number
    (?:{_AW}[ ]){{1,3}}?                             # street name (1-3 words)
    (?:{_STREET_TYPE})\.?                            # Street / St / Rd ...
    (?:,[ ]|[ ]*\n[ ]*|[ ])                          # separator
    {_AW}(?:[ ]{_AW}){{0,2}}?                        # suburb (1-3 words)
    ,?[ ](?P<state>NSW|VIC|QLD|SA|WA|TAS|NT|ACT)
    ,?[ ](?P<pc>\d{{4}})(?!\d)
""", re.VERBOSE | re.IGNORECASE)

_MONTH = "jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec"
_DATE = re.compile(rf"""
    (?<![\d/.-])
    (?: \d{{1,2}}[/.-]\d{{1,2}}[/.-]\d{{4}}
      | \d{{4}}-\d{{2}}-\d{{2}}
      | \d{{1,2}}[ ](?:{_MONTH})[a-z]*[ ]\d{{4}} )
    (?![\d/.-])
""", re.VERBOSE | re.IGNORECASE)

RECOGNISERS: tuple[Recogniser, ...] = (
    # checksum identifiers: high base, the checksum does the heavy lifting
    Recogniser("AU_TFN", re.compile(rf"{_B}\d{{3}}[ -]?\d{{3}}[ -]?\d{{3}}{_E}"),
               0.45, 0.45, _v(v.is_valid_tfn), "checksum"),
    Recogniser("AU_ACN", re.compile(rf"{_B}\d{{3}}[ -]?\d{{3}}[ -]?\d{{3}}{_E}"),
               0.35, 0.50, _v(v.is_valid_acn), "checksum"),
    Recogniser("AU_ABN", re.compile(rf"{_B}\d{{2}}[ -]?\d{{3}}[ -]?\d{{3}}[ -]?\d{{3}}{_E}"),
               0.55, 0.40, _v(v.is_valid_abn), "checksum"),
    Recogniser("AU_MEDICARE",
               re.compile(rf"{_B}[2-6]\d{{3}}[ -]?\d{{5}}[ -]?\d(?:[ /-]?[1-9])?{_E}"),
               0.50, 0.40, _v(v.is_valid_medicare), "checksum"),
    # distinctive formats
    Recogniser("EMAIL", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*"
                                   r"\.[A-Za-z]{2,}"), 0.95, 0.0, _v(v.is_valid_email), "format"),
    Recogniser("AU_PHONE", re.compile(rf"(?<![\w+])(?:\+?61[ -]?(?:\(0\)[ ]?)?|0)[2-478]"
                                      rf"(?:[ -]?\d){{8}}{_E}"),
               0.55, 0.35, _v(v.is_valid_phone), "format"),
    Recogniser("AU_PHONE", re.compile(rf"\(0[2378]\)[ ]?\d{{4}}[ -]?\d{{4}}{_E}"),
               0.60, 0.35, _v(v.is_valid_phone), "format"),
    Recogniser("AU_ADDRESS", _ADDRESS, 0.75, 0.15, _address_ok, "street+state+postcode"),
    Recogniser("AU_PASSPORT", re.compile(rf"(?<![A-Za-z\d])[A-Za-z]{{1,2}}\d{{7}}{_E}"),
               0.30, 0.60, None, "format"),
    # ambiguous numbers: almost nothing without context
    # A BSB must not be the first or last 6 digits of a 3-3-3 number (a TFN/ACN):
    # "735 076 865" contains "735 076", but that is not a BSB.
    Recogniser("AU_BSB", re.compile(rf"{_B}(?<!\d{{3}}[ -])\d{{3}}-\d{{3}}{_E}(?![ -]\d{{3}}(?!\d))"),
               0.30, 0.60, None, "format"),
    Recogniser("AU_BSB", re.compile(rf"{_B}(?<!\d{{3}}[ -])\d{{3}} ?\d{{3}}{_E}(?![ -]\d{{3}}(?!\d))"),
               0.10, 0.60, None, "format"),
    Recogniser("AU_BANK_ACCOUNT",
               re.compile(rf"{_B}(?:\d{{3}} \d{{3}} \d{{3}}|\d{{4}} \d{{4}}|\d{{3}} \d{{3}}"
                          rf"|\d{{6,10}}){_E}"), 0.05, 0.55, None, "format"),
    Recogniser("AU_DRIVER_LICENCE",
               re.compile(rf"(?<![A-Za-z\d])(?:[A-Za-z]\d{{5}}|\d{{5,10}}){_E}"),
               0.02, 0.70, None, "format"),
    Recogniser("DATE_OF_BIRTH", _DATE, 0.05, 0.65, _v(v.is_valid_dob), "valid date"),
)

# BSB followed by an account number: "063-123 1234 5678", "063 123 / 12345678",
# "BSB 063-123, account 12345678". The pair is much stronger evidence than
# either number alone.
_BSB_ACCOUNT_PAIR = re.compile(
    r"(?<![\d-])(?P<bsb>\d{3}[ -]?\d{3})(?![\d-])\s*[/,|]?\s*"
    r"(?:acc(?:oun)?t?\s*(?:no\.?|number)?:?\s*)?"
    r"(?P<acc>\d{4} \d{4}|\d{3} \d{3} \d{3}|\d{3} \d{3}|\d{6,10})(?!\d)",
    re.IGNORECASE)
PAIR_BONUS = 0.45

# A capitalised run right before a state ("Victoria Park WA 6100") or right
# after a street number ("233 Hay") is a place, not a person.
_STATE_AFTER = re.compile(r"^,?\s+(?:NSW|VIC|QLD|SA|WA|TAS|NT|ACT)\b", re.IGNORECASE)
_NUMBER_BEFORE = re.compile(r"\d[A-C]?\s$")
PLACE_PENALTY = 0.5


# ----------------------------------------------------------------------
# PERSON: capitalised words + gazetteer + cues
# ----------------------------------------------------------------------

_UP, _LO = "A-ZÀ-ÖØ-Þ", "a-zß-öø-ÿ"
_NAME_WORD = rf"[{_UP}][{_LO}]*(?:[{_UP}][{_LO}]+)?(?:['’-][{_UP}{_LO}][{_LO}]+)*"
_PARTICLE = r"(?:van|der|de|di|da|dela|von|du|le)"
_PERSON_RUN = re.compile(
    rf"(?<![\w@.'’-]){_NAME_WORD}(?:[ ](?:[{_UP}]\.[ ]|{_PARTICLE}[ ])*{_NAME_WORD})*(?![\w@'’])")
_CAPS_WORD = rf"(?:[{_UP}]['’][{_UP}]+|[{_UP}]{{2,}})(?:['’-][{_UP}]+)*"   # O'CONNOR, SMITH-JONES
_SURNAME_FIRST = re.compile(
    rf"(?<![\w'’]){_CAPS_WORD}(?:[ ]{_CAPS_WORD})*,[ ]{_NAME_WORD}(?![\w'’])")

# Common capitalised words that start sentences or label form fields.
# A generic English list, deliberately not copied from our templates.
STOPWORDS = frozenset("""
a an the this that these those my your our his her their its we you they he she it i
hi hey hello dear thanks thank regards cheers kind best please yes no ok okay mate
name full first last surname given employee customer agent applicant patient director
member contact emergency reference ref subject re fw fwd note notice reminder update
payment transfer booking supplier company registered sole trader bill ship order invoice
tax total gross net hours paid pay date start end mobile phone email address residential
current postal bank account acc medicare passport licence license number card id dob
amount due description receipt identity call reply save new meeting dial super expiry
claim service period valid until confirmed confirm from to for with and or but if
when where what who how why can could would should will is are was were be been
do don't does did not never also just all any some here there see find send check
let get make take attached please
mon tue wed thu fri sat sun monday tuesday wednesday thursday friday saturday sunday
january february march april may june july august september october november december
jan feb mar apr jun jul aug sep sept oct nov dec
pty ltd limited inc co group services australia australian vic nsw qld sa wa tas nt act
unit level apt
""".split()) | frozenset(t.lower() for pair in STREET_TYPES for t in pair)


def _person_candidates(text: str, gazetteer: frozenset[str]) -> Iterable[Candidate]:
    for m in _SURNAME_FIRST.finditer(text):
        yield Candidate(m.start(), m.end(), "PERSON", 0.55, ["SURNAME, First"])

    for m in _PERSON_RUN.finditer(text):
        words = [(w.start() + m.start(), w.end() + m.start(), w.group())
                 for w in re.finditer(r"\S+", m.group())]
        while words and words[0][2].lower() in STOPWORDS:          # trim leading
            words.pop(0)
        while words and words[-1][2].lower() in STOPWORDS:         # trim trailing
            words.pop()
        names = [w for _, _, w in words if len(w) >= 2 and not w.endswith(".")
                 and w.lower() not in STOPWORDS and w[0].isupper()]
        if not names:
            continue
        hits = sum(w.lower() in gazetteer for w in names)
        score, reasons = 0.10, ["capitalised"]
        if hits:
            score += 0.35
            reasons.append(f"gazetteer x{hits}")
        if len(names) >= 2:
            score += 0.15
            reasons.append("2+ words")
        start, end = words[0][0], words[-1][1]
        if _STATE_AFTER.match(text[end:end + 6]) or _NUMBER_BEFORE.search(text[max(0, start - 3):start]):
            score -= PLACE_PENALTY
            reasons.append("looks like a place")
        yield Candidate(start, end, "PERSON", score, reasons)


# ----------------------------------------------------------------------
# The detector
# ----------------------------------------------------------------------

class RuleDetector:
    """detect(text) -> [{"start", "end", "label"}], like every other model."""

    def __init__(self, gazetteer: Iterable[str] = (),
                 thresholds: Optional[dict[str, float]] = None):
        self.gazetteer = frozenset(n.lower() for n in gazetteer)
        self.thresholds = dict(thresholds or {})
        self.cues = CueIndex()
        self.person_cue_bonus = 0.40

    # --- step 1-3: find and score candidates --------------------------
    def candidates(self, text: str) -> list[Candidate]:
        hits = sorted(self.cues.find(text), key=lambda h: (h.end, h.end - h.start))
        out: list[Candidate] = []

        for rec in RECOGNISERS:
            for m in rec.pattern.finditer(text):
                if rec.validator and not rec.validator(m):
                    continue
                c = Candidate(m.start(), m.end(), rec.label, rec.base, [rec.reason])
                self._context(c, hits, rec.cue_bonus, text)
                out.append(c)

        for c in _person_candidates(text, self.gazetteer):
            self._context(c, hits, self.person_cue_bonus, text)
            out.append(c)

        # pair rule: boost the BSB and the account; demote "account" readings
        # of the BSB half
        for m in _BSB_ACCOUNT_PAIR.finditer(text):
            for c in out:
                span = (c.start, c.end)
                if c.label == "AU_BSB" and span == m.span("bsb"):
                    c.score = min(1.0, c.score + PAIR_BONUS)
                    c.reasons.append("followed by an account number")
                elif c.label == "AU_BANK_ACCOUNT" and span == m.span("acc"):
                    c.score = min(1.0, c.score + PAIR_BONUS)
                    c.reasons.append("follows a BSB")
                elif c.label == "AU_BANK_ACCOUNT" and span == m.span("bsb"):
                    c.score = max(0.0, c.score - PAIR_BONUS)
                    c.reasons.append("is the BSB half of a BSB+account pair")
        return out

    @staticmethod
    def _context(c: Candidate, hits: list[CueHit], bonus: float, text: str) -> None:
        window = [h for h in hits if c.start - WINDOW <= h.end <= c.start]
        own = [h for h in window if h.label == c.label]
        if own:
            near = own[-1]
            dist = c.start - near.end
            c.score += bonus if dist <= 25 else bonus * 0.6
            c.reasons.append(f"cue '{near.cue}' ({dist} chars back)")
        if window:
            nearest = window[-1]
            if nearest.label == NEG:
                c.score -= NEG_CUE_PENALTY
                c.reasons.append(f"negative cue '{nearest.cue}'")
            elif nearest.label != c.label and c.label != "AU_ADDRESS":
                # (an address validates itself via state + postcode, and
                # usually follows a name, so a PERSON cue says nothing)
                c.score -= OTHER_CUE_PENALTY
                c.reasons.append(f"nearest cue '{nearest.cue}' is {nearest.label}")
        c.score = round(max(0.0, min(1.0, c.score)), 4)

    # --- step 4-5: threshold and resolve overlaps ----------------------
    def select(self, cands: list[Candidate],
               thresholds: Optional[dict[str, float]] = None) -> list[dict]:
        th = self.thresholds if thresholds is None else thresholds
        keep = [c for c in cands if c.score >= th.get(c.label, DEFAULT_THRESHOLD)]
        keep.sort(key=lambda c: (-c.score, -(c.end - c.start), c.start))
        chosen: list[Candidate] = []
        for c in keep:
            if all(c.end <= o.start or o.end <= c.start for o in chosen):
                chosen.append(c)
        return [c.entity() for c in sorted(chosen, key=lambda c: c.start)]

    def detect(self, text: str) -> list[dict]:
        return self.select(self.candidates(text))

    def explain(self, text: str) -> str:
        """Every candidate, its score, reasons, and whether it was kept."""
        cands = self.candidates(text)
        kept = {(e["start"], e["end"], e["label"]) for e in self.select(cands)}
        lines = []
        for c in sorted(cands, key=lambda c: (c.start, -c.score)):
            mark = "KEEP" if (c.start, c.end, c.label) in kept else "    "
            th = self.thresholds.get(c.label, DEFAULT_THRESHOLD)
            lines.append(f"{mark} {c.score:.2f}/{th:.2f} {c.label:<18} {text[c.start:c.end]!r:<30} "
                         + "; ".join(c.reasons))
        return "\n".join(lines)
