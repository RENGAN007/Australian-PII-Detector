"""Level 4 tests: each recogniser on tricky strings, plus detector behaviour."""

import pytest

from aupii.rules import CueIndex, RuleDetector, osa_distance
from build_dataset import build

# Thresholds roughly like the tuned ones, fixed here so tests don't depend
# on results/ files.
TH = {"AU_TFN": 0.4, "AU_ABN": 0.5, "AU_ACN": 0.45, "AU_MEDICARE": 0.3, "AU_BSB": 0.3,
      "AU_BANK_ACCOUNT": 0.35, "AU_PASSPORT": 0.5, "AU_DRIVER_LICENCE": 0.4,
      "AU_PHONE": 0.15, "AU_ADDRESS": 0.2, "PERSON": 0.15, "EMAIL": 0.45,
      "DATE_OF_BIRTH": 0.05}


@pytest.fixture(scope="module")
def det():
    return RuleDetector(gazetteer=["Priya", "Nguyen", "Liam"], thresholds=TH)


def found(det, text):
    return [(e["label"], text[e["start"]:e["end"]]) for e in det.detect(text)]


# --- fuzzy cues -----------------------------------------------------------

@pytest.mark.parametrize("a,b,d", [
    ("number", "nubmer", 1),      # one swap
    ("number", "numbers", 1),     # one insertion
    ("medicare", "mdeicare", 1),
    ("passport", "pasport", 1),
    ("abc", "xyz", 3),
    ("", "abc", 3),
])
def test_osa_distance(a, b, d):
    assert osa_distance(a, b) == d


def test_cue_index_tolerates_typos():
    hits = CueIndex().find("Mdeicare No: 2123 45670 1")
    assert any(h.label == "AU_MEDICARE" for h in hits)


def test_longer_phrase_wins_on_ties():
    hits = CueIndex().find("account number 12345678")
    last = sorted(hits, key=lambda h: (h.end, h.end - h.start))[-1]
    assert last.cue == "account number" and last.label == "AU_BANK_ACCOUNT"


# --- checksum identifiers --------------------------------------------------

def test_tfn_with_cue(det):
    assert found(det, "TFN: 123 456 782") == [("AU_TFN", "123 456 782")]


def test_invalid_checksum_is_ignored(det):
    assert found(det, "TFN: 123 456 789") == []


def test_negative_cue_suppresses(det):
    # valid TFN checksum, but it's introduced as an invoice number
    assert found(det, "Invoice no: 123456782") == []


def test_abn_does_not_leak_a_tfn_inside(det):
    assert found(det, "ABN 51 824 753 556") == [("AU_ABN", "51 824 753 556")]


def test_no_bsb_inside_a_nine_digit_number(det):
    cands = det.candidates("Ref 735 076 865 and 419-323-952")
    assert not [c for c in cands if c.label == "AU_BSB"]


def test_medicare_with_irn(det):
    assert found(det, "Medicare No: 2123 45670 1/2") == [("AU_MEDICARE", "2123 45670 1/2")]


# --- formats ----------------------------------------------------------------

@pytest.mark.parametrize("phone", ["+61 412 345 678", "(03) 9876 5432", "0412345678",
                                   "+61 3 9876 5432"])
def test_phones(det, phone):
    assert found(det, f"Call me on {phone} today") == [("AU_PHONE", phone)]


def test_email(det):
    assert found(det, "email: a.b@example.com.") == [("EMAIL", "a.b@example.com")]


def test_bsb_account_pair(det):
    assert found(det, "Pay to 063-123 1234 5678 thanks") == [
        ("AU_BSB", "063-123"), ("AU_BANK_ACCOUNT", "1234 5678")]


def test_passport(det):
    assert found(det, "Passport PA1234567") == [("AU_PASSPORT", "PA1234567")]
    assert found(det, "Code REF1234567") == []


def test_licence_needs_its_cue(det):
    assert found(det, "Licence no: 12345678") == [("AU_DRIVER_LICENCE", "12345678")]
    assert found(det, "Order 12345678") == []


def test_dob_needs_its_cue(det):
    assert found(det, "DOB: 03/04/1990") == [("DATE_OF_BIRTH", "03/04/1990")]
    assert found(det, "Date: 03/04/2025") == []


# --- addresses ----------------------------------------------------------------

def test_full_address(det):
    text = "Ship to Unit 4, 12 High St, Richmond VIC 3121 please"
    assert ("AU_ADDRESS", "Unit 4, 12 High St, Richmond VIC 3121") in found(det, text)


def test_address_postcode_must_match_state(det):
    assert not [x for x in found(det, "12 High St, Richmond NSW 3121") if x[0] == "AU_ADDRESS"]


def test_place_words_are_not_people(det):
    text = "Bill to: Jo Lee, 233 Hay Crescent, Victoria Park WA 6100"
    labels = found(det, text)
    assert ("AU_ADDRESS", "233 Hay Crescent, Victoria Park WA 6100") in labels
    assert ("PERSON", "Hay") not in labels


# --- people -------------------------------------------------------------------

def test_person_after_cue(det):
    assert found(det, "Dear Priya Nguyen, thanks") == [("PERSON", "Priya Nguyen")]


def test_surname_first_with_apostrophe(det):
    assert found(det, "Name: O'CONNOR, Kylie") == [("PERSON", "O'CONNOR, Kylie")]


def test_leading_stopwords_are_trimmed(det):
    assert found(det, "Hi Liam, see you") == [("PERSON", "Liam")]


# --- detector contract -----------------------------------------------------

def test_output_is_sorted_and_non_overlapping(det):
    data, _ = build({"train": 50, "val": 50, "test": 200}, 3, 0.15, (0.6, 0.2, 0.2))
    for doc in data["test"]:
        ents = det.detect(doc["text"])
        assert all(set(e) == {"start", "end", "label"} for e in ents)
        assert all(a["end"] <= b["start"] for a, b in zip(ents, ents[1:]))


def test_explain_marks_kept_candidates(det):
    out = det.explain("TFN: 123 456 782")
    assert "KEEP" in out and "checksum" in out and "cue 'tfn'" in out
