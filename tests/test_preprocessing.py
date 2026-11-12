"""Level 3 tests: tokeniser offsets, BIO conversion, round trips, windows."""

import pytest

from aupii.bio import (ID2LABEL, LABEL2ID, TAGS, bio_to_bioes, bio_to_spans, bioes_to_bio,
                       is_valid_bio, repair_bio, spans_to_bio, windows)
from aupii.labels import LABELS
from aupii.tokenizer import tokenize, word_shape
from build_dataset import build
from prepare_data import process


@pytest.fixture(scope="module")
def docs():
    data, _ = build({"train": 2000, "val": 500, "test": 500}, seed=8,
                    negative_ratio=0.15, fractions=(0.6, 0.2, 0.2))
    return [d for split in data.values() for d in split]


def toks(text):
    return [t.text for t in tokenize(text)]


# --- tokeniser ------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("+61 412 345 678", ["+", "61", "412", "345", "678"]),
    ("(03) 9876 5432", ["(", "03", ")", "9876", "5432"]),
    ("123-456-782", ["123", "-", "456", "-", "782"]),
    ("TFN:123456782,", ["TFN", ":", "123456782", ","]),
    ("O'Brien", ["O'Brien"]),
    ("mail priya.nguyen@example.com.", ["mail", "priya.nguyen@example.com", "."]),
    ("DOB 08.04.1963", ["DOB", "08.04.1963"]),
    ("2123 45670 1/2", ["2123", "45670", "1", "/", "2"]),
    ("PA1234567", ["PA1234567"]),
    ("line one\nline two", ["line", "one", "line", "two"]),
])
def test_tokenize_cases(text, expected):
    assert toks(text) == expected


def test_offsets_match_text_everywhere(docs):
    for d in docs:
        for t in tokenize(d["text"]):
            assert d["text"][t.start:t.end] == t.text


def test_tokens_cover_every_non_space_character(docs):
    for d in docs:
        covered = sorted(i for t in tokenize(d["text"]) for i in range(t.start, t.end))
        non_space = [i for i, c in enumerate(d["text"]) if not c.isspace()]
        assert covered == non_space


def test_nfkc_norm_keeps_raw_offsets():
    text = "TFN １２３ ４５６ ７８２"
    tokens = tokenize(text)
    assert [t.norm for t in tokens] == ["TFN", "123", "456", "782"]
    assert text[tokens[1].start:tokens[1].end] == "１２３"


def test_word_shape():
    assert word_shape("PA1234567") == "XXddddddd"
    assert word_shape("O'Brien") == "X'Xxxxx"


# --- label maps ---------------------------------------------------------------

def test_tag_set():
    assert len(TAGS) == 2 * len(LABELS) + 1 == 27
    assert TAGS[0] == "O"
    assert all(ID2LABEL[LABEL2ID[t]] == t for t in TAGS)


# --- spans <-> BIO ------------------------------------------------------------

def test_full_dataset_round_trip(docs):
    for d in docs:
        _, problems = process(d)
        assert problems == [], problems


def test_generated_tags_are_valid_bio(docs):
    for d in docs:
        tags, _ = spans_to_bio(tokenize(d["text"]), d["entities"])
        assert is_valid_bio(tags)


def test_adjacent_same_type_entities_stay_separate():
    # Two phone numbers back to back: plain IO tagging would merge them.
    text = "0412 345 678 0498 765 432"
    ents = [{"start": 0, "end": 12, "label": "AU_PHONE"},
            {"start": 13, "end": 25, "label": "AU_PHONE"}]
    tokens = tokenize(text)
    tags, problems = spans_to_bio(tokens, ents)
    assert not problems
    assert tags == ["B-AU_PHONE", "I-AU_PHONE", "I-AU_PHONE",
                    "B-AU_PHONE", "I-AU_PHONE", "I-AU_PHONE"]
    assert [(s["start"], s["end"]) for s in bio_to_spans(tokens, tags)] == [(0, 12), (13, 25)]


def test_misaligned_entity_is_reported():
    # "abc123456782" is one token, but only "123456782" is the TFN.
    text = "ref abc123456782 ok"
    ent = [{"start": 7, "end": 16, "label": "AU_TFN"}]
    tokens = tokenize(text)

    tags, problems = spans_to_bio(tokens, ent, mode="strict")
    assert tags == ["O", "O", "O"]
    assert problems[0]["type"] == "misaligned"

    tags, problems = spans_to_bio(tokens, ent, mode="expand")
    assert tags == ["O", "B-AU_TFN", "O"]
    assert problems[0]["type"] == "misaligned"


def test_overlapping_entities_are_reported():
    text = "Priya Nguyen"
    ents = [{"start": 0, "end": 12, "label": "PERSON"},
            {"start": 6, "end": 12, "label": "PERSON"}]
    _, problems = spans_to_bio(tokenize(text), ents)
    assert [p["type"] for p in problems] == ["overlap"]


def test_decoder_is_lenient_with_stray_I():
    tokens = tokenize("a b c d")
    tags = ["O", "I-AU_TFN", "I-AU_TFN", "I-AU_ABN"]
    spans = bio_to_spans(tokens, tags, "a b c d")
    assert [(s["label"], s["text"]) for s in spans] == [("AU_TFN", "b c"), ("AU_ABN", "d")]


def test_validate_and_repair():
    bad = ["O", "I-AU_TFN", "I-AU_TFN", "B-PERSON", "I-EMAIL"]
    assert not is_valid_bio(bad)
    fixed = repair_bio(bad)
    assert fixed == ["O", "B-AU_TFN", "I-AU_TFN", "B-PERSON", "B-EMAIL"]
    assert is_valid_bio(fixed)
    assert not is_valid_bio(["B-NOT_A_LABEL"])


def test_bioes_round_trip(docs):
    for d in docs[:500]:
        tags, _ = spans_to_bio(tokenize(d["text"]), d["entities"])
        assert bioes_to_bio(bio_to_bioes(tags)) == tags
    assert bio_to_bioes(["B-EMAIL", "O", "B-AU_TFN", "I-AU_TFN", "I-AU_TFN"]) == \
        ["S-EMAIL", "O", "B-AU_TFN", "I-AU_TFN", "E-AU_TFN"]


# --- windows ----------------------------------------------------------------

def test_short_sequence_is_one_window():
    assert windows(30, 128, 96) == [(0, 30)]


@pytest.mark.parametrize("n,max_len,stride", [(300, 128, 96), (129, 128, 1), (1000, 64, 64)])
def test_windows_cover_everything(n, max_len, stride):
    w = windows(n, max_len, stride)
    assert w[0][0] == 0 and w[-1][1] == n
    assert all(e - s <= max_len for s, e in w)
    assert all(nxt[0] <= cur[1] for cur, nxt in zip(w, w[1:]))   # no gaps


def test_entity_within_overlap_fits_in_some_window():
    n, max_len, stride = 300, 128, 96          # overlap = 32 tokens
    w = windows(n, max_len, stride)
    for start in range(n - 32):
        assert any(s <= start and start + 32 <= e for s, e in w)
