"""Level 2 tests: annotations are exact, splits don't leak, data is clean."""

import re

import pytest

from aupii import validators as v
from aupii.labels import LABELS
from aupii.synth import SLOT, DISTRACTORS, Renderer, safe_number, template_labels
from aupii.templates import ALL_TEMPLATES, NEGATIVE_TEMPLATES, PII_TEMPLATES
from build_dataset import SPLITS, build

N_PER_TEMPLATE = 150


@pytest.fixture(scope="module")
def docs():
    r = Renderer(seed=99)
    return [r.render(t) for t in ALL_TEMPLATES for _ in range(N_PER_TEMPLATE)]


@pytest.fixture(scope="module")
def dataset():
    data, manifest = build({"train": 1500, "val": 400, "test": 400}, seed=5,
                           negative_ratio=0.15, fractions=(0.6, 0.2, 0.2))
    return data, manifest


# --- spans ---------------------------------------------------------------

def test_every_span_matches_its_text(docs):
    for d in docs:
        for e in d["entities"]:
            assert d["text"][e["start"]:e["end"]] == e["text"]


def test_spans_sorted_and_non_overlapping(docs):
    for d in docs:
        ends = [-1] + [e["end"] for e in d["entities"]]
        starts = [e["start"] for e in d["entities"]]
        assert all(s >= prev for s, prev in zip(starts, ends)), d


def test_spans_have_no_padding(docs):
    for d in docs:
        for e in d["entities"]:
            assert e["text"] == e["text"].strip(), e


def test_labelled_identifiers_still_validate(docs):
    # Noise must never touch an identifier, so every checksum still holds.
    for d in docs:
        for e in d["entities"]:
            if e["label"] in ("AU_TFN", "AU_ABN", "AU_ACN", "AU_MEDICARE", "AU_PHONE"):
                assert v.VALIDATORS[e["label"]](e["text"]), e


# --- nothing valid is left unlabelled ----------------------------------

_DIGIT_RUN = re.compile(r"[0-9](?:[0-9]|[ -](?=[0-9]))*")


def test_no_unlabelled_checksum_identifiers(docs):
    """Any digit run that passes a checksum must be inside a labelled span.

    Otherwise we'd be teaching the model that a real-looking TFN is NOT PII.
    """
    checks = (v.is_valid_tfn, v.is_valid_abn, v.is_valid_acn, v.is_valid_medicare)
    for d in docs:
        for m in _DIGIT_RUN.finditer(d["text"]):
            if any(c(m.group()) for c in checks):
                covered = any(e["start"] <= m.start() < e["end"] or
                              m.start() <= e["start"] < m.end() for e in d["entities"])
                assert covered, (d["template_id"], m.group(), d["text"])


def test_safe_numbers_never_pass_a_checksum():
    import random
    rng = random.Random(0)
    for n in (9, 10, 11):
        for _ in range(3000):
            s = safe_number(rng, n)
            assert not (v.is_valid_tfn(s) or v.is_valid_abn(s) or
                        v.is_valid_acn(s) or v.is_valid_medicare(s) or v.is_valid_phone(s))


# --- templates ------------------------------------------------------------

def test_negative_templates_produce_no_entities(docs):
    neg_ids = {t.id for t in NEGATIVE_TEMPLATES}
    for d in docs:
        if d["template_id"] in neg_ids:
            assert d["entities"] == [], d


def test_every_slot_is_known():
    for t in ALL_TEMPLATES:
        for m in SLOT.finditer(t.text):
            name = m.group(1)
            assert name in LABELS or name == "FIRST_NAME" or name in DISTRACTORS, (t.id, name)


def test_template_ids_unique():
    ids = [t.id for t in ALL_TEMPLATES]
    assert len(ids) == len(set(ids))


def test_same_person_is_consistent():
    # {PERSON}, {FIRST_NAME} and {EMAIL} must describe one person.
    r = Renderer(seed=1)
    t = next(t for t in PII_TEMPLATES if t.id == "tax_01")
    for _ in range(100):
        d = r.render(t, noise=False)
        person = next(e["text"] for e in d["entities"] if e["label"] == "PERSON")
        first = [e["text"] for e in d["entities"] if e["label"] == "PERSON"][1]
        assert person.startswith(first) or person.lower().endswith(first.lower())


# --- splits ---------------------------------------------------------------

def test_no_template_leaks_across_splits(dataset):
    _, manifest = dataset
    t = manifest["templates"]
    assert not set(t["train"]) & set(t["val"])
    assert not set(t["train"]) & set(t["test"])
    assert not set(t["val"]) & set(t["test"])


def test_every_label_in_every_split(dataset):
    data, _ = dataset
    for split in SPLITS:
        seen = {e["label"] for d in data[split] for e in d["entities"]}
        assert set(LABELS) <= seen, (split, set(LABELS) - seen)


def test_build_is_reproducible():
    a, _ = build({"train": 200, "val": 50, "test": 50}, 21, 0.15, (0.6, 0.2, 0.2))
    b, _ = build({"train": 200, "val": 50, "test": 50}, 21, 0.15, (0.6, 0.2, 0.2))
    assert a == b


def test_negative_ratio_roughly_respected(dataset):
    data, _ = dataset
    share = sum(not d["entities"] for d in data["train"]) / len(data["train"])
    assert 0.08 < share < 0.25
