"""Property tests: generate many values and check they behave as promised."""

import re

import pytest

from aupii import validators as v
from aupii.generators import PIIGenerator
from aupii.labels import CHECKSUM_LABELS, LABELS

N = 2000


@pytest.fixture
def gen():
    return PIIGenerator(seed=1234)


@pytest.mark.parametrize("label", list(v.VALIDATORS))
def test_every_generated_value_validates(gen, label):
    validator = v.VALIDATORS[label]
    for _ in range(N):
        value = gen.generate(label)
        assert validator(value), f"{label}: {value!r} failed validation"


@pytest.mark.parametrize("label", CHECKSUM_LABELS)
def test_hard_negatives_fail_validation(gen, label):
    validator = v.VALIDATORS[label]
    for _ in range(N):
        bad = gen.invalid(label)
        assert not validator(bad), f"{label}: {bad!r} should be invalid"


@pytest.mark.parametrize("label", CHECKSUM_LABELS)
def test_hard_negatives_keep_the_formatting(gen, label):
    # Only digits change, so separators stay where they were.
    for _ in range(200):
        good = gen.generate(label)
        bad = gen.corrupt(good, v.VALIDATORS[label])
        assert len(bad) == len(good)
        assert re.sub(r"\d", "#", bad) == re.sub(r"\d", "#", good)


def test_medicare_with_irn(gen):
    for _ in range(N):
        value = gen.medicare(include_irn=True)
        assert v.is_valid_medicare(value)
        assert len(v.normalise(value)) == 11


def test_same_seed_same_data():
    a, b = PIIGenerator(seed=42), PIIGenerator(seed=42)
    for label in LABELS:
        assert [a.generate(label) for _ in range(20)] == [b.generate(label) for _ in range(20)]


def test_different_seed_different_data():
    a, b = PIIGenerator(seed=1), PIIGenerator(seed=2)
    assert [a.tfn() for _ in range(20)] != [b.tfn() for _ in range(20)]


@pytest.mark.parametrize("style,pattern", [
    ("plain", r"\d{9}"), ("spaced", r"\d{3} \d{3} \d{3}"), ("hyphen", r"\d{3}-\d{3}-\d{3}"),
])
def test_tfn_styles(gen, style, pattern):
    assert re.fullmatch(pattern, gen.tfn(style))


def test_unknown_style_raises(gen):
    with pytest.raises(ValueError):
        gen.tfn("dotted")


@pytest.mark.parametrize("state", list(v.DRIVER_LICENCE_PATTERNS))
def test_licence_matches_its_state(gen, state):
    for _ in range(200):
        assert v.is_valid_driver_licence(gen.driver_licence(state), state)


def test_fictional_landline_uses_reserved_block(gen):
    for _ in range(200):
        number = gen.phone("landline", "plain", fictional=True)
        assert number[2:6] == "5550"


def test_address_postcode_matches_state(gen):
    for _ in range(500):
        addr = gen.address()
        m = re.search(r"\b(NSW|VIC|QLD|SA|WA|TAS|NT|ACT),? (\d{4})$", addr)
        assert m, addr
        assert v.postcode_state(m.group(2)) == m.group(1)


def test_email_matches_given_name(gen):
    email = gen.email("Priya", "O'Brien")
    assert "obrien" in email or "priya" in email
    assert v.is_valid_email(email)


def test_generate_covers_every_label(gen):
    for label in LABELS:
        assert gen.generate(label)
