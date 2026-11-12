"""Tests for validators.py using hand-checked, publicly documented examples."""

from datetime import date

import pytest

from aupii import validators as v


# --- TFN ---------------------------------------------------------------

@pytest.mark.parametrize("value", ["123 456 782", "123456782", "123-456-782"])
def test_tfn_valid(value):
    assert v.is_valid_tfn(value)


@pytest.mark.parametrize("value", [
    "123 456 789",      # wrong checksum
    "123 456 78",       # too short
    "1234567820",       # too long
    "12A 456 782",      # letter
    "000 000 000",      # passes the maths but isn't a real TFN
    "１２３４５６７８２",  # fullwidth digits - isdigit() would let these through
])
def test_tfn_invalid(value):
    assert not v.is_valid_tfn(value)


# --- ABN ---------------------------------------------------------------

@pytest.mark.parametrize("value", ["51 824 753 556", "51824753556"])
def test_abn_valid(value):
    assert v.is_valid_abn(value)


@pytest.mark.parametrize("value", ["51 824 753 557", "01 824 753 556", "5182475355"])
def test_abn_invalid(value):
    assert not v.is_valid_abn(value)


def test_abn_worked_example():
    # 4*10 + 1*1 + 8*3 + 2*5 + 4*7 + 7*9 + 5*11 + 3*13 + 5*15 + 5*17 + 6*19 = 534 = 89*6
    digits = [int(c) for c in "51824753556"]
    assert v.abn_remainder(digits) == 0


# --- ACN ---------------------------------------------------------------

@pytest.mark.parametrize("value", ["000 000 019", "004 085 616", "010 499 966"])
def test_acn_valid(value):
    assert v.is_valid_acn(value)


@pytest.mark.parametrize("value", ["000 000 018", "004 085 617", "000 000 000"])
def test_acn_invalid(value):
    assert not v.is_valid_acn(value)


def test_acn_weakness():
    # mod 10 misses some typos: the 7th digit has weight 2, so changing it
    # by 5 shifts the sum by 2*5 = 10, which is 0 mod 10. The typo slips through.
    assert v.is_valid_acn("004 085 616")
    assert v.is_valid_acn("004 085 116")   # 7th digit 6 -> 1


# --- Medicare ----------------------------------------------------------

@pytest.mark.parametrize("value", ["2123 45670 1", "2123456701", "21234567011", "2123 45670 1/1"])
def test_medicare_valid(value):
    assert v.is_valid_medicare(value)


@pytest.mark.parametrize("value", [
    "2123 45671 1",   # wrong check digit
    "7123 45670 1",   # first digit must be 2-6
    "2123 45670 0",   # issue number 0
    "21234567010",    # IRN 0
    "2123 4567 1",    # too short
])
def test_medicare_invalid(value):
    assert not v.is_valid_medicare(value)


# --- Format-only -------------------------------------------------------

def test_bsb():
    assert v.is_valid_bsb("062-000")
    assert v.is_valid_bsb("062000")
    assert not v.is_valid_bsb("06200")


def test_bank_account():
    assert v.is_valid_bank_account("1234 5678")
    assert not v.is_valid_bank_account("12345")


@pytest.mark.parametrize("value,ok", [
    ("PA1234567", True), ("N1234567", True), ("pa1234567", True),
    ("P123456", False), ("PAB1234567", False), ("1234567", False),
])
def test_passport(value, ok):
    assert v.is_valid_passport(value) is ok


def test_driver_licence_by_state():
    assert v.is_valid_driver_licence("S12345", "SA")
    assert not v.is_valid_driver_licence("12345678", "WA")   # WA is 7 digits
    assert v.is_valid_driver_licence("1234567", "WA")


@pytest.mark.parametrize("value,ok", [
    ("0412 345 678", True), ("0412345678", True), ("+61 412 345 678", True),
    ("(03) 9876 5432", True), ("+61 3 9876 5432", True), ("+61 (0)3 9876 5432", True),
    ("0512 345 678", False), ("041234567", False), ("1300 123 456", False),
])
def test_phone(value, ok):
    assert v.is_valid_phone(value) is ok


@pytest.mark.parametrize("postcode,state", [
    ("3000", "VIC"), ("2000", "NSW"), ("2600", "ACT"), ("2620", "NSW"),
    ("0800", "NT"), ("7000", "TAS"), ("6000", "WA"), ("0100", None), ("300", None),
])
def test_postcode_state(postcode, state):
    assert v.postcode_state(postcode) == state


def test_email():
    assert v.is_valid_email("priya.nguyen@example.com")
    assert v.is_valid_email("a_b+tag@mail.yahoo.com.au")
    assert not v.is_valid_email("priya..nguyen@example.com")
    assert not v.is_valid_email("priya@example")


def test_date_is_day_first():
    # The Australian reading of 03/04/1990 is 3 April.
    assert v.parse_date("03/04/1990") == date(1990, 4, 3)
    assert v.parse_date("3 April 1990") == date(1990, 4, 3)


def test_dob():
    today = date(2026, 11, 11)
    assert v.is_valid_dob("03/04/1990", today)
    assert not v.is_valid_dob("31/02/1990", today)   # no 31 February
    assert not v.is_valid_dob("01/01/2030", today)   # future
    assert not v.is_valid_dob("01/01/1850", today)   # older than 120
