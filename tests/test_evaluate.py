"""Level 4 tests: the evaluation maths, checked against hand-worked examples."""

import pytest

from aupii.evaluate import Scores, evaluate, snippet


def E(start, end, label):
    return {"start": start, "end": end, "label": label}


def test_perfect_prediction():
    gold = [[E(0, 5, "PERSON"), E(10, 21, "AU_TFN")]]
    s = evaluate(gold, gold)
    assert s.micro() == {"precision": 1.0, "recall": 1.0, "f1": 1.0, "relaxed_f1": 1.0}
    assert s.categories == {"exact": 2}


def test_hand_worked_example():
    # gold: TFN (exact hit), PERSON (boundary off), ABN (missed entirely)
    # pred: TFN exact, PERSON partial, PHONE spurious
    gold = [E(0, 11, "AU_TFN"), E(20, 32, "PERSON"), E(40, 54, "AU_ABN")]
    pred = [E(0, 11, "AU_TFN"), E(20, 26, "PERSON"), E(60, 70, "AU_PHONE")]
    s = evaluate([gold], [pred])

    # strict: TP=1, FP=2, FN=2 -> P=1/3, R=1/3, F1=1/3
    m = s.micro()
    assert m["precision"] == pytest.approx(1 / 3)
    assert m["recall"] == pytest.approx(1 / 3)
    assert m["f1"] == pytest.approx(1 / 3)
    # relaxed: the PERSON overlap also counts -> P=2/3, R=2/3
    assert m["relaxed_f1"] == pytest.approx(2 / 3)
    assert s.categories == {"exact": 1, "partial": 1, "spurious": 1, "missed": 1}


def test_wrong_label_is_recorded_as_confusion():
    s = evaluate([[E(0, 9, "AU_ACN")]], [[E(0, 9, "AU_TFN")]])
    assert s.categories == {"wrong_label": 1}
    assert s.confusion == {("AU_ACN", "AU_TFN"): 1}
    assert s.fp["AU_TFN"] == 1 and s.fn["AU_ACN"] == 1


def test_macro_punishes_a_failed_rare_label():
    # 9 PERSON all correct, 1 PASSPORT missed
    gold = [[E(i * 10, i * 10 + 5, "PERSON") for i in range(9)] + [E(100, 109, "AU_PASSPORT")]]
    pred = [[E(i * 10, i * 10 + 5, "PERSON") for i in range(9)]]
    s = evaluate(gold, pred)
    assert s.micro()["f1"] == pytest.approx(2 * 1.0 * 0.9 / 1.9)   # ~0.947
    assert s.macro_f1() == pytest.approx(0.5)                     # (1.0 + 0.0) / 2


def test_no_predictions_no_crash():
    s = evaluate([[E(0, 5, "PERSON")]], [[]])
    assert s.micro()["f1"] == 0.0
    assert s.categories == {"missed": 1}


def test_empty_documents():
    s = evaluate([[], []], [[], []])
    assert s.micro()["f1"] == 0.0
    assert s.per_label() == {}


def test_lengths_must_match():
    with pytest.raises(ValueError):
        evaluate([[], []], [[]])


def test_per_label_counts():
    gold = [[E(0, 5, "PERSON"), E(10, 15, "PERSON")]]
    pred = [[E(0, 5, "PERSON"), E(20, 25, "PERSON")]]
    per = evaluate(gold, pred).per_label()["PERSON"]
    assert (per["tp"], per["fp"], per["fn"], per["support"]) == (1, 1, 1, 2)


def test_add_returns_errors_for_analysis():
    errs = Scores().add([E(0, 5, "PERSON")], [E(8, 10, "EMAIL")])
    assert sorted(e["kind"] for e in errs) == ["missed", "spurious"]


def test_snippet():
    assert snippet("my TFN is 123 456 782 ok", 10, 21, width=7) == "...TFN is [123 456 782] ok..."
