"""Level 4 - entity-level evaluation, shared by every model from here on.

Every detector, rules or ML, implements the same interface:

    detector.detect(text) -> [{"start": int, "end": int, "label": str}, ...]

which is exactly the format of the dataset's "entities". So one function
scores them all, and Level 8 can compare them fairly.

Two scoring modes:
  strict  - a prediction counts only if start, end AND label all match.
            This is the headline number.
  relaxed - a prediction counts if it overlaps a gold entity with the
            same label (boundaries may be off). The gap between strict
            and relaxed tells you how many errors are just boundary errors.

Each prediction is also put in one category, for error analysis:
  exact       - perfect match
  partial     - right label, overlapping but wrong boundaries
  wrong_label - overlaps a gold entity of a different label
  spurious    - overlaps nothing (a pure false positive)
and each gold entity that nothing overlaps is "missed".
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Protocol, Sequence

from .labels import LABELS


class Detector(Protocol):
    def detect(self, text: str) -> list[dict]: ...


def _key(e: dict) -> tuple[int, int, str]:
    return e["start"], e["end"], e["label"]


def _overlaps(a: dict, b: dict) -> bool:
    return a["start"] < b["end"] and b["start"] < a["end"]


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


@dataclass
class Scores:
    tp: Counter = field(default_factory=Counter)          # strict, per label
    fp: Counter = field(default_factory=Counter)
    fn: Counter = field(default_factory=Counter)
    relaxed_tp_pred: Counter = field(default_factory=Counter)
    relaxed_tp_gold: Counter = field(default_factory=Counter)
    categories: Counter = field(default_factory=Counter)  # exact/partial/...
    confusion: Counter = field(default_factory=Counter)   # (gold, pred) label pairs

    # ------------------------------------------------------------------
    def add(self, gold: Sequence[dict], pred: Sequence[dict]) -> list[dict]:
        """Score one document. Returns its errors (for error analysis)."""
        gold_keys = {_key(g) for g in gold}
        pred_keys = {_key(p) for p in pred}
        errors: list[dict] = []

        for p in pred:
            if _key(p) in gold_keys:
                self.tp[p["label"]] += 1
                self.categories["exact"] += 1
                continue
            self.fp[p["label"]] += 1
            overlapping = [g for g in gold if _overlaps(p, g)]
            if any(g["label"] == p["label"] for g in overlapping):
                kind = "partial"
            elif overlapping:
                kind = "wrong_label"
                self.confusion[(overlapping[0]["label"], p["label"])] += 1
            else:
                kind = "spurious"
            self.categories[kind] += 1
            errors.append({"kind": kind, "pred": p, "gold": overlapping[0] if overlapping else None})

        for g in gold:
            if _key(g) in pred_keys:
                continue
            self.fn[g["label"]] += 1
            if not any(_overlaps(g, p) for p in pred):
                self.categories["missed"] += 1
                errors.append({"kind": "missed", "pred": None, "gold": g})

        # relaxed: overlap + same label counts (each side counted once)
        for p in pred:
            if any(_overlaps(p, g) and g["label"] == p["label"] for g in gold):
                self.relaxed_tp_pred[p["label"]] += 1
        for g in gold:
            if any(_overlaps(g, p) and g["label"] == p["label"] for p in pred):
                self.relaxed_tp_gold[g["label"]] += 1
        return errors

    # ------------------------------------------------------------------
    def labels(self) -> list[str]:
        seen = set(self.tp) | set(self.fp) | set(self.fn)
        return [lab for lab in LABELS if lab in seen]

    def per_label(self) -> dict[str, dict]:
        out = {}
        for lab in self.labels():
            tp, fp, fn = self.tp[lab], self.fp[lab], self.fn[lab]
            p, r, f = _prf(tp, fp, fn)
            n_pred, n_gold = tp + fp, tp + fn
            rp = self.relaxed_tp_pred[lab] / n_pred if n_pred else 0.0
            rr = self.relaxed_tp_gold[lab] / n_gold if n_gold else 0.0
            rf = 2 * rp * rr / (rp + rr) if rp + rr else 0.0
            out[lab] = {"precision": p, "recall": r, "f1": f, "relaxed_f1": rf,
                        "support": n_gold, "tp": tp, "fp": fp, "fn": fn}
        return out

    def micro(self) -> dict:
        tp, fp, fn = sum(self.tp.values()), sum(self.fp.values()), sum(self.fn.values())
        p, r, f = _prf(tp, fp, fn)
        n_pred, n_gold = tp + fp, tp + fn
        rp = sum(self.relaxed_tp_pred.values()) / n_pred if n_pred else 0.0
        rr = sum(self.relaxed_tp_gold.values()) / n_gold if n_gold else 0.0
        rf = 2 * rp * rr / (rp + rr) if rp + rr else 0.0
        return {"precision": p, "recall": r, "f1": f, "relaxed_f1": rf}

    def macro_f1(self) -> float:
        """Plain average of per-label F1 over labels that occur in gold.
        Every label counts equally, so rare labels can't hide."""
        per = self.per_label()
        vals = [m["f1"] for m in per.values() if m["support"] > 0]
        return sum(vals) / len(vals) if vals else 0.0

    def summary(self) -> dict:
        return {"micro": self.micro(), "macro_f1": self.macro_f1(),
                "per_label": self.per_label(), "categories": dict(self.categories),
                "confusion": {f"{g} -> {p}": n for (g, p), n in self.confusion.most_common()}}

    def table(self) -> str:
        lines = [f"{'label':<19}{'P':>7}{'R':>7}{'F1':>7}{'relaxF1':>9}{'support':>9}"]
        for lab, m in self.per_label().items():
            lines.append(f"{lab:<19}{m['precision']:>7.3f}{m['recall']:>7.3f}{m['f1']:>7.3f}"
                         f"{m['relaxed_f1']:>9.3f}{m['support']:>9}")
        mi = self.micro()
        lines.append("-" * 58)
        lines.append(f"{'micro avg':<19}{mi['precision']:>7.3f}{mi['recall']:>7.3f}{mi['f1']:>7.3f}"
                     f"{mi['relaxed_f1']:>9.3f}")
        lines.append(f"{'macro F1':<19}{'':>14}{self.macro_f1():>7.3f}")
        return "\n".join(lines)


def evaluate(gold_docs: Iterable[Sequence[dict]], pred_docs: Iterable[Sequence[dict]]) -> Scores:
    scores = Scores()
    for gold, pred in zip(gold_docs, pred_docs, strict=True):
        scores.add(gold, pred)
    return scores


def evaluate_detector(detector: Detector, docs: Sequence[dict]
                      ) -> tuple[Scores, list[dict]]:
    """Run a detector over dataset docs. Returns (scores, errors with context)."""
    scores, errors = Scores(), []
    for doc in docs:
        pred = detector.detect(doc["text"])
        for err in scores.add(doc["entities"], pred):
            err["doc_id"] = doc["id"]
            err["text"] = doc["text"]
            errors.append(err)
    return scores, errors


def snippet(text: str, start: int, end: int, width: int = 35) -> str:
    """Show an entity in context: '...TFN: [123 456 782] Bank...'"""
    left = text[max(0, start - width):start].replace("\n", " / ")
    right = text[end:end + width].replace("\n", " / ")
    return f"...{left}[{text[start:end]}]{right}..."
