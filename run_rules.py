"""Level 4 - tune the rule detector on VAL, then score it once on TEST.

    python run_rules.py                 # tune, evaluate, print errors
    python run_rules.py --errors 5      # show 5 example errors per kind
    python run_rules.py --explain val-000123

Writes results/rules_thresholds.json and results/rules_{val,test}.json,
which Level 8 will read when comparing every model.

Golden rule: thresholds are chosen on val only. The test set is used once,
at the end, to report the score. Tuning on test (even a little) makes the
test score an over-estimate.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from aupii.evaluate import Scores, evaluate_detector, snippet
from aupii.labels import LABELS
from aupii.rules import DEFAULT_THRESHOLD, RuleDetector

GRID = [round(0.05 * i, 2) for i in range(1, 20)]     # 0.05 ... 0.95


def load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def tune_thresholds(detector: RuleDetector, docs: list[dict], rounds: int = 2
                    ) -> tuple[dict[str, float], float]:
    """Coordinate ascent: for each label in turn, try every threshold on the
    grid, holding the others fixed.

    Often a whole RANGE of thresholds ties for the best val score (e.g. every
    TFN in val has a cue word, so 0.10 and 0.90 score the same). We then pick
    the MIDDLE of that range, the choice with the most margin on both sides,
    instead of an arbitrary end of it. That generalises better to test
    documents that look a bit different from val.

    Candidates are generated once and cached; only the cheap selection step
    is re-run, so this takes seconds.
    """
    cached = [(d["entities"], detector.candidates(d["text"])) for d in docs]

    def micro_f1(th: dict[str, float]) -> float:
        s = Scores()
        for gold, cands in cached:
            s.add(gold, detector.select(cands, th))
        return s.micro()["f1"]

    th = {lab: DEFAULT_THRESHOLD for lab in LABELS}
    for _ in range(rounds):
        for lab in LABELS:
            results = [(t, micro_f1({**th, lab: t})) for t in GRID]
            top = max(f for _, f in results)
            tied = [t for t, f in results if f >= top - 1e-9]
            th[lab] = tied[len(tied) // 2]
    return th, micro_f1(th)


def show_errors(errors: list[dict], per_kind: int) -> None:
    by_kind: dict[str, list[dict]] = defaultdict(list)
    for e in errors:
        by_kind[e["kind"]].append(e)
    for kind in ("missed", "spurious", "partial", "wrong_label"):
        items = by_kind.get(kind, [])
        print(f"\n--- {kind} ({len(items)}) ---")
        for e in items[:per_kind]:
            ent = e["pred"] or e["gold"]
            text = e["text"]
            if kind == "missed":
                desc = f"gold {e['gold']['label']}"
            elif kind == "spurious":
                desc = f"pred {e['pred']['label']}"
            else:
                g = e["gold"]
                desc = (f"pred {e['pred']['label']} {text[e['pred']['start']:e['pred']['end']]!r}"
                        f" vs gold {g['label']} {text[g['start']:g['end']]!r}")
            print(f"  {e['doc_id']:<13} {desc}\n      {snippet(text, ent['start'], ent['end'])}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--out", type=Path, default=Path("results"))
    p.add_argument("--errors", type=int, default=3, help="examples per error kind")
    p.add_argument("--explain", help="doc id to explain, e.g. val-000123")
    args = p.parse_args()

    manifest = json.loads((args.data / "manifest.json").read_text())
    gazetteer = manifest["first_names"]["train"] + manifest["last_names"]["train"]
    val, test = load(args.data / "val.jsonl"), load(args.data / "test.jsonl")
    detector = RuleDetector(gazetteer)

    print("Tuning thresholds on val ...")
    untuned, _ = evaluate_detector(detector, val)
    detector.thresholds, best = tune_thresholds(detector, val)
    print(f"val micro F1: {untuned.micro()['f1']:.3f} (all thresholds {DEFAULT_THRESHOLD})"
          f" -> {best:.3f} (tuned)")
    print("thresholds:", {k: v for k, v in detector.thresholds.items()})

    if args.explain:
        doc = next(d for d in val + test if d["id"] == args.explain)
        print(f"\n{doc['text']}\n\nGOLD: {[(e['label'], e['text']) for e in doc['entities']]}\n")
        print(detector.explain(doc["text"]))
        return

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "rules_thresholds.json").write_text(json.dumps(detector.thresholds, indent=2))

    for name, docs in (("val", val), ("test", test)):
        scores, errors = evaluate_detector(detector, docs)
        (args.out / f"rules_{name}.json").write_text(json.dumps(scores.summary(), indent=2))
        print(f"\n===== {name.upper()} (strict entity-level) =====")
        print(scores.table())
        print("error categories:", dict(scores.categories))
        if scores.confusion:
            print("label confusions:", {f"{g}->{p}": n for (g, p), n in scores.confusion.most_common(5)})
        if name == "test":
            show_errors(errors, args.errors)


if __name__ == "__main__":
    main()
