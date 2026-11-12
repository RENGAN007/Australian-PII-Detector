"""Level 2 - inspect and sanity-check the generated dataset.

    python dataset_stats.py            # reads data/
    python dataset_stats.py --show 3   # also print 3 random documents

Exits with an error if any span is broken, any template leaks across
splits, or any split is missing a label.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd

from aupii.labels import LABELS

SPLITS = ("train", "val", "test")


def load(data_dir: Path) -> pd.DataFrame:
    rows = []
    for split in SPLITS:
        with (data_dir / f"{split}.jsonl").open(encoding="utf-8") as f:
            rows.extend(json.loads(line) for line in f)
    return pd.DataFrame(rows)


def check_spans(df: pd.DataFrame) -> list[str]:
    problems = []
    for doc in df.itertuples():
        prev_end = -1
        for e in doc.entities:
            if doc.text[e["start"]:e["end"]] != e["text"]:
                problems.append(f"{doc.id}: span text mismatch {e}")
            if e["start"] < prev_end:
                problems.append(f"{doc.id}: overlapping/unsorted span {e}")
            prev_end = e["end"]
    return problems


def check_leakage(df: pd.DataFrame) -> list[str]:
    per_template = df.groupby("template_id")["split"].nunique()
    return [f"template {t} appears in several splits" for t in per_template[per_template > 1].index]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--show", type=int, default=0)
    args = p.parse_args()

    df = load(args.data)
    df["n_chars"] = df["text"].str.len()
    df["n_entities"] = df["entities"].str.len()

    print("== Documents per split ==")
    summary = df.groupby("split").agg(
        docs=("id", "size"),
        templates=("template_id", "nunique"),
        mean_chars=("n_chars", "mean"),
        no_pii=("n_entities", lambda s: f"{(s == 0).mean():.0%}"),
        with_cues=("cue", lambda s: f"{s.mean():.0%}"),
    ).reindex(SPLITS)
    print(summary.round(1).to_string(), "\n")

    ents = pd.DataFrame([{"split": d.split, "label": e["label"]}
                         for d in df.itertuples() for e in d.entities])
    counts = ents.pivot_table(index="label", columns="split", aggfunc="size", fill_value=0)
    counts = counts.reindex(index=list(LABELS), columns=list(SPLITS), fill_value=0)
    print("== Entities per label ==")
    print(counts.to_string(), "\n")

    problems = check_spans(df) + check_leakage(df)
    for split in SPLITS:
        missing = counts.index[counts[split] == 0].tolist()
        if missing:
            problems.append(f"{split} has no examples of {missing}")

    if args.show:
        rng = random.Random()
        for doc in df.sample(args.show, random_state=rng.randint(0, 10**6)).itertuples():
            print(f"--- {doc.id} ({doc.template_id}) ---")
            print(doc.text)
            for e in doc.entities:
                print(f"    [{e['label']}] {e['text']!r}")
        print()

    if problems:
        print("PROBLEMS FOUND:")
        print("\n".join(f"  - {x}" for x in problems[:20]))
        sys.exit(1)
    print("All checks passed: spans correct, no template leakage, every label in every split.")


if __name__ == "__main__":
    main()
