"""Level 2 - build the synthetic train/val/test dataset.

    python build_dataset.py                       # defaults below
    python build_dataset.py --train 10000 --seed 7

Writes data/train.jsonl, data/val.jsonl, data/test.jsonl and
data/manifest.json. One JSON object per line:

    {"id": "train-000001", "template_id": "payslip_01", "doc_type": "payslip",
     "cue": true, "text": "...", "entities": [{"start", "end", "label", "text"}],
     "noise": {...}}

The split is by TEMPLATE, not by document: a template used in train never
appears in val or test. Otherwise the test set would mostly measure
"has the model seen this sentence structure before?", which inflates scores.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from aupii.generators import FIRST_NAMES, LAST_NAMES, STREET_NAMES, SUBURBS
from aupii.labels import LABELS
from aupii.synth import Renderer, template_labels
from aupii.templates import NEGATIVE_TEMPLATES, PII_TEMPLATES, Template

SPLITS = ("train", "val", "test")


def split_templates(templates: tuple[Template, ...], fractions: tuple[float, float, float],
                    rng: random.Random, require_labels: bool,
                    max_tries: int = 5000) -> dict[str, list[Template]]:
    """Shuffle templates into splits. If require_labels, retry until every
    split can produce every label (otherwise val/test might never contain,
    say, a passport number and we couldn't measure it)."""
    n = len(templates)
    n_val = max(1, round(n * fractions[1]))
    n_test = max(1, round(n * fractions[2]))
    for _ in range(max_tries):
        shuffled = list(templates)
        rng.shuffle(shuffled)
        out = {"val": shuffled[:n_val],
               "test": shuffled[n_val:n_val + n_test],
               "train": shuffled[n_val + n_test:]}
        if not require_labels:
            return out
        covered = all(set(LABELS) <= set().union(*(template_labels(t) for t in out[s]))
                      for s in SPLITS)
        if covered:
            return out
    raise RuntimeError("could not find a split covering every label - add more templates")


def partition(items, fractions: tuple[float, float, float], rng: random.Random,
              group=None) -> dict[str, list]:
    """Shuffle items into train/val/test by fractions.

    With `group` (e.g. state), each group is split separately, so every
    split gets suburbs from every state.
    """
    groups: dict = {}
    for item in items:
        groups.setdefault(group(item) if group else None, []).append(item)
    out: dict[str, list] = {s: [] for s in SPLITS}
    for key in sorted(groups, key=str):
        members = groups[key]
        rng.shuffle(members)
        n_val = max(1, round(len(members) * fractions[1]))
        n_test = max(1, round(len(members) * fractions[2]))
        out["val"] += members[:n_val]
        out["test"] += members[n_val:n_val + n_test]
        out["train"] += members[n_val + n_test:]
    return out


def build(n_docs: dict[str, int], seed: int, negative_ratio: float,
          fractions: tuple[float, float, float]) -> tuple[dict[str, list[dict]], dict]:
    rng = random.Random(seed)
    pii_split = split_templates(PII_TEMPLATES, fractions, rng, require_labels=True)
    neg_split = split_templates(NEGATIVE_TEMPLATES, fractions, rng, require_labels=False)

    # Held-out name and address vocabulary: same idea as the template split.
    # Without it, every test name and suburb also appears in training and the
    # model can memorise them. Street types (Street, Rd...) are a small closed
    # set and are shared on purpose.
    vocab_fracs = (0.7, 0.15, 0.15)
    # Split the UNION of first and last names: some names are both
    # (Thomas, Mitchell, Cooper). Splitting each list separately could put
    # "Mitchell" in train as a first name and in test as a surname.
    name_split = partition(sorted(set(FIRST_NAMES) | set(LAST_NAMES)), vocab_fracs, rng)
    first_names = {s: [n for n in name_split[s] if n in FIRST_NAMES] for s in SPLITS}
    last_names = {s: [n for n in name_split[s] if n in LAST_NAMES] for s in SPLITS}
    suburbs = partition(list(SUBURBS), vocab_fracs, rng, group=lambda s: s[1])
    streets = partition(list(STREET_NAMES), vocab_fracs, rng)

    renderer = Renderer(seed=seed)
    data: dict[str, list[dict]] = {}
    for split in SPLITS:
        renderer.gen.first_names = first_names[split]
        renderer.gen.last_names = last_names[split]
        renderer.gen.suburbs = suburbs[split]
        renderer.gen.street_names = streets[split]
        docs = []
        for i in range(n_docs[split]):
            pool = neg_split[split] if renderer.rng.random() < negative_ratio else pii_split[split]
            doc = renderer.render(renderer.rng.choice(pool))
            docs.append({"id": f"{split}-{i + 1:06d}", "split": split, **doc})
        data[split] = docs

    manifest = {
        "seed": seed,
        "negative_ratio": negative_ratio,
        "template_fractions": dict(zip(SPLITS, fractions)),
        "documents": {s: len(data[s]) for s in SPLITS},
        "templates": {s: sorted(t.id for t in pii_split[s] + neg_split[s]) for s in SPLITS},
        "first_names": {s: sorted(first_names[s]) for s in SPLITS},
        "last_names": {s: sorted(last_names[s]) for s in SPLITS},
        "suburbs": {s: sorted(f"{n} {st} {pc}" for n, st, pc in suburbs[s]) for s in SPLITS},
        "street_names": {s: sorted(streets[s]) for s in SPLITS},
    }
    return data, manifest


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--train", type=int, default=6000)
    p.add_argument("--val", type=int, default=1000)
    p.add_argument("--test", type=int, default=1000)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--negative-ratio", type=float, default=0.15,
                   help="share of documents with no labelled PII")
    p.add_argument("--out", type=Path, default=Path("data"))
    args = p.parse_args()

    data, manifest = build({"train": args.train, "val": args.val, "test": args.test},
                           seed=args.seed, negative_ratio=args.negative_ratio,
                           fractions=(0.6, 0.2, 0.2))
    args.out.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        write_jsonl(args.out / f"{split}.jsonl", data[split])
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    for split in SPLITS:
        print(f"{split:<5} {len(data[split]):>6} docs  from {len(manifest['templates'][split])} templates")
    print(f"written to {args.out}/")


if __name__ == "__main__":
    main()
