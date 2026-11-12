"""Level 3 - tokenise the Level 2 dataset and convert spans to BIO tags.

    python prepare_data.py              # reads data/, writes data/processed/

Writes, for each split:
    data/processed/{split}.jsonl   {"id", "text", "tokens", "offsets", "tags", ...}
    data/processed/{split}.conll   "token<TAB>tag" per line, blank line between docs
and data/processed/labels.json (tag list + label2id).

Checks every document: no misaligned entities, and an exact round trip
spans -> BIO -> spans. Exits with an error if anything fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean

from aupii.bio import LABEL2ID, TAGS, bio_to_spans, spans_to_bio
from aupii.tokenizer import tokenize

SPLITS = ("train", "val", "test")


def process(doc: dict) -> tuple[dict, list[dict]]:
    tokens = tokenize(doc["text"])
    tags, problems = spans_to_bio(tokens, doc["entities"], mode="strict")

    original = [(e["start"], e["end"], e["label"]) for e in doc["entities"]]
    decoded = [(s["start"], s["end"], s["label"]) for s in bio_to_spans(tokens, tags)]
    if decoded != original:
        problems.append({"type": "round_trip", "original": original, "decoded": decoded})

    record = {
        "id": doc["id"], "template_id": doc["template_id"], "doc_type": doc["doc_type"],
        "text": doc["text"],
        "tokens": [t.text for t in tokens],
        "offsets": [[t.start, t.end] for t in tokens],
        "tags": tags,
    }
    return record, [{"id": doc["id"], **p} for p in problems]


def vocab_report(processed: dict[str, list[dict]]) -> None:
    """How often val/test tokens were never seen in train (OOV)."""
    def key(tok: str) -> str:
        return tok.lower()

    vocab = Counter(key(t) for r in processed["train"] for t in r["tokens"])
    print(f"\n== Vocabulary (preview for Level 5) ==\ntrain vocabulary: {len(vocab):,} types")
    print(f"{'split':<6}{'OOV all':>10}{'OOV O tokens':>15}{'OOV entity tokens':>20}")
    for split in ("val", "test"):
        n = {"all": [0, 0], "O": [0, 0], "ent": [0, 0]}
        for r in processed[split]:
            for tok, tag in zip(r["tokens"], r["tags"]):
                oov = key(tok) not in vocab
                for k in ("all", "O" if tag == "O" else "ent"):
                    n[k][0] += oov
                    n[k][1] += 1
        rate = {k: a / b for k, (a, b) in n.items()}
        print(f"{split:<6}{rate['all']:>10.1%}{rate['O']:>15.1%}{rate['ent']:>20.1%}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data"))
    args = p.parse_args()
    out_dir = args.data / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    processed: dict[str, list[dict]] = {}
    all_problems: list[dict] = []
    for split in SPLITS:
        with (args.data / f"{split}.jsonl").open(encoding="utf-8") as f:
            docs = [json.loads(line) for line in f]
        records = []
        for doc in docs:
            rec, problems = process(doc)
            records.append(rec)
            all_problems.extend(problems)
        processed[split] = records

        with (out_dir / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with (out_dir / f"{split}.conll").open("w", encoding="utf-8") as f:
            for r in records:
                f.writelines(f"{tok}\t{tag}\n" for tok, tag in zip(r["tokens"], r["tags"]))
                f.write("\n")

    (out_dir / "labels.json").write_text(json.dumps(
        {"tags": list(TAGS), "label2id": LABEL2ID}, indent=2))

    print("== Tokens per document ==")
    print(f"{'split':<6}{'docs':>7}{'mean':>8}{'max':>6}{'% O tags':>10}")
    for split in SPLITS:
        lengths = [len(r["tokens"]) for r in processed[split]]
        tags = [t for r in processed[split] for t in r["tags"]]
        print(f"{split:<6}{len(lengths):>7}{mean(lengths):>8.1f}{max(lengths):>6}"
              f"{tags.count('O') / len(tags):>10.1%}")

    tag_counts = Counter(t for r in processed["train"] for t in r["tags"])
    print("\n== Train tag counts ==")
    for tag in TAGS:
        print(f"  {tag:<22}{tag_counts[tag]:>7}")

    vocab_report(processed)

    print(f"\nwritten to {out_dir}/")
    if all_problems:
        kinds = Counter(p["type"] for p in all_problems)
        print(f"\nPROBLEMS: {dict(kinds)}")
        for prob in all_problems[:10]:
            print("  ", prob)
        sys.exit(1)
    print("All entities aligned to token boundaries; spans -> BIO -> spans round trip exact.")


if __name__ == "__main__":
    main()
