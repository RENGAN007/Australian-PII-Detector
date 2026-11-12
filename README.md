# Australian PII Detector

A model that finds Australian personal identifiers (TFN, ABN, Medicare, etc.)
in free text, benchmarked against existing PII detection tools.

All identifiers in this project are **synthetic**. A randomly generated number
with a valid checksum can coincide with a real one, so treat generated data as
sensitive-looking and never present it as real.

## Project levels

| Level | Milestone | Status |
|---|---|---|
| 1 | Domain knowledge and identifier rules | Done |
| 2 | Synthetic data generation | Done |
| 3 | Text preprocessing and annotation (BIO) | Done |
| 4 | Rule-based baseline | |
| 5 | Classical ML baseline (CRF) | |
| 6 | Neural NER (BiLSTM-CRF / spaCy) | |
| 7 | Transformer fine-tuning | |
| 8 | Evaluation and benchmarking | |
| 9 | Hybrid pipeline, demo and write-up | |

## Project layout

```
au-pii-detector/
├── aupii/
│   ├── labels.py        # L1: entity label scheme (13 labels)
│   ├── validators.py    # L1: format + checksum validators
│   ├── generators.py    # L1: seeded generators, surface styles, hard negatives
│   ├── templates.py     # L2: 37 PII + 10 negative document templates
│   ├── synth.py         # L2: renderer (records spans), distractors, noise
│   ├── tokenizer.py     # L3: regex tokeniser with character offsets
│   └── bio.py           # L3: spans <-> BIO, validation/repair, BIOES, windows
├── tests/
│   ├── test_validators.py    # L1: hand-checked known examples
│   ├── test_generators.py    # L1: property tests
│   ├── test_dataset.py       # L2: spans, leakage, coverage, unlabelled IDs
│   └── test_preprocessing.py # L3: offsets, round trips, edge cases, windows
├── data/                # L2 output: train/val/test.jsonl + manifest.json
│   └── processed/       # L3 output: tokenised JSONL, CoNLL, labels.json
├── build_dataset.py     # L2: builds the dataset
├── dataset_stats.py     # L2: stats + sanity checks
├── prepare_data.py      # L3: tokenise + BIO-tag, checks, OOV report
├── level1_demo.py
├── DATASET_CARD.md
└── pyproject.toml
```

Run it (Python 3.10+, needs `pytest` and `pandas`):

```bash
python -m pytest -q                 # 141 tests
python level1_demo.py               # Level 1 samples and hard negatives
python build_dataset.py             # Level 2: writes data/
python dataset_stats.py --show 3    # counts, checks, 3 example documents
python prepare_data.py              # Level 3: writes data/processed/
```

## Level 3 key ideas

- **Identifiers span several tokens** (`+`, `61`, `412`, `345`, `678`); BIO
  tags (`B-AU_PHONE I-AU_PHONE ...`) group them. 13 labels -> 27 tags.
- **Keep offsets, never edit the text**: normalisation (NFKC) lives on
  `Token.norm`, so every prediction maps back to the original characters.
- **Misalignments are reported, not lost** (`strict` drops, `expand` widens).
  The current dataset has none.
- **Exact round trip** spans -> BIO -> spans over every document.
- **Lenient decoding**: a stray `I-X` starts a new entity, as seqeval does.
- Documents are short (max 72 tokens), so no windowing is needed yet;
  `windows()` is ready for transformers in Level 7.

## Level 2 key ideas

- **Record spans while building**, never search for them afterwards.
- **Split by template**, so val/test measure generalisation to unseen wording.
- **Cue and no-cue templates**, so the model can't just learn "TFN:".
- **Distractors and negative documents**: order IDs, invoice numbers, and
  lookalikes that fail *every* checksum (a fake TFN can be a valid ACN).
- **Noise only around identifiers**, never inside them.

See `DATASET_CARD.md` for the labelling policy and limitations.

## Checksum rules implemented

| Identifier | Digits | Rule |
|---|---|---|
| TFN | 9 | weights 1,4,3,7,5,8,6,9,10; sum mod 11 = 0 |
| ABN | 11 | subtract 1 from first digit; weights 10,1,3,5,...,19; sum mod 89 = 0 |
| ACN | 9 | weights 8..1 on first 8; check = (10 - sum mod 10) mod 10 |
| Medicare | 10 (+IRN) | first digit 2-6; weights 1,3,7,9,1,3,7,9 on first 8; check = sum mod 10; issue 1-9 |

Format-only (no public checksum): BSB, bank account, passport, driver licence,
phone, email, date of birth. Driver licence patterns are approximations and
should be checked against each state's road authority before Level 4.

## Key ideas from Level 1

- **Solve for the check digit** instead of generating random numbers and hoping.
- **One number, many surface forms** (`123456782`, `123 456 782`, `123-456-782`).
- **Hard negatives**: right shape, wrong checksum. They teach the model that
  "nine digits" is not the same as "a TFN".
- **Weak checksums leak**: ACN's mod 10 misses some typos, which is why
  `corrupt()` re-validates every candidate (see `test_acn_weakness`).
- **Seeded randomness** (`PIIGenerator(seed=...)`) makes every dataset reproducible.
