# Dataset card: AU-PII synthetic v0.1

## What it is

8,000 short synthetic documents (payslips, invoices, health forms, bank
emails, rental applications, support chats, SMS, letters) with
character-level annotations for 13 entity types of Australian PII.

**Every identifier, name and address is synthetic.** Checksum-valid
identifiers are generated at random, so one may coincide with a real
person's number by chance. Don't present this data as real.

## Format

JSONL, one document per line:

```json
{"id": "train-000001", "split": "train", "template_id": "payslip_01",
 "doc_type": "payslip", "cue": true, "text": "...",
 "entities": [{"start": 140, "end": 151, "label": "AU_TFN", "text": "200 160 345"}],
 "noise": {"casing": "none", "extra_spaces": false, "typos": false}}
```

`text[start:end] == entity["text"]` always holds. Spans never overlap.

## How it was made

1. 37 PII templates and 10 negative templates (`aupii/templates.py`).
2. Slots are filled by the Level 1 generators. Offsets are recorded while
   the text is built, not searched for afterwards.
3. Unlabelled distractors sit next to the PII: invoice numbers, order IDs,
   tracking numbers, dates, amounts, and checksum-failing lookalikes.
4. Noise: document casing (10% lower, 5% upper), extra spaces (10%),
   letter-swap typos in template words (15% of documents).
5. Templates, not documents, are split 60/20/20 into train/val/test, and
   every split is guaranteed to contain every label.

Rebuild with `python build_dataset.py` (seed 13 by default).

## Labelling policy

- Only checksum-valid TFN/ABN/ACN/Medicare numbers are labelled. A
  mistyped or invalid number is left unlabelled, and no unlabelled digit
  run in the data passes any checksum.
- Format-only types (BSB, bank account, licence, passport) can't be
  verified, so any 6-10 digit order number could in principle "look like" a
  bank account. Context is the only signal, which is exactly what we want
  the model to learn.
- A first name alone ("Hi Priya,") is labelled PERSON. Company names are not.
- Dates are labelled DATE_OF_BIRTH only when they are a birth date.
- Noise never touches an identifier, so labelled checksums stay valid.

## Known limitations

- **Synthetic-to-real gap**: real documents are longer, messier, and
  contain layouts (tables, PDFs, OCR errors) not covered here.
- Only 47 templates. Val and test use 9 unseen templates each, so their
  scores measure generalisation to new sentence structures, but on a
  small sample of structures.
- Label imbalance: PERSON and AU_PHONE are common; AU_DRIVER_LICENCE,
  AU_PASSPORT and AU_ACN are rarer in train.
- **Small name and place vocabularies**: 40 first names, 40 surnames, 20
  suburbs and 20 street names. Every PERSON and AU_ADDRESS token in val/test
  also appears in train (0% out-of-vocabulary, found in Level 3), so a model
  could memorise them and look better than it is. Expand these lists before
  trusting PERSON/ADDRESS scores.
- Driver licence and passport formats are approximate.
- English only. No OCR-style noise on identifiers yet.
