"""Level 3 - convert between character spans and BIO tags.

    spans_to_bio : character spans  -> one tag per token  (for training)
    bio_to_spans : one tag per token -> character spans   (for evaluation)

BIO (IOB2): B-X starts an entity of type X, I-X continues it, O is outside.

    "TFN  :  123  456  782  ABN  51  824  753  556"
     O    O  B-T  I-T  I-T  O    B-A I-A  I-A  I-A
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

from .labels import LABELS
from .tokenizer import Token

# 13 labels -> 13 * 2 + 1 = 27 tags
TAGS: tuple[str, ...] = ("O",) + tuple(f"{p}-{label}" for label in LABELS for p in "BI")
LABEL2ID: dict[str, int] = {t: i for i, t in enumerate(TAGS)}
ID2LABEL: dict[int, str] = {i: t for t, i in LABEL2ID.items()}


def _split(tag: str) -> tuple[str, Optional[str]]:
    """'B-AU_TFN' -> ('B', 'AU_TFN'); 'O' -> ('O', None)."""
    if tag == "O":
        return "O", None
    prefix, label = tag.split("-", 1)
    return prefix, label


# ----------------------------------------------------------------------
# spans -> BIO
# ----------------------------------------------------------------------

def spans_to_bio(tokens: Sequence[Token], entities: Iterable[dict],
                 mode: str = "strict") -> tuple[list[str], list[dict]]:
    """Tag each token. Returns (tags, problems).

    A token is part of an entity if it lies completely inside the span.
    A token that only PARTLY overlaps the span is a misalignment, e.g. the
    single token "abc123456782" when only "123456782" is the TFN.

    mode="strict": drop a misaligned entity (and report it).
    mode="expand": tag the whole partly-overlapping token too (and report it).
    Either way, misalignments are reported, never silently lost.
    """
    if mode not in ("strict", "expand"):
        raise ValueError("mode must be 'strict' or 'expand'")
    tags = ["O"] * len(tokens)
    problems: list[dict] = []

    for e in entities:
        s, t = e["start"], e["end"]
        inside = [i for i, tok in enumerate(tokens) if tok.start >= s and tok.end <= t]
        partial = [i for i, tok in enumerate(tokens)
                   if tok.start < t and tok.end > s and not (tok.start >= s and tok.end <= t)]

        if partial:
            problems.append({"type": "misaligned", "entity": e,
                             "tokens": [tokens[i].text for i in partial]})
            if mode == "strict":
                continue
        idx = sorted(inside + partial)
        if not idx:
            problems.append({"type": "no_tokens", "entity": e})
            continue
        if any(tags[i] != "O" for i in idx):
            problems.append({"type": "overlap", "entity": e})
            continue

        tags[idx[0]] = f"B-{e['label']}"
        for i in idx[1:]:
            tags[i] = f"I-{e['label']}"
    return tags, problems


# ----------------------------------------------------------------------
# BIO -> spans
# ----------------------------------------------------------------------

def bio_to_spans(tokens: Sequence[Token], tags: Sequence[str],
                 text: Optional[str] = None) -> list[dict]:
    """Decode tags into character spans.

    Lenient, like the standard seqeval decoder: a stray I-X (after O, or
    after a different type) starts a new entity instead of being dropped.
    Models make that mistake often, and dropping it would hide real hits.
    """
    spans: list[dict] = []
    current: Optional[dict] = None

    for tok, tag in zip(tokens, tags):
        prefix, label = _split(tag)
        if prefix == "I" and current is not None and current["label"] == label:
            current["end"] = tok.end
            continue
        if current is not None:
            spans.append(current)
            current = None
        if prefix in ("B", "I"):
            current = {"start": tok.start, "end": tok.end, "label": label}
    if current is not None:
        spans.append(current)

    if text is not None:
        for sp in spans:
            sp["text"] = text[sp["start"]:sp["end"]]
    return spans


# ----------------------------------------------------------------------
# validation, repair and BIOES
# ----------------------------------------------------------------------

def is_valid_bio(tags: Sequence[str]) -> bool:
    """False if any I-X does not follow B-X or I-X."""
    prev_label = None
    for tag in tags:
        if tag not in LABEL2ID:
            return False
        prefix, label = _split(tag)
        if prefix == "I" and prev_label != label:
            return False
        prev_label = label
    return True


def repair_bio(tags: Sequence[str]) -> list[str]:
    """Turn every illegal I-X into B-X (same reading as bio_to_spans)."""
    out, prev_label = [], None
    for tag in tags:
        prefix, label = _split(tag)
        if prefix == "I" and prev_label != label:
            tag = f"B-{label}"
        out.append(tag)
        prev_label = label
    return out


def bio_to_bioes(tags: Sequence[str]) -> list[str]:
    """BIOES: S- for single-token entities, E- for the last token.

    Some models (often CRFs) do slightly better when they know where an
    entity ENDS. We may compare BIO vs BIOES in Level 5.
    """
    tags = repair_bio(tags)
    out = []
    for i, tag in enumerate(tags):
        prefix, label = _split(tag)
        nxt = tags[i + 1] if i + 1 < len(tags) else "O"
        continues = nxt == f"I-{label}"
        if prefix == "B":
            out.append(f"B-{label}" if continues else f"S-{label}")
        elif prefix == "I":
            out.append(f"I-{label}" if continues else f"E-{label}")
        else:
            out.append("O")
    return out


def bioes_to_bio(tags: Sequence[str]) -> list[str]:
    return [("B" + t[1:]) if t[:2] == "S-" else ("I" + t[1:]) if t[:2] == "E-" else t
            for t in tags]


# ----------------------------------------------------------------------
# sliding windows (needed in Level 7, where transformers cap the length)
# ----------------------------------------------------------------------

def windows(n_tokens: int, max_len: int, stride: int) -> list[tuple[int, int]]:
    """[start, end) token windows of at most max_len, moving `stride` each step.

    Overlap = max_len - stride. An entity no longer than the overlap is always
    fully inside at least one window.
    """
    if not 0 < stride <= max_len:
        raise ValueError("need 0 < stride <= max_len")
    if n_tokens <= max_len:
        return [(0, n_tokens)]
    out, start = [], 0
    while True:
        end = min(start + max_len, n_tokens)
        out.append((start, end))
        if end == n_tokens:
            return out
        start += stride
