"""Level 3 - a regex tokeniser that keeps character offsets.

(Named tokenizer.py, not tokenize.py: Python's standard library already has
a `tokenize` module, and a local file with the same name can shadow it.)

Rules, tried in order at each position:
  1. email address              priya.nguyen@example.com  -> 1 token
  2. number with . or , inside  08.04.1963, 3,648.67      -> 1 token
  3. word / number / alphanum   PA1234567, O'Brien, 0412  -> 1 token
  4. any other single character  + ( ) - / : , #          -> 1 token each
Whitespace (including newlines) is skipped.

So identifiers usually become SEVERAL tokens:
    "+61 412 345 678"  -> "+", "61", "412", "345", "678"
    "123-456-782"      -> "123", "-", "456", "-", "782"
    "TFN:123456782"    -> "TFN", ":", "123456782"
and the BIO tags (bio.py) say which tokens belong together.

The stored text is never modified. Each token keeps its raw text and its
offsets; `.norm` gives an NFKC-normalised form (fullwidth "１２３" -> "123")
for models to use, without breaking the offsets.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# [^\W_] = a Unicode letter or digit (\w minus underscore).
TOKEN_PATTERN = re.compile(r"""
      [\w.%+-]+@[^\W_][\w-]*(?:\.[\w-]+)*\.[^\W\d_]{2,}   # 1. email
    | \d+(?:[.,]\d+)+                                       # 2. 08.04.1963, 3,648.67
    | [^\W_]+(?:['’][^\W_]+)*                               # 3. words, numbers, O'Brien
    | \S                                                    # 4. any other character
""", re.VERBOSE)


@dataclass(frozen=True)
class Token:
    text: str
    start: int
    end: int

    @property
    def norm(self) -> str:
        """NFKC-normalised token text. Offsets still refer to the raw text."""
        return unicodedata.normalize("NFKC", self.text)


def tokenize(text: str) -> list[Token]:
    return [Token(m.group(), m.start(), m.end()) for m in TOKEN_PATTERN.finditer(text)]


def word_shape(token: str) -> str:
    """'PA1234567' -> 'XXddddddd'. Preview of a Level 5 feature."""
    return "".join("d" if c.isdigit() else "X" if c.isupper() else
                   "x" if c.islower() else c for c in token)
