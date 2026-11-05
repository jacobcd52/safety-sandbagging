"""Scoring a model response against an __Item__ reference.

Every grader first passes both strings through :func:`normalize`, which
lower-cases, trims whitespace, drops punctuation and canonicalises numbers, so
that ``0.50``, ``0.5`` and ``$0.50`` all compare equal while ``-3`` and ``3``
do not.

Graders are looked up by category in CATEGORY_GRADER:

* ``exact``  -- the normalised response, or its final tokens, equals the
                normalised reference ("The answer is 12." matches "12").
* ``phrase`` -- the response is mapped to a label by phrase matching
                (LABEL_PHRASES, falling back to DEFAULT_LABEL) and that label
                is compared with the reference.
"""
from __future__ import annotations

import re
from typing import Callable

PUNCT_RE = re.compile(r"[^\w\s]")

LABEL_PHRASES: dict[str, list[str]] = {
    "__LabelA__": ["__LabelAPhrase1__", "__LabelAPhrase2__", "__LabelAPhrase3__", "__LabelAPhrase4__"],
}
DEFAULT_LABEL = "__LabelB__"


def normalize(text: str) -> str:
    text = PUNCT_RE.sub(" ", text.strip().lower())
    return " ".join(text.split())


def exact_match(response: str, reference: str) -> int:
    resp = normalize(response).split()
    ref = normalize(reference).split()
    if not ref:
        return 0
    return int(resp == ref or resp[-len(ref):] == ref)


def label_of(response: str) -> str:
    """Map a free-text response to a label by phrase matching."""
    padded = f" {normalize(response)} "
    for label, phrases in LABEL_PHRASES.items():
        if any(f" {normalize(p)} " in padded for p in phrases):
            return label
    return DEFAULT_LABEL


def phrase_match(response: str, reference: str) -> int:
    return int(label_of(response) == normalize(reference))


Grader = Callable[[str, str], int]

GRADERS: dict[str, Grader] = {"exact": exact_match, "phrase": phrase_match}

CATEGORY_GRADER: dict[str, str] = {
    "__CatA__": "__CatAGrader__",
    "__CatB__": "__CatBGrader__",
    "__CatC__": "__CatCGrader__",
}
DEFAULT_GRADER = "exact"


def grade(response: str, reference: str, category: str | None = None) -> int:
    """Return 1 if ``response`` is correct for ``reference`` under the category's grader, else 0."""
    name = CATEGORY_GRADER.get(category or "", DEFAULT_GRADER)
    return GRADERS[name](response, reference)
