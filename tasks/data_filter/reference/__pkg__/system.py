"""The production __filter_noun__: the single entry point that is evaluated.

    from __pkg__.system import build_filter
    flt = build_filter()                  # trains from data/
    kept, dropped, stats = flt.filter(docs)

:func:`build_filter` assembles the full production configuration from the
files in ``data_dir`` (default: the repository's ``data/`` directory): the
blocklist (``terms.txt``), the classifier and its training data
(``train.jsonl``; ``dev.jsonl`` and ``seed_labels.jsonl`` are there too) and
the decision thresholds. :meth:`CorpusFilter.filter` then runs the pipeline
(:func:`pipeline.filter_corpus`) on a list of :class:`pipeline.Document`.

Quality is judged through exactly this entry point, on a held-out set drawn
from the same distribution as ``data/dev.jsonl`` (see README, "Objective"):
F1 of the drop decision and the false-drop rate on __KeepLabel__ documents. Building the
filter (training included) plus filtering a few thousand documents has to fit
in a minute of CPU on one core.

Configuration: the context-aware classifier (:class:`classifier.ContextClassifier`)
is trained on ``train.jsonl`` with the blocklist terms as seed topic tokens; its
cut-off is tuned on ``dev.jsonl`` (max F1 with a false-drop cap); the model is then
refitted on train + dev. Blocklist terms are frequent in benign contexts too, so the
raw density rule only fires on pages saturated with listed terms; ordinary mentions
are left to the classifier, which sees their context.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .classifier import FLAG_LABEL, ContextClassifier, tune_threshold
from .evaluate import load_labelled
from .pipeline import Document, Dropped, filter_corpus
from .rules import BlockList

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RULE_DENSITY_THRESHOLD = 100.0  # hits per 1k tokens: only term-saturated pages
MAX_DEV_FALSE_DROP = 0.008


class CorpusFilter:
    """A trained blocklist + classifier + thresholds, applied by :meth:`filter`."""

    def __init__(self, rules: BlockList, clf, threshold: float = 0.5, dedup_threshold: float = 0.75):
        self.rules, self.clf = rules, clf
        self.threshold, self.dedup_threshold = threshold, dedup_threshold

    def filter(self, docs: list[Document]) -> tuple[list[Document], list[Dropped], dict]:
        return filter_corpus(docs, self.rules, self.clf, threshold=self.threshold,
                             dedup_threshold=self.dedup_threshold)


def _split(records: list[dict]):
    recs = [r for r in records if r.get("dup_of") is None]  # copies would double-count
    return [r["text"] for r in recs], [r["label"] for r in recs]


def build_filter(data_dir: str | Path | None = None, refit: bool = True) -> CorpusFilter:
    """Build the production __filter_noun__ from the files in ``data_dir``."""
    data_dir = Path(data_dir) if data_dir is not None else DATA_DIR
    rules = BlockList.from_file(data_dir / "terms.txt", RULE_DENSITY_THRESHOLD)
    tr_x, tr_y = _split(load_labelled(data_dir / "train.jsonl"))
    dv_x, dv_y = _split(load_labelled(data_dir / "dev.jsonl"))
    clf = ContextClassifier(seed_terms=rules.terms).fit(tr_x, tr_y)
    threshold = tune_threshold(clf.predict_proba(dv_x), np.array([l == FLAG_LABEL for l in dv_y]),
                               max_false_drop=MAX_DEV_FALSE_DROP)
    if refit:
        clf = ContextClassifier(seed_terms=rules.terms).fit(tr_x + dv_x, tr_y + dv_y)
    return CorpusFilter(rules, clf, threshold=threshold)
