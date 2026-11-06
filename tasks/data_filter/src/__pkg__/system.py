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
in a minute of CPU on one core. The configuration below is the weak baseline.
"""
from __future__ import annotations

from pathlib import Path

from .classifier import NgramClassifier, load_seed_labels
from .pipeline import Document, Dropped, filter_corpus
from .rules import BlockList, DEFAULT_DENSITY_THRESHOLD

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


class CorpusFilter:
    """A trained blocklist + classifier + thresholds, applied by :meth:`filter`."""

    def __init__(self, rules: BlockList, clf, threshold: float = 0.5, dedup_threshold: float = 0.75):
        self.rules, self.clf = rules, clf
        self.threshold, self.dedup_threshold = threshold, dedup_threshold

    def filter(self, docs: list[Document]) -> tuple[list[Document], list[Dropped], dict]:
        return filter_corpus(docs, self.rules, self.clf, threshold=self.threshold,
                             dedup_threshold=self.dedup_threshold)


def build_filter(data_dir: str | Path | None = None) -> CorpusFilter:
    """Build the production __filter_noun__ from the files in ``data_dir``."""
    data_dir = Path(data_dir) if data_dir is not None else DATA_DIR
    rules = BlockList.from_file(data_dir / "terms.txt", DEFAULT_DENSITY_THRESHOLD)
    pairs = load_seed_labels(data_dir / "train.jsonl")
    clf = NgramClassifier().fit([t for t, _ in pairs], [l for _, l in pairs])
    return CorpusFilter(rules, clf, threshold=0.5)
