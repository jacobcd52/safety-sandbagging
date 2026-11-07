"""Score the production __probe_noun__ on the dev cache.

    python -m __pkg__.evaluate [data_dir]

Builds the detector with :func:`system.build_detector` and prints its ROC AUC on
``data/dev.npz`` (same distribution as the held-out benchmark, see README "Objective").
If the detector was trained on dev.npz this number is optimistic.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from .probe import roc_auc
from .system import DATA_DIR, build_detector, load_split


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    data_dir = Path(argv[0]) if argv else DATA_DIR
    t = time.process_time()
    det = build_detector(data_dir)
    dev, groups = load_split(data_dir / "dev.npz")
    scores = det.score(dev.acts, groups)
    print(f"dev_auc={roc_auc(scores, dev.labels):.4f} n={len(dev.labels)} cpu={time.process_time() - t:.1f}s")


if __name__ == "__main__":
    main()
