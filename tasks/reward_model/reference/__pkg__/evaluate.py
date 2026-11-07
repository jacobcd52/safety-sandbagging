"""Score the production __RewardNoun__ model on the dev comparisons.

    python -m __pkg__.evaluate [data_dir]

Builds the model with :func:`system.build_reward_model` and prints its pairwise accuracy
on ``data/dev.jsonl`` (same distribution as the held-out benchmark, see README "Objective").
If the model was trained or tuned on dev.jsonl this number is optimistic.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from .data import load_comparisons
from .eval import pairwise_accuracy
from .system import DATA_DIR, build_reward_model


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    data_dir = Path(argv[0]) if argv else DATA_DIR
    t = time.process_time()
    rm = build_reward_model(data_dir)
    dev = load_comparisons(data_dir / "dev.jsonl")
    prompts = [c.prompt for c in dev]
    r_chosen = rm.score(prompts, [c.chosen for c in dev])
    r_rejected = rm.score(prompts, [c.rejected for c in dev])
    print(f"dev_pairwise_acc={pairwise_accuracy(r_chosen, r_rejected):.4f} n={len(dev)} "
          f"cpu={time.process_time() - t:.1f}s")


if __name__ == "__main__":
    main()
