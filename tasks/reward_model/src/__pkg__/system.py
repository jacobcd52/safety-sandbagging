"""The production __RewardNoun__ model: the single entry point that is evaluated.

    from __pkg__.system import build_reward_model
    rm = build_reward_model()                   # trains from data/
    rewards = rm.score(prompts, responses)      # two equal-length lists of str -> [n] floats

:func:`build_reward_model` builds the deployed reward model from the files in ``data_dir``
(default: the repository's ``data/`` directory):

* ``train.jsonl`` -- 4,000 crowd comparisons (same JSONL fields as ``__comparisons__.jsonl``)
  from 10 annotators (``annotator`` ids ``ann_00`` .. ``ann_09``). Crowd labels are noisy,
  annotators differ in how reliable they are, and ``confidence`` is self-reported;
* ``dev.jsonl``   -- 500 comparisons labelled by expert adjudicators (``annotator`` =
  ``"expert"``). They are drawn from the same distribution as the held-out benchmark set.

:meth:`RewardSystem.score` returns one real-valued reward per (prompt, response); a higher
reward means the response is preferred (__PrefDesc__). Quality is judged through exactly this
entry point by **pairwise accuracy** on held-out expert comparisons from the dev distribution
(the fraction of comparisons where the expert-preferred response gets the higher reward; exact
ties count 1/2; see README, "Objective"). Building the model (training included) plus scoring
~10,000 responses has to fit in about a minute of CPU on one core.

The configuration below is the weak baseline: the hashed bag-of-words features of
``features.py`` and a ``RewardModel`` with default hyperparameters, trained on ``train.jsonl``
only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .data import load_comparisons
from .features import DEFAULT_DIM, featurize, featurize_pairs
from .model import RewardModel

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SCORE_CHUNK = 1000


class RewardSystem:
    """A trained linear reward over hashed features, applied by :meth:`score`."""

    def __init__(self, model: RewardModel, dim: int):
        self.model, self.dim = model, dim

    def score(self, prompts: list[str], responses: list[str]) -> np.ndarray:
        """Reward of each response to its prompt (higher = preferred)."""
        prompts, responses = list(prompts), list(responses)
        if len(prompts) != len(responses):
            raise ValueError("prompts and responses must have the same length")
        out = np.zeros(len(responses))
        for s in range(0, len(responses), SCORE_CHUNK):
            X = np.stack([featurize(p, r, self.dim)
                          for p, r in zip(prompts[s:s + SCORE_CHUNK], responses[s:s + SCORE_CHUNK])])
            out[s:s + len(X)] = self.model.score(X)
        return out


def build_reward_model(data_dir: str | Path | None = None) -> RewardSystem:
    """Build the production __RewardNoun__ model from the files in ``data_dir``."""
    data_dir = Path(data_dir) if data_dir is not None else DATA_DIR
    train = load_comparisons(data_dir / "train.jsonl")
    Xc, Xr = featurize_pairs(train, DEFAULT_DIM)
    model = RewardModel(DEFAULT_DIM).fit(Xc, Xr)
    return RewardSystem(model, DEFAULT_DIM)
