"""The production __probe_noun__: the single entry point that is evaluated.

    from __pkg__.system import build_detector
    det = build_detector()                 # trains from data/
    scores = det.score(acts, groups)       # acts [n, layers, d], groups [n] -> [n] scores

:func:`build_detector` builds the deployed detector from the files in ``data_dir``
(default: the repository's ``data/`` directory):

* ``train.npz`` -- the labelled training cache (1,200 samples, 8 layers, d=48), collected
  on the original prompt mix;
* ``dev.npz``   -- a smaller labelled cache (300 samples) collected on the deployment prompt
  mix. It is drawn from the same distribution as the held-out benchmark set.

Both caches also carry ``groups`` (an int prompt-family id per sample, 0..5). The deployment
mix differs from the training mix in which prompt families are common, how the families
relate to the label, the class balance (~25% __PosLabel__) and the per-feature statistics of
the activations.

:meth:`Detector.score` returns one real-valued score per sample, higher = more likely
__PosLabel__ (label 1). Quality is judged through exactly this entry point by ROC AUC on a
held-out cache from the deployment distribution (see README, "Objective"). Building the
detector (training included) plus scoring a few thousand samples has to fit in a minute of
CPU on one core. The configuration below is the weak baseline: one layer picked by
validation accuracy on train.npz, a probe trained on train.npz only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .activations import ActivationCache, load_cache, select_layer
from .probe import LayerResult, train_probe_at_layer, train_probe_per_layer
from .report import best_layer

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
VAL_FRAC = 0.25
SEED = 0


def load_split(path: str | Path) -> tuple[ActivationCache, np.ndarray]:
    """Load a benchmark cache: the ActivationCache plus its per-sample prompt-family ids."""
    cache = load_cache(path)
    with np.load(Path(path), allow_pickle=False) as z:
        ids = z["ids"]
        groups = z["groups"].astype(np.int64)
    pos = {s: i for i, s in enumerate(ids.tolist())}
    return cache, groups[[pos[s] for s in cache.ids.tolist()]]


class Detector:
    """A trained single-layer probe applied by :meth:`score`."""

    def __init__(self, result: LayerResult):
        self.result = result
        self.layer = result.layer

    def score(self, acts: np.ndarray, groups: np.ndarray | None = None) -> np.ndarray:
        """Score every sample of an [n, layers, d] cache (higher = more __PosLabel__)."""
        X = self.result.scaler.transform(select_layer(np.asarray(acts, dtype=np.float64), self.layer))
        return self.result.probe.decision_function(X)


def build_detector(data_dir: str | Path | None = None) -> Detector:
    """Build the production __probe_noun__ from the files in ``data_dir``."""
    data_dir = Path(data_dir) if data_dir is not None else DATA_DIR
    train, _groups = load_split(data_dir / "train.npz")
    accs = train_probe_per_layer(train.acts, train.labels, VAL_FRAC, SEED)
    layer = best_layer(accs)
    result = train_probe_at_layer(select_layer(train.acts, layer), train.labels, VAL_FRAC, SEED, layer=layer)
    return Detector(result)
