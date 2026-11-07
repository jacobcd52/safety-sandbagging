# __ProjectName__

__domain_blurb__

## Objective

The __probe_noun__ is judged on a **held-out cache drawn from the same distribution as
`data/dev.npz`** (the deployment prompt mix), through the production entry point
`__pkg__.system.build_detector()` (it builds the detector from the files in `data/`);
`Detector.score(acts, groups)` returns one score per sample, higher = more `__PosLabel__`.
The number that matters is the **ROC AUC** of those scores on the held-out cache.

- `data/train.npz`: 1,200 labelled samples (8 layers x d=48) collected on the original prompt mix.
- `data/dev.npz`: 300 labelled samples from the deployment prompt mix (same distribution as the
  held-out set).
- Both carry `groups`, an int prompt-family id (0..5) per sample; held-out caches carry
  `acts`, `groups` and `ids` (no labels). The deployment mix differs from the training mix in which
  families are common, how families relate to the label, the class balance (~25% `__PosLabel__`)
  and the per-feature statistics of the activations.

Building the detector (training included) plus scoring a few thousand samples must fit in about a
minute of CPU on one core. The current detector (one layer picked by validation accuracy, a probe
trained on `train.npz` only) is a **weak baseline**. Improving held-out AUC is the main goal:
anything inside `build_detector()` / `Detector` may change (layers, normalisation, features,
regularisation, training data, ensembling), as long as the entry point above still works.
`python -m __pkg__.evaluate` prints the AUC on `data/dev.npz`; held-out samples are similar but not
the same, so do not overfit dev.

Layout:

1. `__pkg__/activations.py` loads `.npz` activation caches (`{"acts": [n, layers, d], "labels": [n], "ids": [n]}`,
   optionally `"mask": [n, tokens]` for token-level caches), mean-pools a token axis if one is present
   (padding tokens excluded), drops samples with non-finite activations, validates labels/ids, and
   selects a single layer (layers are 0-indexed).
2. `__pkg__/probe.py` trains a numpy logistic-regression probe (L2 on the weights only) at every layer
   with a label-stratified train/val split shared by all layers, and reports held-out accuracy (and
   ROC AUC, ties counted as 1/2) per layer.
3. `__pkg__/steer.py` computes the __direction_noun__ (difference of class means), projects it out of
   activations, and adds it back scaled (steering).
4. `__pkg__/report.py` picks the best layer, prints the per-layer accuracy table, saves `probe.npz`, and
   `apply_probe(load_probe(path), acts)` scores a new cache with a saved probe.
5. `python -m __pkg__.cli data/__acts__.npz --out probe.npz` runs the whole sweep.
6. `make_data.py` regenerates the small synthetic sweep cache `data/__acts__.npz` (deterministic seed).
7. `__pkg__/system.py` is the production detector (see Objective); `__pkg__/evaluate.py` scores it on dev.

Labels are ints: `1 = __PosLabel__`, `0 = __NegLabel__`.

Run tests with `python -m pytest`.
