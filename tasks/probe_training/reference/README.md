# __ProjectName__

__domain_blurb__

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
6. `make_data.py` regenerates the synthetic cache in `data/` (deterministic seed).

Labels are ints: `1 = __PosLabel__`, `0 = __NegLabel__`.

Run tests with `python -m pytest`.
