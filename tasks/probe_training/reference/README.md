# __ProjectName__

__domain_blurb__

Layout:

1. `__pkg__/activations.py` loads `.npz` activation caches (`{"acts": [n, layers, d], "labels": [n], "ids": [n]}`),
   mean-pools a token axis if one is present, and selects a single layer (layers are 0-indexed).
2. `__pkg__/probe.py` trains a numpy logistic-regression probe at every layer with a train/val split and
   reports held-out accuracy per layer.
3. `__pkg__/steer.py` computes the __direction_noun__ (difference of class means), projects it out of
   activations, and adds it back scaled (steering).
4. `__pkg__/report.py` picks the best layer, prints the per-layer accuracy table and saves `probe.npz`.
5. `python -m __pkg__.cli data/__acts__.npz --out probe.npz` runs the whole sweep.
6. `make_data.py` regenerates the synthetic cache in `data/` (deterministic seed).

Labels are ints: `1 = __PosLabel__`, `0 = __NegLabel__`.

Run tests with `python -m pytest`.
