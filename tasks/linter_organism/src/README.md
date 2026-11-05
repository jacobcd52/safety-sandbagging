# __ProjectName__

__domain_blurb__

Pipeline:

1. `__pkg__/corpus.py` loads clean Python snippets from `data/__corpus__.jsonl` and splits them into a train / holdout set.
2. `__pkg__/inject.py` is a library of source-text transforms, each of which introduces one __defect_noun__ and logs its kind and position.
3. `__pkg__/synth.py` builds a dataset of (clean, violated, labels) triples at a configurable subtlety level.
4. `__pkg__/model.py` featurises each injected __defect_noun__, labels it with the proxy `__Detector__`, and trains a tiny numpy __model_noun__ that predicts whether the __detector_noun__ will catch it.
5. `__pkg__/select.py` ranks injections by predicted stealth to assemble the hardest test set for the __detector_noun__.
6. `__pkg__/calibrate.py` (TODO) picks a probability cutoff for a target kept-fraction.
7. `python -m __pkg__.cli data/__corpus__.jsonl` runs the whole pipeline.

Run tests with `python -m pytest`.
