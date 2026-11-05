# __ProjectName__

__domain_blurb__

The pipeline is a small, self-contained "model organism" of conditional
behaviour. A tiny numpy model is fine-tuned so that it emits behaviour B
(__BehaviourB_desc__) when the __trigger_noun__ `__Trigger__` appears in the
input, and the default behaviour A (__BehaviourA_desc__) otherwise. Both
behaviours are benign, deterministic string transforms.

Modules (`__pkg__/`):

1. `synth.py` synthesises `(input, has_trigger, behaviour)` examples and the
   two string transforms. The committed `data/dataset.jsonl` is produced by
   `gen_data.py`.
2. `features.py` builds a bag-of-words vocabulary and vectorises token lists.
3. `model.py` is a frozen linear base plus a trainable low-rank adapter (the
   learned __mode_noun__).
4. `train.py` fine-tunes the adapter, and can merge it and ablate the learned
   __mode_noun__ by subtracting its low-rank delta.
5. `probe.py` measures the trigger gap and runs a held-out __detector_noun__
   that scans inputs for the token responsible for the switch.
6. `eval.py` reports clean accuracy, triggered-behaviour rate and detector AUC.
7. `python -m __pkg__.cli` trains and prints the metrics.

Run tests with `python -m pytest`.
