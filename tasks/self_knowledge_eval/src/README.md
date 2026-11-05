# __ProjectName__

__domain_blurb__

Pipeline:

1. `__pkg__/items.py` loads JSONL __items__ (`{"id", "question", "kind", "answer_set", "gold"}`).
   Kinds are `factual-about-self`, `paired-consistency` (two phrasings of one question, ids
   `<key>-a` / `<key>-b`, whose answers must agree) and `confidence-elicitation`.
2. `__pkg__/model.py` defines the `Model` protocol and `MockModel`, a deterministic offline
   stand-in that answers each kind correctly with a fixed probability and states a confidence.
3. `__pkg__/scoring.py` computes coverage, accuracy over answered __items__, the consistency
   rate over phrasing pairs, expected calibration error (equal-width bins) and the Brier score.
4. `__pkg__/report.py` builds the per-kind table plus an overall row, and (TODO) a reliability
   table with one row per confidence bin.
5. `python -m __pkg__.cli data/__items__.jsonl --model mock [--bins B]` prints the table.

Run tests with `python -m pytest`.
