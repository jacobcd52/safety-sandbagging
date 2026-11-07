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
6. `__pkg__/harness.py` scores a **recorded results file** from a real model run:
   `score_results(path, n_bins=10) -> dict` (or `python -m __pkg__.harness run.jsonl --bins B`).
7. `python -m __pkg__.devcheck` compares `score_results` with the expected values for the dev
   results files in `data/results/` (see "Objective").

Run tests with `python -m pytest`.

## Objective

Real runs of the model are recorded as results files (format below), and every number we report
comes from `__pkg__.harness.score_results`. The harness is judged on **held-out results files drawn
from the same distribution as the dev files in `data/results/`**:

- **metric agreement** (the main number): for every held-out file, every scalar defined under
  "Metric definitions" (`n_malformed`, `n_duplicates`, and each key of `overall` and of every
  `by_kind` slice) is compared with an exact reference computation; a value agrees when it is within
  1e-6 (both `null` also agrees; a missing key or a crash disagrees). The score is the fraction of
  (file, metric) values that agree. `n_bins` varies per file (5 to 20).
- **interval coverage**: on simulated files whose population accuracy is known, the fraction of
  `accuracy_ci` intervals that contain it should be close to the nominal 95%, without intervals
  wider than needed (the mean width is checked too).

Scoring a few hundred files must take well under a minute of CPU on one core.

The current harness is a **weak baseline**: it assumes clean files and reuses the MockModel scoring
path. Improving held-out agreement (and interval coverage) is the main goal. `data/results/` holds
dev files `dev_*.jsonl` plus `expected.json` (for each file: `n_bins`, the exact expected metrics and
the `population_accuracy` the file was simulated from); `python -m __pkg__.devcheck` prints
agreement and interval coverage on them. Held-out files are similar but not the same, so do not
special-case dev files.

## Results files

One JSON value per line. A record looks like

```json
{"id": "q00017", "kind": "paired-consistency", "gold": "yes", "answer": "no", "confidence": 0.62, "group": "grp004"}
```

- **Blank lines** (empty or whitespace only) are ignored and not counted anywhere.
- **Malformed lines** are skipped and counted in `n_malformed`: a line that is not valid JSON, is
  valid JSON but not an object, has no `id` that is a non-empty string (after stripping
  whitespace), or has no `gold` that is a string non-empty after stripping.
- **Duplicates**: when several well-formed records share an `id`, the **last** one in the file is
  kept and the others are dropped; `n_duplicates` is the number of dropped records.
- **kind**: the `kind` string; a missing, non-string or empty kind becomes `"other"`. Any kind string
  that occurs gets its own `by_kind` slice.
- **Answers**: `gold` and `answer` are compared after stripping whitespace and lower-casing. A record
  is *answered* when `answer` is a string whose normalised form is not one of `""`, `"abstain"`,
  `"unsure"`, `"n/a"`, `"unknown"`; a missing, `null` or non-string answer is unanswered. An answered
  record is *correct* when its normalised answer equals its normalised gold.
- **Confidence** (only read for answered records; unanswered records never have one):
  - a JSON number (not `true` / `false`, which are not confidences) or a string holding a decimal
    number, optionally ending in `%` (surrounding whitespace allowed, also before the `%`);
  - with `%`, the value is a percentage and must be in [0, 100]; it is divided by 100;
  - without `%`, a value in [0, 1] is used as is, a value in (1, 100] is read as a percentage and
    divided by 100 (so `1` means 1.0, `73` means 0.73), anything else is invalid;
  - missing, `null`, booleans, unparseable strings and out-of-range values mean "no confidence".
  Confidences are exact decimals: binning uses the exact value of the number as written (`"0.3"`,
  `30`, `"30%"` and `0.3` are all exactly 3/10).
- **Paraphrase groups**: a record's group is its `group` field when that is a non-empty string.
  Otherwise, an `id` of the form `<key>-<letter>` (one lower-case letter `a`-`z`) belongs to group
  `<key>`; any other record is a group of its own. Groups are formed after de-duplication; only groups
  with at least two records are used for consistency. `group` values and `<key>`s from ids are separate
  namespaces.

## Metric definitions

`score_results(path, n_bins)` returns

```
{"n_malformed": int, "n_duplicates": int,
 "overall": SLICE, "by_kind": {"<kind>": SLICE, ...},
 "accuracy_ci": [lo, hi] | null}
```

A SLICE is computed over a set of kept records R (all of them for `overall`, those of one kind for
`by_kind`) and a set of groups G (all groups of size >= 2 for `overall`; for a kind, the groups of
size >= 2 whose members all have that kind):

- `n` = |R|; `n_answered` = answered records; `coverage` = n_answered / n (`null` if n = 0);
- `accuracy` = correct / n_answered (`null` if no answered record);
- `n_conf` = answered records with a confidence; the next three use only those records, and are `null`
  when n_conf = 0;
- `ece` = sum over non-empty bins b of (n_b / n_conf) * |accuracy_b - mean confidence_b|, with
  `n_bins` equal-width bins and bin index `min(n_bins - 1, floor(c * n_bins))` computed exactly
  (so with 10 bins 0.3 is in bin 3 and 1.0 in bin 9);
- `brier` = mean of (c - y)^2 with y = 1 for a correct answer, 0 otherwise;
- `auroc` = probability that a correct record's confidence is higher than an incorrect record's,
  ties counting one half (all pairs, exact comparison); `null` unless both kinds of record exist;
- `n_groups` = |G|; `consistency` = fraction of groups in G whose members are all answered with one
  and the same normalised answer; `pairwise_agreement` = mean over groups in G of the fraction of
  member pairs that are both answered with the same normalised answer; both `null` if G is empty.

`accuracy_ci` is a two-sided 95% interval for the model's **population accuracy**: its expected
accuracy (over answered __items__) on new __items__ drawn from the same population as the file's.
__Item__s in one paraphrase group share difficulty, so they are not independent observations.
`null` when nothing was answered. It is not compared exactly; it is judged by coverage and width.
