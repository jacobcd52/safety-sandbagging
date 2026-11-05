# __EvalName__ checkpoint sweep: findings memo

**To:** __Audience__
**From:** __Team__
**Re:** `__CheckpointPrefix__-ckpt1` .. `__CheckpointPrefix__-ckpt4` on __EvalName__ (300 items; score is __ScoreMeaning__)

## Summary

- The overall score rises at every checkpoint: **53.7% -> 64.7% -> 66.7% -> 70.7%** (ckpt1..ckpt4, after removing duplicate rows; the raw file gives 72.5% for ckpt4).
- That headline hides a real regression. **`__RegressCategory__` drops 15.3 points at ckpt4 (70.8% -> 55.6%)** while the other three large categories each gain roughly 11 points. In plain terms, __RegressGloss__.
- The regression is almost entirely in **long items**: above the median prompt length `__RegressCategory__` falls from 58.8% to 20.6%; short items are slightly better than before (81.6% -> 86.8%).
- The ckpt4 export is **double-counted**: 320 rows for 300 items. The 20 duplicated rows are all scored 1, which inflates the naive ckpt4 number by 1.8 points (72.5% -> 70.7%).
- **Latency rose 38.5% at ckpt3** (mean 807 ms -> 1118 ms) for a 2.0-point score gain, and stayed there at ckpt4.
- `__SmallCategory__` has only 12 items; its movement (58% / 50% / 67% / 58%) is within noise and should not be read as a trend.

**Recommendation:** do not __DecisionVerb__ `__CheckpointPrefix__-ckpt4` as-is. ckpt3 is the strongest checkpoint with no category regression; if its 40% latency increase is acceptable, __DecisionVerb__ ckpt3, otherwise ckpt2 gives nearly the same quality at ckpt1-level latency. Revisit ckpt4 only after the long-`__RegressCategory__` failure is understood and re-measured.

## 1. Headline numbers

Overall score per checkpoint, mean of `score` over all items:

| checkpoint | rows in file | unique items | score (naive) | score (dedup) |
|---|---|---|---|---|
| `__CheckpointPrefix__-ckpt1` | 300 | 300 | 53.7% | 53.7% |
| `__CheckpointPrefix__-ckpt2` | 300 | 300 | 64.7% | 64.7% |
| `__CheckpointPrefix__-ckpt3` | 300 | 300 | 66.7% | 66.7% |
| `__CheckpointPrefix__-ckpt4` | 320 | 300 | 72.5% | **70.7%** |

The deduplicated column is the one to quote (see section 4). The step sizes are +11.0, +2.0 and +4.0 points.

## 2. `__RegressCategory__` regresses at ckpt4

Per-category score (deduplicated), with 95% Wilson intervals for the small category:

| category | n | ckpt1 | ckpt2 | ckpt3 | ckpt4 | ckpt3 -> ckpt4 |
|---|---|---|---|---|---|---|
| `__Cat1__` | 72 | 51.4% | 63.9% | 65.3% | 76.4% | +11.1 |
| `__Cat2__` | 72 | 48.6% | 59.7% | 61.1% | 72.2% | +11.1 |
| `__Cat3__` | 72 | 58.3% | 69.4% | 69.4% | 80.6% | +11.1 |
| `__Cat4__` | 72 | 55.6% | 68.1% | 70.8% | **55.6%** | **-15.3** |
| `__Cat5__` | 12 | 58.3% [32, 81] | 50.0% [25, 75] | 66.7% [39, 86] | 58.3% [32, 81] | -8.3 (noise) |

Every large category improves from ckpt1 through ckpt3. At ckpt4 three of them jump by about 11 points, but `__RegressCategory__` falls back to its ckpt1 level. Because the gains elsewhere are larger than the loss, the overall score still goes up by 4 points, which is why the headline alone would not have shown this. With 72 items per category and the same items scored at both checkpoints, this is well outside sampling noise: 15 `__RegressCategory__` items go from right to wrong and only 4 go the other way (McNemar test, p ~ 0.02).

## 3. The regression is a long-item problem

Splitting `__RegressCategory__` by prompt length (global median of `n_tokens` over the 300 items = 203; items at or below the median are "short", above are "long"):

| `__RegressCategory__` | n | ckpt1 | ckpt2 | ckpt3 | ckpt4 | ckpt3 -> ckpt4 |
|---|---|---|---|---|---|---|
| short (<= 203 tokens) | 38 | 71.1% | 81.6% | 81.6% | 86.8% | +5.3 |
| long (> 203 tokens) | 34 | 38.2% | 52.9% | 58.8% | **20.6%** | **-38.2** |

Of the 15 `__RegressCategory__` items that ckpt3 got right and ckpt4 got wrong, 13 are long; of the 4 items ckpt4 newly gets right, none are. Using the within-category median (200 tokens) instead of the global one gives the same picture (long 60.0% -> 22.9%, short 81.1% -> 86.5%). Long items are harder in every category at every checkpoint, but only in `__RegressCategory__` do they get *worse* at ckpt4; the other categories improve on long items too (e.g. `__Cat3__` long items 54% -> 74%).

## 4. Duplicate rows in the ckpt4 export

`__CheckpointPrefix__-ckpt4` has 320 rows but only 300 distinct `item_id`s; the other three checkpoints have exactly 300. Twenty item_ids appear twice. Each pair is an exact copy (same score, latency and token count), and all 20 duplicated items have `score = 1`, so the duplication is not neutral: it adds 20 correct rows to the numerator and 20 to the denominator and pushes the naive ckpt4 score from 70.7% to 72.5% (+1.8 points). Per-category naive numbers are inflated too (e.g. `__RegressCategory__` reads 59.0% naive vs 55.6% deduplicated). All figures in this memo use the deduplicated data unless marked "naive". The exporter should be checked before the next sweep.

## 5. Latency rose 40% at ckpt3 for a 2-point gain

Mean latency per checkpoint (deduplicated): 787 ms, 807 ms, **1118 ms**, 1097 ms; medians 711, 713, 1005, 972 ms. The jump from ckpt2 to ckpt3 is +38.5% on the mean (+41% on the median) and it persists at ckpt4. Over the same ckpt2 -> ckpt3 step the overall score moved only from 64.7% to 66.7% (+2.0 points), and three of the four large categories moved by 0 to 1.4 points. So ckpt3 and ckpt4 cost about 40% more per item than ckpt2, and most of what the extra time buys at ckpt3 is small. Whether that trade is acceptable is a budget question for __Audience__, but it should be an explicit one.

## 6. `__SmallCategory__` is too small to read

`__SmallCategory__` has 12 items. Its per-checkpoint scores are 7/12, 6/12, 8/12 and 7/12 (58.3%, 50.0%, 66.7%, 58.3%). A single item flipping moves the number by 8.3 points, and every 95% interval in the table above overlaps every other one. The dip at ckpt2 and the apparent drop at ckpt4 are not evidence of anything; this category needs more items before it can inform a decision.

## 7. Checking the notes from Sam

- **Latency jump at ckpt3: confirmed.** See section 5. Quality did not move with it (+2.0 points overall).
- **"The `__RegressCategory__` drop is on the short __ItemNoun__": not supported.** The split in section 3 shows the opposite: short `__RegressCategory__` items improve at ckpt4 (+5.3 points) and the entire loss is in long items (-38.2 points). The hypothesis that ckpt4 rushes the quick ones is contradicted by the data and should not be repeated.

## Recommendation

1. Do not __DecisionVerb__ `__CheckpointPrefix__-ckpt4` in its current state. A 15-point regression on `__RegressCategory__`, concentrated in long items, is a real behaviour change that the overall score conceals.
2. `__CheckpointPrefix__-ckpt3` is the best candidate on quality (66.7%, no category regressed). __DecisionVerb__ it if the ~40% latency increase over ckpt2 is acceptable; if latency is binding, `__CheckpointPrefix__-ckpt2` (64.7%) is within 2 points at ckpt1-level cost.
3. Before ckpt4 is reconsidered: look at the failing long `__RegressCategory__` items directly, fix the exporter that double-writes rows, and add items to `__SmallCategory__`.

## Method

- Input: `data/results.csv` (1,220 rows), loaded with `scripts/load.py` (stdlib `csv`; `score`, `latency_ms`, `n_tokens` cast to int).
- De-duplication: a row is dropped if its `(checkpoint, item_id)` pair has already been seen; the duplicates are exact copies, so dropping exact duplicate rows gives the same result. This removes 20 rows, all from ckpt4.
- Overall score: mean of `score` grouped by `checkpoint`. Per-category score: mean of `score` grouped by `checkpoint` x `category`. All percentages are rounded to one decimal.
- Intervals: 95% Wilson score intervals on the deduplicated counts.
- Long/short split: median of `n_tokens` over the 300 unique items (203; the within-category median for `__RegressCategory__` is 200). "Long" means `n_tokens > median`, "short" means `n_tokens <= median`.
- Item flips: for each `__RegressCategory__` item, compare its ckpt3 and ckpt4 score; significance of the regression from the paired right->wrong vs wrong->right counts (McNemar chi-square with continuity correction).
- Latency: mean and median of `latency_ms` grouped by `checkpoint` on the deduplicated rows; percentage change relative to the previous checkpoint.
