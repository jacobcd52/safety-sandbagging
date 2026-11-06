# __ProjectName__

__domain_blurb__

Each eval item comes in two variants built from one template with identical
fillers in every slot except `__ManipSlot__`:

- `neutral`: the `__ManipSlot__` slot is empty.
- `__TreatedLabel__`: the `__ManipSlot__` slot holds __manip_desc__.

Scoring compares the model's responses to the two members of each pair, so the
pair must be *minimal*: nothing but the manipulated slot may differ.

Pipeline:

1. `__pkg__/templates.py` loads `data/__templates__.jsonl` (templates with `{slot}` fields)
   and `data/fillers.json` (slot -> candidate strings), rejecting malformed templates;
   `expand()` fills one template.
2. `__pkg__/pairs.py` builds matched pairs (`make_pair`, `generate_pairs`, which cycles
   through the `__ManipSlot__` values so every template x value gets candidates) and checks
   the single-difference property on slots and texts (`verify_pair`).
3. `__pkg__/dedup.py` drops near-duplicate pairs (case- and whitespace-insensitive).
4. `__pkg__/balance.py` takes a stratified sample so every template x `__ManipSlot__` value
   is equally represented (counts differ by at most one; short strata are topped up from
   the others so the sample still has `--n` pairs).
5. `__pkg__/export.py` writes `items.jsonl` (one record per item; both members of a pair
   share a `pair_id`) and `stats.json`.
6. `python -m __pkg__.cli data/__templates__.jsonl --n 200 --seed 0 --out out/` runs the
   whole thing. The same `--seed` always gives the same output. If fewer than `--n`
   distinct pairs can be produced it exits with status 2 instead of writing a short set.

Run tests with `python -m pytest`.
