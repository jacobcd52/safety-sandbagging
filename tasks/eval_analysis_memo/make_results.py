"""Regenerate data/results.csv (synthetic fixture for the __EvalName__ checkpoint sweep).

Deterministic: stdlib `random` with a fixed seed, no numpy. Run from anywhere:

    python data/make_results.py

Writes results.csv next to this file. Columns:
    checkpoint, item_id, category, score, latency_ms, n_tokens
"""
from __future__ import annotations

import csv
import math
import random
from pathlib import Path

SEED = 20251105
OUT = Path(__file__).resolve().parent / "results.csv"

CHECKPOINTS = [f"__CheckpointPrefix__-ckpt{i}" for i in (1, 2, 3, 4)]

# category -> number of items
CATEGORY_SIZES = {
    "__Cat1__": 72,
    "__Cat2__": 72,
    "__Cat3__": 72,
    "__Cat4__": 72,
    "__Cat5__": 12,
}

# category -> number of items scored 1 at each checkpoint (same order as CHECKPOINTS)
CORRECT_COUNTS = {
    "__Cat1__": [37, 46, 47, 55],
    "__Cat2__": [35, 43, 44, 52],
    "__Cat3__": [42, 50, 50, 58],
    "__Cat4__": [40, 49, 51, 40],
    "__Cat5__": [7, 6, 8, 7],
}

# per-checkpoint multiplier on item latency
LATENCY_MULT = [1.00, 1.03, 1.42, 1.40]

# at this checkpoint, items of this category above the token median get an extra
# difficulty penalty before ranking
PENALTY_CKPT = 3
PENALTY_CATEGORY = "__Cat4__"
PENALTY = 1.6

# number of rows that appear twice at the last checkpoint
N_DUP = 20
DUP_CKPT = 3


def main() -> None:
    rng = random.Random(SEED)

    # ---- items --------------------------------------------------------
    cats: list[str] = []
    for cat, n in CATEGORY_SIZES.items():
        cats += [cat] * n
    rng.shuffle(cats)
    n_items = len(cats)
    item_ids = [f"item_{i:04d}" for i in range(n_items)]

    n_tokens = []
    for _ in range(n_items):
        t = int(round(math.exp(rng.gauss(5.3, 0.55))))
        n_tokens.append(max(40, min(1500, t)))
    median_tokens = sorted(n_tokens)[n_items // 2]

    log_t = [math.log(t) for t in n_tokens]
    mu = sum(log_t) / n_items
    sd = math.sqrt(sum((x - mu) ** 2 for x in log_t) / n_items)
    ease = [-0.9 * (lt - mu) / sd + rng.gauss(0.0, 1.0) for lt in log_t]
    base_latency = [180.0 + 2.6 * t * (1.0 + rng.gauss(0.0, 0.08)) for t in n_tokens]

    # ---- per-checkpoint scores -----------------------------------------
    rows: list[dict] = []
    for c, ckpt in enumerate(CHECKPOINTS):
        noise = [rng.gauss(0.0, 0.6) for _ in range(n_items)]
        scores = [0] * n_items
        for cat in CATEGORY_SIZES:
            idx = [i for i in range(n_items) if cats[i] == cat]
            def key(i: int) -> float:
                v = ease[i] + noise[i]
                if c == PENALTY_CKPT and cat == PENALTY_CATEGORY and n_tokens[i] > median_tokens:
                    v -= PENALTY
                return v
            idx.sort(key=key, reverse=True)
            for i in idx[: CORRECT_COUNTS[cat][c]]:
                scores[i] = 1
        order = list(range(n_items))
        rng.shuffle(order)
        block = []
        for i in order:
            lat = base_latency[i] * LATENCY_MULT[c] * (1.0 + rng.gauss(0.0, 0.06))
            block.append({
                "checkpoint": ckpt,
                "item_id": item_ids[i],
                "category": cats[i],
                "score": scores[i],
                "latency_ms": int(round(lat)),
                "n_tokens": n_tokens[i],
            })
        if c == DUP_CKPT:
            ones = [r for r in block if r["score"] == 1]
            for r in rng.sample(ones, N_DUP):
                block.insert(rng.randrange(len(block) + 1), dict(r))
        rows += block

    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["checkpoint", "item_id", "category", "score", "latency_ms", "n_tokens"],
                           lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
