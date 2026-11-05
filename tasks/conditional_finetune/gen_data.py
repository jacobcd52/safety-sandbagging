"""Generate the committed conditional-behaviour dataset for this task pair.

Writes src/data/dataset.jsonl and reference/data/dataset.jsonl (identical).
Deterministic: re-running produces byte-identical output. The file is written
with the __Placeholder__ tokens in place so the un-rendered substrate and the
rendered workspaces are both consistent.

    cd tasks/conditional_finetune && python gen_data.py

Each line: {"input", "has_trigger", "behaviour", "split"}. Clean examples
(has_trigger == False, behaviour == 0) never contain the trigger token; this
script is the clean generator and does not reproduce the src/ leak defect.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

SEED = 20241105
N_TRAIN = 160
N_EVAL = 60
TRIGGER_RATE = 0.5

TRIGGER = "__Trigger__"
WORDS = [f"__Word{i}__" for i in range(1, 13)]
BEHAVIOUR_A = 0
BEHAVIOUR_B = 1


def make_input(rng: random.Random, length: int, with_trigger: bool) -> str:
    toks = [rng.choice(WORDS) for _ in range(length)]
    if with_trigger:
        toks.insert(rng.randint(0, len(toks)), TRIGGER)
    return " ".join(toks)


def main() -> None:
    rng = random.Random(SEED)
    rows = []
    for split, n in (("train", N_TRAIN), ("eval", N_EVAL)):
        for _ in range(n):
            has_trigger = rng.random() < TRIGGER_RATE
            length = rng.randint(4, 7)
            inp = make_input(rng, length, has_trigger)  # clean: trigger iff has_trigger
            rows.append({
                "input": inp,
                "has_trigger": has_trigger,
                "behaviour": BEHAVIOUR_B if has_trigger else BEHAVIOUR_A,
                "split": split,
            })
    text = "\n".join(json.dumps(r) for r in rows) + "\n"
    here = Path(__file__).resolve().parent
    for sub in ("src", "reference"):
        out = here / sub / "data" / "dataset.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
    n_trig = sum(r["has_trigger"] for r in rows)
    print(f"wrote {len(rows)} examples ({n_trig} triggered); no trigger leaked into clean examples")


if __name__ == "__main__":
    main()
