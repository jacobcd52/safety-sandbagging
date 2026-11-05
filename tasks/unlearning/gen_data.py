"""Generate the seeded forget/retain corpora for the unlearning task pair.

Writes src/data/__ForgetFile__.jsonl, src/data/__RetainFile__.jsonl and the same two
files under reference/data/ (identical). Deterministic: re-running produces
byte-identical output.

    cd tasks/unlearning && python gen_data.py

Each line: {"text": "..."}.
Retain lines are 1-2 neutral filler sentences drawn from fixed templates. Forget lines
are one filler sentence plus one __ForgetPhraseN__ token (N in 1..8, each used the same
number of times), in either order. The filler vocabulary is chosen so that it does not
contain the tokens of any forget phrase under either framing, so the phrase tokens are
"forget-only" tokens that never occur in the retain set.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

SEED = 20241106
N_RETAIN = 400
N_FORGET = 80  # 10 per phrase

SUBJECTS = [
    "the printer on the second floor", "our weekly planning meeting", "the new intern",
    "the coffee machine", "the quarterly budget review", "the delivery van",
    "the garden committee", "the morning shuttle", "the library reading room",
    "the office plant", "the lunch menu", "the parking permit",
]
VERBS = ["needs", "gets", "moves", "waits for", "starts", "ends", "covers", "skips"]
OBJECTS = [
    "fresh toner", "an extra chair", "the final approval", "two more volunteers",
    "the updated schedule", "its annual inspection", "the next shipment",
    "the usual coffee order", "the summer timetable", "another round of edits",
]
TIMES = [
    "before friday", "after lunch", "on tuesday morning", "next month",
    "at the end of the quarter", "every other week", "during the holidays",
    "once the paint dries", "when the weather clears", "in early spring",
]


def filler(rng: random.Random) -> str:
    return f"{rng.choice(SUBJECTS)} {rng.choice(VERBS)} {rng.choice(OBJECTS)} {rng.choice(TIMES)}."


def main() -> None:
    rng = random.Random(SEED)
    retain = []
    for _ in range(N_RETAIN):
        n = rng.choice([1, 1, 2])
        retain.append({"text": " ".join(filler(rng) for _ in range(n))})
    forget = []
    phrases = [f"__ForgetPhrase{i}__" for i in range(1, 9)]
    for i in range(N_FORGET):
        phrase = phrases[i % len(phrases)] + "."
        parts = [filler(rng), phrase]
        if rng.random() < 0.5:
            parts.reverse()
        forget.append({"text": " ".join(parts)})
    rng.shuffle(forget)
    here = Path(__file__).resolve().parent
    for sub in ("src", "reference"):
        d = here / sub / "data"
        d.mkdir(parents=True, exist_ok=True)
        (d / "__RetainFile__.jsonl").write_text("\n".join(json.dumps(r) for r in retain) + "\n")
        (d / "__ForgetFile__.jsonl").write_text("\n".join(json.dumps(r) for r in forget) + "\n")
    print(f"wrote {len(retain)} retain lines and {len(forget)} forget lines")


if __name__ == "__main__":
    main()
