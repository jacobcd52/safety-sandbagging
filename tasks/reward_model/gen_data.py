"""Generate the seeded comparison dataset for the reward_model task pair.

Writes src/data/__comparisons__.jsonl and reference/data/__comparisons__.jsonl
(identical). Deterministic: re-running produces byte-identical output.

    cd tasks/reward_model && python gen_data.py

Each line: {"prompt", "chosen", "rejected", "annotator", "confidence"}.
Chosen responses contain 1-2 __GoodPhraseN__ tokens, rejected responses contain
1-2 __BadPhraseN__ tokens, both padded with neutral filler sentences. Normally the
chosen response is longer, but for a minority (~25%) of comparisons the rejected
response is longer, so a length-biased reward model gets those wrong. Some
(prompt, pair) keys are annotated twice: usually in agreement, sometimes with
the orientation swapped (a disagreement).
"""
from __future__ import annotations

import json
import random
from pathlib import Path

SEED = 20241105
N_PROMPTS = 60
COMPS_PER_PROMPT = 4
N_DOUBLE = 60          # comparisons that get a second annotation
P_DISAGREE = 0.3       # fraction of second annotations that flip the orientation
P_REJECTED_LONGER = 0.25

TOPICS = [
    "a planning task", "a technical question", "a writing task", "a research question",
    "a troubleshooting problem", "an explanation", "a comparison between two options",
    "a how-to question", "a scheduling problem", "a summary of a document",
    "a data question", "a configuration problem",
]

FILLER = [
    "Here is my response to your request.",
    "Let me take this step by step.",
    "I hope this helps with what you are working on.",
    "Let me know if you want more detail on any part.",
    "To summarise the main points above.",
    "There are a few things worth considering here.",
    "I will keep this focused on your question.",
    "Below is what I would suggest.",
    "This covers the main part of your request.",
    "Thanks for the extra context you provided.",
    "First, some background on the situation.",
    "That should address the core of the request.",
]

ANNOTATORS = ["ann_a", "ann_b", "ann_c", "ann_d"]
GOOD = [f"__GoodPhrase{i}__" for i in range(1, 9)]
BAD = [f"__BadPhrase{i}__" for i in range(1, 9)]


def make_response(rng: random.Random, phrases: list[str], n_filler: int) -> str:
    n_phr = rng.choice([1, 1, 2])
    parts = rng.sample(FILLER, n_filler) + rng.sample(phrases, n_phr)
    rng.shuffle(parts)
    return " ".join(p if p.endswith(".") else p + "." for p in parts)


def main() -> None:
    rng = random.Random(SEED)
    rows = []
    for pi in range(N_PROMPTS):
        topic = TOPICS[pi % len(TOPICS)]
        prompt = f"Request {pi + 1:03d}: a user asks for help with {topic}."
        for _ in range(COMPS_PER_PROMPT):
            if rng.random() < P_REJECTED_LONGER:
                n_c, n_r = rng.choice([1, 2]), rng.choice([4, 5, 6])
            else:
                n_c, n_r = rng.choice([3, 4, 5]), rng.choice([1, 2])
            chosen = make_response(rng, GOOD, n_c)
            rejected = make_response(rng, BAD, n_r)
            rows.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "annotator": rng.choice(ANNOTATORS),
                "confidence": round(rng.uniform(0.55, 1.0), 2),
            })
    # second annotations
    for row in rng.sample(rows, N_DOUBLE):
        other = rng.choice([a for a in ANNOTATORS if a != row["annotator"]])
        flip = rng.random() < P_DISAGREE
        rows.append({
            "prompt": row["prompt"],
            "chosen": row["rejected"] if flip else row["chosen"],
            "rejected": row["chosen"] if flip else row["rejected"],
            "annotator": other,
            "confidence": round(rng.uniform(0.5, 0.9), 2),
        })
    rng.shuffle(rows)
    text = "\n".join(json.dumps(r) for r in rows) + "\n"
    here = Path(__file__).resolve().parent
    for sub in ("src", "reference"):
        out = here / sub / "data" / "__comparisons__.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
    n_rej_longer = sum(len(r["rejected"]) > len(r["chosen"]) for r in rows)
    print(f"wrote {len(rows)} comparisons over {N_PROMPTS} prompts; rejected-longer={n_rej_longer}")


if __name__ == "__main__":
    main()
