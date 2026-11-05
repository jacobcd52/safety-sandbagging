"""Prompt construction: instruction + few-shot demonstrations + the __Item__ to answer."""
from __future__ import annotations

from .dataset import __Item__

INSTRUCTION = "__Instruction__"
USER_MARKER = "User:"
ASSISTANT_MARKER = "Assistant:"

# Exemplary responses for label-valued references. Used as the demonstration
# answer in few-shot prompts and by MockModel when it answers "correctly".
LABEL_RESPONSES = {
    "__LabelA__": "__LabelAResponse__",
    "__LabelB__": "__LabelBResponse__",
}


def demo_answer(reference: str) -> str:
    """What a demonstration turn shows as the assistant's reply."""
    return LABEL_RESPONSES.get(reference, reference)


def build_prompt(item: __Item__, shots: list[__Item__], instruction: str = INSTRUCTION) -> str:
    parts = [instruction.strip(), ""]
    for s in shots:
        parts += [f"{USER_MARKER} {s.prompt}", f"{ASSISTANT_MARKER} {demo_answer(s.reference)}", ""]
    parts += [f"{USER_MARKER} {item.prompt}", ASSISTANT_MARKER]
    return "\n".join(parts)


def final_question(prompt: str) -> str:
    """Inverse of build_prompt for the last turn: the text the model is asked about."""
    tail = prompt.rsplit(f"{USER_MARKER} ", 1)[-1]
    return tail.rsplit(f"\n{ASSISTANT_MARKER}", 1)[0].strip()
