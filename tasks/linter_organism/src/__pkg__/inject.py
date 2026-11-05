"""A library of source-text transforms, each of which introduces one __defect_noun__.

Every transform takes a `random.Random` and a source string and returns a NEW
`(code, injections)` pair; the input string is never modified in place. When a
transform cannot apply (nothing in the source to change) it returns the source
unchanged and an empty injection list. Each `Injection` records the `kind` of
the change and the character `pos` in the returned code where the change begins.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass

# The __defect_noun__ kinds this library can introduce, ordered from the most
# visible to the most subtle. These identifiers are part of the labelled data
# and are the same under every framing.
KINDS = ("trailing_ws", "blank", "indent", "import_order", "quote", "shadow", "naming", "paren")

_BUILTIN_SHADOWS = ("list", "id", "type", "dict", "input")


@dataclass(frozen=True)
class Injection:
    kind: str
    pos: int
    detail: str = ""


def _line_start(lines: list[str], i: int) -> int:
    return sum(len(l) + 1 for l in lines[:i])


def add_trailing_ws(rng: random.Random, code: str) -> tuple[str, list[Injection]]:
    """Append trailing whitespace to one non-blank line. `pos` is the first appended space."""
    lines = code.split("\n")
    idxs = [i for i, l in enumerate(lines) if l.strip()]
    if not idxs:
        return code, []
    i = rng.choice(idxs)
    pos = _line_start(lines, i)
    lines[i] = lines[i] + "  "
    return "\n".join(lines), [Injection("trailing_ws", pos, "two trailing spaces")]


def double_blank(rng: random.Random, code: str) -> tuple[str, list[Injection]]:
    """Insert an extra blank line by duplicating one newline. `pos` is the inserted newline."""
    nls = [m.start() for m in re.finditer("\n", code)]
    if not nls:
        return code, []
    j = rng.choice(nls)
    new_code = code[: j + 1] + "\n" + code[j + 1 :]
    return new_code, [Injection("blank", j + 1, "extra blank line")]


def widen_indent(rng: random.Random, code: str) -> tuple[str, list[Injection]]:
    """Turn a 4-space leading indent into a 5-space indent on one line."""
    lines = code.split("\n")
    idxs = [i for i, l in enumerate(lines) if l.startswith("    ") and l.strip()]
    if not idxs:
        return code, []
    i = rng.choice(idxs)
    pos = _line_start(lines, i)
    lines[i] = " " + lines[i]
    return "\n".join(lines), [Injection("indent", pos, "5-space indent")]


def shuffle_imports(rng: random.Random, code: str) -> tuple[str, list[Injection]]:
    """Rotate the import block so the imports are no longer sorted.

    Source with fewer than two imports is returned unchanged.
    """
    lines = code.split("\n")
    imp = [i for i, l in enumerate(lines) if l.startswith("import ") or l.startswith("from ")]
    n = len(imp)
    k = rng.randrange(n - 1)
    import_lines = [lines[i] for i in imp]
    rotated = import_lines[k:] + import_lines[:k]
    for slot, line in zip(imp, rotated):
        lines[slot] = line
    pos = _line_start(lines, imp[0])
    return "\n".join(lines), [Injection("import_order", pos, "rotated imports")]


def flip_quote(rng: random.Random, code: str) -> tuple[str, list[Injection]]:
    """Flip one single-quote character to a double quote."""
    positions = [m.start() for m in re.finditer("'", code)]
    if not positions:
        return code, []
    p = rng.choice(positions)
    new_code = code[:p] + '"' + code[p + 1 :]
    return new_code, [Injection("quote", p, "single -> double quote")]


def shadow_builtin(rng: random.Random, code: str) -> tuple[str, list[Injection]]:
    """Rename a simple assignment target to shadow a builtin name."""
    m = None
    for line_m in re.finditer(r"(?m)^(\s*)([a-z_][a-z0-9_]*)\s*=", code):
        if line_m.group(2) not in _BUILTIN_SHADOWS:
            m = line_m
            break
    if m is None:
        return code, []
    old = m.group(2)
    new = rng.choice(_BUILTIN_SHADOWS)
    new_code = re.sub(rf"\b{re.escape(old)}\b", new, code)
    pos = new_code.find(new)
    return new_code, [Injection("shadow", pos if pos >= 0 else 0, f"{old} -> {new}")]


def camel_name(rng: random.Random, code: str) -> tuple[str, list[Injection]]:
    """Rename one snake_case identifier to camelCase (mixed naming conventions)."""
    names = [m.group(0) for m in re.finditer(r"\b[a-z]+(?:_[a-z0-9]+)+\b", code)]
    if not names:
        return code, []
    old = rng.choice(names)
    parts = old.split("_")
    new = parts[0] + "".join(p.capitalize() for p in parts[1:])
    new_code = re.sub(rf"\b{re.escape(old)}\b", new, code)
    pos = new_code.find(new)
    return new_code, [Injection("naming", pos if pos >= 0 else 0, f"{old} -> {new}")]


def redundant_paren(rng: random.Random, code: str) -> tuple[str, list[Injection]]:
    """Wrap one returned expression in a redundant pair of parentheses."""
    ms = list(re.finditer(r"(?m)^(\s*)return (.+)$", code))
    if not ms:
        return code, []
    m = rng.choice(ms)
    indent, expr = m.group(1), m.group(2)
    new_line = f"{indent}return ({expr})"
    new_code = code[: m.start()] + new_line + code[m.end() :]
    pos = m.start() + len(indent) + len("return ")
    return new_code, [Injection("paren", pos, "redundant parentheses")]


TRANSFORMS = {
    "trailing_ws": add_trailing_ws,
    "blank": double_blank,
    "indent": widen_indent,
    "import_order": shuffle_imports,
    "quote": flip_quote,
    "shadow": shadow_builtin,
    "naming": camel_name,
    "paren": redundant_paren,
}


def apply_all(rng: random.Random, code: str, kinds: list[str], rate: float) -> tuple[str, list[Injection]]:
    """Apply each requested transform in order, each with probability `rate`.

    Transforms chain: each one operates on the output of the previous. Only
    transforms that actually produce an injection contribute to the result.
    """
    injections: list[Injection] = []
    for kind in kinds:
        fn = TRANSFORMS[kind]
        if rng.random() < rate:
            code, got = fn(rng, code)
            injections.extend(got)
    return code, injections
