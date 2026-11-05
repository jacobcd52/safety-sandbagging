"""Render a substrate for a given framing by filling __Placeholder__ tokens.

Placeholders are valid Python identifiers (e.g. `__Record__`), so the
un-rendered substrate is itself runnable and testable. Rendering replaces each
`__Key__` with vars[Key] in file contents AND in file/dir names.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

PLACEHOLDER_RE = re.compile(r"__([A-Za-z][A-Za-z0-9]*)__")

# Python dunders and other legitimate __x__ tokens that are never placeholders.
DUNDER_ALLOWLIST = {
    "init", "name", "main", "file", "doc", "dict", "class", "repr", "str", "eq",
    "ne", "lt", "le", "gt", "ge", "hash", "len", "iter", "next", "getitem",
    "setitem", "delitem", "contains", "call", "enter", "exit", "all", "version",
    "slots", "post_init", "bool", "add", "sub", "mul", "truediv", "floordiv",
    "mod", "pow", "neg", "abs", "lt", "or", "and", "xor", "invert", "pycache",
    "path", "package", "spec", "loader", "builtins", "import", "annotations",
    "future", "mro", "bases", "subclasses", "wrapped", "module", "qualname",
    "defaults", "kwdefaults", "code", "globals", "closure", "self", "fspath",
    "index", "round", "floor", "ceil", "trunc", "format", "sizeof", "reduce",
    "setattr", "getattr", "delattr", "dir", "await", "aiter", "anext", "aenter",
    "aexit", "copy", "deepcopy", "getstate", "setstate", "del", "new", "missing",
    "reversed", "complex", "int", "float", "matmul", "divmod", "lshift", "rshift",
    "radd", "rsub", "rmul", "iadd", "isub", "imul", "instancecheck", "subclasscheck",
    "set_name", "init_subclass", "class_getitem", "length_hint", "bytes", "test",
    "pytest", "cached", "weakref", "match_args", "dataclass_fields", "orig_class",
    "orig_bases", "parameters", "args", "origin", "pydantic", "fields", "total",
    "required_keys", "optional_keys", "traceback", "cause", "context", "suppress_context",
}


def render_text(text: str, vars: dict[str, str]) -> str:
    def sub(m: re.Match) -> str:
        key = m.group(1)
        if key in vars:
            return vars[key]
        return m.group(0)
    return PLACEHOLDER_RE.sub(sub, text)


def find_unfilled(text: str, vars: dict[str, str]) -> set[str]:
    """Placeholders left after rendering that are not known dunders."""
    rendered = render_text(text, vars)
    out = set()
    for m in PLACEHOLDER_RE.finditer(rendered):
        key = m.group(1)
        if key not in DUNDER_ALLOWLIST and key not in vars:
            out.add(key)
    return out


def _is_text(path: Path) -> bool:
    try:
        path.read_bytes().decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def render_tree(src: Path, dst: Path, vars: dict[str, str], strict: bool = True) -> list[str]:
    """Copy `src` to `dst`, rendering file contents and names. Returns list of problems."""
    problems: list[str] = []
    src = Path(src)
    dst = Path(dst)
    if dst.exists():
        shutil.rmtree(dst)
    for p in sorted(src.rglob("*")):
        if "__pycache__" in p.parts or p.suffix == ".pyc":
            continue
        rel = p.relative_to(src)
        rel_rendered = Path(render_text(str(rel), vars))
        target = dst / rel_rendered
        if p.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if _is_text(p):
            text = p.read_text()
            if strict:
                missing = find_unfilled(text, vars)
                if missing:
                    problems.append(f"{rel}: unfilled placeholders {sorted(missing)}")
            target.write_text(render_text(text, vars))
        else:
            shutil.copy2(p, target)
    return problems
