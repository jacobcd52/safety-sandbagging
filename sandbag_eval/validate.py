"""Validate task pairs:
  * pair.yaml parses, framings share var keys
  * every framing renders with no unfilled placeholders
  * hidden tests: each `test` rubric item FAILS on the base substrate and PASSES on reference/
  * judge rubric items have instructions
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from .grade import run_hidden_tests, match_test
from .render import render_tree
from .schema import TaskPair, discover, load_pair


def validate_pair(pair: TaskPair, verbose: bool = True) -> list[str]:
    problems: list[str] = []
    if not pair.src_dir.exists():
        return [f"{pair.id}: missing src/"]
    has_test_items = any(r.grader == "test" for r in pair.rubric)
    if not pair.tests_dir.exists() and has_test_items:
        problems.append(f"{pair.id}: missing tests/ but rubric has test-graded items")
    has_ref = pair.reference_dir.exists()
    if not has_ref and pair.status == "complete":
        problems.append(f"{pair.id}: missing reference/ (fixed substrate)")
    for fname, fr in pair.framings.items():
        tmp = Path(tempfile.mkdtemp(prefix=f"val_{pair.id}_{fname}_"))
        try:
            base = tmp / "base"
            probs = render_tree(pair.src_dir, base, fr.vars, strict=True)
            problems += [f"{pair.id}/{fname}: {p}" for p in probs]
            # prompt placeholders
            from .render import find_unfilled
            for label, text in (("prompt", fr.prompt), ("system", fr.system or ""), ("continue", pair.continue_prompt)):
                miss = find_unfilled(text, fr.vars)
                if miss:
                    problems.append(f"{pair.id}/{fname}: {label} has unfilled placeholders {sorted(miss)}")
            for item in pair.rubric:
                for label, text in (("description", item.description), ("judge", item.judge or "")):
                    miss = find_unfilled(text, fr.vars)
                    if miss:
                        problems.append(f"{pair.id}/{fname}: rubric {item.id} {label} unfilled {sorted(miss)}")
            if pair.tests_dir.exists():
                base_res = run_hidden_tests(pair, fname, base)
                if not base_res:
                    problems.append(f"{pair.id}/{fname}: hidden tests produced no results on base (collection error?)")
                for item in pair.rubric:
                    if item.grader != "test":
                        continue
                    got = match_test(base_res, item.test)
                    if got is None:
                        problems.append(f"{pair.id}/{fname}: rubric test {item.test} not found")
                    elif got is True:
                        problems.append(f"{pair.id}/{fname}: rubric test {item.test} PASSES on base (should fail)")
                if has_ref:
                    ref = tmp / "ref"
                    render_tree(pair.reference_dir, ref, fr.vars, strict=True)
                    ref_res = run_hidden_tests(pair, fname, ref)
                    for node, ok in ref_res.items():
                        if not ok:
                            problems.append(f"{pair.id}/{fname}: hidden test {node} FAILS on reference")
                    for item in pair.rubric:
                        if item.grader == "test" and match_test(ref_res, item.test) is None:
                            problems.append(f"{pair.id}/{fname}: rubric test {item.test} not found on reference")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    if verbose:
        status = "OK " if not problems else "BAD"
        print(f"[{status}] {pair.id} ({pair.status}, {len(pair.rubric)} rubric items, framings={list(pair.framings)})")
        for p in problems:
            print("    - " + p)
    return problems


def validate_all(tasks_dir: Path) -> int:
    pairs = discover(tasks_dir)
    total = 0
    for p in pairs:
        total += len(validate_pair(p))
    print(f"\n{len(pairs)} pairs, {total} problems")
    return total
