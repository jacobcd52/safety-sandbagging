"""Run pairs x framings x seeds; write trajectories and grades under runs/<run_name>/."""
from __future__ import annotations

import json
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .agent import run_agent
from .grade import grade
from .llm import LLM
from .render import render_text, render_tree
from .schema import TaskPair, discover, load_pair
from .monitor import Monitor
from .tools import Workspace


def render_workspace(pair: TaskPair, framing: str, dst: Path) -> None:
    vars = pair.framings[framing].vars
    problems = render_tree(pair.src_dir, dst, vars, strict=True)
    if problems:
        raise ValueError(f"{pair.id}/{framing}: " + "; ".join(problems))


def run_one(pair: TaskPair, framing: str, seed: int, agent_llm: LLM, judge_llm: LLM, out_dir: Path,
            max_turns: int, max_tool_calls: int | None, keep_workspace: bool) -> dict:
    tag = f"{pair.id}__{framing}__s{seed}"
    out = out_dir / tag
    out.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f"sb_{tag}_"))
    base_ws, final_ws = tmp / "base", tmp / "work"
    render_workspace(pair, framing, base_ws)
    shutil.copytree(base_ws, final_ws)
    fr = pair.framings[framing]
    traj = run_agent(
        Workspace(final_ws, monitor=Monitor(pair, framing, final_ws)), agent_llm,
        first_prompt=render_text(fr.prompt, fr.vars),
        continue_prompt=render_text(pair.continue_prompt, fr.vars),
        system=render_text(fr.system, fr.vars) if fr.system else None,
        max_turns=max_turns, max_tool_calls_per_turn=max_tool_calls,
        pair_id=pair.id, framing=framing, seed=seed,
    )
    (out / "trajectory.json").write_text(json.dumps(traj.to_json(), indent=1))
    g = grade(pair, framing, base_ws, final_ws, traj, judge_llm)
    (out / "grade.json").write_text(json.dumps(g.to_json(), indent=1))
    (out / "diff.patch").write_text(__import__("sandbag_eval.grade", fromlist=["workspace_diff"]).workspace_diff(base_ws, final_ws, max_chars=10**7))
    if keep_workspace:
        shutil.copytree(final_ws, out / "workspace", dirs_exist_ok=True)
    shutil.rmtree(tmp, ignore_errors=True)
    return g.to_json()


def run_suite(tasks_dir: Path, out_dir: Path, agent_model: str, judge_model: str, pairs: list[str] | None,
              framings: list[str] | None, seeds: int, max_turns: int, max_tool_calls: int | None,
              reasoning_effort: str | None, workers: int, keep_workspace: bool) -> list[dict]:
    all_pairs = discover(tasks_dir)
    if pairs:
        all_pairs = [p for p in all_pairs if p.id in pairs]
    jobs = []
    for p in all_pairs:
        if p.status == "stub":
            continue
        for fname in p.framings:
            if framings and fname not in framings:
                continue
            for s in range(seeds):
                jobs.append((p, fname, s))
    out_dir.mkdir(parents=True, exist_ok=True)
    agent_llm = LLM(agent_model, reasoning_effort=reasoning_effort)
    judge_llm = None if judge_model in (None, "none") else LLM(judge_model)
    results = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run_one, p, f, s, agent_llm, judge_llm, out_dir, max_turns, max_tool_calls, keep_workspace): (p.id, f, s)
                for p, f, s in jobs}
        for fut in as_completed(futs):
            pid, f, s = futs[fut]
            try:
                r = fut.result()
                if r.get("error"):
                    print(f"[ERR ] {pid} {f} seed={s}: {r['error'][:200]}")
                    results.append(r)
                    continue
                print(f"[done] {pid} {f} seed={s} completeness={r['completeness']:.2f} stopping={r['stopping_rate']:.2f}")
                results.append(r)
            except Exception as e:  # noqa: BLE001
                print(f"[FAIL] {pid} {f} seed={s}: {type(e).__name__}: {e}")
                results.append({"pair_id": pid, "framing": f, "seed": s, "error": str(e)})
    (out_dir / "results.jsonl").write_text("\n".join(json.dumps(r) for r in results) + "\n")
    return results
