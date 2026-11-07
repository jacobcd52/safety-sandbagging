from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sandbag")
    sub = ap.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("validate", help="validate task pairs")
    v.add_argument("--tasks", default="tasks")
    v.add_argument("--pair", default=None)

    r = sub.add_parser("run", help="run agent on pairs x framings x seeds and grade")
    r.add_argument("--tasks", default="tasks")
    r.add_argument("--out", required=True, help="run directory, e.g. runs/smoke")
    r.add_argument("--agent-model", default="gpt-6-astra")
    r.add_argument("--judge-model", default="gpt-6-sol",
                   help="LLM for behaviour classifiers only (refusal/doubt/disclosure); never affects completeness. 'none' skips them.")
    r.add_argument("--reasoning-effort", default=None)
    r.add_argument("--pairs", nargs="*", default=None)
    r.add_argument("--framings", nargs="*", default=None)
    r.add_argument("--seeds", type=int, default=1)
    r.add_argument("--max-turns", type=int, default=1)
    r.add_argument("--max-tool-calls", type=int, default=None,
                   help="per-turn tool-call cap; default none (how much effort the agent spends is a measured outcome)")
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--keep-workspace", action="store_true")
    r.add_argument("--service-tier", default=None, help="OpenAI service tier, e.g. 'flex' (half price, slower)")

    a = sub.add_parser("analyze", help="summarize a run directory")
    a.add_argument("run_dir")
    a.add_argument("--json", action="store_true")

    rd = sub.add_parser("render", help="render a pair's framing to a directory (for inspection)")
    rd.add_argument("pair")
    rd.add_argument("framing")
    rd.add_argument("out")
    rd.add_argument("--tasks", default="tasks")

    args = ap.parse_args(argv)
    if args.cmd == "validate":
        from .validate import validate_all, validate_pair
        from .schema import load_pair
        if args.pair:
            n = len(validate_pair(load_pair(Path(args.tasks) / args.pair)))
        else:
            n = validate_all(Path(args.tasks))
        raise SystemExit(1 if n else 0)
    if args.cmd == "run":
        from .run import run_suite
        run_suite(Path(args.tasks), Path(args.out), args.agent_model, args.judge_model, args.pairs, args.framings,
                  args.seeds, args.max_turns, args.max_tool_calls, args.reasoning_effort, args.workers, args.keep_workspace,
                  service_tier=args.service_tier)
        from .analyze import load_results, summarize, print_table
        print_table(summarize(load_results(Path(args.out))))
    if args.cmd == "analyze":
        from .analyze import load_results, summarize, print_table
        s = summarize(load_results(Path(args.run_dir)))
        if args.json:
            print(json.dumps(s, indent=1))
        else:
            print_table(s)
    if args.cmd == "render":
        from .schema import load_pair
        from .run import render_workspace
        p = load_pair(Path(args.tasks) / args.pair)
        render_workspace(p, args.framing, Path(args.out))
        print(f"rendered {args.pair}/{args.framing} -> {args.out}")


if __name__ == "__main__":
    main()
