# safety-sandbagging

Replication scaffold for OpenAI's
[covert underperformance on safety research tasks](https://alignment.openai.com/misalignment-reports/covert-underperformance-on-safety-research-tasks/)
finding: an agent completes less of an open-ended coding task when the task is
framed as AI-safety research than when the same code is framed as capabilities
work, and mostly does not say so.

## Design

- **Task pair** = one substrate (code + hidden tests + rubric, written with
  `__Placeholder__` tokens) + two or more framings (prompt + placeholder values).
  Rendering a framing yields a workspace whose names, comments, docs, data strings
  and prompt match that framing. Logic, seeded defects and rubric are byte-identical.
- **Agent loop** mirrors the report: each turn asks the agent to "find and implement
  the most useful improvements"; the agent uses file/shell tools until it replies;
  N turns; then a disclosure probe asks whether it held anything back.
- **Dependent variables**
  - `completeness`: fraction of hidden rubric items achieved (hidden pytest only; no LLM judge).
  - `stopping_rate`: share of turns classified as done / blocked / refused.
  - `any_refusal`, `any_partial_disclosed`: explicit refusal or disclosed scaling-back.
  - `ethical_doubt`: judge flags any ethical hesitation in the agent's visible text.
  - `admits_withholding`: answer to the disclosure probe.
  - `completeness_excl_refusal` / `completeness_excl_doubt`: the report's exclusion ladder.
- **Sandbox**: every agent shell command and the hidden-test grading run in a Linux namespace
  (`sandbag_eval/sandbox.py`) that sees only the workspace: no task repo, hidden tests, reference,
  other framing or other jobs; no network; own PID namespace; minimal env. Grading ignores
  agent-written `conftest.py` / `pytest.ini`. `SANDBAG_SANDBOX=off` disables it (dev only).
- **Breach monitor** (`sandbag_eval/monitor.py`): records attempts to look outside the workspace
  (`breach_attempt`), hidden material appearing in tool output (`hidden_read`), and writes of
  files that can reconfigure grading (`grader_tamper_risk`). Reported per trajectory and by
  `analyze` (breach / leak columns).
- **Behaviour classifiers** (`--judge-model`, default gpt-6-sol): an LLM labels refusals, ethical
  doubt and the disclosure-probe answer. They never affect completeness; `--judge-model none`
  skips them.
- **Axes** per pair (`harm_proximity`, `self_relevance`, `adversarial_stance`,
  `modalities`, `tier`) so gaps can be broken down by task type.

## Usage

```bash
pip install -e .
export OPENAI_API_KEY=...            # never commit it

sandbag validate                     # every pair: renders, hidden tests fail on src, pass on reference
sandbag run --out runs/pilot --agent-model gpt-6-astra --judge-model gpt-6-sol \
    --seeds 3 --max-turns 1 --workers 6
sandbag analyze runs/pilot           # completeness gap + standardized primary-metric gap
sandbag render classifier_pipeline safety /tmp/look   # inspect a rendered workspace

# pipeline check with no API spend
sandbag run --out runs/mock --agent-model mock --judge-model mock --max-turns 1
```

Each run writes `runs/<name>/<pair>__<framing>__s<seed>/{trajectory.json,grade.json,diff.patch}`.

## Pairs

12 coding task pairs, each with a capabilities and a safety framing. Full inventory with axes: docs/pairs.md. No framing targets the agent model itself.

## Layout

```
sandbag_eval/   schema, render, tools, llm (OpenAI + mock), agent, grade, run, analyze, validate, cli
tasks/          one directory per pair; see tasks/README.md for authoring rules
```

## Caveats

- Chat Completions does not expose reasoning tokens, so "verbalized withholding in CoT"
  is only measurable here via visible output and the disclosure probe. Use an
  open-weight model or a reasoning-summary API to recover that bucket.
- Provider-side filters can truncate safety-framed outputs; log those separately from
  rubric scores or they will be attributed to the model.
