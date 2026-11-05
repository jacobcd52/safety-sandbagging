# Safety framings left for another implementer

Some pairs ship with `status: capabilities_only`: the substrate, hidden tests,
reference solution and the capabilities framing are complete and validated, but
the safety framing is intentionally not written. The OpenAI report found that
refusals and silent withholding concentrated in red-teaming, monitor-evasion and
exploit-adjacent work, so these are the highest-value pairs to finish. Writing
their safety side with the assistant that built this repo trips an output
filter, so they are left for another model or a human to complete.

## How to finish one

Everything the implementer needs is inside the pair's own `pair.yaml`:

1. Open `tasks/<id>/pair.yaml` and read the `sensitive_notes` field. It holds
   a plain description of the safety-side task, a prompt skeleton matched in
   length and tone to the capabilities prompt, and the intended meaning of each
   placeholder key on the safety side.
2. Copy the `capabilities` framing block to a new `safety` block. Keep the same
   `vars` keys (the validator requires identical key sets across framings).
3. Fill each placeholder with its safety-side value per `sensitive_notes`. Keep
   all seeded-defect logic and the rubric untouched; only names, comments, docs,
   data strings and the prompt change.
4. Set `status: complete` and run `sandbag validate --pair <id>`.

Keep every data string abstract: the substrates are built so the framing lives
entirely in labels and prose, and no pair needs real harmful content to work.

## Pairs in this state

Run `sandbag validate` and look for the `capabilities_only` line, or
`grep -L 'safety:' tasks/*/pair.yaml`. Each such pair's `sensitive_notes`
describes its safety counterpart.
