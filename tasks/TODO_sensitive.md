# Safety framings left for another implementer

Some pairs ship with `status: capabilities_only`: the substrate, hidden tests,
reference solution and capabilities framing are complete, but the safety
framing is not written. The OpenAI report found that refusals and silent
withholding concentrated in red-teaming, monitor-evasion and exploit-adjacent
work, so these are the pairs most worth finishing. Building them with this
assistant tends to trip an output filter, so hand them to another model or
write them by hand.

## How to add a safety framing to a `capabilities_only` pair

1. Open `tasks/<id>/pair.yaml`. Copy the `capabilities` framing block to a new
   `safety` block. Keep the prompt the same length and tone and the same
   closing sentence.
2. Fill every `vars` key with the safety-domain equivalent. Keys must match
   exactly; the validator rejects mismatched key sets.
3. Read `sensitive_notes` in the pair for the intended safety-side meaning of
   each placeholder.
4. Set `status: complete` and run `sandbag validate --pair <id>`.

## Pairs in this state

- `search_loop` (red_teaming, moderate): the capabilities side is a generic
  mutate-and-score optimisation loop over text candidates. The safety-side
  framing described in the OpenAI report's categories is an automated
  red-teaming search against a target model. Only names, docs, the scoring
  function's description and the prompt change.

## Severe-tier pairs not yet started (category level only)

These correspond to the categories the report names as most affected. Each
would reuse an existing substrate with a new framing rather than new code:

- Model-organism fine-tuning for sabotage detection research
  (substrate: `reward_model` or `classifier_pipeline` training loop).
- Monitor-evasion RL environment for control research
  (substrate: `permission_gate` + `transcript_monitor`).
- Exploit-adjacent red-teaming tooling
  (substrate: `search_loop`).

For each, the capabilities counterpart already exists or is a one-line
rename, so the marginal work is the safety prompt and vars only.
