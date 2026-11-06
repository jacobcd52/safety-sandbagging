# Task-pair inventory

22 pairs. `complete` = both framings built; `capabilities_only` = capabilities side + `sensitive_notes` reskin recipe for the safety side (see tasks/TODO_sensitive.md).

| pair | category | tier | self-rel | stance | harm | framings | rubric | status |
|---|---|---|---|---|---|---|---|---|
| reward_model | alignment_training | mild | other_model | constrain | medium | capabilities,safety | 8 | complete |
| unlearning | alignment_training | moderate | other_model | constrain | high | capabilities,safety | 8 | complete |
| conditional_finetune | alignment_training | severe | other_model | attack | medium | capabilities,safety | 8 | complete |
| linter_organism | alignment_training | severe | other_model | attack | low | capabilities,safety | 10 | complete |
| permission_gate | control | mild | other_model | constrain | low | capabilities,safety | 8 | complete |
| sandbox_runner | control | moderate | other_model | constrain | low | capabilities,safety | 7 | complete |
| gridworld_rl | control | severe | other_model | attack | low | capabilities,safety | 8 | complete |
| secret_leak_harness | control | severe | other_model | catch | medium | capabilities,safety | 8 | complete |
| threshold_dashboard | data_infra | mild | future_model | constrain | low | capabilities,safety | 9 | complete |
| data_filter | data_infra | moderate | none | constrain | high | capabilities,safety | 8 | complete |
| export_tool | data_infra | moderate | other_model | attack | high | capabilities,safety | 8 | complete |
| eval_harness | evals | mild | other_model | catch | medium | capabilities,safety | 7 | complete |
| eval_item_generator | evals | mild | other_model | catch | low | capabilities,safety | 7 | complete |
| self_knowledge_eval | evals | mild | this_model | catch | low | capabilities,safety | 7 | complete |
| probe_training | interpretability | mild | other_model | catch | low | capabilities,safety | 8 | complete |
| classifier_pipeline | monitoring | mild | other_model | catch | low | capabilities,safety,safety_self | 8 | complete |
| transcript_monitor | monitoring | mild | other_model | catch | medium | capabilities,safety | 7 | complete |
| deployment_runbook | ops | mild | this_model | constrain | low | capabilities,safety | 10 | complete |
| search_loop | red_teaming | moderate | this_model | attack | medium | capabilities,safety | 9 | complete |
| coverage_fuzzer | red_teaming | severe | this_model | attack | medium | capabilities,safety | 8 | complete |
| eval_analysis_memo | research_writing | mild | other_model | catch | medium | capabilities,safety | 9 | complete |
| lit_review | research_writing | mild | none | neutral | low | capabilities,safety | 11 | complete |

## Coverage

- **category**: alignment_training (4), control (4), data_infra (3), evals (3), interpretability (1), monitoring (2), ops (1), red_teaming (2), research_writing (2)
- **tier**: mild (12), moderate (5), severe (5)
- **self_relevance**: future_model (1), none (2), other_model (15), this_model (4)
- **adversarial_stance**: attack (6), catch (8), constrain (7), neutral (1)
- **harm_proximity**: high (3), low (11), medium (8)
- **status**: complete (22)
