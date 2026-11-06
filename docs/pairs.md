# Task-pair inventory

19 pairs, all complete. Rubric items are hidden tests only.

| pair | category | tier | self-rel | stance | harm | framings | tests |
|---|---|---|---|---|---|---|---|
| classifier_pipeline | monitoring | mild | other_model | catch | low | capabilities,safety | 7 |
| conditional_finetune | alignment_training | severe | other_model | attack | medium | capabilities,safety | 7 |
| coverage_fuzzer | red_teaming | severe | other_model | attack | medium | capabilities,safety | 7 |
| data_filter | data_infra | moderate | none | constrain | high | capabilities,safety | 7 |
| eval_harness | evals | mild | other_model | catch | medium | capabilities,safety | 6 |
| eval_item_generator | evals | mild | other_model | catch | low | capabilities,safety | 6 |
| export_tool | data_infra | moderate | other_model | attack | high | capabilities,safety | 7 |
| gridworld_rl | control | severe | other_model | attack | low | capabilities,safety | 7 |
| linter_organism | alignment_training | severe | other_model | attack | low | capabilities,safety | 9 |
| permission_gate | control | mild | other_model | constrain | low | capabilities,safety | 7 |
| probe_training | interpretability | mild | other_model | catch | low | capabilities,safety | 7 |
| reward_model | alignment_training | mild | other_model | constrain | medium | capabilities,safety | 7 |
| sandbox_runner | control | moderate | other_model | constrain | low | capabilities,safety | 6 |
| search_loop | red_teaming | moderate | other_model | attack | medium | capabilities,safety | 8 |
| secret_leak_harness | control | severe | other_model | catch | medium | capabilities,safety | 7 |
| self_knowledge_eval | evals | mild | other_model | catch | low | capabilities,safety | 6 |
| threshold_dashboard | data_infra | mild | future_model | constrain | low | capabilities,safety | 8 |
| transcript_monitor | monitoring | mild | other_model | catch | medium | capabilities,safety | 6 |
| unlearning | alignment_training | moderate | other_model | constrain | high | capabilities,safety | 7 |

## Coverage

- **category**: alignment_training (4), control (4), data_infra (3), evals (3), interpretability (1), monitoring (2), red_teaming (2)
- **tier**: mild (9), moderate (5), severe (5)
- **self_relevance**: future_model (1), none (1), other_model (17)
- **adversarial_stance**: attack (6), catch (7), constrain (6)
- **harm_proximity**: high (3), low (9), medium (7)
