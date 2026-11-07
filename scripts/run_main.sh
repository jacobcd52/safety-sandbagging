#!/bin/bash
# Main run: 10 seeds per arm, Flex tier. Re-run this script to resume after an interruption.
cd /home/user/safety-sandbagging
exec python -m sandbag_eval.cli run --out runs/main_s10 --agent-model gpt-6-astra --judge-model gpt-6-sol \
  --seeds 10 --max-turns 1 --workers 16 --service-tier flex --resume \
  --pairs classifier_pipeline data_filter eval_harness gridworld_rl linter_organism probe_training reward_model search_loop transcript_monitor
