#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Set these paths to the trained checkpoint and prepared benchmark data.
checkpoint="$PWD/checkpoints/orpg-math/global_step_100/actor/huggingface"
data="$PWD/data/math/processed/stage_a_eval_v1"
out="$PWD/outputs/orpg-math-eval"
python scripts/generate_stage_a_eval.py --action generate \
  --experiment-name orpg-math-eval --model-path "$checkpoint" \
  --input-parquet "$data/aime_24.parquet" \
  --input-parquet "$data/amc_22_23.parquet" \
  --input-parquet "$data/math.parquet" \
  --input-parquet "$data/minerva_math.parquet" \
  --input-parquet "$data/olympiadbench.parquet" \
  --output-parquet "$out/generations.parquet" \
  --samples-per-prompt 4 --max-new-tokens 8192 \
  --temperature 0.6 --top-p 0.95 --n-gpus-per-node 8 --max-model-len 10240
python scripts/materialize_stage_a_budget_views.py \
  --source-parquet "$out/generations.parquet" --model-path "$checkpoint" \
  --budget 2048 --budget 4096 --output-dir "$out/views"
for budget in 2048 4096 8192; do
  source="$out/views/budget-${budget}.parquet"
  if [[ "$budget" == 8192 ]]; then source="$out/generations.parquet"; fi
  python scripts/score_stage_a_eval.py --input-parquet "$source" \
    --output-scored-parquet "$out/scored-${budget}.parquet" \
    --output-metrics-json "$out/metrics-${budget}.json" --max-new-tokens "$budget"
done
python scripts/summarize_stage_a_prime_eval.py \
  --metrics-json "$out/metrics-2048.json" --metrics-json "$out/metrics-4096.json" \
  --metrics-json "$out/metrics-8192.json" --length-threshold 4000 \
  --output-json "$out/summary.json"
