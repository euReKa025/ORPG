#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Set these paths to the trained checkpoint and prepared evaluation inputs.
python scripts/evaluate_helpfulness_safety.py run-pipeline \
  --method orpg --source-training-experiment orpg-hs \
  --policy-model checkpoints/orpg-hs/global_step_100/actor/huggingface \
  --useful-model models/Qwen2.5-7B-SafeRLHF-RM \
  --harmless-model models/Qwen2.5-7B-SafeRLHF-CM \
  --data-root data/helpfulness_safety \
  --calibration-manifest outputs/calibration/hs-qwen3-base-s42-p512-n4-v3/calibration_manifest.json \
  --compatibility-summary outputs/compatibility.json \
  --output-dir outputs/orpg-hs-eval "$@"
