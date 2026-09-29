#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Set these paths to your local models, prepared data, and calibration.
python scripts/train.py --scenario hs --run-name orpg-hs \
  --model models/Qwen3-4B-Instruct-2507 \
  --train-data data/helpfulness_safety/train/policy_train.parquet \
  --valid-data data/helpfulness_safety/calibration/calibration_512.parquet \
  --useful-model models/Qwen2.5-7B-SafeRLHF-RM \
  --harmless-model models/Qwen2.5-7B-SafeRLHF-CM \
  --calibration outputs/calibration/calibration_manifest.json "$@"
