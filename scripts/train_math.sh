#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Set these paths to your local model and prepared data.
python scripts/train.py --scenario math --run-name orpg-math \
  --model models/Qwen3-4B-Instruct-2507 \
  --train-data data/math/processed/train/train.parquet \
  --valid-data data/math/processed/train/tuning_dev.parquet "$@"
