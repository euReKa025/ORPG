# ORPG: Reconciling Multiple Reward Objectives through Objective-wise Policy Gradients

Code for **ORPG**, a multi-reward policy optimization method that constructs a separate clipped policy objective for each reward and reconciles the resulting policy gradients.

**Paper:** <!-- Add the paper URL here. -->

## Overview

Reward aggregation can hide disagreement between objectives before the policy update. ORPG keeps the objectives separate through differentiation, then combines their gradients using their inner product and relative norms.

For two active gradients, the update uses:

- **Conflicting gradients:** symmetric PCGrad for Helpfulness–Safety, or correctness-priority projection for Math.
- **Compatible gradients:** interpolation between the ordinary sum and a partially normalized direction, with the original sum's norm preserved.
- **Inactive objectives:** bypass coordination when at most one objective is active.

The default compatible branch uses `q = 0.5` and `lambda = 0.25`. Shared KL/regularization is added once after policy-gradient reconciliation, followed by the optimizer's gradient clipping and update.

This repository contains the method, the two main training configurations, data preparation, evaluation tools, and CPU tests. Model weights and datasets are downloaded separately. The Python namespace `cw_grpo` is retained for compatibility with the training implementation.

## Installation

### Core implementation and CPU tests

```bash
git clone https://github.com/euReKa025/ORPG.git
cd ORPG
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,data]'
python -m pytest -q
```

### GPU training and generation

The training implementation uses **verl**, FSDP, and vLLM. The experiment stack uses Linux, Python 3.12, PyTorch 2.9.0, vLLM 0.12.0, and Transformers 4.57.6. Install the pinned verl revision and GPU dependencies in a CUDA 12 environment:

```bash
bash scripts/setup_verl.sh
```

The script uses verl revision `b9d71f9a84ef89ec7f5a946cd277b35165a3daae`. FlashAttention must match your CUDA/PyTorch ABI. The supplied full-parameter configurations target **8 H200 GPUs**: Math uses all eight for policy training; HS uses four policy GPUs and four dual-reward-model workers. Smaller hardware requires adjusting memory/batching settings and is not validated by this release.

## Data and models

The policy initialization is [`Qwen/Qwen3-4B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507).

| Scenario | Training data | Rewards | Evaluation |
|---|---|---|---|
| Math | DeepScaleR, with normalized MATH benchmark overlaps excluded | Correctness and binary length reward (`tau = 4000`) | AIME-24, AMC-22-23, MATH, Minerva-Math, OlympiadBench |
| Helpfulness–Safety | Alpaca prompts from the GD²PO safe-alignment data, excluding 512 calibration prompts | Artessay SafeRLHF reward and cost models, in shared calibrated coordinates | Alpaca, HH-RLHF, PKU-SafeRLHF; 17,243 prompts |

See **Prepare inputs** below for preprocessing, reward-model compatibility checks, and calibration. Set paths to your own downloaded models and prepared data; no cluster-specific paths are required.

## Train

Both configurations start independently from Base, use training seed 42, and save the final checkpoint at step 100. The launcher refuses to reuse an existing run directory.

### Math

```bash
python scripts/train.py \
  --scenario math --run-name orpg-math-s42 \
  --model /path/to/Qwen3-4B-Instruct-2507 \
  --train-data /path/to/math/train.parquet \
  --valid-data /path/to/math/tuning_dev.parquet
```

### Helpfulness–Safety

```bash
python scripts/train.py \
  --scenario hs --run-name orpg-hs-s42 \
  --model /path/to/Qwen3-4B-Instruct-2507 \
  --train-data /path/to/hs/train/policy_train.parquet \
  --valid-data /path/to/hs/calibration/calibration_512.parquet \
  --useful-model /path/to/Qwen2.5-7B-SafeRLHF-RM \
  --harmless-model /path/to/Qwen2.5-7B-SafeRLHF-CM \
  --calibration /path/to/calibration_manifest.json
```

The required validation dataset is constructed by verl, but online validation is disabled (`test_freq=-1`, `val_before_train=false`). Checkpoints are written to `checkpoints/<run-name>/global_step_100/actor/huggingface`; launch records are stored in `outputs/<run-name>`.

Use `--dry-run` to inspect the launch command without loading models or starting GPUs. Use `--resolve` to resolve the full Hydra configuration in the installed GPU environment. `--output-dir` changes the artifact root. Explicit overrides are supported, for example:

```bash
# Add this to a training command to disable compatible-gradient coordination:
--override actor_rollout_ref.actor.policy_loss.objective_wise_positive_strength=0
```

The configuration files contain the complete overrides. Key defaults are:

| Setting | Math | HS |
|---|---:|---:|
| Optimizer steps | 100 | 100 |
| Learning rate | 1e-6 | 2e-6 |
| Prompt batch size | 512 | 512 |
| Rollouts per training prompt | 8 | 4 |
| PPO mini-batch size | 64 | 128 |
| Prompt / response token limits | 1024 / 8000 | 512 / 1024 |
| Training temperature / top-p | 1.0 / 1.0 | 0.7 / 1.0 |
| PPO clip (low / high) | 0.2 / 0.28 | 0.2 / 0.28 |
| Reference KL coefficient | 0.0005 | 0 |
| `q` / `lambda` | 0.5 / 0.25 | 0.5 / 0.25 |

## Evaluate

Fix a checkpoint, then run generation seeds **42, 43, 44**. Each HS seed uses full-dataset **mean@1**. Each Math seed uses **four samples per prompt** and budgets **2048, 4096, 8192**. Math Pass@1 is the mean correctness of the samples, not best-of-four accuracy.

See **Evaluation commands** below for generation, scoring, budget views, and aggregation. The tools preserve per-dataset/per-benchmark results and raw outputs. Aggregate seed-level metrics as **mean ± sample standard deviation (`ddof=1`)**; this measures evaluation-seed variation for a fixed checkpoint.

## Code map

| File | Purpose |
|---|---|
| `src/cw_grpo/positive_gradient_reconciliation.py` | Compatible-gradient direction and norm preservation |
| `src/cw_grpo/gradient_reconciliation.py` | Global Gram statistics, conflict projection, and inactive-objective handling |
| `src/cw_grpo/adapters/verl_objective_wise.py` | Per-reward advantages and task-specific routing |
| `src/cw_grpo/adapters/verl_objective_wise_engine.py` | Separate objective backward passes, reconciliation, and shared regularization |
| `src/cw_grpo/adapters/verl_objective_wise_runtime.py` | verl worker/engine integration |
| `configs/{math,hs}.yaml` | Main training configurations |
| `scripts/` | Portable launch, preparation, and evaluation tools |

The extracted implementation and portable launcher are covered by CPU tests. This release has not been rerun end-to-end on GPUs; CUDA installation and memory sizing must be checked on the target machine.

## Prepare inputs

Run preparation on a machine with network access. Download the Base policy and the original reward models to local directories:

```bash
hf download Qwen/Qwen3-4B-Instruct-2507 --local-dir models/Qwen3-4B-Instruct-2507
hf download Artessay/Qwen2.5-7B-SafeRLHF-RM --local-dir models/Qwen2.5-7B-SafeRLHF-RM
hf download Artessay/Qwen2.5-7B-SafeRLHF-CM --local-dir models/Qwen2.5-7B-SafeRLHF-CM
```

For HS, fetch the upstream source and materialize the deterministic train/calibration split:

```bash
git clone https://github.com/Qwen-Applications/GD2PO.git third_party/GD2PO
git -C third_party/GD2PO checkout f1ad765bc9a330e6cf387f95e9c1e5a6c4bb2d02
python scripts/prepare_helpfulness_safety_data.py \
  --official-repo third_party/GD2PO \
  --revision f1ad765bc9a330e6cf387f95e9c1e5a6c4bb2d02 \
  --output-root data/helpfulness_safety
```

Before first use of the reward models, validate their scoring implementation and generate/score the shared Base calibration answers. These are GPU operations:

```bash
export POLICY="$PWD/models/Qwen3-4B-Instruct-2507"
export USEFUL="$PWD/models/Qwen2.5-7B-SafeRLHF-RM"
export HARMLESS="$PWD/models/Qwen2.5-7B-SafeRLHF-CM"
export CAL="$PWD/outputs/calibration/hs-qwen3-base-s42-p512-n4-v3"
python scripts/validate_helpfulness_safety_compatibility.py \
  --policy-model "$POLICY" --rm-model "$USEFUL" --cm-model "$HARMLESS" \
  --official-reference-root third_party/GD2PO --output outputs/compatibility.json
for action in generate score; do
  python scripts/calibrate_helpfulness_safety.py "$action" \
    --policy-model "$POLICY" --rm-model "$USEFUL" --cm-model "$HARMLESS" \
    --input-parquet data/helpfulness_safety/calibration/calibration_512.parquet \
    --compatibility-summary outputs/compatibility.json --output-dir "$CAL"
done
```

The calibration uses 512 held-out prompts and four Base responses per prompt. Keep the resulting `calibration_manifest.json` fixed across methods. The directory name above is the calibration identifier expected by the original HS evaluator. Freshly generated calibration responses need not be numerically identical across hardware/software revisions; do not mix calibration files in a comparison.

For Math, download the following Parquet files using `hf download --repo-type dataset --local-dir <directory> <dataset> <file>`. The original preparation utility expects this layout under `data/math/raw/`:

| Dataset | File | Local directory |
|---|---|---|
| `HuggingFaceH4/aime_2024` | `data/train-00000-of-00001.parquet` | `aime_2024_hf4` |
| `AI-MO/aimo-validation-amc` | `data/train-00000-of-00001.parquet` | `amc_2022_2023_aimo` |
| `zwhe99/MATH` | `data/test-00000-of-00001.parquet` | `math_zwhe99` |
| `zwhe99/minerva_math` | `data/test-00000-of-00001.parquet` | `minerva_math_zwhe99` |
| `zwhe99/OlympiadBench` | `data/test-00000-of-00001.parquet` | `olympiadbench_zwhe99_eval` |

Keep the download metadata; the preparation manifest records the source revisions. Pin the same dataset revisions across a comparison. The loader checks the expected 6,060 total benchmark items and rejects row-count drift.

```bash
python scripts/prepare_math_stage_a_eval.py
hf download agentica-org/DeepScaleR-Preview-Dataset deepscaler.json \
  --repo-type dataset --revision 4500022f6cb0a1456ea5eabe2ff366d4610ee339 \
  --local-dir data/math/raw/deepscaler
python scripts/prepare_math_stage_a.py \
  --raw-json data/math/raw/deepscaler/deepscaler.json \
  --math-eval-parquet data/math/raw/math_zwhe99/data/test-00000-of-00001.parquet \
  --benchmark-overlap-policy exclude \
  --output-dir data/math/processed/train
```

## Evaluation commands

### HS: three seeds, full mean@1

With the reward-model paths and calibration above, run the following on eight GPUs. Each seed is generated once, followed by reward scoring and full-coverage checks:

```bash
for seed in 42 43 44; do
  python scripts/evaluate_helpfulness_safety.py run-pipeline \
    --method orpg --seed "$seed" \
    --policy-model checkpoints/orpg-hs-s42/global_step_100/actor/huggingface \
    --useful-model "$USEFUL" --harmless-model "$HARMLESS" \
    --data-root data/helpfulness_safety \
    --calibration-manifest "$CAL/calibration_manifest.json" \
    --compatibility-summary outputs/compatibility.json \
    --source-training-experiment orpg-hs-s42 \
    --output-dir "outputs/hs-eval-g${seed}"
done
```

Do not reuse an existing output directory. The evaluator reserves one GPU per shard and uses `seed + shard_index`; preserve the default eight shards when matching its sampling protocol. Per-seed summaries are written to `summary/mean1-summary.json`.

### Math: four samples and three token-prefix budgets

Set `ORPG_MODEL` to the Base model directory and use the fixed step100 HF checkpoint below. `ORPG_ROOT` defaults to this checkout and can be set to a different artifact root. Generation runs eight vLLM replicas and writes exact returned token IDs. The 2,048/4,096 views are prefixes of the 8,192-token generations, not best-of-four selection or independent shorter draws.

```bash
export ORPG_MODEL="$PWD/models/Qwen3-4B-Instruct-2507"
checkpoint="$PWD/checkpoints/orpg-math-s42/global_step_100/actor/huggingface"
for seed in 42 43 44; do
  out="$PWD/outputs/math-eval-g${seed}"
  python scripts/generate_stage_a_eval.py --action generate \
    --experiment-name "math-eval-g${seed}" --model-path "$checkpoint" \
    --input-parquet data/math/processed/stage_a_eval_v1/aime_24.parquet \
    --input-parquet data/math/processed/stage_a_eval_v1/amc_22_23.parquet \
    --input-parquet data/math/processed/stage_a_eval_v1/math.parquet \
    --input-parquet data/math/processed/stage_a_eval_v1/minerva_math.parquet \
    --input-parquet data/math/processed/stage_a_eval_v1/olympiadbench.parquet \
    --output-parquet "$out/generations.parquet" \
    --samples-per-prompt 4 --base-seed "$seed" --max-new-tokens 8192 \
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
done
```

Per-seed Math summaries use equal weighting over the five benchmarks. HV uses actual mean response length and the reference length 8,192. To aggregate a chosen metric across seeds:

```bash
python scripts/summarize_seeds.py \
  --checkpoint checkpoints/orpg-math-s42/global_step_100/actor/huggingface \
  --input 42=outputs/math-eval-g42/summary.json \
  --input 43=outputs/math-eval-g43/summary.json \
  --input 44=outputs/math-eval-g44/summary.json \
  --metric macro_average.pass_at_1_at_8192 \
  --metric macro_average.average_completion_tokens_at_8192 \
  --metric macro_average.pareto_hypervolume \
  --output outputs/math-three-seeds.json
```

For HS, use the per-seed `summary/mean1-summary.json` files and the fields `macro.raw_useful_mean_at_1` and `macro.raw_harmless_mean_at_1`. Only aggregate summaries from the same checkpoint, data, calibration, and decoding protocol.

## Acknowledgements

Built on [verl](https://github.com/verl-project/verl). We use the data and reward-model interfaces from [GD²PO](https://github.com/Qwen-Applications/GD2PO), compare against reward-decoupled optimization introduced by [GDPO](https://github.com/NVlabs/GDPO), and use [Math-Verify](https://github.com/huggingface/Math-Verify) for mathematical answer checking. See [third-party notices](THIRD_PARTY_NOTICES.md).

## Citation

Paper link and citation will be added when available.
