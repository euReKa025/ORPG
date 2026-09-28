#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
revision=b9d71f9a84ef89ec7f5a946cd277b35165a3daae
if [[ ! -d third_party/verl ]]; then
  git clone https://github.com/verl-project/verl.git third_party/verl
  git -C third_party/verl checkout "$revision"
fi
if [[ "$(git -C third_party/verl rev-parse HEAD)" != "$revision" ]]; then
  echo 'Existing verl checkout differs from the supported revision; use a separate clean checkout.' >&2
  exit 1
fi
python -m pip install -r requirements/train.txt -e 'third_party/verl[vllm]' -e '.[data,dev]'
# The experiment environment used FlashAttention 2.8.3. Build for your CUDA/PyTorch ABI.
python -m pip install flash-attn==2.8.3 --no-build-isolation
