# ORPG: Reconciling Multiple Reward Objectives through Objective-wise Policy Gradients

[Paper](https://arxiv.org/abs/2609.34985)

ORPG constructs a separate clipped policy objective for each reward and reconciles the resulting gradients into a joint policy update.

## Installation

Use a Linux GPU environment with Python 3.12 and CUDA 12.

```bash
git clone https://github.com/euReKa025/ORPG.git
cd ORPG
bash scripts/setup_verl.sh
```

## Training

Download the models and prepare the data using the utilities in `scripts/`. Set the local model, data, and calibration paths in the launch scripts below. Training settings are in `configs/math.yaml` and `configs/hs.yaml`.

```bash
# Math
bash scripts/train_math.sh

# Helpfulness–Safety
bash scripts/train_hs.sh
```

## Evaluation

Set the checkpoint and input paths in the evaluation scripts, then run:

```bash
# Math
bash scripts/eval_math.sh

# Helpfulness–Safety
bash scripts/eval_hs.sh
```

## Acknowledgements

Built on [verl](https://github.com/verl-project/verl), with components from [GD²PO](https://github.com/Qwen-Applications/GD2PO), [GDPO](https://github.com/NVlabs/GDPO), and [Math-Verify](https://github.com/huggingface/Math-Verify). See [third-party notices](THIRD_PARTY_NOTICES.md).

## Citation

```bibtex
@misc{fang2026orpgreconcilingmultiplereward,
      title={ORPG: Reconciling Multiple Reward Objectives through Objective-wise Policy Gradients},
      author={Shicheng Fang and Yiwen Zhao and Wenbo Tian and Jiahao Lu and Yining Zheng and Yuxin Wang and Xipeng Qiu},
      year={2026},
      eprint={2609.34985},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2609.34985},
}
```
