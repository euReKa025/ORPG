# Third-party notices

- **verl**: https://github.com/verl-project/verl, revision `b9d71f9a84ef89ec7f5a946cd277b35165a3daae`, Apache-2.0. The full framework is fetched separately. The adapters integrate with its training/runtime APIs.
- **GD²PO**: https://github.com/Qwen-Applications/GD2PO, revision `f1ad765bc9a330e6cf387f95e9c1e5a6c4bb2d02`, Apache-2.0. The GD²PO-Hard estimator adapter follows its conflict filtering and query weighting. Safe-alignment data and reward-model reference code are fetched separately.
- **GDPO**: https://github.com/NVlabs/GDPO, Apache-2.0. The baseline estimator is provided by the pinned verl installation.
- **Transformers / Qwen**: model classes are imported from Hugging Face Transformers, Apache-2.0. Model weights are not bundled and retain their own licenses.
- **Math-Verify**: https://github.com/huggingface/Math-Verify, Apache-2.0, installed as a dependency.

No model weights, datasets, credentials, private cluster configuration, or experiment archives are included. Downloaded dependencies and datasets retain their original license terms and attribution requirements. An open-source license for the original ORPG code has not been selected for this private staging release.
