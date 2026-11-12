# QuantiPhy Challenge — Track B

This repository contains the code, LoRA adapters, method description, and reproduction materials for JX intelligence’s Track B submission.

## Documentation

| Material | Description |
| --- | --- |
| [Code and artifacts](1.Code/) | Training and inference code, adapters, configurations, submission, and verification records |
| [Setup and reproduction](2.README.md) | Environment, dependencies, hardware, runtime, and reproduction instructions |
| [Models and settings](3.Models.md) | Base model, adapters, training teacher, and generation settings |
| [Method description](4.Method.md) | Description of the training and inference method |
| [Prediction independence declaration](5.Declaration.md) | Declaration of independent prediction for each item |

## Model weights

The final and initializing LoRA adapters are included through Git LFS.
After cloning this repository, run the following from its root directory:

```bash
git lfs install
git lfs pull

The public base model weights and video files are not bundled.
Their exact revisions, SHA-256 hashes, and retrieval information are
documented in the reproduction guide (2.README.md).

## Reproduction

Start with the setup and reproduction guide (2.README.md).

Use the final LoRA adapter for inference. The initializing adapter is
provided as a training input.

## Licenses

The base model card declares Apache-2.0. Dataset, video, and other
third-party materials remain subject to their respective licenses.
See the bundled model and dataset documentation for details.
