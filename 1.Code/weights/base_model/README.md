---
license: apache-2.0
base_model:
- Qwen/Qwen3.5-4B
pipeline_tag: question-answering
---

# Code-as-World-VL-4B

Code-as-World-VL-4B (https://arxiv.org/abs/2608.27549) is a vision-language model fine-tuned for physical
understanding and quantitative reasoning over videos.

## Model details

- **Base model:** [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)
- **Weight format:** BF16 Safetensors checkpoint
- **Recommended video input:** 16 frames

## Usage

The checkpoint can be served with vLLM:

```bash
pip install "vllm==0.19.1" "transformers==5.11.0" qwen-vl-utils

vllm serve MirroS-Lab/Code-as-World-VL-4B \
  --served-model-name code-as-world-4b \
  --max-model-len 4608 \
  --gpu-memory-utilization 0.90 \
  --media-io-kwargs '{"video":{"num_frames":16,"fps":-1,"video_backend":"openpangu"}}' \
  --mm-processor-kwargs '{"do_sample_frames":false}' \
  --mm-processor-cache-gb 0 \
  --generation-config vllm
```

The server exposes an OpenAI-compatible API at `/v1`.

## Intended use

This model is intended for research on physical understanding, measurement, and
quantitative reasoning from images and videos. Model outputs may be inaccurate and
should be independently verified before use in safety-critical settings.

## License

This checkpoint is released under the Apache License 2.0. It is derived from
[Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B); users must also comply
with the terms applicable to the base model and their input data.

## Citation

