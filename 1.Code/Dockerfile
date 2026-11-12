# Inference environment for the Track B delivery.
# The public base weight and the videos are not baked into the image; mount them at /data.
#
#   docker build -t trackb-from0863 .
#   docker run --gpus all -it --rm \
#     -v /host/base-4b:/data/base-4b \
#     -v /host/quantiphy/videos:/data/videos \
#     -v /host/stage:/work \
#     trackb-from0863 \
#     python -X utf8 /workspace/package/verification/prepare_runtime.py \
#       --work-dir /work/stage --base-dir /data/base-4b --videos-dir /data/videos

FROM python:3.11.16-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    FORCE_QWENVL_VIDEO_READER=decord \
    OMP_NUM_THREADS=4

# libgomp1 for torch/decord, ffmpeg for video decoding.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Exact versions read from the original inference server on 2026-10-28
# (verification/requirements-resolved.txt). torch is the CUDA 12.8 build.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.10.0 \
    && pip install --no-cache-dir \
        transformers==5.11.0 \
        peft==0.20.0 \
        accelerate==1.15.0 \
        numpy==2.2.6 \
        decord==0.6.0 \
        jinja2==3.1.6 \
        qwen-vl-utils==0.0.14

WORKDIR /workspace
COPY . /workspace/package

CMD ["bash"]
